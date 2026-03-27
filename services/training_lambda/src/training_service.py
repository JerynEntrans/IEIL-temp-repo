from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import boto3
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from shared.schema.db import ProcessName, ProcessingState
from shared.utils.db import Db
from shared.utils.s3 import sha256_bytes, parse_s3_uri
from shared.utils.sagemaker import create_or_update_endpoint, get_sagemaker_client
from shared.utils.logging import set_logging

logger = set_logging(__name__)

AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
UTC = timezone.utc
IST = ZoneInfo("Asia/Kolkata")

MODEL_DIR = os.getenv("GOAL_SEEK_MODEL_DIR", "/opt/airflow/include/model")
MODEL_PREFIX = os.getenv("GOAL_SEEK_MODEL_PREFIX", "desalter_model")
MODEL_S3_BUCKET = os.getenv("MODEL_S3_BUCKET", "ieil-raw")
MODEL_S3_PREFIX = os.getenv("MODEL_S3_PREFIX", "models").strip("/")

DEFAULT_MANIPULATED_VARS = [
    "chemical_consumption_demulsifier_ppm_unnamed_85_level_2",
    "desalter_monitoring_press_kg_cm2",
    "desalter_monitoring_w_w_temp_deg_c",
]

DEFAULT_TARGETS = [
    "desalter_monitoring_interface_level",
    "desalter_2_monitoring_interface_level",
    "desalter_salt_ptb_o_l",
    "desalter_brine_water_oil_ppm",
    "o_h_boot_water_analysis_chloride_ppm",
]


def _safe_model_dir() -> str:
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        return MODEL_DIR
    except PermissionError:
        fallback = "/tmp/model"
        os.makedirs(fallback, exist_ok=True)
        logger.warning("Model dir %s not writable, using fallback %s", MODEL_DIR, fallback)
        return fallback


def _get_s3_client():
    endpoint_url = os.getenv("AWS_ENDPOINT_URL_S3") or os.getenv("AWS_ENDPOINT_URL")
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
        region_name=os.getenv("AWS_DEFAULT_REGION", AWS_REGION),
    )


def _upload_goal_seek_model_to_s3(*, model_bytes: bytes, device_id: str, model_version: str) -> tuple[str, str]:
    s3 = _get_s3_client()
    s3_key = f"{MODEL_S3_PREFIX}/device_id={device_id}/{model_version}/desalter_model.pkl"

    try:
        s3.head_bucket(Bucket=MODEL_S3_BUCKET)
    except Exception:
        logger.info("Creating S3 bucket: %s", MODEL_S3_BUCKET)
        s3.create_bucket(Bucket=MODEL_S3_BUCKET)

    logger.info("Uploading goal-seek model to s3://%s/%s", MODEL_S3_BUCKET, s3_key)
    s3.put_object(
        Bucket=MODEL_S3_BUCKET,
        Key=s3_key,
        Body=model_bytes,
        ContentType="application/octet-stream",
    )
    return f"s3://{MODEL_S3_BUCKET}/{s3_key}", s3_key


def _fetch_validated_df(db: Db, *, device_id: str, run_id: str | None = None) -> pd.DataFrame:
    where = ["device_id = %s"]
    params: list[str] = [device_id]

    if run_id:
        where.append("run_id = %s")
        params.append(run_id)

    with db.cursor() as cur:
        cur.execute(
            f"""
            SELECT *
            FROM validated_desalter_data
            WHERE {' AND '.join(where)}
            ORDER BY recorded_at ASC
            """,
            tuple(params),
        )
        rows = cur.fetchall()
        cols = [d.name for d in cur.description]

    return pd.DataFrame(rows, columns=cols)


def _resolve_columns(df: pd.DataFrame, requested: list[str], aliases: dict[str, str]) -> list[str]:
    resolved: list[str] = []
    for col in requested:
        if col in df.columns:
            resolved.append(col)
            continue
        mapped = aliases.get(col)
        if mapped and mapped in df.columns:
            resolved.append(mapped)
    return resolved


def _ensure_goal_seek_training_tables(db: Db) -> None:
    db.execute("CREATE SCHEMA IF NOT EXISTS training;")
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS training.test_goalseek_metrics (
            id BIGSERIAL PRIMARY KEY,
            run_id TEXT,
            device_id TEXT,
            model_version TEXT,
            unit_name TEXT,
            location_name TEXT,
            plant_name TEXT,
            target TEXT NOT NULL,
            mae DOUBLE PRECISION,
            rmse DOUBLE PRECISION,
            r2 DOUBLE PRECISION,
            run_timestamp TIMESTAMPTZ NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS training.test_goalseek_results (
            id BIGSERIAL PRIMARY KEY,
            run_id TEXT,
            device_id TEXT,
            model_version TEXT,
            unit_name TEXT,
            location_name TEXT,
            plant_name TEXT,
            recorded_at TIMESTAMPTZ,
            run_timestamp TIMESTAMPTZ NOT NULL,
            predictions_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
    )


def run_goal_seek_training(event: dict, *, db: Db) -> dict:
    run_id = event.get("run_id") or datetime.now(UTC).strftime("goal-seek-train-%Y%m%dT%H%M%SZ")
    device_id = event.get("device_id", "desalter")
    validated_run_id = event.get("validated_run_id") or event.get("validation_run_id")
    model_version = event.get("model_version") or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    activate_model = bool(event.get("activate_model", True))

    db.upsert_tracker(
        run_id=run_id,
        process_name=ProcessName.TRAINING,
        parent_run_id=validated_run_id,
        device_id=device_id,
        state=ProcessingState.RUNNING,
        data_start_ts=event.get("data_start_ts"),
        data_end_ts=event.get("data_end_ts"),
        meta={"action": "train_goal_seek", "validated_run_id": validated_run_id},
        end_now=False,
    )

    try:
        df = _fetch_validated_df(db, device_id=device_id, run_id=validated_run_id)
        if df.empty:
            db.upsert_tracker(
                run_id=run_id,
                process_name=ProcessName.TRAINING,
                parent_run_id=validated_run_id,
                device_id=device_id,
                state=ProcessingState.SKIPPED,
                data_start_ts=event.get("data_start_ts"),
                data_end_ts=event.get("data_end_ts"),
                meta={"reason": "NO_VALIDATED_DATA", "action": "train_goal_seek"},
                end_now=True,
            )
            return {
                "run_id": run_id,
                "validated_run_id": validated_run_id,
                "device_id": device_id,
                "skipped": True,
                "reason": "NO_VALIDATED_DATA",
            }

        df["recorded_at"] = pd.to_datetime(df["recorded_at"], errors="coerce")
        df = df.dropna(subset=["recorded_at"]).sort_values("recorded_at")
        if df.empty:
            raise ValueError("validated_desalter_data has no usable recorded_at values")

        meta_cols = ["unit_name", "location_name", "plant_name"]
        meta_info = {col: df[col].iloc[-1] if col in df.columns else None for col in meta_cols}

        aliases = {
            "chemical_consumption_demulsifier_ppm": "chemical_consumption_demulsifier_ppm_unnamed_85_level_2",
            "cdu_o_h_boot_water_analysis_chloride_ppm": "o_h_boot_water_analysis_chloride_ppm",
            "Date": "recorded_at",
        }

        requested_manipulated_vars = event.get("manipulated_vars") or DEFAULT_MANIPULATED_VARS
        manipulated_vars = _resolve_columns(df, requested_manipulated_vars, aliases)
        if not manipulated_vars:
            raise ValueError("No manipulated variables resolved from validated_desalter_data")

        requested_targets = event.get("targets") or DEFAULT_TARGETS
        targets = _resolve_columns(df, requested_targets, aliases)
        if not targets:
            raise ValueError("No target columns resolved from validated_desalter_data")

        exclude_cols = set(manipulated_vars + targets + meta_cols + ["recorded_at"])
        non_feature_cols = {
            "id",
            "run_id",
            "parent_run_id",
            "device_id",
            "extras_json",
            "created_at",
        }
        disturbance_vars = [
            col
            for col in df.select_dtypes(include=[np.number]).columns
            if col not in exclude_cols and col not in non_feature_cols
        ]
        features = manipulated_vars + disturbance_vars
        if not features:
            raise ValueError("No features resolved for goal-seek training")

        work_df = df.dropna(subset=features + targets).copy()
        if len(work_df) < int(event.get("min_train_rows") or 25):
            raise ValueError(
                f"Insufficient clean rows for training. got={len(work_df)}, "
                f"required>={int(event.get('min_train_rows') or 25)}"
            )

        X = work_df[features]
        y = work_df[targets]

        split = int(len(X) * float(event.get("train_split") or 0.8))
        if split <= 0 or split >= len(X):
            raise ValueError(f"Invalid split index {split} for dataset length {len(X)}")

        X_train = X.iloc[:split]
        X_test = X.iloc[split:]
        y_train = y.iloc[:split]
        y_test = y.iloc[split:]

        model = XGBRegressor(
            n_estimators=int(event.get("n_estimators") or 400),
            learning_rate=float(event.get("learning_rate") or 0.05),
            max_depth=int(event.get("max_depth") or 6),
            subsample=float(event.get("subsample") or 0.8),
            colsample_bytree=float(event.get("colsample_bytree") or 0.8),
            random_state=int(event.get("seed") or 42),
            n_jobs=int(event.get("n_jobs") or -1),
        )
        model.fit(X_train, y_train)

        preds = np.asarray(model.predict(X_test))
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)

        run_timestamp = datetime.now(IST)

        metrics_rows: list[dict[str, Any]] = []
        for i, target in enumerate(targets):
            mae = float(mean_absolute_error(y_test[target], preds[:, i]))
            rmse = float(np.sqrt(mean_squared_error(y_test[target], preds[:, i])))
            r2 = float(r2_score(y_test[target], preds[:, i]))
            metrics_rows.append(
                {
                    "run_id": run_id,
                    "device_id": device_id,
                    "model_version": None,
                    **meta_info,
                    "target": target,
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                    "run_timestamp": run_timestamp,
                }
            )

        test_dates = work_df.iloc[split:]["recorded_at"].reset_index(drop=True)
        prediction_rows: list[dict[str, Any]] = []
        for i in range(len(preds)):
            pred_obj = {targets[j]: float(preds[i][j]) for j in range(len(targets))}
            prediction_rows.append(
                {
                    "run_id": run_id,
                    "device_id": device_id,
                    "model_version": None,
                    **meta_info,
                    "recorded_at": test_dates.iloc[i],
                    "run_timestamp": run_timestamp,
                    "predictions_json": json.dumps(pred_obj),
                }
            )

        _ensure_goal_seek_training_tables(db)

        with db.cursor() as cur:
            for row in metrics_rows:
                cur.execute(
                    """
                    INSERT INTO training.test_goalseek_metrics (
                        run_id, device_id, model_version,
                        unit_name, location_name, plant_name,
                        target, mae, rmse, r2, run_timestamp
                    )
                    VALUES (
                        %(run_id)s, %(device_id)s, %(model_version)s,
                        %(unit_name)s, %(location_name)s, %(plant_name)s,
                        %(target)s, %(mae)s, %(rmse)s, %(r2)s, %(run_timestamp)s
                    );
                    """,
                    row,
                )

            for row in prediction_rows:
                cur.execute(
                    """
                    INSERT INTO training.test_goalseek_results (
                        run_id, device_id, model_version,
                        unit_name, location_name, plant_name,
                        recorded_at, run_timestamp, predictions_json
                    )
                    VALUES (
                        %(run_id)s, %(device_id)s, %(model_version)s,
                        %(unit_name)s, %(location_name)s, %(plant_name)s,
                        %(recorded_at)s, %(run_timestamp)s, %(predictions_json)s::jsonb
                    );
                    """,
                    row,
                )

        model_dir = _safe_model_dir()
        timestamp = datetime.now(IST).strftime("%Y_%m_%d_%H_%M")
        model_filename = f"{MODEL_PREFIX}_{timestamp}.pkl"
        model_path = os.path.join(model_dir, model_filename)
        joblib.dump(model, model_path)

        with open(model_path, "rb") as f:
            model_bytes = f.read()

        artifact_sha256 = sha256_bytes(model_bytes)
        artifact_s3_uri, artifact_s3_key = _upload_goal_seek_model_to_s3(
            model_bytes=model_bytes,
            device_id=device_id,
            model_version=model_version,
        )

        feature_schema = {
            "manipulated_vars": manipulated_vars,
            "disturbance_vars": disturbance_vars,
            "features": features,
            "targets": targets,
            "model_path": model_path,
        }
        metrics_json = {"per_target": metrics_rows}

        with db.cursor() as cur:
            cur.execute(
                """
                INSERT INTO model_registry (
                    device_id, model_type, model_version,
                    s3_uri, artifact_sha256,
                    feature_schema_json, metrics_json, trained_data_json,
                    status
                )
                VALUES (
                    %(device_id)s, 'DESALTER_GOAL_SEEK'::model_type_enum, %(model_version)s,
                    %(s3_uri)s, %(artifact_sha256)s,
                    %(feature_schema)s::jsonb, %(metrics)s::jsonb, %(trained_data)s::jsonb,
                    'STAGED'
                )
                ON CONFLICT DO NOTHING;
                """,
                {
                    "device_id": device_id,
                    "model_version": model_version,
                    "s3_uri": artifact_s3_uri,
                    "artifact_sha256": artifact_sha256,
                    "feature_schema": json.dumps(feature_schema),
                    "metrics": json.dumps(metrics_json),
                    "trained_data": json.dumps(
                        {
                            "source": "training_lambda",
                            "run_id": run_id,
                            "validated_run_id": validated_run_id,
                            "artifact_s3_key": artifact_s3_key,
                            "saved_local_path": model_path,
                        }
                    ),
                },
            )

            if activate_model:
                cur.execute(
                    """
                    UPDATE model_registry
                    SET status = 'DEPRECATED'
                    WHERE device_id = %s AND model_type = 'DESALTER_GOAL_SEEK' AND status = 'ACTIVE';
                    """,
                    (device_id,),
                )
                cur.execute(
                    """
                    UPDATE model_registry
                    SET status = 'ACTIVE', activated_at = now()
                    WHERE device_id = %s AND model_type = 'DESALTER_GOAL_SEEK' AND model_version = %s;
                    """,
                    (device_id, model_version),
                )

            cur.execute(
                """
                SELECT id
                FROM model_registry
                WHERE device_id=%s AND model_type='DESALTER_GOAL_SEEK' AND model_version=%s
                ORDER BY created_at DESC
                LIMIT 1;
                """,
                (device_id, model_version),
            )
            reg_row = cur.fetchone()
            model_registry_id = reg_row[0] if reg_row else None

        db._conn.commit()

        db.upsert_tracker(
            run_id=run_id,
            process_name=ProcessName.TRAINING,
            parent_run_id=validated_run_id,
            device_id=device_id,
            state=ProcessingState.SUCCESS,
            data_start_ts=event.get("data_start_ts"),
            data_end_ts=event.get("data_end_ts"),
            meta={
                "action": "train_goal_seek",
                "validated_run_id": validated_run_id,
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "feature_count": int(len(features)),
                "target_count": int(len(targets)),
                "model_path": model_path,
                "model_version": model_version,
                "model_registry_id": model_registry_id,
                "artifact_s3_uri": artifact_s3_uri,
                "artifact_sha256": artifact_sha256,
            },
            end_now=True,
        )

        return {
            "run_id": run_id,
            "validated_run_id": validated_run_id,
            "device_id": device_id,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "feature_count": int(len(features)),
            "target_count": int(len(targets)),
            "metrics_rows": len(metrics_rows),
            "prediction_rows": len(prediction_rows),
            "model_path": model_path,
            "model_version": model_version,
            "model_registry_id": model_registry_id,
            "artifact_s3_uri": artifact_s3_uri,
            "artifact_sha256": artifact_sha256,
            "activated": activate_model,
            "skipped": False,
        }

    except Exception as exc:
        db.upsert_tracker(
            run_id=run_id,
            process_name=ProcessName.TRAINING,
            parent_run_id=validated_run_id,
            device_id=device_id,
            state=ProcessingState.FAILED,
            data_start_ts=event.get("data_start_ts"),
            data_end_ts=event.get("data_end_ts"),
            error_message=str(exc),
            meta={"action": "train_goal_seek", "validated_run_id": validated_run_id},
            end_now=True,
        )
        raise


def run_training_register(event: dict, *, db: Db) -> dict:
    """
    Register a completed SageMaker training job in model_registry and
    deploy (create or update) the SageMaker real-time endpoint.

    Expected event keys:
      run_id                  – pipeline run id
      device_id               – e.g. "desalter"
      model_type              – "DESALTER_FORECAST" | "DESALTER_GOAL_SEEK"
      training_job_name       – completed SageMaker training job name
      endpoint_name           – target SageMaker endpoint name
      image_uri               – ECR image URI used for training (reused for endpoint)
      role_arn                – SageMaker IAM role ARN
      instance_type           – endpoint instance type (default ml.m5.large)
      model_version           – semantic version string to store in model_registry
      data_start_ts           – optional ISO timestamp
      data_end_ts             – optional ISO timestamp
    """
    run_id = event.get("run_id")
    if not run_id:
        raise ValueError("training register requires run_id")

    device_id = event.get("device_id", "desalter")
    model_type = event.get("model_type")
    if not model_type:
        raise ValueError("model_type is required (DESALTER_FORECAST or DESALTER_GOAL_SEEK)")
    if model_type not in ("DESALTER_FORECAST", "DESALTER_GOAL_SEEK"):
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            "Supported: DESALTER_FORECAST, DESALTER_GOAL_SEEK."
        )

    # Goal-seek does NOT get a real-time inference endpoint — it is loaded directly from S3.
    deploy_endpoint = model_type == "DESALTER_FORECAST"

    training_job_name = event.get("training_job_name")
    if not training_job_name:
        raise ValueError("training_job_name is required")

    endpoint_name = event.get("endpoint_name") or os.getenv(
        f"SAGEMAKER_ENDPOINT_{model_type}", f"desalter-{model_type.lower().replace('_', '-')}"
    )
    image_uri = event.get("image_uri") or os.getenv("SAGEMAKER_IMAGE_URI")
    role_arn = event.get("role_arn") or os.getenv("SAGEMAKER_ROLE_ARN")
    instance_type = event.get("instance_type") or os.getenv("SAGEMAKER_ENDPOINT_INSTANCE_TYPE", "ml.m5.large")
    model_version = event.get("model_version") or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    if not image_uri:
        raise ValueError("image_uri (or SAGEMAKER_IMAGE_URI env var) is required")
    if not role_arn:
        raise ValueError("role_arn (or SAGEMAKER_ROLE_ARN env var) is required")

    db.upsert_tracker(
        run_id=run_id,
        process_name=ProcessName.TRAINING,
        device_id=device_id,
        state=ProcessingState.RUNNING,
        data_start_ts=event.get("data_start_ts"),
        data_end_ts=event.get("data_end_ts"),
        meta={
            "model_type": model_type,
            "training_job_name": training_job_name,
            "action": "register",
        },
        end_now=False,
    )

    try:
        sm = get_sagemaker_client()

        # --- Describe completed training job ---
        logger.info(f"Describing SageMaker training job: {training_job_name}")
        job = sm.describe_training_job(TrainingJobName=training_job_name)
        job_status = job["TrainingJobStatus"]
        if job_status != "Completed":
            raise RuntimeError(
                f"Training job '{training_job_name}' is not Completed (status={job_status}). "
                "Ensure Airflow waits for job completion before calling this Lambda."
            )

        model_artifact_s3 = job["ModelArtifacts"]["S3ModelArtifacts"]
        logger.info(f"Model artifact: {model_artifact_s3}")

        # --- Compute SHA256 of model artifact ---
        s3_client = boto3.client("s3", region_name=AWS_REGION)
        bucket, key = parse_s3_uri(model_artifact_s3)
        body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
        artifact_sha256 = sha256_bytes(body)
        logger.info(f"Model artifact SHA256: {artifact_sha256}")

        # --- Read feature_schema + metrics from training output ---
        # The training script writes training_output.json under SM_OUTPUT_DATA_DIR,
        # which SageMaker packages into output.tar.gz in the same S3 output prefix.
        feature_schema: dict = {}
        metrics_json: dict = {}
        output_s3_prefix = job.get("OutputDataConfig", {}).get("S3OutputPath", "")
        output_key = f"{output_s3_prefix.rstrip('/')}/{training_job_name}/output/output.tar.gz"
        try:
            import io, tarfile
            out_bucket, out_key = parse_s3_uri(
                f"s3://{output_s3_prefix.split('s3://')[-1].split('/')[0]}/{out_key}"
                if "s3://" in output_s3_prefix
                else f"s3://{output_key}"
            )
            # Prefer clean path from described job
            out_s3_raw = (
                f"{output_s3_prefix.rstrip('/')}/{training_job_name}/output/output.tar.gz"
            )
            out_bucket2, out_key2 = parse_s3_uri(out_s3_raw)
            tar_bytes = s3_client.get_object(Bucket=out_bucket2, Key=out_key2)["Body"].read()
            with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
                try:
                    member = tf.getmember("training_output.json")
                    training_output = json.loads(tf.extractfile(member).read().decode("utf-8"))
                    feature_schema = training_output.get("feature_schema", {})
                    metrics_json = {"per_target": training_output.get("metrics", [])}
                    logger.info("Loaded feature_schema and metrics from training_output.json")
                except KeyError:
                    logger.warning("training_output.json not found in output.tar.gz – using empty schema")
        except Exception as exc:
            logger.warning(f"Could not load training output JSON: {exc} – proceeding with empty schema")

        # Embed the endpoint name in feature_schema only for forecast models
        if deploy_endpoint:
            feature_schema["sagemaker_endpoint_name"] = endpoint_name

        # --- Register in model_registry (STAGED status) ---
        logger.info(f"Registering model in model_registry (version={model_version})")
        db.execute(
            """
            INSERT INTO model_registry (
                device_id, model_type, model_version,
                s3_uri, artifact_sha256,
                feature_schema_json, metrics_json,
                trained_data_json, status
            )
            VALUES (
                %(device_id)s, %(model_type)s::model_type_enum, %(model_version)s,
                %(s3_uri)s, %(artifact_sha256)s,
                %(feature_schema)s::jsonb, %(metrics)s::jsonb,
                %(trained_data)s::jsonb, 'STAGED'
            );
            """,
            {
                "device_id": device_id,
                "model_type": model_type,
                "model_version": model_version,
                "s3_uri": model_artifact_s3,
                "artifact_sha256": artifact_sha256,
                "feature_schema": json.dumps(feature_schema),
                "metrics": json.dumps(metrics_json),
                "trained_data": json.dumps({"training_job_name": training_job_name}),
            },
        )

        # Fetch the newly inserted registry id
        reg_row = db.fetch_one(
            """
            SELECT id FROM model_registry
            WHERE device_id=%s AND model_type=%s AND model_version=%s
            ORDER BY created_at DESC LIMIT 1;
            """,
            (device_id, model_type, model_version),
        )
        model_registry_id = reg_row[0] if reg_row else None
        logger.info(f"model_registry id={model_registry_id}")

        # --- Deploy / update SageMaker endpoint (FORECAST only) ---
        if deploy_endpoint:
            sm_model_name = f"{endpoint_name}-{model_version.replace(':', '-')}"
            create_or_update_endpoint(
                sm_client=sm,
                endpoint_name=endpoint_name,
                model_name=sm_model_name,
                execution_role_arn=role_arn,
                model_s3_uri=model_artifact_s3,
                image_uri=image_uri,
                instance_type=instance_type,
            )
        else:
            logger.info("DESALTER_GOAL_SEEK: skipping endpoint deployment (model loaded directly from S3)")

        db.upsert_tracker(
            run_id=run_id,
            process_name=ProcessName.TRAINING,
            device_id=device_id,
            state=ProcessingState.SUCCESS,
            data_start_ts=event.get("data_start_ts"),
            data_end_ts=event.get("data_end_ts"),
            meta={
                "model_type": model_type,
                "model_version": model_version,
                "model_registry_id": model_registry_id,
                "training_job_name": training_job_name,
                "endpoint_name": endpoint_name if deploy_endpoint else None,
                "artifact_sha256": artifact_sha256,
            },
            end_now=True,
        )

        return {
            "run_id": run_id,
            "device_id": device_id,
            "model_type": model_type,
            "model_version": model_version,
            "model_registry_id": model_registry_id,
            "training_job_name": training_job_name,
            "endpoint_name": endpoint_name if deploy_endpoint else None,
            "artifact_s3_uri": model_artifact_s3,
            "skipped": False,
        }

    except Exception as exc:
        logger.exception(f"Training registration failed: {exc}")
        db.upsert_tracker(
            run_id=run_id,
            process_name=ProcessName.TRAINING,
            device_id=device_id,
            state=ProcessingState.FAILED,
            data_start_ts=event.get("data_start_ts"),
            data_end_ts=event.get("data_end_ts"),
            error_message=str(exc),
            meta={"model_type": model_type, "training_job_name": training_job_name},
            end_now=True,
        )
        raise
