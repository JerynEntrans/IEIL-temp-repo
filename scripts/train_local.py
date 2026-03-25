#!/usr/bin/env python3
"""
Local training script — trains DESALTER_FORECAST or DESALTER_GOAL_SEEK XGBoost models,
uploads the artifact to S3 (or localstack S3), and registers it in model_registry.

Usage (from repo root):
    python scripts/train_local.py [options]

Environment variables (same as the lambda stack, from .env.local):
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DB_SSLMODE
    AWS_ENDPOINT_URL_S3   — set to http://localhost:4566 for localstack
    MODEL_S3_BUCKET       — bucket to store model artifact  (default: ieil-raw)
    MODEL_S3_PREFIX       — S3 key prefix                   (default: models)

Options:
    --model-type        DESALTER_FORECAST or DESALTER_GOAL_SEEK (default: DESALTER_FORECAST)
    --device-id         device identifier (default: desalter)
    --model-version     semantic version string (default: local-<timestamp>)
    --n-estimators      XGBoost n_estimators (default: 300)
    --learning-rate     XGBoost learning rate (default: 0.05)
    --max-depth         XGBoost max depth (default: 6)
    --activate          immediately set model status to ACTIVE in model_registry

FORECAST-only options:
    --lookback          lookback window (default: 10)
    --horizons          comma-separated horizon minutes (default: 30,60,120)

GOAL_SEEK-only options:
    --manipulated-vars  comma-separated manipulated variable column names
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone

import boto3
import joblib
import numpy as np
import psycopg
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from xgboost import XGBRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

UTC = timezone.utc

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_FEATURES = [
    "desalter_monitoring_press_kg_cm2",
    "desalter_monitoring_w_w_temp_deg_c",
    "chemical_consumption_demulsifier_ppm_unnamed_85_level_2",
    "crude_details_crude_details_unnamed_2_level_2",
    "crude_details_api_unnamed_4_level_2",
    "crude_details_density_unnamed_5_level_2",
]

DEFAULT_TARGETS = [
    "desalter_monitoring_interface_level",
    "desalter_2_monitoring_interface_level",
    "o_h_boot_water_analysis_chloride_ppm",
    "desalter_salt_ptb_o_l",
    "desalter_brine_water_ph_ppm",
    "desalter_brine_water_oil_ppm",
]

DEFAULT_FORECAST_HORIZONS = [30, 60, 120]

DEFAULT_GOAL_SEEK_MANIPULATED_VARS = [
    "chemical_consumption_demulsifier_ppm_unnamed_85_level_2",
    "desalter_monitoring_press_kg_cm2",
    "desalter_monitoring_w_w_temp_deg_c",
]

DEFAULT_GOAL_SEEK_TARGETS = [
    "desalter_monitoring_interface_level",
    "desalter_2_monitoring_interface_level",
    "desalter_salt_ptb_o_l",
    "desalter_brine_water_oil_ppm",
    "o_h_boot_water_analysis_chloride_ppm",
]

# Columns that are metadata / non-feature by definition
_NON_FEATURE_COLS: frozenset = frozenset({
    "recorded_at", "run_id", "parent_run_id", "device_id",
    "plant_name", "unit_name", "location_name", "extras_json", "created_at",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_s3_client():
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL")
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "test"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "ap-south-1"),
    )


def _get_db_conn():
    return psycopg.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=os.environ.get("DB_PORT", "5432"),
        dbname=os.environ.get("DB_NAME", "ieil"),
        user=os.environ.get("DB_USER", "ieil"),
        password=os.environ.get("DB_PASSWORD", "ieil"),
        sslmode=os.environ.get("DB_SSLMODE", "disable"),
    )


def _make_supervised(X, Y, lookback, horizons):
    T = X.shape[0]
    H = max(horizons)
    rows, targets = [], []
    for t in range(lookback, T - H):
        past_y = Y[t - lookback:t, :].ravel()
        past_x = X[t - lookback:t, :].ravel()
        rows.append(np.hstack([past_y, past_x]))
        future_vals = []
        for h in horizons:
            future_vals.extend(Y[t + h, :])
        targets.append(future_vals)
    return np.array(rows), np.array(targets)


def _normalize_horizons(horizons: list[int]) -> list[int]:
    """Normalize and validate forecast horizons as positive integer steps."""
    normalized = sorted({int(h) for h in horizons})
    if not normalized:
        raise ValueError("At least one forecast horizon is required")
    if any(h <= 0 for h in normalized):
        raise ValueError(f"Forecast horizons must be > 0. got={normalized}")
    return normalized


def _resolve_feature_columns(df: pd.DataFrame, requested: list[str]) -> tuple[list[str], list[str]]:
    """Return usable feature columns and dropped feature names (missing/all-null)."""
    usable: list[str] = []
    dropped: list[str] = []
    for col in requested:
        if col not in df.columns:
            dropped.append(col)
            continue
        if df[col].notna().sum() == 0:
            dropped.append(col)
            continue
        usable.append(col)
    return usable, dropped


def _validate_target_columns(df: pd.DataFrame, requested: list[str]) -> list[str]:
    """Require all targets to exist and have at least one non-null row."""
    missing = [col for col in requested if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required target columns: {missing}")

    empty = [col for col in requested if df[col].notna().sum() == 0]
    if empty:
        raise ValueError(f"Target columns contain no non-null values: {empty}")

    return requested


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _train_and_register_forecast(args):
    device_id = args.device_id
    model_version = args.model_version or f"local-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    horizons_raw = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    horizons = _normalize_horizons(horizons_raw)
    bucket = os.environ.get("MODEL_S3_BUCKET", "ieil-raw")
    prefix = os.environ.get("MODEL_S3_PREFIX", "models").strip("/")

    # --- Load data ---
    logger.info("Connecting to Postgres and loading validated_desalter_data...")
    conn = _get_db_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM validated_desalter_data WHERE device_id=%s ORDER BY recorded_at",
            (device_id,),
        )
        rows = cur.fetchall()
        col_names = [desc[0] for desc in cur.description]
    conn.close()

    df = pd.DataFrame(rows, columns=col_names)
    if "recorded_at" not in df.columns:
        raise ValueError("validated_desalter_data is missing required column: recorded_at")

    features, dropped_features = _resolve_feature_columns(df, DEFAULT_FEATURES)
    if dropped_features:
        logger.warning("Dropping unusable forecast features (missing/all-null): %s", dropped_features)
    if not features:
        raise RuntimeError("No usable forecast features after filtering missing/all-null columns")

    targets = _validate_target_columns(df, DEFAULT_TARGETS)

    df = df.set_index("recorded_at")
    df = df[features + targets].apply(pd.to_numeric, errors="coerce").dropna()
    logger.info(f"Loaded {len(df)} clean rows for device_id='{device_id}' | features={len(features)} targets={len(targets)}")

    if len(df) < args.lookback + max(horizons) + 10:
        logger.error(
            f"Not enough data to train. Need at least {args.lookback + max(horizons) + 10} rows, got {len(df)}."
        )
        sys.exit(1)

    X_values = df[features].values
    Y_values = df[targets].values

    X_design, Y_target = _make_supervised(X_values, Y_values, args.lookback, horizons)
    logger.info(f"Supervised dataset: X={X_design.shape}, Y={Y_target.shape}")

    split = int(0.8 * len(X_design))
    X_train, X_test = X_design[:split], X_design[split:]
    Y_train, Y_test = Y_target[:split], Y_target[split:]

    # --- Train ---
    logger.info(f"Training XGBRegressor (n_estimators={args.n_estimators}, lr={args.learning_rate}, max_depth={args.max_depth})...")
    model = XGBRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        enable_categorical=False,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, Y_train)

    # --- Evaluate ---
    Y_pred = model.predict(X_test)
    n_targets = len(targets)
    metrics = []
    for h_idx, h in enumerate(horizons):
        for j, tgt in enumerate(targets):
            idx = h_idx * n_targets + j
            mse = float(mean_squared_error(Y_test[:, idx], Y_pred[:, idx]))
            r2 = float(r2_score(Y_test[:, idx], Y_pred[:, idx]))
            metrics.append({"target": tgt, "horizon": h, "mse": mse, "rmse": float(mse**0.5), "r2": r2})
            logger.info(f"  [{tgt}] h={h}  rmse={mse**0.5:.4f}  r2={r2:.4f}")

    # --- Save model to temp file, upload to S3 ---
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name
    model.get_booster().save_model(tmp_path)
    with open(tmp_path, "rb") as f:
        model_bytes = f.read()
    os.unlink(tmp_path)

    artifact_sha256 = _sha256(model_bytes)
    s3_key = f"{prefix}/device_id={device_id}/{model_version}/model.json"

    s3 = _get_s3_client()
    # Ensure bucket exists (localstack)
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        logger.info(f"Creating S3 bucket: {bucket}")
        s3.create_bucket(Bucket=bucket)

    logger.info(f"Uploading model to s3://{bucket}/{s3_key}")
    s3.put_object(Bucket=bucket, Key=s3_key, Body=model_bytes, ContentType="application/octet-stream")

    s3_uri = f"s3://{bucket}/{s3_key}"
    logger.info(f"Model uploaded: {s3_uri}  sha256={artifact_sha256}")

    # --- Register in model_registry ---
    feature_schema = {
        "features": features,
        "lookback": args.lookback,
        "output": {
            "targets": targets,
            "horizons_minutes": horizons,
        },
        # No sagemaker_endpoint_name — local mode loads directly from S3
    }

    conn = _get_db_conn()
    conn.autocommit = False
    with conn.cursor() as cur:
        # Insert with STAGED status first
        cur.execute(
            """
            INSERT INTO model_registry (
                device_id, model_type, model_version,
                s3_uri, artifact_sha256,
                feature_schema_json, metrics_json, trained_data_json,
                status
            )
            VALUES (%s, 'DESALTER_FORECAST'::model_type_enum, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, 'STAGED')
            ON CONFLICT DO NOTHING;
            """,
            (
                device_id,
                model_version,
                s3_uri,
                artifact_sha256,
                json.dumps(feature_schema),
                json.dumps({"per_target": metrics}),
                json.dumps({"trained_locally": True, "trained_at": datetime.now(UTC).isoformat()}),
            ),
        )

        if args.activate:
            # Clear any existing ACTIVE model for this device+type, then activate new one
            cur.execute(
                """
                UPDATE model_registry
                SET status = 'DEPRECATED'
                WHERE device_id = %s AND model_type = 'DESALTER_FORECAST' AND status = 'ACTIVE';
                """,
                (device_id,),
            )
            cur.execute(
                """
                UPDATE model_registry
                SET status = 'ACTIVE', activated_at = now()
                WHERE device_id = %s AND model_type = 'DESALTER_FORECAST' AND model_version = %s;
                """,
                (device_id, model_version),
            )
            logger.info(f"Model activated: device_id={device_id} version={model_version}")

    conn.commit()
    conn.close()

    status = "ACTIVE" if args.activate else "STAGED"
    logger.info(f"Registered in model_registry: device_id={device_id} version={model_version} status={status}")
    logger.info("")
    logger.info("Done. To activate this model later run:")
    logger.info(f"  python scripts/train_local.py --model-type DESALTER_FORECAST --activate --model-version {model_version}")
    logger.info("")
    logger.info("Set these in your lambda environment (or .env.local) to use the model:")
    logger.info("  USE_ML_MODELS=true")
    logger.info("  USE_SAGEMAKER_ENDPOINT=false")

    return s3_uri, model_version


# ---------------------------------------------------------------------------
# Goal-seek training
# ---------------------------------------------------------------------------

def _train_and_register_goal_seek(args):
    """Train a flat XGBRegressor for goal-seek and register it in model_registry."""
    device_id = args.device_id
    model_version = args.model_version or f"local-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    bucket = os.environ.get("MODEL_S3_BUCKET", "ieil-raw")
    prefix = os.environ.get("MODEL_S3_PREFIX", "models").strip("/")

    manipulated_vars: list[str] = (
        [v.strip() for v in args.manipulated_vars.split(",") if v.strip()]
        if getattr(args, "manipulated_vars", None)
        else DEFAULT_GOAL_SEEK_MANIPULATED_VARS
    )
    targets = DEFAULT_GOAL_SEEK_TARGETS

    # --- Load data (all numeric columns) ---
    logger.info("Connecting to Postgres and loading validated_desalter_data...")
    conn = _get_db_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM validated_desalter_data WHERE device_id=%s ORDER BY recorded_at",
            (device_id,),
        )
        col_names = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
    conn.close()

    df = pd.DataFrame(rows, columns=col_names)
    if "recorded_at" in df.columns:
        df = df.set_index("recorded_at")

    # Drop metadata and non-numeric columns; auto-detect disturbance vars
    df = df.drop(columns=[c for c in _NON_FEATURE_COLS if c in df.columns], errors="ignore")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    excl = set(targets) | set(manipulated_vars)
    disturbance_vars = [c for c in numeric_cols if c not in excl]

    feature_cols = manipulated_vars + disturbance_vars
    all_needed = feature_cols + targets
    missing = [c for c in all_needed if c not in df.columns]
    if missing:
        logger.warning(f"Missing columns (will be dropped from features/targets): {missing}")
        feature_cols = [c for c in feature_cols if c in df.columns]
        targets = [c for c in targets if c in df.columns]

    df = df[feature_cols + targets].apply(pd.to_numeric, errors="coerce").dropna()
    logger.info(f"Loaded {len(df)} clean rows | features={len(feature_cols)} | targets={len(targets)}")

    if len(df) < 50:
        logger.error(f"Not enough data to train (need ≥50, got {len(df)}).")
        sys.exit(1)

    X = df[feature_cols].values
    Y = df[targets].values

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    # --- Train ---
    logger.info(
        f"Training XGBRegressor goal-seek "
        f"(n_estimators={args.n_estimators}, lr={args.learning_rate}, max_depth={args.max_depth})..."
    )
    model = XGBRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        enable_categorical=False,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, Y_train)

    # --- Evaluate ---
    Y_pred = model.predict(X_test)
    metrics = []
    for j, tgt in enumerate(targets):
        mae = float(mean_absolute_error(Y_test[:, j], Y_pred[:, j]))
        mse = float(mean_squared_error(Y_test[:, j], Y_pred[:, j]))
        r2 = float(r2_score(Y_test[:, j], Y_pred[:, j]))
        metrics.append({"target": tgt, "mae": mae, "rmse": mse ** 0.5, "r2": r2})
        logger.info(f"  [{tgt}]  mae={mae:.4f}  rmse={mse**0.5:.4f}  r2={r2:.4f}")

    # --- Save model as pkl, upload to S3 ---
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        joblib.dump(model, tmp_path)
        with open(tmp_path, "rb") as f:
            model_bytes = f.read()
    finally:
        os.unlink(tmp_path)

    artifact_sha256 = _sha256(model_bytes)
    s3_key = f"{prefix}/device_id={device_id}/{model_version}/desalter_model.pkl"

    s3 = _get_s3_client()
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        logger.info(f"Creating S3 bucket: {bucket}")
        s3.create_bucket(Bucket=bucket)

    logger.info(f"Uploading model to s3://{bucket}/{s3_key}")
    s3.put_object(Bucket=bucket, Key=s3_key, Body=model_bytes, ContentType="application/octet-stream")

    s3_uri = f"s3://{bucket}/{s3_key}"
    logger.info(f"Model uploaded: {s3_uri}  sha256={artifact_sha256}")

    # --- Register ---
    feature_schema = {
        "manipulated_vars": manipulated_vars,
        "disturbance_vars": disturbance_vars,
        "features": feature_cols,
        "targets": targets,
    }

    conn = _get_db_conn()
    conn.autocommit = False
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO model_registry (
                device_id, model_type, model_version,
                s3_uri, artifact_sha256,
                feature_schema_json, metrics_json, trained_data_json,
                status
            )
            VALUES (%s, 'DESALTER_GOAL_SEEK'::model_type_enum, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, 'STAGED')
            ON CONFLICT DO NOTHING;
            """,
            (
                device_id,
                model_version,
                s3_uri,
                artifact_sha256,
                json.dumps(feature_schema),
                json.dumps({"per_target": metrics}),
                json.dumps({"trained_locally": True, "trained_at": datetime.now(UTC).isoformat()}),
            ),
        )

        if args.activate:
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
            logger.info(f"Model activated: device_id={device_id} version={model_version}")

    conn.commit()
    conn.close()

    status = "ACTIVE" if args.activate else "STAGED"
    logger.info(f"Registered in model_registry: device_id={device_id} version={model_version} status={status}")
    logger.info("")
    logger.info("Done. To activate this model later run:")
    logger.info(f"  python scripts/train_local.py --model-type DESALTER_GOAL_SEEK --activate --model-version {model_version}")
    logger.info("")
    logger.info("Set these in your lambda environment (or .env.local) to use the model:")
    logger.info("  USE_ML_MODELS=true")

    return s3_uri, model_version


def parse_args():
    parser = argparse.ArgumentParser(description="Train and register a desalter model locally")
    parser.add_argument(
        "--model-type",
        default="DESALTER_FORECAST",
        choices=["DESALTER_FORECAST", "DESALTER_GOAL_SEEK"],
        help="Which model to train (default: DESALTER_FORECAST)",
    )
    parser.add_argument("--device-id", default="desalter")
    parser.add_argument("--model-version", default=None, help="Version string (auto-generated if omitted)")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    # FORECAST-only
    parser.add_argument("--lookback", type=int, default=10)
    parser.add_argument(
        "--horizons",
        default=",".join(str(h) for h in DEFAULT_FORECAST_HORIZONS),
        help="Comma-separated horizon minutes",
    )
    # GOAL_SEEK-only
    parser.add_argument(
        "--manipulated-vars",
        default=None,
        help="Comma-separated manipulated variable column names (goal-seek only)",
    )
    parser.add_argument("--activate", action="store_true", help="Set model status to ACTIVE after registration")
    return parser.parse_args()


if __name__ == "__main__":
    # Load .env.local if present (for running outside docker)
    env_file = os.path.join(os.path.dirname(__file__), "..", ".env.local")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())

    parsed = parse_args()
    if parsed.model_type == "DESALTER_GOAL_SEEK":
        _train_and_register_goal_seek(parsed)
    else:
        _train_and_register_forecast(parsed)
