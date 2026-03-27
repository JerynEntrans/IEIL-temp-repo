from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from shared.schema.db import ProcessName, ProcessingState
from shared.utils.db import fetch_model_spec

UTC = timezone.utc
logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("GOAL_SEEK_MODEL_DIR", "/opt/airflow/include/model")
MODEL_PREFIX = os.getenv("GOAL_SEEK_MODEL_PREFIX", "desalter_model")

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

DEFAULT_OBJECTIVE_TARGETS: dict[str, tuple[float, float]] = {
    "desalter_salt_ptb_o_l": (4.0, 2.0),
    "desalter_brine_water_oil_ppm": (80.0, 30.0),
    "o_h_boot_water_analysis_chloride_ppm": (7.0, 3.0),
}


def _parse_ts(ts: str | None) -> datetime:
    if not ts:
        return datetime.now(UTC)
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(UTC)


def _load_latest_model(model_dir: str = MODEL_DIR, prefix: str = MODEL_PREFIX):
    """
    Load the goal-seek model.  Priority:
      1. Most-recent ``<prefix>_*.pkl`` file in ``model_dir`` (local disk / EFS).
      2. ACTIVE DESALTER_GOAL_SEEK model from model_registry loaded via S3.
    The ``db`` keyword is only needed for path 2 and is injected by ``run_goal_seek``.
    """
    if os.path.isdir(model_dir):
        files = [f for f in os.listdir(model_dir) if f.startswith(prefix) and f.endswith(".pkl")]
        if files:
            latest = sorted(files)[-1]
            model_path = os.path.join(model_dir, latest)
            logger.info("Loading goal-seek model from local disk: %s", model_path)
            return joblib.load(model_path), latest

    raise FileNotFoundError(
        f"No model files found in {model_dir} with prefix '{prefix}'. "
        "Train a goal-seek model first (SageMaker or local) and ensure the "
        "artifact is available at GOAL_SEEK_MODEL_DIR."
    )


def _load_model_from_registry(db, *, device_id: str) -> tuple[Any, str]:
    """Load the ACTIVE DESALTER_GOAL_SEEK model from model_registry via S3."""
    from shared.utils.s3 import load_joblib_model_from_s3
    spec = fetch_model_spec(db, device_id=device_id, model_type="DESALTER_GOAL_SEEK")
    logger.info("Loading goal-seek model from model_registry (s3_uri=%s)", spec.s3_uri)
    model = load_joblib_model_from_s3(spec)
    return model, spec.model_version


def _fetch_validated_df(db, *, device_id: str, run_ts: datetime) -> pd.DataFrame:
    with db.cursor() as cur:
        cur.execute(
            """
            SELECT *
            FROM validated_desalter_data
            WHERE device_id = %s
              AND recorded_at <= %s
            ORDER BY recorded_at ASC;
            """,
            (device_id, run_ts),
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


def _bounds_around(v: float, low_mult: float, high_mult: float) -> tuple[float, float]:
    if v == 0:
        return (-1.0, 1.0)
    lo = v * low_mult
    hi = v * high_mult
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def run_goal_seek(event: dict, *, db) -> dict:
    run_id = event.get("run_id")
    if not run_id:
        raise ValueError("goal_seek requires run_id")

    device_id = event.get("device_id", "desalter")
    run_ts = _parse_ts(event.get("data_end_ts") or event.get("run_timestamp"))
    validated_run_id = event.get("validated_run_id") or event.get("validation_run_id") or run_id

    db.upsert_tracker(
        run_id=run_id,
        parent_run_id=validated_run_id,
        process_name=ProcessName.GOAL_SEEK,
        device_id=device_id,
        state=ProcessingState.RUNNING,
        data_start_ts=event.get("data_start_ts"),
        data_end_ts=event.get("data_end_ts"),
        meta={"validated_run_id": validated_run_id},
        end_now=False,
    )

    try:
        df = _fetch_validated_df(db, device_id=device_id, run_ts=run_ts)
        if df.empty:
            db.upsert_tracker(
                run_id=run_id,
                parent_run_id=validated_run_id,
                process_name=ProcessName.GOAL_SEEK,
                device_id=device_id,
                state=ProcessingState.SKIPPED,
                data_start_ts=event.get("data_start_ts"),
                data_end_ts=event.get("data_end_ts"),
                meta={"reason": "NO_VALIDATED_DATA"},
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

        latest_date = df["recorded_at"].iloc[-1]
        meta_cols = ["unit_name", "location_name", "plant_name"]
        meta_info = {col: df[col].iloc[-1] if col in df.columns else None for col in meta_cols}

        aliases = {
            "chemical_consumption_demulsifier_ppm": "chemical_consumption_demulsifier_ppm_unnamed_85_level_2",
            "cdu_o_h_boot_water_analysis_chloride_ppm": "o_h_boot_water_analysis_chloride_ppm",
        }

        requested_manipulated_vars = event.get("manipulated_vars") or DEFAULT_MANIPULATED_VARS
        manipulated_vars = _resolve_columns(df, requested_manipulated_vars, aliases)
        if not manipulated_vars:
            raise ValueError("No manipulated_vars were resolved in validated_desalter_data")

        requested_targets = event.get("targets") or DEFAULT_TARGETS
        targets = _resolve_columns(df, requested_targets, aliases)
        if not targets:
            raise ValueError("No target columns were resolved in validated_desalter_data")

        logger.info("Metadata: %s", meta_info)

        recent_window = int(event.get("stable_window") or 10)
        recent_df = df.tail(max(1, recent_window))
        numeric_cols = [c for c in recent_df.select_dtypes(include=[np.number]).columns if c != "id"]
        if not numeric_cols:
            raise ValueError("No numeric columns available to infer stable plant state")

        median_vals = recent_df[numeric_cols].median()
        distances = (recent_df[numeric_cols] - median_vals).abs().sum(axis=1)
        current_state = recent_df.loc[distances.idxmin()]

        non_feature_cols = {
            "id",
            "run_id",
            "parent_run_id",
            "recorded_at",
            "device_id",
            "plant_name",
            "unit_name",
            "location_name",
            "extras_json",
            "created_at",
        }
        disturbance_vars = [
            col
            for col in numeric_cols
            if col not in manipulated_vars and col not in targets and col not in non_feature_cols
        ]

        features = manipulated_vars + disturbance_vars
        if not features:
            raise ValueError("No model features resolved for goal seek")

        try:
            model, model_version = _load_model_from_registry(db, device_id=device_id)
            model_source = "model_registry"
        except Exception as exc:
            logger.warning(
                "Could not load ACTIVE goal-seek model from model_registry (%s). Falling back to local model directory.",
                exc,
            )
            model, model_version = _load_latest_model(
                model_dir=event.get("model_dir") or MODEL_DIR,
                prefix=event.get("model_prefix") or MODEL_PREFIX,
            )
            model_source = "local_disk"

        def build_features(manipulated_values: np.ndarray) -> np.ndarray:
            x: dict[str, float] = {}

            for i, col in enumerate(manipulated_vars):
                x[col] = float(manipulated_values[i])

            for col in disturbance_vars:
                x[col] = float(current_state[col])

            return np.array([[x[col] for col in features]], dtype=float)

        initial_guess = current_state[manipulated_vars].astype(float).values

        if np.isnan(initial_guess).any():
            missing_cols = [
                manipulated_vars[i]
                for i, value in enumerate(initial_guess)
                if np.isnan(value)
            ]
            raise ValueError(f"NaN in manipulated_vars current state: {missing_cols}")

        baseline_preds = np.asarray(model.predict(build_features(initial_guess))).reshape(-1)
        if baseline_preds.size != len(targets):
            raise ValueError(
                f"Model output size mismatch: expected {len(targets)}, got {baseline_preds.size}. "
                "Ensure model was trained on selected target list."
            )
        logger.info("Initial guess: %s", initial_guess.tolist())
        logger.info("Initial prediction: %s", dict(zip(targets, [float(v) for v in baseline_preds])))

        debug_perturbations = int(event.get("debug_perturbations") or 0)
        if debug_perturbations > 0:
            logger.info("Testing %s random perturbations...", debug_perturbations)
            for i in range(debug_perturbations):
                test = initial_guess * np.random.uniform(0.85, 1.15, len(initial_guess))
                test_preds = np.asarray(model.predict(build_features(test))).reshape(-1)
                logger.info(
                    "Perturbation %s inputs=%s predictions=%s",
                    i + 1,
                    [float(v) for v in test],
                    dict(zip(targets, [float(v) for v in test_preds])),
                )

        if hasattr(model, "feature_importances_"):
            try:
                fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).head(20)
                logger.info("Top feature importances: %s", fi.to_dict())
            except Exception:
                logger.debug("Could not compute feature importances", exc_info=True)

        objective_targets = dict(DEFAULT_OBJECTIVE_TARGETS)
        user_objective = event.get("objective_targets") or {}
        for k, pair in user_objective.items():
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                objective_targets[k] = (float(pair[0]), float(pair[1]))

        def objective(manipulated_values: np.ndarray) -> float:
            preds = np.asarray(model.predict(build_features(manipulated_values))).reshape(-1)
            result = dict(zip(targets, preds))

            loss = 0.0
            for metric, (target_val, scale) in objective_targets.items():
                if metric in result and scale > 0:
                    loss += ((float(result[metric]) - float(target_val)) / float(scale)) ** 2

            if result.get("desalter_brine_water_oil_ppm", 0.0) < 0:
                loss += 10000
            if result.get("desalter_salt_ptb_o_l", 0.0) < 0:
                loss += 10000
            if result.get("desalter_brine_water_oil_ppm", 0.0) < 60:
                loss += 500

            if result.get("desalter_monitoring_interface_level", 9999.0) < float(event.get("min_interface_1") or 55):
                loss += 5000
            if result.get("desalter_2_monitoring_interface_level", 9999.0) < float(event.get("min_interface_2") or 30):
                loss += 5000

            return float(loss)

        low_mult = float(event.get("bound_low_multiplier") or 0.85)
        high_mult = float(event.get("bound_high_multiplier") or 1.15)
        bounds = [_bounds_around(float(current_state[col]), low_mult, high_mult) for col in manipulated_vars]

        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=int(event.get("maxiter") or 200),
            popsize=int(event.get("popsize") or 20),
            tol=float(event.get("tol") or 0.01),
            seed=int(event.get("seed") or 42),
        )

        optimal_inputs = dict(zip(manipulated_vars, [float(v) for v in result.x]))
        opt_preds = np.asarray(model.predict(build_features(result.x))).reshape(-1)
        outputs = dict(zip(targets, [float(v) for v in opt_preds]))

        result_json = {
            "validated_run_id": validated_run_id,
            "latest_validated_ts": latest_date.isoformat() if hasattr(latest_date, "isoformat") else str(latest_date),
            "model_version": model_version,
            "model_source": model_source,
            "features": features,
            "disturbance_vars": disturbance_vars,
            "manipulated_vars": manipulated_vars,
            "targets": targets,
            "objective_value": float(result.fun),
            "optimization_success": bool(result.success),
            "optimization_message": str(result.message),
            "baseline_outputs": dict(zip(targets, [float(v) for v in baseline_preds])),
            "optimal_inputs": optimal_inputs,
            "predicted_outputs": outputs,
        }

        with db.cursor() as cur:
            cur.execute(
                """
                INSERT INTO desalter_goal_seek_results (
                    device_id,
                    run_timestamp,
                    desalter_monitoring_press_kg_cm2,
                    desalter_monitoring_w_w_temp_deg_c,
                    chemical_consumption_demulsifier_ppm,
                    crude_details_api,
                    crude_details_density,
                    crude_details_crude_details,
                    result_json,
                    model_version,
                    run_id
                )
                VALUES (
                    %(device_id)s,%(run_timestamp)s,%(press)s,%(wwt)s,%(demuls)s,
                    %(api)s,%(density)s,%(crude)s,%(result)s::jsonb,%(model_version)s,%(run_id)s
                )
                ON CONFLICT (run_id, device_id, run_timestamp)
                DO UPDATE SET
                    result_json=EXCLUDED.result_json,
                    model_version=EXCLUDED.model_version,
                    chemical_consumption_demulsifier_ppm=EXCLUDED.chemical_consumption_demulsifier_ppm;
                """,
                {
                    "device_id": device_id,
                    "run_timestamp": run_ts,
                    "press": float(current_state.get("desalter_monitoring_press_kg_cm2"))
                    if pd.notna(current_state.get("desalter_monitoring_press_kg_cm2"))
                    else None,
                    "wwt": float(current_state.get("desalter_monitoring_w_w_temp_deg_c"))
                    if pd.notna(current_state.get("desalter_monitoring_w_w_temp_deg_c"))
                    else None,
                    "demuls": float(optimal_inputs.get("chemical_consumption_demulsifier_ppm_unnamed_85_level_2"))
                    if "chemical_consumption_demulsifier_ppm_unnamed_85_level_2" in optimal_inputs
                    else None,
                    "api": float(current_state.get("crude_details_api_unnamed_4_level_2"))
                    if pd.notna(current_state.get("crude_details_api_unnamed_4_level_2"))
                    else None,
                    "density": float(current_state.get("crude_details_density_unnamed_5_level_2"))
                    if pd.notna(current_state.get("crude_details_density_unnamed_5_level_2"))
                    else None,
                    "crude": float(current_state.get("crude_details_crude_details_unnamed_2_level_2"))
                    if pd.notna(current_state.get("crude_details_crude_details_unnamed_2_level_2"))
                    else None,
                    "result": json.dumps(result_json),
                    "model_version": model_version,
                    "run_id": run_id,
                },
            )
        db._conn.commit()

        db.upsert_tracker(
            run_id=run_id,
            parent_run_id=validated_run_id,
            process_name=ProcessName.GOAL_SEEK,
            device_id=device_id,
            state=ProcessingState.SUCCESS,
            data_start_ts=event.get("data_start_ts"),
            data_end_ts=event.get("data_end_ts"),
            meta={
                "goal_seek": True,
                "model_version": model_version,
                "model_source": model_source,
                "latest_validated_ts": latest_date.isoformat() if hasattr(latest_date, "isoformat") else str(latest_date),
                "objective_value": float(result.fun),
                "optimization_success": bool(result.success),
                **meta_info,
            },
            end_now=True,
        )

        return {
            "run_id": run_id,
            "validated_run_id": validated_run_id,
            "device_id": device_id,
            "run_timestamp": run_ts.isoformat(),
            "model_version": model_version,
            "optimal_inputs": optimal_inputs,
            "predicted_outputs": outputs,
            "skipped": False,
        }

    except Exception as exc:
        db.upsert_tracker(
            run_id=run_id,
            parent_run_id=validated_run_id,
            process_name=ProcessName.GOAL_SEEK,
            device_id=device_id,
            state=ProcessingState.FAILED,
            data_start_ts=event.get("data_start_ts"),
            data_end_ts=event.get("data_end_ts"),
            error_message=str(exc),
            meta={"goal_seek": True},
            end_now=True,
        )
        raise
