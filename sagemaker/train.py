#!/usr/bin/env python3
"""
SageMaker training script for Desalter XGBoost models.

Supports two model types (controlled via --model-type):
  DESALTER_FORECAST  – lookback-window supervised dataset, multi-horizon multi-target output
  DESALTER_GOAL_SEEK – flat feature set, single-step multi-target output

Environment variables (DB connection – injected by the training Lambda):
  DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DB_SSLMODE

SageMaker paths:
  SM_MODEL_DIR       – where to write the serialised model   (default /opt/ml/model)
  SM_OUTPUT_DATA_DIR – where to write metrics JSON           (default /opt/ml/output/data)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sqlalchemy import create_engine
from xgboost import XGBRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SageMaker directories
# ---------------------------------------------------------------------------
MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
OUTPUT_DIR = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")

UTC = timezone.utc

# ---------------------------------------------------------------------------
# Defaults (can be overridden via CLI args / SageMaker hyperparameters)
# ---------------------------------------------------------------------------
DEFAULT_FORECAST_FEATURES = [
    "desalter_monitoring_press_kg_cm2",
    "desalter_monitoring_w_w_temp_deg_c",
    "chemical_consumption_demulsifier_ppm_unnamed_85_level_2",
    "crude_details_crude_details_unnamed_2_level_2",
    "crude_details_api_unnamed_4_level_2",
    "crude_details_density_unnamed_5_level_2",
]

DEFAULT_FORECAST_TARGETS = [
    "desalter_monitoring_interface_level",
    "desalter_2_monitoring_interface_level",
    "o_h_boot_water_analysis_chloride_ppm",
    "desalter_salt_ptb_o_l",
    "desalter_brine_water_ph_ppm",
    "desalter_brine_water_oil_ppm",
]

DEFAULT_FORECAST_HORIZONS = [30, 60, 120]

# Goal-seek defaults — flat (no lookback), multi-target, single-step
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

# Columns excluded from auto-detected disturbance vars
_NON_FEATURE_COLS = {
    "recorded_at", "run_id", "parent_run_id", "device_id",
    "plant_name", "unit_name", "location_name", "extras_json", "created_at",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_db_engine():
    host = os.environ["DB_HOST"]
    port = os.environ.get("DB_PORT", "5432")
    dbname = os.environ["DB_NAME"]
    user = os.environ["DB_USER"]
    password = os.environ["DB_PASSWORD"]
    sslmode = os.environ.get("DB_SSLMODE", "disable")
    url = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{dbname}?sslmode={sslmode}"
    return create_engine(url)


def _make_supervised_forecast(X: np.ndarray, Y: np.ndarray, lookback: int, horizons: list[int]):
    """Create lookback-window supervised dataset for multi-horizon forecasting."""
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


def _evaluate_metrics(Y_test, Y_pred, targets, horizons):
    """Return list of per-target-per-horizon metric dicts."""
    records = []
    n_targets = len(targets)
    for h_idx, h in enumerate(horizons):
        for j, tgt in enumerate(targets):
            idx = h_idx * n_targets + j
            y_true = Y_test[:, idx]
            y_pred = Y_pred[:, idx]
            mse = float(mean_squared_error(y_true, y_pred))
            records.append({
                "target": tgt,
                "horizon": h,
                "mse": mse,
                "rmse": float(mse ** 0.5),
                "r2": float(r2_score(y_true, y_pred)),
            })
    return records


def _evaluate_flat_metrics(Y_test: np.ndarray, Y_pred: np.ndarray, targets: list[str]) -> list[dict]:
    """Per-target metrics for flat (non-horizon) goal-seek models."""
    from sklearn.metrics import mean_absolute_error
    records = []
    for j, tgt in enumerate(targets):
        mse = float(mean_squared_error(Y_test[:, j], Y_pred[:, j]))
        records.append({
            "target": tgt,
            "mse": mse,
            "rmse": float(mse ** 0.5),
            "mae": float(mean_absolute_error(Y_test[:, j], Y_pred[:, j])),
            "r2": float(r2_score(Y_test[:, j], Y_pred[:, j])),
        })
    return records


# ---------------------------------------------------------------------------
# Model training functions
# ---------------------------------------------------------------------------

def train_goal_seek(args, engine):
    """
    Train DESALTER_GOAL_SEEK model.

    - Flat feature set: manipulated vars + auto-detected numeric disturbance vars.
    - Single-step multi-target regression (no lookback / horizons).
    - Saved as desalter_model.pkl (joblib) so the goal-seek Lambda can load it
      directly. An XGBoost JSON copy is also written for inspection.
    """
    manipulated_vars = json.loads(args.manipulated_vars) if args.manipulated_vars else DEFAULT_GOAL_SEEK_MANIPULATED_VARS
    targets = json.loads(args.targets) if args.targets else DEFAULT_GOAL_SEEK_TARGETS

    logger.info("Loading data from validated_desalter_data (GOAL_SEEK)...")
    df = pd.read_sql("SELECT * FROM validated_desalter_data ORDER BY recorded_at", engine)
    logger.info(f"Loaded {len(df)} rows")

    df = df.set_index("recorded_at")

    exclude = set(manipulated_vars) | set(targets) | _NON_FEATURE_COLS
    disturbance_vars = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]
    features = manipulated_vars + disturbance_vars
    logger.info(f"Features: {len(features)} ({len(manipulated_vars)} manipulated, {len(disturbance_vars)} disturbance)")

    work = df[features + targets].dropna()
    if len(work) < 25:
        raise RuntimeError(f"Insufficient clean rows for goal-seek training: {len(work)}")

    X = work[features].values
    Y = work[targets].values
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    logger.info(f"Training XGBRegressor goal-seek (n_estimators={args.n_estimators})...")
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

    Y_pred = model.predict(X_test)
    if Y_pred.ndim == 1:
        Y_pred = Y_pred.reshape(-1, 1)
    if Y_test.ndim == 1:
        Y_test = Y_test.reshape(-1, 1)

    metrics = _evaluate_flat_metrics(Y_test, Y_pred, targets)
    feature_schema = {
        "features": features,
        "manipulated_vars": manipulated_vars,
        "disturbance_vars": disturbance_vars,
        "output": {"targets": targets},
    }
    return model, metrics, feature_schema


def train_forecast(args, engine):
    """Train DESALTER_FORECAST model (lookback supervised, multi-horizon)."""
    requested_features = json.loads(args.features) if args.features else DEFAULT_FORECAST_FEATURES
    requested_targets = json.loads(args.targets) if args.targets else DEFAULT_FORECAST_TARGETS
    lookback = args.lookback
    horizons_raw = json.loads(args.horizons_minutes) if args.horizons_minutes else DEFAULT_FORECAST_HORIZONS
    horizons = _normalize_horizons(horizons_raw)

    logger.info("Loading data from validated_desalter_data (FORECAST)...")
    df = pd.read_sql("SELECT * FROM validated_desalter_data ORDER BY recorded_at", engine)
    logger.info(f"Loaded {len(df)} rows")

    features, dropped_features = _resolve_feature_columns(df, requested_features)
    if dropped_features:
        logger.warning("Dropping unusable forecast features (missing/all-null): %s", dropped_features)
    if not features:
        raise RuntimeError("No usable forecast features after filtering missing/all-null columns")

    targets = _validate_target_columns(df, requested_targets)

    df = df.set_index("recorded_at").dropna()
    X_values = df[features].values
    Y_values = df[targets].values

    X_design, Y_target = _make_supervised_forecast(X_values, Y_values, lookback, horizons)
    logger.info(f"Supervised dataset: X={X_design.shape}, Y={Y_target.shape}")
    if len(X_design) < 2:
        raise RuntimeError(
            f"Insufficient supervised rows after lookback/horizon transform: got={len(X_design)} "
            f"(lookback={lookback}, horizons={horizons})."
        )

    split = int(0.8 * len(X_design))
    X_train, X_test = X_design[:split], X_design[split:]
    Y_train, Y_test = Y_target[:split], Y_target[split:]

    logger.info(f"Training XGBRegressor (n_estimators={args.n_estimators})...")
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

    Y_pred = model.predict(X_test)
    metrics = _evaluate_metrics(Y_test, Y_pred, targets, horizons)

    feature_schema = {
        "features": features,
        "lookback": lookback,
        "output": {
            "targets": targets,
            "horizons_minutes": horizons,
        },
    }
    return model, metrics, feature_schema


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, required=True,
                        choices=["DESALTER_FORECAST", "DESALTER_GOAL_SEEK"])
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    # FORECAST-only
    parser.add_argument("--lookback", type=int, default=10,
                        help="Lookback window length (FORECAST only)")
    parser.add_argument("--horizons-minutes", type=str, default=None,
                        help='JSON list of forecast horizons in minutes (FORECAST only)')
    parser.add_argument("--features", type=str, default=None,
                        help="JSON list of feature column names")
    parser.add_argument("--targets", type=str, default=None,
                        help="JSON list of target column names")
    # GOAL_SEEK-only
    parser.add_argument("--manipulated-vars", type=str, default=None,
                        help='JSON list of manipulated variable column names (GOAL_SEEK only)')
    return parser.parse_args()


def main():
    import joblib

    args = parse_args()
    logger.info(f"Starting training: model_type={args.model_type}")

    engine = _get_db_engine()

    if args.model_type == "DESALTER_GOAL_SEEK":
        model, metrics, feature_schema = train_goal_seek(args, engine)
    else:
        model, metrics, feature_schema = train_forecast(args, engine)

    os.makedirs(MODEL_DIR, exist_ok=True)

    if args.model_type == "DESALTER_GOAL_SEEK":
        # Primary artifact: joblib pkl — loaded directly by goal_seek_service
        pkl_path = os.path.join(MODEL_DIR, "desalter_model.pkl")
        joblib.dump(model, pkl_path)
        logger.info(f"Goal-seek model (pkl) saved: {pkl_path}")
        # Secondary: XGBoost JSON for human inspection / fallback
        json_path = os.path.join(MODEL_DIR, "model.json")
        model.get_booster().save_model(json_path)
        logger.info(f"Goal-seek model (json) saved: {json_path}")
    else:
        model_path = os.path.join(MODEL_DIR, "model.json")
        model.get_booster().save_model(model_path)
        logger.info(f"Forecast model saved: {model_path}")

    # Write metrics + feature schema so registration Lambda can read them from output.tar.gz
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output = {
        "model_type": args.model_type,
        "trained_at": datetime.now(UTC).isoformat(),
        "metrics": metrics,
        "feature_schema": feature_schema,
    }
    output_path = os.path.join(OUTPUT_DIR, "training_output.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Training output written to {output_path}")

    for m in metrics:
        if "horizon" in m:
            logger.info(f"  [{m['target']}] h={m['horizon']} rmse={m['rmse']:.4f} r2={m['r2']:.4f}")
        else:
            logger.info(f"  [{m['target']}] rmse={m['rmse']:.4f} r2={m['r2']:.4f}")


if __name__ == "__main__":
    main()
