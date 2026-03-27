from datetime import datetime, timedelta, timezone
import os

import numpy as np

from shared.schema.db import ProcessName, ProcessingState
from shared.utils.db import Db, fetch_model_spec
from shared.utils.s3 import load_booster_from_s3
from shared.utils.sagemaker import invoke_sagemaker_endpoint

# When False the lambda loads the booster directly from S3 (localstack / non-AWS envs).
# In production with a deployed SageMaker endpoint this should be True.
_USE_SAGEMAKER_ENDPOINT = os.getenv("USE_SAGEMAKER_ENDPOINT", "true").lower() == "true"

FORECAST_TARGETS = [
    "desalter_monitoring_interface_level",
    "desalter_2_monitoring_interface_level",
    "o_h_boot_water_analysis_chloride_ppm",
    "desalter_salt_ptb_o_l",
    "desalter_brine_water_ph_ppm",
    "desalter_brine_water_oil_ppm",
]
DEFAULT_FORECAST_HORIZONS = [30, 60, 120]
UTC = timezone.utc


def _parse_ts(ts) -> datetime:
    if not ts:
        return datetime.now(UTC)
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(UTC)


def _coerce_float(v):
    return None if v is None else float(v)


def _normalize_horizons(horizons) -> list[int]:
    normalized = sorted({int(h) for h in horizons})
    if not normalized:
        raise ValueError("At least one forecast horizon is required")
    if any(h <= 0 for h in normalized):
        raise ValueError(f"Forecast horizons must be > 0. got={normalized}")
    return normalized


def run_forecast(event: dict, *, db: Db) -> dict:
    run_id = event.get("run_id")
    if not run_id:
        raise ValueError("forecast requires run_id")

    device_id = event.get("device_id", "desalter")
    validated_run_id = event.get("validated_run_id") or event.get("validation_run_id")
    if not validated_run_id:
        raise ValueError("validated_run_id (or validation_run_id) is required for run-based forecast")

    forecast_base = _parse_ts(event.get("data_end_ts") or event.get("forecast_timestamp"))
    requested_version = event.get("model_version")
    spec = fetch_model_spec(
        db,
        device_id=device_id,
        model_type="DESALTER_FORECAST",
        model_version=requested_version,
    )

    if _USE_SAGEMAKER_ENDPOINT:
        endpoint_name = spec.sagemaker_endpoint_name
        if not endpoint_name:
            raise ValueError(
                "model_registry.feature_schema_json.sagemaker_endpoint_name is required. "
                "Ensure the training registration Lambda sets this field."
            )
        booster = None
    else:
        # Local / non-SageMaker mode: load booster directly from S3 (e.g. localstack)
        booster = load_booster_from_s3(spec)
        endpoint_name = None

    schema = spec.feature_schema or {}
    features = schema.get("features") or []
    lookback = int(schema.get("lookback") or 10)
    output = schema.get("output") or {}
    trained_horizons = _normalize_horizons(output.get("horizons_minutes") or DEFAULT_FORECAST_HORIZONS)
    requested_horizons = event.get("horizons_minutes")
    horizons = _normalize_horizons(requested_horizons) if requested_horizons is not None else trained_horizons
    targets = output.get("targets") or FORECAST_TARGETS

    if not features:
        raise ValueError("model_registry.feature_schema_json.features is required for forecast model")
    if lookback <= 0:
        raise ValueError(f"feature_schema_json.lookback must be > 0. got={lookback}")
    if list(targets) != FORECAST_TARGETS:
        raise ValueError(f"Forecast targets mismatch. expected={FORECAST_TARGETS}, got={targets}")
    unknown_horizons = [h for h in horizons if h not in trained_horizons]
    if unknown_horizons:
        raise ValueError(
            f"Requested horizons {unknown_horizons} are not in model trained horizons {trained_horizons}."
        )

    db.upsert_tracker(
        run_id=run_id,
        parent_run_id=validated_run_id,
        process_name=ProcessName.FORECAST,
        device_id=device_id,
        state=ProcessingState.RUNNING,
        data_start_ts=event.get("data_start_ts"),
        data_end_ts=event.get("data_end_ts"),
        meta={
            "model_version": spec.model_version,
            "model_registry_id": spec.id,
            "validated_run_id": validated_run_id,
        },
        end_now=False,
    )

    # Metadata is persisted in tracker meta for observability; forecast table has no meta columns.
    meta_row = db.fetch_one(
        """
        SELECT unit_name, location_name, plant_name
        FROM validated_desalter_data
        WHERE run_id=%s AND device_id=%s
        ORDER BY recorded_at DESC
        LIMIT 1;
        """,
        (validated_run_id, device_id),
    )
    meta_info = {
        "unit_name": meta_row[0] if meta_row else None,
        "location_name": meta_row[1] if meta_row else None,
        "plant_name": meta_row[2] if meta_row else None,
    }

    cols = ", ".join(["recorded_at"] + features + targets)
    history_rows = db.fetch_all(
        f"""
        SELECT {cols}
        FROM validated_desalter_data
        WHERE run_id=%s AND device_id=%s
          AND recorded_at <= %s
        ORDER BY recorded_at DESC
        LIMIT %s;
        """,
        (validated_run_id, device_id, forecast_base, lookback),
    )

    if not history_rows:
        db.upsert_tracker(
            run_id=run_id,
            parent_run_id=validated_run_id,
            process_name=ProcessName.FORECAST,
            device_id=device_id,
            state=ProcessingState.SKIPPED,
            data_start_ts=event.get("data_start_ts"),
            data_end_ts=event.get("data_end_ts"),
            meta={"reason": "NO_VALIDATED_DATA_FOR_RUN", "validated_run_id": validated_run_id},
            end_now=True,
        )
        return {
            "run_id": run_id,
            "validated_run_id": validated_run_id,
            "device_id": device_id,
            "skipped": True,
            "reason": "NO_VALIDATED_DATA_FOR_RUN",
        }

    if len(history_rows) < lookback:
        db.upsert_tracker(
            run_id=run_id,
            parent_run_id=validated_run_id,
            process_name=ProcessName.FORECAST,
            device_id=device_id,
            state=ProcessingState.SKIPPED,
            data_start_ts=event.get("data_start_ts"),
            data_end_ts=event.get("data_end_ts"),
            meta={
                "reason": "INSUFFICIENT_LOOKBACK_HISTORY",
                "required_lookback": lookback,
                "available_rows": len(history_rows),
                "validated_run_id": validated_run_id,
                **meta_info,
            },
            end_now=True,
        )
        return {
            "run_id": run_id,
            "validated_run_id": validated_run_id,
            "device_id": device_id,
            "skipped": True,
            "reason": "INSUFFICIENT_LOOKBACK_HISTORY",
            "required_lookback": lookback,
            "available_rows": len(history_rows),
        }

    latest_validated_ts = history_rows[0][0]
    last_forecast_base = db.fetch_one(
        """
        SELECT MAX(forecast_timestamp - make_interval(mins => horizon_minutes))
        FROM desalter_forecast_results
        WHERE device_id=%s;
        """,
        (device_id,),
    )
    last_forecast_base = last_forecast_base[0] if last_forecast_base else None
    if last_forecast_base is not None and latest_validated_ts <= last_forecast_base:
        db.upsert_tracker(
            run_id=run_id,
            parent_run_id=validated_run_id,
            process_name=ProcessName.FORECAST,
            device_id=device_id,
            state=ProcessingState.SKIPPED,
            data_start_ts=event.get("data_start_ts"),
            data_end_ts=event.get("data_end_ts"),
            meta={
                "reason": "NO_NEW_VALIDATED_DATA",
                "latest_validated_ts": latest_validated_ts.isoformat() if hasattr(latest_validated_ts, "isoformat") else str(latest_validated_ts),
                "last_forecast_base_ts": last_forecast_base.isoformat() if hasattr(last_forecast_base, "isoformat") else str(last_forecast_base),
                "validated_run_id": validated_run_id,
                **meta_info,
            },
            end_now=True,
        )
        return {
            "run_id": run_id,
            "validated_run_id": validated_run_id,
            "device_id": device_id,
            "skipped": True,
            "reason": "NO_NEW_VALIDATED_DATA",
        }

    history_rows = list(reversed(history_rows))
    feature_hist = []
    target_hist = []
    for row in history_rows:
        feat_vals = [_coerce_float(v) for v in row[1 : 1 + len(features)]]
        tgt_vals = [_coerce_float(v) for v in row[1 + len(features) : 1 + len(features) + len(targets)]]
        if any(v is None for v in feat_vals + tgt_vals):
            raise ValueError(
                "Missing values in lookback window for forecast features/targets. "
                "Ensure validation outputs non-null training columns."
            )
        feature_hist.append(feat_vals)
        target_hist.append(tgt_vals)

    X = np.array(
        [np.hstack([np.array(target_hist, dtype=float).ravel(), np.array(feature_hist, dtype=float).ravel()])],
        dtype=float,
    )
    if _USE_SAGEMAKER_ENDPOINT:
        pred = invoke_sagemaker_endpoint(endpoint_name, X).reshape(-1)
    else:
        import xgboost as xgb
        pred = np.array(booster.predict(xgb.DMatrix(X))).reshape(-1)
    expected = len(trained_horizons) * len(targets)
    if pred.size != expected:
        raise ValueError(
            f"Unexpected prediction size: got={pred.size}, expected={expected} "
            f"(trained_horizons={trained_horizons}, targets={targets})"
        )
    pred = pred.reshape(len(trained_horizons), len(targets))
    horizon_to_idx = {h: i for i, h in enumerate(trained_horizons)}

    inserted = 0
    with db.cursor() as cur:
        for i, h in enumerate(horizons):
            ts = forecast_base + timedelta(minutes=int(h))
            pred_idx = horizon_to_idx[h]
            values = {t: float(pred[pred_idx, j]) for j, t in enumerate(targets)}
            cur.execute(
                f"""
                INSERT INTO desalter_forecast_results (
                    run_id, device_id, forecast_timestamp, horizon_minutes,
                    {','.join(FORECAST_TARGETS)}, model_version
                )
                VALUES (
                    %(run_id)s, %(device_id)s, %(forecast_timestamp)s, %(horizon_minutes)s,
                    {','.join([f'%({c})s' for c in FORECAST_TARGETS])}, %(model_version)s
                )
                ON CONFLICT (run_id, device_id, forecast_timestamp, horizon_minutes)
                DO UPDATE SET
                    {','.join([f'{c}=EXCLUDED.{c}' for c in FORECAST_TARGETS])},
                    model_version=EXCLUDED.model_version;
                """,
                {
                    "run_id": run_id,
                    "device_id": device_id,
                    "forecast_timestamp": ts,
                    "horizon_minutes": int(h),
                    **values,
                    "model_version": spec.model_version,
                },
            )
            inserted += 1
    db._conn.commit()

    db.upsert_tracker(
        run_id=run_id,
        parent_run_id=validated_run_id,
        process_name=ProcessName.FORECAST,
        device_id=device_id,
        state=ProcessingState.SUCCESS if inserted else ProcessingState.SKIPPED,
        data_start_ts=event.get("data_start_ts"),
        data_end_ts=event.get("data_end_ts"),
        meta={
            "forecast_rows": inserted,
            "horizons": horizons,
            "trained_horizons": trained_horizons,
            "lookback": lookback,
            "validated_run_id": validated_run_id,
            "model_version": spec.model_version,
            "model_registry_id": spec.id,
            **meta_info,
        },
        end_now=True,
    )

    return {
        "run_id": run_id,
        "validated_run_id": validated_run_id,
        "device_id": device_id,
        "forecast_timestamp": forecast_base.isoformat(),
        "model_version": spec.model_version,
        "model_registry_id": spec.id,
        "forecast_rows": inserted,
        "horizons_minutes": horizons,
        "skipped": inserted == 0,
    }
