from datetime import datetime, timedelta, timezone

import numpy as np
import xgboost as xgb

from shared.schema.db import ProcessName, ProcessingState
from shared.utils.db import Db, fetch_model_spec
from shared.utils.s3 import load_booster_from_s3

FORECAST_TARGETS = [
    "desalter_monitoring_interface_level",
    "desalter_2_monitoring_interface_level",
    "o_h_boot_water_analysis_chloride_ppm",
    "desalter_salt_ptb_o_l",
    "desalter_brine_water_ph_ppm",
    "desalter_brine_water_oil_ppm",
]
UTC = timezone.utc


def _parse_ts(ts) -> datetime:
    if not ts:
        return datetime.now(UTC)
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(UTC)


def _coerce_float(v):
    return None if v is None else float(v)


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
    booster = load_booster_from_s3(spec)

    schema = spec.feature_schema or {}
    features = schema.get("features") or []
    output = schema.get("output") or {}
    horizons = event.get("horizons_minutes") or output.get("horizons_minutes") or [0, 30, 60, 120]
    targets = output.get("targets") or FORECAST_TARGETS

    if not features:
        raise ValueError("model_registry.feature_schema_json.features is required for forecast model")
    if list(targets) != FORECAST_TARGETS:
        raise ValueError(f"Forecast targets mismatch. expected={FORECAST_TARGETS}, got={targets}")

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

    cols = ", ".join(["recorded_at"] + features)
    row = db.fetch_one(
        f"""
        SELECT {cols}
        FROM validated_desalter_data
        WHERE run_id=%s AND device_id=%s
        ORDER BY recorded_at DESC
        LIMIT 1;
        """,
        (validated_run_id, device_id),
    )

    if not row:
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

    x_vals = [_coerce_float(v) for v in row[1:]]
    if any(v is None for v in x_vals):
        raise ValueError("Missing feature values for forecast. Add fill strategy or ensure validation outputs non-null features.")

    X = np.array([x_vals], dtype=float)
    pred = np.array(booster.predict(xgb.DMatrix(X))).reshape(-1)
    expected = len(horizons) * len(targets)
    if pred.size != expected:
        raise ValueError(
            f"Unexpected prediction size: got={pred.size}, expected={expected} "
            f"(horizons={horizons}, targets={targets})"
        )
    pred = pred.reshape(len(horizons), len(targets))

    inserted = 0
    with db.cursor() as cur:
        for i, h in enumerate(horizons):
            ts = forecast_base + timedelta(minutes=int(h))
            values = {t: float(pred[i, j]) for j, t in enumerate(targets)}
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
            "validated_run_id": validated_run_id,
            "model_version": spec.model_version,
            "model_registry_id": spec.id,
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
