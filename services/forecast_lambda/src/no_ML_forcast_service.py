from __future__ import annotations

from datetime import datetime, timezone, timedelta

UTC = timezone.utc

FORECAST_TARGETS = [
    "desalter_monitoring_interface_level",
    "desalter_2_monitoring_interface_level",
    "o_h_boot_water_analysis_chloride_ppm",
    "desalter_salt_ptb_o_l",
    "desalter_brine_water_ph_ppm",
    "desalter_brine_water_oil_ppm",
]
DEFAULT_FORECAST_HORIZONS = [30, 60, 120]


def run_forecast(event: dict, *, db) -> dict:
    run_id = event.get("run_id")
    device_id = event.get("device_id", "desalter")
    horizons = sorted({int(h) for h in (event.get("horizons_minutes") or DEFAULT_FORECAST_HORIZONS)})
    if any(h <= 0 for h in horizons):
        raise ValueError(f"horizons_minutes must contain only positive integers. got={horizons}")
    model_version = event.get("model_version", "naive-carry-forward")

    # Determine timestamp to forecast from
    end_ts_raw = event.get("data_end_ts") or event.get("forecast_timestamp")
    forecast_base = datetime.fromisoformat(str(end_ts_raw).replace("Z", "+00:00")).astimezone(UTC) if end_ts_raw else datetime.now(UTC)

    db.upsert_tracker(
        run_id=run_id or "UNKNOWN",
        process_name="FORECAST",
        device_id=device_id,
        state="RUNNING",
        data_start_ts=event.get("data_start_ts"),
        data_end_ts=event.get("data_end_ts"),
        meta={"model_version": model_version},
        end_now=False,
    )

    # Get latest validated row at/before base time
    row = db.fetch_one(
        f"""
        SELECT recorded_at, {",".join(FORECAST_TARGETS)}
        FROM validated_desalter_data
        WHERE device_id = %s
          AND recorded_at <= %s
        ORDER BY recorded_at DESC
        LIMIT 1;
        """,
        (device_id, forecast_base),
    )

    if not row:
        db.upsert_tracker(
            run_id=run_id or "UNKNOWN",
            process_name="FORECAST",
            device_id=device_id,
            state="SKIPPED",
            data_start_ts=event.get("data_start_ts"),
            data_end_ts=event.get("data_end_ts"),
            meta={"reason": "NO_VALIDATED_DATA"},
            end_now=True,
        )
        return {"run_id": run_id, "device_id": device_id, "skipped": True, "reason": "NO_VALIDATED_DATA"}

    latest_validated_ts = row[0]
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
            run_id=run_id or "UNKNOWN",
            process_name="FORECAST",
            device_id=device_id,
            state="SKIPPED",
            data_start_ts=event.get("data_start_ts"),
            data_end_ts=event.get("data_end_ts"),
            meta={
                "reason": "NO_NEW_VALIDATED_DATA",
                "latest_validated_ts": latest_validated_ts.isoformat() if hasattr(latest_validated_ts, "isoformat") else str(latest_validated_ts),
                "last_forecast_base_ts": last_forecast_base.isoformat() if hasattr(last_forecast_base, "isoformat") else str(last_forecast_base),
            },
            end_now=True,
        )
        return {"run_id": run_id, "device_id": device_id, "skipped": True, "reason": "NO_NEW_VALIDATED_DATA"}

    latest = dict(zip(FORECAST_TARGETS, row[1:]))
    inserted = 0

    with db.cursor() as cur:
        for h in horizons:
            ts = forecast_base + timedelta(minutes=int(h))
            cur.execute(
                f"""
                INSERT INTO desalter_forecast_results (
                  device_id, forecast_timestamp, horizon_minutes,
                  {",".join(FORECAST_TARGETS)},
                  model_version, run_id
                )
                VALUES (
                  %(device_id)s, %(forecast_timestamp)s, %(horizon_minutes)s,
                  {",".join([f"%({c})s" for c in FORECAST_TARGETS])},
                  %(model_version)s, %(run_id)s
                )
                                ON CONFLICT (run_id, device_id, forecast_timestamp, horizon_minutes)
                DO UPDATE SET
                  {",".join([f"{c}=EXCLUDED.{c}" for c in FORECAST_TARGETS])},
                  model_version=EXCLUDED.model_version,
                  run_id=EXCLUDED.run_id;
                """,
                {
                    "device_id": device_id,
                    "forecast_timestamp": ts,
                    "horizon_minutes": int(h),
                    **latest,
                    "model_version": model_version,
                    "run_id": run_id,
                },
            )
            inserted += 1

    db._conn.commit()

    db.upsert_tracker(
        run_id=run_id or "UNKNOWN",
        process_name="FORECAST",
        device_id=device_id,
        state="SUCCESS" if inserted else "SKIPPED",
        data_start_ts=event.get("data_start_ts"),
        data_end_ts=event.get("data_end_ts"),
        meta={"forecast_rows": inserted, "horizons": horizons},
        end_now=True,
    )

    return {
        "run_id": run_id,
        "device_id": device_id,
        "forecast_timestamp": forecast_base.isoformat(),
        "model_version": model_version,
        "forecast_rows": inserted,
        "skipped": inserted == 0,
        "horizons_minutes": horizons,
    }
