from __future__ import annotations

from datetime import datetime, timezone
import json

UTC = timezone.utc

# Minimal goal-seek: snapshot of latest validated values + simple 'recommendation' placeholder


def run_goal_seek(event: dict, *, db) -> dict:
    run_id = event.get("run_id")
    device_id = event.get("device_id", "desalter")
    model_version = event.get("model_version", "rules-v0")

    base_ts_raw = event.get("data_end_ts") or event.get("run_timestamp")
    run_ts = datetime.fromisoformat(str(base_ts_raw).replace("Z", "+00:00")).astimezone(UTC) if base_ts_raw else datetime.now(UTC)

    db.upsert_tracker(
        run_id=run_id or "UNKNOWN",
        process_name="GOAL_SEEK",
        device_id=device_id,
        state="RUNNING",
        data_start_ts=event.get("data_start_ts"),
        data_end_ts=event.get("data_end_ts"),
        meta={"model_version": model_version},
        end_now=False,
    )

    # Latest validated row
    row = db.fetch_one(
        """
        SELECT
          desalter_monitoring_press_kg_cm2,
          desalter_monitoring_w_w_temp_deg_c,
          chemical_consumption_demulsifier_ppm_unnamed_85_level_2,
          crude_details_api_unnamed_4_level_2,
          crude_details_density_unnamed_5_level_2,
          crude_details_crude_details_unnamed_2_level_2,
          desalter_monitoring_interface_level,
          desalter_2_monitoring_interface_level
        FROM validated_desalter_data
        WHERE device_id=%s
          AND recorded_at <= %s
        ORDER BY recorded_at DESC
        LIMIT 1;
        """,
        (device_id, run_ts),
    )
    if not row:
        db.upsert_tracker(
            run_id=run_id or "UNKNOWN",
            process_name="GOAL_SEEK",
            device_id=device_id,
            state="SKIPPED",
            data_start_ts=event.get("data_start_ts"),
            data_end_ts=event.get("data_end_ts"),
            meta={"reason": "NO_VALIDATED_DATA"},
            end_now=True,
        )
        return {"run_id": run_id, "device_id": device_id, "skipped": True, "reason": "NO_VALIDATED_DATA"}

    (
        press,
        wwt,
        demuls_ppm,
        api,
        density,
        crude,
        int1,
        int2,
    ) = row

    # Placeholder 'goal' and 'recommended_demulsifier_ppm'
    target_interface = event.get("target_interface_level")
    if target_interface is None:
        target_interface = 50.0

    current_interface = int1 if int1 is not None else 0.0
    delta = (target_interface - current_interface)

    recommended_demulsifier = None
    if demuls_ppm is not None:
        recommended_demulsifier = max(0.0, demuls_ppm + (delta * 0.1))

    result_json = {
        "target_interface_level": target_interface,
        "current_interface_level": current_interface,
        "delta": delta,
        "recommended_demulsifier_ppm": recommended_demulsifier,
        "notes": "placeholder rules-v0 goal-seek. replace with real optimizer later.",
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
              desalter_monitoring_interface_level,
              desalter_2_monitoring_interface_level,
              result_json,
              model_version,
              run_id
            )
            VALUES (%(device_id)s,%(run_timestamp)s,%(press)s,%(wwt)s,%(demuls)s,%(api)s,%(density)s,%(crude)s,%(int1)s,%(int2)s,%(result)s::jsonb,%(model_version)s,%(run_id)s)
            ON CONFLICT (device_id, run_timestamp)
            DO UPDATE SET
              result_json=EXCLUDED.result_json,
              model_version=EXCLUDED.model_version,
              run_id=EXCLUDED.run_id;
            """,
            {
                "device_id": device_id,
                "run_timestamp": run_ts,
                "press": press,
                "wwt": wwt,
                "demuls": recommended_demulsifier,
                "api": api,
                "density": density,
                "crude": crude,
                "int1": int1,
                "int2": int2,
                "result": json.dumps(result_json),
                "model_version": model_version,
                "run_id": run_id,
            },
        )
    db._conn.commit()

    db.upsert_tracker(
        run_id=run_id or "UNKNOWN",
        process_name="GOAL_SEEK",
        device_id=device_id,
        state="SUCCESS",
        data_start_ts=event.get("data_start_ts"),
        data_end_ts=event.get("data_end_ts"),
        meta={"goal_seek": True},
        end_now=True,
    )

    return {
        "run_id": run_id,
        "device_id": device_id,
        "run_timestamp": run_ts.isoformat(),
        "model_version": model_version,
        "result": result_json,
        "skipped": False,
    }
