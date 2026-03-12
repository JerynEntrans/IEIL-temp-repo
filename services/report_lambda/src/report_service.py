from __future__ import annotations

import json
from datetime import datetime, timezone

import boto3

from shared.schema.db import ProcessName, ProcessingState

UTC = timezone.utc
s3 = boto3.client("s3")


def _parse_ts(value):
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def generate_report(event: dict, *, db):
    run_id = event.get("run_id")
    if not run_id:
        raise ValueError("report generation requires run_id")

    device_id = event.get("device_id", "desalter")
    report_type = event.get("report_type") or "daily_summary"
    reports_bucket = event.get("reports_s3_bucket")
    reports_prefix = (event.get("reports_s3_prefix") or "reports").strip("/")
    if not reports_bucket:
        raise ValueError("reports_s3_bucket is required")

    db.upsert_tracker(
        run_id=run_id,
        parent_run_id=event.get("validated_run_id") or run_id,
        process_name=ProcessName.REPORT_GENERATION,
        device_id=device_id,
        state=ProcessingState.RUNNING,
        data_start_ts=event.get("data_start_ts"),
        data_end_ts=event.get("data_end_ts"),
        meta={"report_type": report_type},
        end_now=False,
    )

    forecast_rows = db.fetch_all(
        """
        SELECT forecast_timestamp, horizon_minutes,
               desalter_monitoring_interface_level,
               desalter_2_monitoring_interface_level,
               o_h_boot_water_analysis_chloride_ppm,
               desalter_salt_ptb_o_l,
               desalter_brine_water_ph_ppm,
               desalter_brine_water_oil_ppm,
               model_version
        FROM desalter_forecast_results
        WHERE run_id=%s AND device_id=%s
        ORDER BY forecast_timestamp, horizon_minutes;
        """,
        (run_id, device_id),
    )

    goal_seek_row = db.fetch_one(
        """
        SELECT run_timestamp, chemical_consumption_demulsifier_ppm, result_json, model_version
        FROM desalter_goal_seek_results
        WHERE run_id=%s AND device_id=%s
        ORDER BY run_timestamp DESC
        LIMIT 1;
        """,
        (run_id, device_id),
    )

    report = {
        "run_id": run_id,
        "device_id": device_id,
        "report_type": report_type,
        "generated_at": datetime.now(UTC).isoformat(),
        "window": {
            "data_start_ts": (_parse_ts(event.get("data_start_ts")) or ""),
            "data_end_ts": (_parse_ts(event.get("data_end_ts")) or ""),
        },
        "forecast": [
            {
                "forecast_timestamp": row[0].isoformat() if hasattr(row[0], "isoformat") else str(row[0]),
                "horizon_minutes": row[1],
                "desalter_monitoring_interface_level": row[2],
                "desalter_2_monitoring_interface_level": row[3],
                "o_h_boot_water_analysis_chloride_ppm": row[4],
                "desalter_salt_ptb_o_l": row[5],
                "desalter_brine_water_ph_ppm": row[6],
                "desalter_brine_water_oil_ppm": row[7],
                "model_version": row[8],
            }
            for row in forecast_rows
        ],
        "goal_seek": None,
        "summary": {
            "forecast_row_count": len(forecast_rows),
            "has_goal_seek": goal_seek_row is not None,
        },
    }

    if goal_seek_row:
        report["goal_seek"] = {
            "run_timestamp": goal_seek_row[0].isoformat() if hasattr(goal_seek_row[0], "isoformat") else str(goal_seek_row[0]),
            "chemical_consumption_demulsifier_ppm": goal_seek_row[1],
            "result": goal_seek_row[2] or {},
            "model_version": goal_seek_row[3],
        }

    key = f"{reports_prefix}/device_id={device_id}/run_id={run_id}/{report_type}.json"
    body = json.dumps(report, default=str, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    s3.put_object(Bucket=reports_bucket, Key=key, Body=body, ContentType="application/json")
    s3_uri = f"s3://{reports_bucket}/{key}"

    db.insert_report_registry(
        run_id=run_id,
        device_id=device_id,
        report_type=report_type,
        s3_uri=s3_uri,
        meta={
            "forecast_row_count": len(forecast_rows),
            "has_goal_seek": goal_seek_row is not None,
        },
    )

    db.upsert_tracker(
        run_id=run_id,
        parent_run_id=event.get("validated_run_id") or run_id,
        process_name=ProcessName.REPORT_GENERATION,
        device_id=device_id,
        state=ProcessingState.SUCCESS,
        data_start_ts=event.get("data_start_ts"),
        data_end_ts=event.get("data_end_ts"),
        meta={"report_type": report_type, "s3_uri": s3_uri},
        end_now=True,
    )

    return {
        "run_id": run_id,
        "device_id": device_id,
        "report_type": report_type,
        "s3_uri": s3_uri,
        "forecast_row_count": len(forecast_rows),
        "has_goal_seek": goal_seek_row is not None,
    }
