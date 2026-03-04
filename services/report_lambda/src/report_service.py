from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone

import boto3

from shared.schema.db import ProcessingState, ProcessName

UTC = timezone.utc
s3 = boto3.client("s3")


def _put_text(bucket: str, key: str, text: str, content_type: str = "text/plain") -> str:
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"), ContentType=content_type)
    return f"s3://{bucket}/{key}"


def generate_report(event: dict, *, db) -> dict:
    run_id = event.get("run_id")
    if not run_id:
        raise ValueError("report generation requires run_id")

    device_id = event.get("device_id", "desalter")
    report_type = event.get("report_type", "daily_summary")
    out_bucket = event.get("reports_s3_bucket") or event.get("s3_bucket") or ""
    out_prefix = (event.get("reports_s3_prefix") or "reports").strip("/")

    if not out_bucket:
        raise ValueError("report generation requires reports_s3_bucket (or s3_bucket)")

    generated_at = datetime.now(UTC)
    dt = generated_at.strftime("%Y-%m-%d")
    ts = generated_at.strftime("%Y%m%dT%H%M%SZ")
    key = f"{out_prefix}/device_id={device_id}/dt={dt}/run_id={run_id}/{report_type}_{ts}.csv"

    db.upsert_tracker(
        run_id=run_id,
        parent_run_id=run_id,
        process_name=ProcessName.REPORT_GENERATION,
        device_id=device_id,
        state=ProcessingState.RUNNING,
        data_start_ts=event.get("data_start_ts"),
        data_end_ts=event.get("data_end_ts"),
        meta={"report_type": report_type},
        end_now=False,
    )

    # Forecast rows for THIS RUN
    forecasts = db.fetch_all(
        """
        SELECT forecast_timestamp, horizon_minutes,
               desalter_monitoring_interface_level,
               desalter_2_monitoring_interface_level,
               o_h_boot_water_analysis_chloride_ppm,
               desalter_salt_ptb_o_l,
               desalter_brine_water_ph_ppm,
               desalter_brine_water_oil_ppm
        FROM desalter_forecast_results
        WHERE device_id=%s
          AND run_id=%s
        ORDER BY forecast_timestamp DESC, horizon_minutes ASC
        LIMIT 50;
        """,
        (device_id, run_id),
    )

    # Goal seek for THIS RUN
    goal = db.fetch_one(
        """
        SELECT run_timestamp, result_json, model_version
        FROM desalter_goal_seek_results
        WHERE device_id=%s
          AND run_id=%s
        ORDER BY run_timestamp DESC
        LIMIT 1;
        """,
        (device_id, run_id),
    )

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["section", "key", "value"])

    writer.writerow(["meta", "device_id", device_id])
    writer.writerow(["meta", "run_id", run_id])
    writer.writerow(["meta", "generated_at", generated_at.isoformat()])
    writer.writerow(["meta", "report_type", report_type])

    if goal:
        goal_ts, goal_json, goal_model = goal
        writer.writerow(["goal_seek", "run_timestamp", goal_ts.isoformat()])
        writer.writerow(["goal_seek", "model_version", goal_model])
        try:
            gj = goal_json if isinstance(goal_json, dict) else json.loads(goal_json)
        except Exception:
            gj = {"raw": str(goal_json)}
        for k, v in gj.items():
            writer.writerow(["goal_seek", k, v])
    else:
        writer.writerow(["goal_seek", "note", "no goal_seek row found for this run_id"])

    if not forecasts:
        writer.writerow(["forecast", "note", "no forecast rows found for this run_id"])

    for f in forecasts:
        ft, h, i1, i2, cl, salt, ph, oil = f
        writer.writerow(["forecast", f"ts+{h}m", ft.isoformat()])
        writer.writerow(["forecast", f"interface_level+{h}m", i1])
        writer.writerow(["forecast", f"interface2_level+{h}m", i2])
        writer.writerow(["forecast", f"chloride_ppm+{h}m", cl])
        writer.writerow(["forecast", f"salt_ptb+{h}m", salt])
        writer.writerow(["forecast", f"brine_ph+{h}m", ph])
        writer.writerow(["forecast", f"brine_oil_ppm+{h}m", oil])

    csv_text = buf.getvalue()
    s3_uri = _put_text(out_bucket, key, csv_text, content_type="text/csv")

    db.execute(
        """
        INSERT INTO report_registry (run_id, device_id, report_type, s3_uri, meta_json)
        VALUES (%s,%s,%s,%s,%s::jsonb);
        """,
        (run_id, device_id, report_type, s3_uri, json.dumps({"rows": len(csv_text.splitlines())})),
    )

    db.upsert_tracker(
        run_id=run_id,
        parent_run_id=run_id,
        process_name=ProcessName.REPORT_GENERATION,
        device_id=device_id,
        state=ProcessingState.SUCCESS,
        data_start_ts=event.get("data_start_ts"),
        data_end_ts=event.get("data_end_ts"),
        meta={"report_s3_uri": s3_uri},
        end_now=True,
    )

    return {
        "run_id": run_id,
        "device_id": device_id,
        "report_type": report_type,
        "report_s3_uri": s3_uri,
        "skipped": False,
    }
