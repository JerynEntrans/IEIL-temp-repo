from __future__ import annotations

import json
from datetime import datetime, timezone

from shared.utils.s3 import parse_s3_uri, get_json
from shared.schema.db import ProcessingState, ProcessName
from shared.utils.logging import set_logging

UTC = timezone.utc
logger = set_logging(__name__)

KNOWN_COLUMNS = {
    "desalter_monitoring_press_kg_cm2",
    "desalter_monitoring_w_w_temp_deg_c",
    "chemical_consumption_demulsifier_ppm_unnamed_85_level_2",
    "crude_details_crude_details_unnamed_2_level_2",
    "crude_details_api_unnamed_4_level_2",
    "crude_details_density_unnamed_5_level_2",
    "desalter_monitoring_interface_level",
    "desalter_2_monitoring_interface_level",
    "o_h_boot_water_analysis_chloride_ppm",
    "desalter_salt_ptb_o_l",
    "desalter_brine_water_ph_ppm",
    "desalter_brine_water_oil_ppm",
}

METRIC_ALIASES = {
    # Zoho metric labels -> internal validated_desalter_data columns
    "Boot Water Analysis Chloride": "o_h_boot_water_analysis_chloride_ppm",
    "O/H Boot Water Analysis Chloride PPM": "o_h_boot_water_analysis_chloride_ppm",
    "Desalter Monitoring Press": "desalter_monitoring_press_kg_cm2",
    "Desalter Monitoring Interface Level": "desalter_monitoring_interface_level",
    "Desalter 2 Monitoring Interface Level": "desalter_2_monitoring_interface_level",
}


def _to_float(v):
    if v is None:
        return None
    try:
        if isinstance(v, str) and v.strip() == "":
            return None
        return float(v)
    except Exception:
        return None


def _parse_ts(v, fallback=None) -> datetime:
    if v:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).astimezone(UTC)
    if fallback:
        return datetime.fromisoformat(str(fallback).replace("Z", "+00:00")).astimezone(UTC)
    return datetime.now(UTC)


def _extract_records(raw: dict) -> list[dict]:
    # Zoho series payload shape:
    # {"data": {"result": [{"Metric Name": [{"value":..., "timestamp":...}, ...]}]}}
    if (
        isinstance(raw, dict)
        and isinstance(raw.get("data"), dict)
        and isinstance(raw["data"].get("result"), list)
        and raw["data"]["result"]
        and isinstance(raw["data"]["result"][0], dict)
    ):
        by_ts: dict[int, dict] = {}
        metric_group = raw["data"]["result"][0]

        for metric_name, points in metric_group.items():
            if not isinstance(points, list):
                continue
            out_name = METRIC_ALIASES.get(metric_name, metric_name)

            for p in points:
                if not isinstance(p, dict):
                    continue
                ts_ms = p.get("timestamp")
                value = p.get("value")
                if ts_ms is None:
                    continue
                try:
                    ts_ms = int(ts_ms)
                except Exception:
                    continue

                row = by_ts.setdefault(ts_ms, {"recorded_at": None, "metrics": {}})
                row["recorded_at"] = datetime.fromtimestamp(ts_ms / 1000, tz=UTC).isoformat()
                row["metrics"][out_name] = value

        return [by_ts[k] for k in sorted(by_ts.keys())]

    if isinstance(raw, dict) and isinstance(raw.get("data"), list):
        recs = []
        for item in raw["data"]:
            if not isinstance(item, dict):
                continue
            ts = item.get("recorded_at") or item.get("timestamp")
            metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else item
            recs.append({"recorded_at": ts, "metrics": metrics})
        return recs

    if isinstance(raw, dict) and isinstance(raw.get("rows"), list):
        return [
            {"recorded_at": r.get("recorded_at") or r.get("timestamp"), "metrics": r}
            for r in raw["rows"]
            if isinstance(r, dict)
        ]

    return [{"recorded_at": raw.get("recorded_at") or raw.get("timestamp"), "metrics": raw}] if isinstance(raw, dict) else []


def run_validation(event: dict, *, db) -> dict:
    run_id = event.get("run_id")
    if not run_id:
        raise ValueError("validation requires run_id (should come from ingestion output)")

    device_id = event.get("device_id", "desalter")
    raw_s3_uri = event.get("raw_s3_uri") or event.get("s3_uri")
    if not raw_s3_uri:
        raise ValueError("validation requires raw_s3_uri (from ingestion output)")

    logger.info(
        "Validation started: run_id=%s device_id=%s raw_s3_uri=%s",
        run_id,
        device_id,
        raw_s3_uri,
    )

    bucket, key = parse_s3_uri(raw_s3_uri)
    logger.info("Loading raw payload from s3://%s/%s", bucket, key)
    raw = get_json(bucket, key)
    logger.info("Raw payload loaded: type=%s top_level_keys=%s", type(raw).__name__, list(raw.keys()) if isinstance(raw, dict) else None)

    db.upsert_tracker(
        run_id=run_id,
        parent_run_id=run_id,
        process_name=ProcessName.VALIDATION,
        device_id=device_id,
        state=ProcessingState.RUNNING,
        data_start_ts=event.get("data_start_ts"),
        data_end_ts=event.get("data_end_ts"),
        meta={"raw_s3_uri": raw_s3_uri},
        end_now=False,
    )

    meta = event.get("metadata") or {}
    plant_name = meta.get("plant_name")
    unit_name = meta.get("unit_name")
    location_name = meta.get("location_name")

    records = _extract_records(raw)
    inserted = 0
    logger.info("Extracted %d records for validation", len(records))

    cols_sorted = sorted(KNOWN_COLUMNS)

    try:
        with db.cursor() as cur:
            for idx, rec in enumerate(records, start=1):
                ts = _parse_ts(rec.get("recorded_at"), fallback=event.get("data_end_ts"))

                metrics = rec.get("metrics") or {}
                if not isinstance(metrics, dict):
                    logger.warning("Skipping non-dict metrics at record %d", idx)
                    continue

                known = {k: _to_float(metrics.get(k)) for k in KNOWN_COLUMNS}
                extras = {
                    k: metrics.get(k)
                    for k in metrics.keys()
                    if k not in KNOWN_COLUMNS and k not in ("timestamp", "recorded_at")
                }

                cur.execute(
                    f"""
                    INSERT INTO validated_desalter_data (
                      run_id, parent_run_id, recorded_at, device_id,
                      plant_name, unit_name, location_name,
                      {",".join(cols_sorted)},
                      extras_json
                    )
                    VALUES (
                      %(run_id)s, %(parent_run_id)s, %(recorded_at)s, %(device_id)s,
                      %(plant_name)s, %(unit_name)s, %(location_name)s,
                      {",".join([f"%({c})s" for c in cols_sorted])},
                      %(extras_json)s::jsonb
                    )
                    ON CONFLICT (run_id, device_id, recorded_at)
                    DO UPDATE SET
                      plant_name = EXCLUDED.plant_name,
                      unit_name = EXCLUDED.unit_name,
                      location_name = EXCLUDED.location_name,
                      {",".join([f"{c}=EXCLUDED.{c}" for c in cols_sorted])},
                      extras_json = validated_desalter_data.extras_json || EXCLUDED.extras_json;
                    """,
                    {
                        "run_id": run_id,
                        "parent_run_id": run_id,
                        "recorded_at": ts,
                        "device_id": device_id,
                        "plant_name": plant_name,
                        "unit_name": unit_name,
                        "location_name": location_name,
                        **known,
                        "extras_json": json.dumps(extras),
                    },
                )
                inserted += 1
                if inserted % 50 == 0:
                    logger.info("Validation progress: inserted=%d/%d", inserted, len(records))

        db._conn.commit()
        logger.info("Validation DB commit complete: inserted=%d", inserted)
    except Exception:
        logger.exception("Validation failed while processing/inserting records")
        raise

    db.upsert_tracker(
        run_id=run_id,
        parent_run_id=run_id,
        process_name=ProcessName.VALIDATION,
        device_id=device_id,
        state=ProcessingState.SUCCESS if inserted else ProcessingState.SKIPPED,
        data_start_ts=event.get("data_start_ts"),
        data_end_ts=event.get("data_end_ts"),
        meta={"validated_rows": inserted},
        end_now=True,
    )

    logger.info("Validation completed: run_id=%s inserted=%d skipped=%s", run_id, inserted, inserted == 0)

    return {
        "run_id": run_id,
        "validated_run_id": run_id,
        "device_id": device_id,
        "raw_s3_uri": raw_s3_uri,
        "validated_rows": inserted,
        "skipped": inserted == 0,
    }
