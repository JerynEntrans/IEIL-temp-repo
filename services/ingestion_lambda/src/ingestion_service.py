import os
from datetime import datetime, timedelta, timezone
from shared.utils.ids import new_run_id
from shared.utils.s3 import S3RawStore
from shared.schema.db import ProcessingState, ProcessName

from .zoho.token_manager import ZohoTokenManager
from .zoho.client import ZohoIoTClient

UTC = timezone.utc


def _parse_ts(v):
    if not v:
        return None
    return datetime.fromisoformat(str(v).replace("Z", "+00:00")).astimezone(UTC)


def ingest_zoho_incremental(event: dict, *, db) -> dict:
    run_id = event.get("run_id") or new_run_id()
    device_id = event["device_id"]
    plant_id = event["plant_id"]

    lookback_hours = int(event.get("lookback_hours", 24))
    now_utc = datetime.now(UTC)

    data_end_utc = _parse_ts(event.get("data_end_ts")) or now_utc

    if event.get("data_start_ts"):
        data_start_utc = _parse_ts(event.get("data_start_ts"))
    else:
        last_success_end = db.get_last_success_end(device_id=device_id)  # must read tracker SUCCESS
        data_start_utc = last_success_end or (data_end_utc - timedelta(hours=lookback_hours))

    db.upsert_tracker(
        run_id=run_id,
        parent_run_id=None,
        process_name=ProcessName.INGESTION,
        device_id=device_id,
        state=ProcessingState.RUNNING,
        data_start_ts=data_start_utc,
        data_end_ts=data_end_utc,
        meta={"plant_id": plant_id},
        end_now=False,
    )

    try:
        if os.environ.get("OFFLINE_JSON_TESTING", "false").lower() == "true":
            import json
            with open(os.environ["OFFLINE_JSON_TESTING_FILE_PATH"], "r") as f:
                payload = json.load(f)
        else:
            token = ZohoTokenManager().get_access_token()
            client = ZohoIoTClient(token)
            payload = client.fetch_custom_range(plant_id=plant_id, from_ts=data_start_utc, to_ts=data_end_utc)

        if payload is None:
            db.upsert_tracker(
                run_id=run_id,
                parent_run_id=None,
                process_name=ProcessName.INGESTION,
                device_id=device_id,
                state=ProcessingState.SKIPPED,
                data_start_ts=data_start_utc,
                data_end_ts=data_end_utc,
                meta={"reason": "NO_DATA"},
                end_now=True,
            )
            return {
                "run_id": run_id,
                "device_id": device_id,
                "data_start_ts": data_start_utc.isoformat(),
                "data_end_ts": data_end_utc.isoformat(),
                "skipped": True,
                "reason": "NO_DATA",
            }

        store = S3RawStore(bucket=event["s3_bucket"], prefix=event["s3_prefix"])
        write_res = store.put_json(device_id=device_id, run_id=run_id, window_end_utc=data_end_utc, payload=payload)

        meta = event.get("metadata") or {}
        db.insert_master_registry(
            run_id=run_id,
            source_timestamp=data_end_utc,
            file_name="payload.json",
            stored_path=write_res.uri,
            plant_name=meta.get("plant_name"),
            unit_name=meta.get("unit_name"),
            location_name=meta.get("location_name"),
            device_id=device_id,
        )

        db.upsert_tracker(
            run_id=run_id,
            parent_run_id=None,
            process_name=ProcessName.INGESTION,
            device_id=device_id,
            state=ProcessingState.SUCCESS,
            data_start_ts=data_start_utc,
            data_end_ts=data_end_utc,
            meta={"raw_s3_uri": write_res.uri},
            end_now=True,
        )

        return {
            "run_id": run_id,
            "device_id": device_id,
            "data_start_ts": data_start_utc.isoformat(),
            "data_end_ts": data_end_utc.isoformat(),
            "raw_s3_uri": write_res.uri,
            "skipped": False,
        }

    except Exception as e:
        db.upsert_tracker(
            run_id=run_id,
            parent_run_id=None,
            process_name=ProcessName.INGESTION,
            device_id=device_id,
            state=ProcessingState.FAILED,
            data_start_ts=data_start_utc,
            data_end_ts=data_end_utc,
            error_message=str(e),
            end_now=True,
        )
        raise
