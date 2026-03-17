import json
import os
from contextlib import contextmanager
from enum import Enum

import psycopg

from shared.schema.db import ModelSpec


class Db:
    def __init__(self, conn):
        self._conn = conn

    @classmethod
    def from_env(cls):
        conn = psycopg.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT", 5432),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            sslmode=os.getenv("DB_SSLMODE", "disable"),
        )
        conn.autocommit = False
        return cls(conn)

    def close(self):
        if self._conn:
            self._conn.close()

    @contextmanager
    def cursor(self):
        with self._conn.cursor() as cur:
            yield cur

    def execute(self, sql, params=None):
        with self.cursor() as cur:
            cur.execute(sql, params)
        self._conn.commit()

    def fetch_one(self, sql, params=None):
        with self.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()

    def fetch_all(self, sql, params=None):
        with self.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def get_last_success_end(self, *, device_id: str, process_name: str = "INGESTION"):
        row = self.fetch_one(
            """
            SELECT data_end_ts
            FROM process_run_tracker
            WHERE process_name = %s
              AND device_id = %s
              AND processing_state = 'SUCCESS'
              AND data_end_ts IS NOT NULL
            ORDER BY data_end_ts DESC
            LIMIT 1;
            """,
            (process_name, device_id),
        )
        return row[0] if row else None

    def upsert_tracker(
        self,
        *,
        run_id: str,
        process_name: str,
        device_id: str | None,
        state: str,
        parent_run_id: str | None = None,
        data_start_ts=None,
        data_end_ts=None,
        error: str | None = None,
        error_message: str | None = None,
        meta: dict | None = None,
        end_now: bool = False,
    ) -> None:
        meta = meta or {}
        final_error = error_message if error_message is not None else error
        self.execute(
            """
            INSERT INTO process_run_tracker (
                run_id,
                parent_run_id,
                process_name,
                device_id,
                data_start_ts,
                data_end_ts,
                processing_state,
                error_message,
                meta_json,
                process_end_ts,
                updated_at
            )
            VALUES (
                %(run_id)s,
                %(parent_run_id)s,
                %(process_name)s::process_name_enum,
                %(device_id)s,
                %(data_start_ts)s,
                %(data_end_ts)s,
                %(state)s::processing_state_enum,
                %(error_message)s,
                %(meta_json)s::jsonb,
                CASE WHEN %(end_now)s THEN now() ELSE NULL END,
                now()
            )
            ON CONFLICT (run_id, process_name) DO UPDATE SET
                parent_run_id = COALESCE(EXCLUDED.parent_run_id, process_run_tracker.parent_run_id),
                device_id = EXCLUDED.device_id,
                data_start_ts = EXCLUDED.data_start_ts,
                data_end_ts = EXCLUDED.data_end_ts,
                processing_state = EXCLUDED.processing_state,
                error_message = EXCLUDED.error_message,
                meta_json = process_run_tracker.meta_json || EXCLUDED.meta_json,
                process_end_ts = CASE
                    WHEN %(end_now)s THEN now()
                    ELSE process_run_tracker.process_end_ts
                END,
                updated_at = now();
            """,
            {
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "process_name": process_name.value if isinstance(process_name, Enum) else str(process_name),
                "device_id": device_id,
                "data_start_ts": data_start_ts,
                "data_end_ts": data_end_ts,
                "state": state.value if isinstance(state, Enum) else str(state),
                "error_message": final_error,
                "meta_json": json.dumps(meta),
                "end_now": end_now,
            },
        )

    def insert_master_registry(
        self,
        *,
        run_id: str | None,
        source_timestamp,
        file_name: str,
        stored_path: str,
        plant_name: str | None = None,
        unit_name: str | None = None,
        location_name: str | None = None,
        device_id: str | None = None,
        source_timestamp_text: str | None = None,
    ) -> None:
        self.execute(
            """
            INSERT INTO master_registry (
                run_id, source_timestamp, source_timestamp_text, file_name, stored_path,
                plant_name, unit_name, location_name, device_id
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s);
            """,
            (
                run_id,
                source_timestamp,
                source_timestamp_text,
                file_name,
                stored_path,
                plant_name,
                unit_name,
                location_name,
                device_id,
            ),
        )

    def insert_report_registry(
        self,
        *,
        run_id: str,
        device_id: str | None,
        report_type: str,
        s3_uri: str,
        meta: dict | None = None,
    ) -> None:
        self.execute(
            """
            INSERT INTO report_registry (run_id, device_id, report_type, s3_uri, meta_json)
            VALUES (%s, %s, %s, %s, %s::jsonb);
            """,
            (run_id, device_id, report_type, s3_uri, json.dumps(meta or {})),
        )


def fetch_model_spec(db, *, device_id: str, model_type: str, model_version: str | None = None) -> ModelSpec:
    if model_version:
        row = db.fetch_one(
            """
            SELECT id, device_id, model_type::text, model_version, s3_uri,
                   artifact_sha256, feature_schema_json, metrics_json
            FROM model_registry
            WHERE device_id=%s AND model_type=%s AND model_version=%s
            ORDER BY created_at DESC
            LIMIT 1;
            """,
            (device_id, model_type, model_version),
        )
        if not row:
            raise ValueError(f"No model found: device_id={device_id} model_type={model_type} model_version={model_version}")
    else:
        row = db.fetch_one(
            """
            SELECT id, device_id, model_type::text, model_version, s3_uri,
                   artifact_sha256, feature_schema_json, metrics_json
            FROM model_registry
            WHERE device_id=%s AND model_type=%s AND status='ACTIVE'
            ORDER BY activated_at DESC NULLS LAST, created_at DESC
            LIMIT 1;
            """,
            (device_id, model_type),
        )
        if not row:
            raise ValueError(f"No ACTIVE model found: device_id={device_id} model_type={model_type}")

    mid, did, mtype, mver, s3_uri, sha, feature_schema, metrics = row
    return ModelSpec(
        id=int(mid),
        device_id=str(did),
        model_type=str(mtype),
        model_version=str(mver),
        s3_uri=str(s3_uri),
        artifact_sha256=str(sha) if sha else None,
        feature_schema=feature_schema or {},
        metrics=metrics or {},
    )
