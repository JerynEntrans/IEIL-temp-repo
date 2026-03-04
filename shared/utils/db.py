import os
import json
import psycopg
from contextlib import contextmanager

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
            sslmode=os.getenv("DB_SSLMODE", "require"),
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

    # ------------------------------------------------------------
    # Pipeline helpers (Flyway V1 schema)
    # ------------------------------------------------------------

    def get_last_success_end(self, *, device_id: str, process_name: str = "INGESTION"):
        """Return latest data_end_ts for a SUCCESS run for this device+process."""
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
        data_start_ts=None,
        data_end_ts=None,
        error: str | None = None,
        meta: dict | None = None,
        end_now: bool = False,
    ) -> None:
        """Insert/update a row in process_run_tracker (unique: run_id+process_name)."""
        meta = meta or {}
        self.execute(
            """
            INSERT INTO process_run_tracker (
              run_id,
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
            ON CONFLICT (run_id, process_name)
            DO UPDATE SET
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
                "process_name": process_name,
                "device_id": device_id,
                "data_start_ts": data_start_ts,
                "data_end_ts": data_end_ts,
                "state": state,
                "error_message": error,
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
              run_id,
              source_timestamp,
              source_timestamp_text,
              file_name,
              stored_path,
              plant_name,
              unit_name,
              location_name,
              device_id
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


def fetch_model_spec(db, *, device_id: str, model_type: str, model_version: str | None = None) -> ModelSpec:
    if model_version:
        row = db.fetch_one(
            """
            SELECT id, device_id, model_type::text, model_version, s3_uri, artifact_sha256,
                   feature_schema_json, metrics_json
            FROM model_registry
            WHERE device_id=%s AND model_type=%s AND model_version=%s
            ORDER BY created_at DESC
            LIMIT 1;
            """,
            (device_id, model_type, model_version),
        )
        if not row:
            raise ValueError(f"No model found: {device_id=} {model_type=} {model_version=}")
    else:
        row = db.fetch_one(
            """
            SELECT id, device_id, model_type::text, model_version, s3_uri, artifact_sha256,
                   feature_schema_json, metrics_json
            FROM model_registry
            WHERE device_id=%s AND model_type=%s AND status='ACTIVE'
            ORDER BY activated_at DESC NULLS LAST, created_at DESC
            LIMIT 1;
            """,
            (device_id, model_type),
        )
        if not row:
            raise ValueError(f"No ACTIVE model found: {device_id=} {model_type=}")

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
