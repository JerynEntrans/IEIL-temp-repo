from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import random
from typing import Any

import boto3
import numpy as np
import xgboost as xgb

UTC = timezone.utc
_s3 = boto3.client("s3")


def _parse_ts(ts) -> datetime:
    if not ts:
        return datetime.now(UTC)
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(UTC)


def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid s3 uri: {uri}")
    rest = uri[len("s3://"):]
    bucket, key = rest.split("/", 1)
    return bucket, key


@dataclass(frozen=True)
class ModelSpec:
    id: int
    device_id: str
    model_type: str
    model_version: str
    s3_uri: str
    artifact_sha256: str | None
    feature_schema: dict[str, Any]


def _fetch_model_spec(db, *, device_id: str, model_type: str, requested_version: str | None) -> ModelSpec:
    if requested_version:
        row = db.fetch_one(
            """
            SELECT id, device_id, model_type::text, model_version, s3_uri, artifact_sha256, feature_schema_json
            FROM model_registry
            WHERE device_id=%s AND model_type=%s AND model_version=%s
            ORDER BY created_at DESC
            LIMIT 1;
            """,
            (device_id, model_type, requested_version),
        )
        if not row:
            raise ValueError(f"No model found for {device_id=} {model_type=} {requested_version=}")
    else:
        row = db.fetch_one(
            """
            SELECT id, device_id, model_type::text, model_version, s3_uri, artifact_sha256, feature_schema_json
            FROM model_registry
            WHERE device_id=%s AND model_type=%s AND status='ACTIVE'
            ORDER BY activated_at DESC NULLS LAST, created_at DESC
            LIMIT 1;
            """,
            (device_id, model_type),
        )
        if not row:
            raise ValueError(f"No ACTIVE model found for {device_id=} {model_type=}")

    mid, did, mtype, mver, s3_uri, sha, schema = row
    return ModelSpec(
        id=int(mid),
        device_id=str(did),
        model_type=str(mtype),
        model_version=str(mver),
        s3_uri=str(s3_uri),
        artifact_sha256=str(sha) if sha else None,
        feature_schema=schema or {},
    )


def _load_booster(spec: ModelSpec) -> xgb.Booster:
    cache_path = f"/tmp/xgb_{spec.id}.json"
    if os.path.exists(cache_path):
        b = xgb.Booster()
        b.load_model(cache_path)
        return b

    bucket, key = _parse_s3_uri(spec.s3_uri)
    obj = _s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()

    got = _sha256_bytes(body)
    if spec.artifact_sha256 and got != spec.artifact_sha256:
        raise ValueError(f"Model sha256 mismatch: expected={spec.artifact_sha256}, got={got}")

    with open(cache_path, "wb") as f:
        f.write(body)

    b = xgb.Booster()
    b.load_model(cache_path)
    return b


def _coerce_float(v):
    return None if v is None else float(v)


def _score_candidate(preds: dict[str, float], *, objective: dict[str, Any]) -> float:
    """
    Lower is better. Implements penalty style similar to your old code.
    """
    total = 0.0

    # Constraints / targets (defaults if missing)
    max_salt = float(objective.get("max_salt_ptb", 5.0))
    max_wio = float(objective.get("max_water_in_oil_ppm", 100.0))
    t_int1 = float(objective.get("target_interface_level", 50.0))
    t_int2 = float(objective.get("target_interface2_level", 25.0))

    salt = preds.get("desalter_salt_ptb_o_l")
    wio = preds.get("desalter_brine_water_oil_ppm")
    int1 = preds.get("desalter_monitoring_interface_level")
    int2 = preds.get("desalter_2_monitoring_interface_level")

    # water-in-oil
    if wio is not None and wio > max_wio:
        total += (wio - max_wio) * 5.0

    # salt
    if salt is not None and salt > max_salt:
        total += (salt - max_salt) * 10.0

    # interface 1: encourage >= t_int1, soft band up to t_int1+10
    if int1 is not None:
        if int1 < t_int1:
            total += (t_int1 - int1) * 5.0
        elif t_int1 <= int1 < (t_int1 + 10.0):
            total += ((t_int1 + 10.0) - int1) * 1.0

    # interface 2: encourage >= t_int2, soft band up to t_int2+5
    if int2 is not None:
        if int2 < t_int2:
            total += (t_int2 - int2) * 5.0
        elif t_int2 <= int2 < (t_int2 + 5.0):
            total += ((t_int2 + 5.0) - int2) * 1.0

    return total


def run_goal_seek(event: dict, *, db) -> dict:
    run_id = event.get("run_id")
    if not run_id:
        raise ValueError("goal_seek requires run_id")

    device_id = event.get("device_id", "desalter")
    run_ts = _parse_ts(event.get("data_end_ts") or event.get("run_timestamp"))

    validated_run_id = event.get("validated_run_id") or event.get("validation_run_id") or run_id
    if not validated_run_id:
        raise ValueError("validated_run_id (or run_id) is required")

    # Pick model: pinned version OR ACTIVE
    requested_version = event.get("model_version")  # optional
    spec = _fetch_model_spec(db, device_id=device_id, model_type="DESALTER_GOAL_SEEK", requested_version=requested_version)
    booster = _load_booster(spec)

    schema = spec.feature_schema or {}
    features: list[str] = schema.get("features") or []
    targets: list[str] = schema.get("targets") or [
        "desalter_monitoring_interface_level",
        "desalter_2_monitoring_interface_level",
        "desalter_salt_ptb_o_l",
        "desalter_brine_water_oil_ppm",
    ]
    controllables: dict[str, dict[str, Any]] = schema.get("controllables") or {
        "chemical_consumption_demulsifier_ppm_unnamed_85_level_2": {"min": 0.0, "max": 200.0}
    }
    objective: dict[str, Any] = schema.get("objective") or {}

    # Allow event override for key objectives
    if "target_interface_level" in event and event["target_interface_level"] is not None:
        objective["target_interface_level"] = float(event["target_interface_level"])

    trials = int((schema.get("search") or {}).get("trials", 200))

    if not features:
        raise ValueError("model_registry.feature_schema_json.features is required for goal-seek model")

    db.upsert_tracker(
        run_id=run_id,
        parent_run_id=validated_run_id,
        process_name="GOAL_SEEK",
        device_id=device_id,
        state="RUNNING",
        data_start_ts=event.get("data_start_ts"),
        data_end_ts=event.get("data_end_ts"),
        meta={"model_version": spec.model_version, "model_registry_id": spec.id, "validated_run_id": validated_run_id},
        end_now=False,
    )

    # Pull latest validated row in the requested window/run
    # Prefer deterministic run-based fetch:
    cols = ", ".join(["recorded_at"] + features + targets)
    row = db.fetch_one(
        f"""
        SELECT {cols}
        FROM validated_desalter_data
        WHERE run_id=%s
          AND device_id=%s
        ORDER BY recorded_at DESC
        LIMIT 1;
        """,
        (validated_run_id, device_id),
    )

    if not row:
        db.upsert_tracker(
            run_id=run_id,
            parent_run_id=validated_run_id,
            process_name="GOAL_SEEK",
            device_id=device_id,
            state="SKIPPED",
            data_start_ts=event.get("data_start_ts"),
            data_end_ts=event.get("data_end_ts"),
            meta={"reason": "NO_VALIDATED_DATA_FOR_RUN", "validated_run_id": validated_run_id},
            end_now=True,
        )
        return {"run_id": run_id, "validated_run_id": validated_run_id, "device_id": device_id, "skipped": True, "reason": "NO_VALIDATED_DATA_FOR_RUN"}

    # row = (recorded_at, features..., targets...)
    recorded_at = row[0]
    feat_vals = [_coerce_float(v) for v in row[1: 1 + len(features)]]

    # baseline values dict
    base_x = dict(zip(features, feat_vals))

    # Fail fast if missing features (or implement fill strategy later)
    missing = [k for k, v in base_x.items() if v is None]
    if missing:
        raise ValueError(f"Missing required feature values in validated data: {missing}")

    # Determine controllable baseline values
    controllable_keys = list(controllables.keys())
    for k in controllable_keys:
        if k not in base_x:
            raise ValueError(f"Controllable '{k}' not present in features list. Add it to feature_schema_json.features.")

    best = None
    best_score = float("inf")

    # Always include current state as a candidate (no-change)
    def eval_candidate(xdict: dict[str, float]) -> tuple[float, dict[str, float]]:
        X = np.array([[xdict[f] for f in features]], dtype=float)
        pred = booster.predict(xgb.DMatrix(X))
        pred = np.array(pred).reshape(-1)

        if pred.size != len(targets):
            raise ValueError(f"Goal-seek model output size mismatch: got={pred.size}, expected={len(targets)}")

        preds = {t: float(pred[i]) for i, t in enumerate(targets)}
        return _score_candidate(preds, objective=objective), preds

    # Evaluate base
    base_score, base_preds = eval_candidate(base_x)
    best = {"x": base_x, "preds": base_preds, "score": base_score, "kind": "baseline"}
    best_score = base_score

    # Random search
    rng = random.Random(int(event.get("seed") or 42))
    for _ in range(trials):
        cand = dict(base_x)

        for k, cfg in controllables.items():
            lo = float(cfg.get("min", 0.0))
            hi = float(cfg.get("max", lo))
            if hi < lo:
                lo, hi = hi, lo

            # sample uniformly; you can switch to gaussian around baseline later
            cand[k] = rng.uniform(lo, hi)

        score, preds = eval_candidate(cand)
        if score < best_score:
            best_score = score
            best = {"x": cand, "preds": preds, "score": score, "kind": "random_search"}

    assert best is not None

    # Extract recommended values (for now, only demulsifier is actionable)
    recommended_demulsifier = best["x"].get("chemical_consumption_demulsifier_ppm_unnamed_85_level_2")

    result_json = {
        "model_registry_id": spec.id,
        "model_version": spec.model_version,
        "validated_run_id": validated_run_id,
        "recorded_at": recorded_at.isoformat() if hasattr(recorded_at, "isoformat") else str(recorded_at),
        "objective": objective,
        "search": {"trials": trials, "best_score": best["score"], "best_kind": best["kind"]},
        "recommended": {
            "chemical_consumption_demulsifier_ppm_unnamed_85_level_2": recommended_demulsifier
        },
        "predicted_kpis": best["preds"],
        "baseline_predicted_kpis": base_preds,
    }

    # Write results
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
            VALUES (
              %(device_id)s,%(run_timestamp)s,%(press)s,%(wwt)s,%(demuls)s,%(api)s,%(density)s,%(crude)s,%(int1)s,%(int2)s,%(result)s::jsonb,%(model_version)s,%(run_id)s
            )
            ON CONFLICT (run_id, device_id, run_timestamp)
            DO UPDATE SET
              result_json=EXCLUDED.result_json,
              model_version=EXCLUDED.model_version;
            """,
            {
                "device_id": device_id,
                "run_timestamp": run_ts,
                "press": base_x.get("desalter_monitoring_press_kg_cm2"),
                "wwt": base_x.get("desalter_monitoring_w_w_temp_deg_c"),
                "demuls": recommended_demulsifier,
                "api": base_x.get("crude_details_api_unnamed_4_level_2"),
                "density": base_x.get("crude_details_density_unnamed_5_level_2"),
                "crude": base_x.get("crude_details_crude_details_unnamed_2_level_2"),
                # these are measured last values (not predicted)
                "int1": row[1 + len(features) + targets.index("desalter_monitoring_interface_level")] if "desalter_monitoring_interface_level" in targets else None,
                "int2": row[1 + len(features) + targets.index("desalter_2_monitoring_interface_level")] if "desalter_2_monitoring_interface_level" in targets else None,
                "result": json.dumps(result_json),
                "model_version": spec.model_version,
                "run_id": run_id,
            },
        )

    db._conn.commit()

    db.upsert_tracker(
        run_id=run_id,
        parent_run_id=validated_run_id,
        process_name="GOAL_SEEK",
        device_id=device_id,
        state="SUCCESS",
        data_start_ts=event.get("data_start_ts"),
        data_end_ts=event.get("data_end_ts"),
        meta={"goal_seek": True, "model_registry_id": spec.id, "model_version": spec.model_version, "best_score": best["score"]},
        end_now=True,
    )

    return {
        "run_id": run_id,
        "validated_run_id": validated_run_id,
        "device_id": device_id,
        "run_timestamp": run_ts.isoformat(),
        "model_version": spec.model_version,
        "model_registry_id": spec.id,
        "skipped": False,
        "result": result_json,
    }
