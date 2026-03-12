from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from typing import Any

import numpy as np
import xgboost as xgb

from shared.schema.db import ProcessName, ProcessingState
from shared.utils.db import fetch_model_spec
from shared.utils.s3 import load_booster_from_s3

UTC = timezone.utc


def _parse_ts(ts) -> datetime:
    if not ts:
        return datetime.now(UTC)
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(UTC)


def _coerce_float(v):
    return None if v is None else float(v)


def _score_candidate(preds: dict[str, float], *, objective: dict[str, Any]) -> float:
    total = 0.0
    max_salt = float(objective.get("max_salt_ptb", 5.0))
    max_wio = float(objective.get("max_water_in_oil_ppm", 100.0))
    t_int1 = float(objective.get("target_interface_level", 50.0))
    t_int2 = float(objective.get("target_interface2_level", 25.0))

    salt = preds.get("desalter_salt_ptb_o_l")
    wio = preds.get("desalter_brine_water_oil_ppm")
    int1 = preds.get("desalter_monitoring_interface_level")
    int2 = preds.get("desalter_2_monitoring_interface_level")

    if wio is not None and wio > max_wio:
        total += (wio - max_wio) * 5.0
    if salt is not None and salt > max_salt:
        total += (salt - max_salt) * 10.0
    if int1 is not None:
        if int1 < t_int1:
            total += (t_int1 - int1) * 5.0
        elif t_int1 <= int1 < (t_int1 + 10.0):
            total += ((t_int1 + 10.0) - int1) * 1.0
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

    requested_version = event.get("model_version")
    spec = fetch_model_spec(
        db,
        device_id=device_id,
        model_type="DESALTER_GOAL_SEEK",
        model_version=requested_version,
    )
    booster = load_booster_from_s3(spec)

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
    if "target_interface_level" in event and event["target_interface_level"] is not None:
        objective["target_interface_level"] = float(event["target_interface_level"])
    trials = int(event.get("trials") or (schema.get("search") or {}).get("trials", 200))

    if not features:
        raise ValueError("model_registry.feature_schema_json.features is required for goal-seek model")

    db.upsert_tracker(
        run_id=run_id,
        parent_run_id=validated_run_id,
        process_name=ProcessName.GOAL_SEEK,
        device_id=device_id,
        state=ProcessingState.RUNNING,
        data_start_ts=event.get("data_start_ts"),
        data_end_ts=event.get("data_end_ts"),
        meta={"model_version": spec.model_version, "model_registry_id": spec.id, "validated_run_id": validated_run_id},
        end_now=False,
    )

    cols = ", ".join(["recorded_at"] + features + targets)
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
            process_name=ProcessName.GOAL_SEEK,
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

    recorded_at = row[0]
    feat_vals = [_coerce_float(v) for v in row[1 : 1 + len(features)]]
    base_x = dict(zip(features, feat_vals))
    missing = [k for k, v in base_x.items() if v is None]
    if missing:
        raise ValueError(f"Missing required feature values in validated data: {missing}")

    controllable_keys = list(controllables.keys())
    for k in controllable_keys:
        if k not in base_x:
            raise ValueError(f"Controllable '{k}' not present in features list. Add it to feature_schema_json.features.")

    def eval_candidate(xdict: dict[str, float]) -> tuple[float, dict[str, float]]:
        X = np.array([[xdict[f] for f in features]], dtype=float)
        pred = np.array(booster.predict(xgb.DMatrix(X))).reshape(-1)
        if pred.size != len(targets):
            raise ValueError(f"Goal-seek model output size mismatch: got={pred.size}, expected={len(targets)}")
        preds = {t: float(pred[i]) for i, t in enumerate(targets)}
        return _score_candidate(preds, objective=objective), preds

    base_score, base_preds = eval_candidate(base_x)
    best = {"x": base_x, "preds": base_preds, "score": base_score, "kind": "baseline"}
    best_score = base_score

    rng = random.Random(int(event.get("seed") or 42))
    for _ in range(trials):
        cand = dict(base_x)
        for k, cfg in controllables.items():
            lo = float(cfg.get("min", 0.0))
            hi = float(cfg.get("max", lo))
            if hi < lo:
                lo, hi = hi, lo
            cand[k] = rng.uniform(lo, hi)
        score, preds = eval_candidate(cand)
        if score < best_score:
            best_score = score
            best = {"x": cand, "preds": preds, "score": score, "kind": "random_search"}

    recommended_demulsifier = best["x"].get("chemical_consumption_demulsifier_ppm_unnamed_85_level_2")
    result_json = {
        "model_registry_id": spec.id,
        "model_version": spec.model_version,
        "validated_run_id": validated_run_id,
        "recorded_at": recorded_at.isoformat() if hasattr(recorded_at, "isoformat") else str(recorded_at),
        "objective": objective,
        "search": {"trials": trials, "best_score": best["score"], "best_kind": best["kind"]},
        "recommended": {"chemical_consumption_demulsifier_ppm_unnamed_85_level_2": recommended_demulsifier},
        "predicted_kpis": best["preds"],
        "baseline_predicted_kpis": base_preds,
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
                result_json,
                model_version,
                run_id
            )
            VALUES (
                %(device_id)s,%(run_timestamp)s,%(press)s,%(wwt)s,%(demuls)s,
                %(api)s,%(density)s,%(crude)s,%(result)s::jsonb,%(model_version)s,%(run_id)s
            )
            ON CONFLICT (run_id, device_id, run_timestamp)
            DO UPDATE SET
                result_json=EXCLUDED.result_json,
                model_version=EXCLUDED.model_version,
                chemical_consumption_demulsifier_ppm=EXCLUDED.chemical_consumption_demulsifier_ppm;
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
                "result": json.dumps(result_json),
                "model_version": spec.model_version,
                "run_id": run_id,
            },
        )
    db._conn.commit()

    db.upsert_tracker(
        run_id=run_id,
        parent_run_id=validated_run_id,
        process_name=ProcessName.GOAL_SEEK,
        device_id=device_id,
        state=ProcessingState.SUCCESS,
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
