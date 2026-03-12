# IEIL local-run patch set

This patch set is aimed at making the repo more runnable locally with the Docker Compose stack.

## Files included
- `requirements.txt`
- `shared/schema/db.py`
- `shared/utils/db.py`
- `shared/utils/s3.py`
- `services/forecast_lambda/src/forecast_service.py`
- `services/goal_seek_lambda/src/goal_seek_service.py`
- `services/report_lambda/src/report_service.py`

## What this fixes
1. `requirements.txt` was effectively unusable as a pip requirements file because everything was collapsed onto one commented line.
2. `Db.upsert_tracker()` now accepts `parent_run_id` and `error_message`, which the service code already passes.
3. `Db.from_env()` defaults to `sslmode=disable`, which is friendlier for local Docker Postgres.
4. `ModelSpec` / model-loading naming mismatches are normalized:
   - `id` vs `model_registry_id`
   - `artifact_sha256` vs `sha256`
5. Forecast service now calls `fetch_model_spec()` with the correct parameter name and uses consistent model metadata fields.
6. Goal-seek service no longer inserts columns that do not exist in `desalter_goal_seek_results`.
7. Report lambda now has a concrete implementation that writes a JSON report to S3 and registers it in `report_registry`.

## How to apply
Copy these files over the matching paths in your repo, then rebuild the local stack.
