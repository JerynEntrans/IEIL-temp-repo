# Added pipelines (Airflow + Lambda)

This repo now supports thin Airflow orchestration with the following Lambda-style services:

- **Ingestion**: `services/ingestion_lambda` → writes raw payload to S3 and registers in `master_registry`
- **Validation / ETL**: `services/validation_lambda` → reads raw payload from S3 and writes to `validated_desalter_data`
- **Forecast**: `services/forecast_lambda` → naive carry-forward forecast into `desalter_forecast_results`
- **Goal Seek**: `services/goal_seek_lambda` → placeholder rule-based goal seek into `desalter_goal_seek_results`
- **Report Generation**: `services/report_lambda` → builds a CSV summary, stores to S3, registers in `report_registry`

## Airflow DAGs

- `zoho_iot_ingestion` (ingestion only)
- `validation_pipeline`
- `forecast_pipeline`
- `goal_seek_pipeline`
- `report_generation_pipeline`
- `desalter_end_to_end` (full chain)

## Next steps

Replace the placeholder logic in forecast/goal-seek with the real model code, keeping the same DB tables and tracker updates.
