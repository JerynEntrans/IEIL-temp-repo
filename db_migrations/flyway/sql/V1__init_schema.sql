-- ============================================================
-- V1__init_schema.sql
-- Generic schema for ingestion + validated timeseries
-- + forecast + goal-seek + reports + model registry
-- Run-id lineage friendly
-- ============================================================

-- ============================================================
-- 0) ENUMS
-- ============================================================

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'processing_state_enum') THEN
    CREATE TYPE processing_state_enum AS ENUM (
      'RUNNING',
      'SUCCESS',
      'FAILED',
      'SKIPPED'
    );
  END IF;
END$$;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'process_name_enum') THEN
    CREATE TYPE process_name_enum AS ENUM (
      'INGESTION',
      'VALIDATION',
      'FORECAST',
      'GOAL_SEEK',
      'REPORT_GENERATION'
    );
  END IF;
END$$;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'model_status_enum') THEN
    CREATE TYPE model_status_enum AS ENUM (
      'STAGED',
      'ACTIVE',
      'DEPRECATED'
    );
  END IF;
END$$;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'model_type_enum') THEN
    CREATE TYPE model_type_enum AS ENUM (
      'DESALTER_FORECAST',
      'DESALTER_GOAL_SEEK'
    );
  END IF;
END$$;

-- ============================================================
-- 1) master_registry
-- ============================================================

CREATE TABLE IF NOT EXISTS master_registry (
  id BIGSERIAL PRIMARY KEY,

  run_id TEXT NOT NULL,

  source_timestamp      TIMESTAMPTZ,
  source_timestamp_text TEXT,

  file_name   TEXT,
  stored_path TEXT,

  plant_name    TEXT,  -- e.g. 'CPCL_Manali', 'Reliance_Jamnagar', 'IOC_Panipat'
  unit_name     TEXT,  -- e.g. 'CDU1', 'CDU2', 'Desalting_Unit_A'
  location_name TEXT,  -- e.g. 'Chennai', 'Jamnagar', 'Panipat'
  device_id     TEXT,  -- e.g. 'desalter', 'desalter_1', 'desalter_A_train'

  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_master_registry_created_at
  ON master_registry(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_master_registry_device
  ON master_registry(device_id);

CREATE INDEX IF NOT EXISTS idx_master_registry_run_id
  ON master_registry(run_id)
  WHERE run_id IS NOT NULL;

-- ============================================================
-- 2) process_run_tracker
-- ============================================================

CREATE TABLE IF NOT EXISTS process_run_tracker (
  id BIGSERIAL PRIMARY KEY,

  run_id TEXT NOT NULL,
  parent_run_id TEXT,

  process_name process_name_enum NOT NULL,
  device_id    TEXT,

  process_start_ts TIMESTAMPTZ NOT NULL DEFAULT now(),
  process_end_ts   TIMESTAMPTZ,

  data_start_ts TIMESTAMPTZ,
  data_end_ts   TIMESTAMPTZ,

  processing_state processing_state_enum NOT NULL DEFAULT 'RUNNING',
  error_message    TEXT,

  meta_json JSONB NOT NULL DEFAULT '{}'::jsonb,

  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

  CONSTRAINT uq_process_run_tracker_run_id_process
    UNIQUE (run_id, process_name)
);

CREATE INDEX IF NOT EXISTS idx_process_run_tracker_run_id
  ON process_run_tracker(run_id);

CREATE INDEX IF NOT EXISTS idx_process_run_tracker_parent_run_id
  ON process_run_tracker(parent_run_id)
  WHERE parent_run_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_process_run_tracker_state
  ON process_run_tracker(processing_state);

-- ============================================================
-- 3) validated_desalter_data
-- ============================================================

CREATE TABLE IF NOT EXISTS validated_desalter_data (
  id BIGSERIAL PRIMARY KEY,

  run_id        TEXT NOT NULL,
  parent_run_id TEXT,

  recorded_at   TIMESTAMPTZ NOT NULL,
  device_id     TEXT NOT NULL,

  plant_name    TEXT,
  unit_name     TEXT,
  location_name TEXT,

  desalter_monitoring_press_kg_cm2                        DOUBLE PRECISION,
  desalter_monitoring_w_w_temp_deg_c                      DOUBLE PRECISION,
  chemical_consumption_demulsifier_ppm_unnamed_85_level_2 DOUBLE PRECISION,
  crude_details_crude_details_unnamed_2_level_2           DOUBLE PRECISION,
  crude_details_api_unnamed_4_level_2                     DOUBLE PRECISION,
  crude_details_density_unnamed_5_level_2                 DOUBLE PRECISION,
  desalter_monitoring_interface_level                     DOUBLE PRECISION,
  desalter_2_monitoring_interface_level                   DOUBLE PRECISION,
  o_h_boot_water_analysis_chloride_ppm                    DOUBLE PRECISION,
  desalter_salt_ptb_o_l                                   DOUBLE PRECISION,
  desalter_brine_water_ph_ppm                             DOUBLE PRECISION,
  desalter_brine_water_oil_ppm                            DOUBLE PRECISION,

  extras_json JSONB NOT NULL DEFAULT '{}'::jsonb,

  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

  CONSTRAINT uq_validated_desalter_data_run_device_time
    UNIQUE (run_id, device_id, recorded_at)
);

CREATE INDEX IF NOT EXISTS idx_validated_desalter_data_run_id
  ON validated_desalter_data(run_id);

CREATE INDEX IF NOT EXISTS idx_validated_desalter_data_parent_run_id
  ON validated_desalter_data(parent_run_id)
  WHERE parent_run_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_validated_desalter_data_device_time
  ON validated_desalter_data(device_id, recorded_at DESC);

-- ============================================================
-- 4) desalter_forecast_results
-- ============================================================

CREATE TABLE IF NOT EXISTS desalter_forecast_results (
  id BIGSERIAL PRIMARY KEY,

  device_id          TEXT NOT NULL,
  forecast_timestamp TIMESTAMPTZ NOT NULL,
  horizon_minutes    INTEGER NOT NULL DEFAULT 0,

  desalter_monitoring_interface_level   DOUBLE PRECISION,
  desalter_2_monitoring_interface_level DOUBLE PRECISION,
  o_h_boot_water_analysis_chloride_ppm  DOUBLE PRECISION,
  desalter_salt_ptb_o_l                 DOUBLE PRECISION,
  desalter_brine_water_ph_ppm           DOUBLE PRECISION,
  desalter_brine_water_oil_ppm          DOUBLE PRECISION,

  model_version TEXT,
  run_id        TEXT NOT NULL,

  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

  CONSTRAINT uq_desalter_forecast_results_run_device_ts
    UNIQUE (run_id, device_id, forecast_timestamp, horizon_minutes)
);

CREATE INDEX IF NOT EXISTS idx_desalter_forecast_results_run_id
  ON desalter_forecast_results(run_id);

-- ============================================================
-- 5) desalter_goal_seek_results
-- ============================================================

CREATE TABLE IF NOT EXISTS desalter_goal_seek_results (
  id BIGSERIAL PRIMARY KEY,

  device_id     TEXT NOT NULL,
  run_timestamp TIMESTAMPTZ NOT NULL,

  "desalter_monitoring_mix_valve_δp_kg_cm2" DOUBLE PRECISION,
  desalter_monitoring_press_kg_cm2          DOUBLE PRECISION,
  desalter_monitoring_w_w_temp_deg_c        DOUBLE PRECISION,
  desalter_monitoring_amp3_amps             DOUBLE PRECISION,
  desalter_2_monitoring_temp_deg_c          DOUBLE PRECISION,
  chemical_consumption_demulsifier_ppm      DOUBLE PRECISION,
  crude_details_api                         DOUBLE PRECISION,
  crude_details_density                     DOUBLE PRECISION,
  crude_details_crude_details               DOUBLE PRECISION,

  result_json JSONB NOT NULL DEFAULT '{}'::jsonb,

  model_version TEXT,
  run_id        TEXT NOT NULL,

  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

  CONSTRAINT uq_goal_seek_run_device_ts
    UNIQUE (run_id, device_id, run_timestamp)
);

CREATE INDEX IF NOT EXISTS idx_desalter_goal_seek_results_run_id
  ON desalter_goal_seek_results(run_id);

-- ============================================================
-- 6) report_registry
-- ============================================================

CREATE TABLE IF NOT EXISTS report_registry (
  id BIGSERIAL PRIMARY KEY,

  run_id    TEXT NOT NULL,
  device_id TEXT,

  report_type TEXT NOT NULL,
  s3_uri     TEXT NOT NULL,

  generated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  meta_json    JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_report_registry_run_id
  ON report_registry(run_id);

-- ============================================================
-- 7) model_registry
-- ============================================================

CREATE TABLE IF NOT EXISTS model_registry (
  id BIGSERIAL PRIMARY KEY,

  device_id     TEXT NOT NULL,
  model_type    model_type_enum NOT NULL,
  model_version TEXT NOT NULL,

  s3_uri          TEXT NOT NULL,
  artifact_sha256 TEXT,

  feature_schema_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  metrics_json        JSONB NOT NULL DEFAULT '{}'::jsonb,
  trained_data_json   JSONB NOT NULL DEFAULT '{}'::jsonb,

  status model_status_enum NOT NULL DEFAULT 'STAGED',
  activated_at TIMESTAMPTZ,

  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_model_registry_active
  ON model_registry(device_id, model_type)
  WHERE status = 'ACTIVE';

CREATE INDEX IF NOT EXISTS idx_model_registry_lookup
  ON model_registry(device_id, model_type, status, created_at DESC);