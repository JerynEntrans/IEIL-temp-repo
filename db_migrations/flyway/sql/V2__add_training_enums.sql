-- V2: Add TRAINING to process_name_enum and DESALTER_TRAINING to model_type_enum

ALTER TYPE process_name_enum ADD VALUE IF NOT EXISTS 'TRAINING';
ALTER TYPE model_type_enum   ADD VALUE IF NOT EXISTS 'DESALTER_TRAINING';
