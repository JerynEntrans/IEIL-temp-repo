from dataclasses import dataclass
from enum import Enum
from typing import Any


class ProcessName(str, Enum):
    INGESTION = "INGESTION"
    VALIDATION = "VALIDATION"
    FORECAST = "FORECAST"
    GOAL_SEEK = "GOAL_SEEK"
    REPORT_GENERATION = "REPORT_GENERATION"


class ProcessingState(str, Enum):
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True)
class ModelSpec:
    id: int
    device_id: str
    model_type: str
    model_version: str
    s3_uri: str
    artifact_sha256: str | None
    feature_schema: dict[str, Any]
    metrics: dict[str, Any]

    @property
    def model_registry_id(self) -> int:
        return self.id

    @property
    def sha256(self) -> str | None:
        return self.artifact_sha256
