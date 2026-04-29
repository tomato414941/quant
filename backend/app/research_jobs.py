from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ResearchJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResearchJobType(str, Enum):
    COMPARISON = "comparison"
    COMPARISON_RUN_SPEC = "comparison_run_spec"
    COMPARISON_RUN_SPEC_RERUN = "comparison_run_spec_rerun"
    PREDICTOR_RUNS = "predictor_runs"
    STRATEGY_RUNS = "strategy_runs"
    CONDITION_SWEEP = "condition_sweep"
    PARAMETER_SWEEP = "parameter_sweep"
    RANKING_EVALUATION = "ranking_evaluation"


class ResearchJobSpec(BaseModel):
    jobType: ResearchJobType
    payload: dict[str, Any] = Field(default_factory=dict)

    @property
    def job_type(self) -> ResearchJobType:
        return self.jobType


class ResearchJobRecord(BaseModel):
    jobId: str
    status: ResearchJobStatus
    jobType: ResearchJobType
    payload: dict[str, Any] = Field(default_factory=dict)
    jobFingerprint: str
    createdAtUtc: str
    updatedAtUtc: str
    resultPath: str | None = None
    error: str | None = None

    @property
    def job_id(self) -> str:
        return self.jobId

    @property
    def job_type(self) -> ResearchJobType:
        return self.jobType

    @property
    def job_fingerprint(self) -> str:
        return self.jobFingerprint

    @property
    def created_at_utc(self) -> str:
        return self.createdAtUtc

    @property
    def updated_at_utc(self) -> str:
        return self.updatedAtUtc

    @property
    def result_path(self) -> str | None:
        return self.resultPath

    def to_public_payload(self) -> dict[str, Any]:
        payload = self.model_dump()
        payload.pop("payload", None)
        return payload


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
