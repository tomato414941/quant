from __future__ import annotations

import traceback
from pathlib import Path
from typing import Callable

from app.comparison_payloads import (
    build_comparison_payload,
    build_comparison_payload_from_run_spec_payload,
    build_comparison_run_spec_payload,
    build_condition_sweep_payload,
    build_predictor_runs_payload,
    build_ranking_evaluation_payload,
    build_strategy_runs_payload,
    generate_parameter_sweep_runs_payload,
)
from app.default_comparison import DEFAULT_COMPARISON_SPEC
from app.market_data import fetch_market_universe_bundle
from app.predictor_registry import REGISTERED_PREDICTOR_SPECS
from app.research_job_store import ResearchJobStore
from app.research_jobs import ResearchJobRecord, ResearchJobType


ResearchJobHandler = Callable[[dict], dict]


def build_research_job_handlers() -> dict[ResearchJobType, ResearchJobHandler]:
    return {
        ResearchJobType.COMPARISON: lambda _payload: build_comparison_payload(
            DEFAULT_COMPARISON_SPEC,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
        ),
        ResearchJobType.COMPARISON_RUN_SPEC: lambda _payload: build_comparison_run_spec_payload(
            DEFAULT_COMPARISON_SPEC,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
        ),
        ResearchJobType.COMPARISON_RUN_SPEC_RERUN: lambda payload: build_comparison_payload_from_run_spec_payload(
            payload,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
        ),
        ResearchJobType.PREDICTOR_RUNS: lambda _payload: build_predictor_runs_payload(
            DEFAULT_COMPARISON_SPEC,
            predictor_specs=REGISTERED_PREDICTOR_SPECS,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
        ),
        ResearchJobType.STRATEGY_RUNS: lambda _payload: build_strategy_runs_payload(
            DEFAULT_COMPARISON_SPEC,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
        ),
        ResearchJobType.CONDITION_SWEEP: lambda _payload: build_condition_sweep_payload(
            DEFAULT_COMPARISON_SPEC,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
        ),
        ResearchJobType.PARAMETER_SWEEP: lambda _payload: generate_parameter_sweep_runs_payload(
            DEFAULT_COMPARISON_SPEC,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
        ),
        ResearchJobType.RANKING_EVALUATION: lambda _payload: build_ranking_evaluation_payload(
            DEFAULT_COMPARISON_SPEC,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
        ),
    }


class ResearchJobRunner:
    def __init__(
        self,
        store: ResearchJobStore,
        handlers: dict[ResearchJobType, ResearchJobHandler] | None = None,
    ) -> None:
        self._store = store
        self._handlers = handlers if handlers is not None else build_research_job_handlers()

    def run_next_queued(self) -> ResearchJobRecord | None:
        job = self._store.claim_next_queued()
        if job is None:
            return None
        return self.run_claimed(job)

    def run_job(self, job_id: str) -> ResearchJobRecord:
        job = self._store.get(job_id)
        if job is None:
            raise ValueError(f"Unknown research job: {job_id}")
        if job.status.value == "queued":
            claimed = self._store.claim_queued(job_id)
            if claimed is None:
                raise ValueError(f"Unknown research job: {job_id}")
            job = claimed
        return self.run_claimed(job)

    def run_claimed(self, job: ResearchJobRecord) -> ResearchJobRecord:
        handler = self._handlers.get(job.job_type)
        if handler is None:
            return self._store.mark_failed(job.job_id, error=f"Unsupported research job type: {job.job_type.value}")
        try:
            result = handler(job.payload)
            result_path = self._store.save_result(job.job_id, result)
            return self._store.mark_succeeded(job.job_id, result_path=str(result_path))
        except Exception as exc:  # pragma: no cover - traceback formatting is tested through failed status.
            error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            return self._store.mark_failed(job.job_id, error=error)


def run_once(job_store_dir: Path) -> ResearchJobRecord | None:
    return ResearchJobRunner(ResearchJobStore(job_store_dir)).run_next_queued()
