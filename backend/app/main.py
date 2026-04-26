from __future__ import annotations

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.default_comparison import DEFAULT_COMPARISON_SPEC
from app.comparison_market_context import build_run_result_store
from app.comparison_payloads import (
    build_condition_sweep_payload,
    build_comparison_payload,
    build_comparison_payload_from_run_spec_payload,
    build_comparison_run_spec_payload,
    build_latest_run_payload,
    build_predictor_run_detail_payload,
    build_predictor_run_index_payload,
    build_predictor_runs_payload,
    build_ranking_evaluation_payload,
    build_run_catalog_payload,
    build_strategy_run_detail_payload,
    build_strategy_run_index_payload,
    build_strategy_runs_payload,
    generate_parameter_sweep_runs_payload,
)
from app.predictor_registry import REGISTERED_PREDICTOR_SPECS
from app.market_data import fetch_market_universe_bundle
from app.strategy_inventory import build_strategy_inventory_payload


app = FastAPI(title="Quant API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|[\w\.-]+)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/comparison")
async def comparison() -> dict:
    try:
        return build_comparison_payload(
            DEFAULT_COMPARISON_SPEC,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/comparison-run-spec")
async def comparison_run_spec() -> dict:
    try:
        return build_comparison_run_spec_payload(
            DEFAULT_COMPARISON_SPEC,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/comparison-run-spec/rerun")
async def rerun_comparison_run_spec(payload: dict = Body(...)) -> dict:
    try:
        return build_comparison_payload_from_run_spec_payload(
            payload,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/predictor-runs")
async def predictor_run_index(
    limit: int = Query(50, ge=1, le=500),
    learner_kind: str | None = Query(None),
    combiner_kind: str | None = Query(None),
    signal_source_kind: str | None = Query(None),
    signal_source_feature_key: str | None = Query(None),
    horizon_value: int | None = Query(None, ge=1),
    strategy_definition_fingerprint: str | None = Query(None),
    market_data_fingerprint: str | None = Query(None),
    evaluation_fingerprint: str | None = Query(None),
    sort_by: str = Query("test_rank_ic"),
) -> dict:
    try:
        return build_predictor_run_index_payload(
            DEFAULT_COMPARISON_SPEC,
            limit=limit,
            learner_kind=learner_kind,
            combiner_kind=combiner_kind,
            signal_source_kind=signal_source_kind,
            signal_source_feature_key=signal_source_feature_key,
            horizon_value=horizon_value,
            strategy_definition_fingerprint=strategy_definition_fingerprint,
            market_data_fingerprint=market_data_fingerprint,
            evaluation_fingerprint=evaluation_fingerprint,
            sort_by=sort_by,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/predictor-runs")
async def predictor_runs() -> dict:
    try:
        return build_predictor_runs_payload(
            DEFAULT_COMPARISON_SPEC,
            predictor_specs=REGISTERED_PREDICTOR_SPECS,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/predictor-runs/{run_key}")
async def predictor_run_detail(run_key: str) -> dict:
    try:
        return build_predictor_run_detail_payload(
            DEFAULT_COMPARISON_SPEC,
            run_key=run_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/strategy-runs")
async def strategy_run_index(
    limit: int = Query(50, ge=1, le=500),
    strategy_definition_fingerprint: str | None = Query(None),
    market_data_fingerprint: str | None = Query(None),
    evaluation_fingerprint: str | None = Query(None),
) -> dict:
    try:
        return build_strategy_run_index_payload(
            DEFAULT_COMPARISON_SPEC,
            limit=limit,
            strategy_definition_fingerprint=strategy_definition_fingerprint,
            market_data_fingerprint=market_data_fingerprint,
            evaluation_fingerprint=evaluation_fingerprint,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/strategy-runs")
async def strategy_runs() -> dict:
    try:
        return build_strategy_runs_payload(
            DEFAULT_COMPARISON_SPEC,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/strategy-runs/{run_key}")
async def strategy_run_detail(run_key: str) -> dict:
    try:
        return build_strategy_run_detail_payload(
            DEFAULT_COMPARISON_SPEC,
            run_key=run_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/condition-sweep")
async def condition_sweep() -> dict:
    try:
        return build_condition_sweep_payload(
            DEFAULT_COMPARISON_SPEC,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/runs/generate-parameter-sweep")
async def generate_parameter_sweep_runs() -> dict:
    try:
        return generate_parameter_sweep_runs_payload(
            DEFAULT_COMPARISON_SPEC,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/run-catalog")
async def run_catalog(
    limit: int = Query(50, ge=1, le=500),
    run_kind: str | None = Query(None),
    generation_method: str | None = Query(None),
    strategy_definition_fingerprint: str | None = Query(None),
    market_data_fingerprint: str | None = Query(None),
    evaluation_fingerprint: str | None = Query(None),
) -> dict:
    try:
        return build_run_catalog_payload(
            DEFAULT_COMPARISON_SPEC,
            limit=limit,
            run_kind=run_kind,
            generation_method=generation_method,
            strategy_definition_fingerprint=strategy_definition_fingerprint,
            market_data_fingerprint=market_data_fingerprint,
            evaluation_fingerprint=evaluation_fingerprint,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/run-catalog/latest")
async def latest_run_catalog_record(
    run_kind: str | None = Query(None),
    generation_method: str | None = Query(None),
    strategy_definition_fingerprint: str | None = Query(None),
    market_data_fingerprint: str | None = Query(None),
    evaluation_fingerprint: str | None = Query(None),
) -> dict:
    try:
        return build_latest_run_payload(
            DEFAULT_COMPARISON_SPEC,
            run_kind=run_kind,
            generation_method=generation_method,
            strategy_definition_fingerprint=strategy_definition_fingerprint,
            market_data_fingerprint=market_data_fingerprint,
            evaluation_fingerprint=evaluation_fingerprint,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/strategy-inventory")
async def strategy_inventory(
    status: str | None = Query(None),
    priority: str | None = Query(None),
    family: str | None = Query(None),
    with_latest_runs: bool = Query(False),
    with_evaluation_matrix: bool = Query(False),
) -> dict:
    try:
        latest_run_records = None
        if with_latest_runs or with_evaluation_matrix:
            latest_run_records = build_run_result_store(DEFAULT_COMPARISON_SPEC).list_compact_records(
                run_kind="strategy_run",
                view="generic",
            )
        return build_strategy_inventory_payload(
            status=status,
            priority=priority,
            family=family,
            with_latest_runs=with_latest_runs,
            latest_run_records=latest_run_records,
            with_evaluation_matrix=with_evaluation_matrix,
            evaluation_run_records=latest_run_records,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/ranking-evaluation")
async def ranking_evaluation() -> dict:
    try:
        return build_ranking_evaluation_payload(
            DEFAULT_COMPARISON_SPEC,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
