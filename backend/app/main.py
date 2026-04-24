from __future__ import annotations

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.default_comparison import DEFAULT_COMPARISON_SPEC
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
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/comparison")
def comparison() -> dict:
    try:
        return build_comparison_payload(
            DEFAULT_COMPARISON_SPEC,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/comparison-run-spec")
def comparison_run_spec() -> dict:
    try:
        return build_comparison_run_spec_payload(
            DEFAULT_COMPARISON_SPEC,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/comparison-run-spec/rerun")
def rerun_comparison_run_spec(payload: dict = Body(...)) -> dict:
    try:
        return build_comparison_payload_from_run_spec_payload(
            payload,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/predictor-runs")
def predictor_run_index(
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
def predictor_runs() -> dict:
    try:
        return build_predictor_runs_payload(
            DEFAULT_COMPARISON_SPEC,
            predictor_specs=REGISTERED_PREDICTOR_SPECS,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/predictor-runs/{run_key}")
def predictor_run_detail(run_key: str) -> dict:
    try:
        return build_predictor_run_detail_payload(
            DEFAULT_COMPARISON_SPEC,
            run_key=run_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/strategy-runs")
def strategy_run_index(
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
def strategy_runs() -> dict:
    try:
        return build_strategy_runs_payload(
            DEFAULT_COMPARISON_SPEC,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/strategy-runs/{run_key}")
def strategy_run_detail(run_key: str) -> dict:
    try:
        return build_strategy_run_detail_payload(
            DEFAULT_COMPARISON_SPEC,
            run_key=run_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/condition-sweep")
def condition_sweep() -> dict:
    try:
        return build_condition_sweep_payload(
            DEFAULT_COMPARISON_SPEC,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/runs/generate-parameter-sweep")
def generate_parameter_sweep_runs() -> dict:
    try:
        return generate_parameter_sweep_runs_payload(
            DEFAULT_COMPARISON_SPEC,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/run-catalog")
def run_catalog(
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
def latest_run_catalog_record(
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
def strategy_inventory(
    status: str | None = Query(None),
    priority: str | None = Query(None),
    family: str | None = Query(None),
) -> dict:
    try:
        return build_strategy_inventory_payload(
            status=status,
            priority=priority,
            family=family,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/ranking-evaluation")
def ranking_evaluation() -> dict:
    try:
        return build_ranking_evaluation_payload(
            DEFAULT_COMPARISON_SPEC,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
