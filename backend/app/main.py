from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.default_comparison import DEFAULT_COMPARISON_SPEC, DEFAULT_PREDICTION_TARGET_SPECS
from app.dashboard_service import (
    build_condition_sweep_payload,
    build_dashboard_payload,
    build_prediction_evaluation_payload,
    build_ranking_evaluation_payload,
    build_run_catalog_payload,
    generate_parameter_sweep_runs_payload,
)
from app.market_data import fetch_market_universe_bundle


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


@app.get("/api/dashboard")
def dashboard() -> dict:
    try:
        return build_dashboard_payload(
            DEFAULT_COMPARISON_SPEC,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
) -> dict:
    try:
        return build_run_catalog_payload(
            DEFAULT_COMPARISON_SPEC,
            limit=limit,
            run_kind=run_kind,
            generation_method=generation_method,
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


@app.get("/api/prediction-evaluation")
def prediction_evaluation() -> dict:
    try:
        return build_prediction_evaluation_payload(
            DEFAULT_COMPARISON_SPEC,
            target_specs=DEFAULT_PREDICTION_TARGET_SPECS,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
