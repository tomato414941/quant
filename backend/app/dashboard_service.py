from __future__ import annotations

from pathlib import Path

from app.portfolio import (
    PortfolioCandidateDefinition,
    compare_portfolio_candidate,
    serialize_portfolio_candidate_definition,
    serialize_portfolio_model_definition,
    serialize_portfolio_state,
    serialize_portfolio_strategy_definition,
)
from app.run_store import FileRunResultStore, RunStoreSummary, build_run_definition
from app.study_models import StudyDefinition


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_dashboard_payload(
    study: StudyDefinition,
    *,
    fetch_market_universe,
) -> dict:
    run_store = build_run_result_store(study)
    closes, metadata = fetch_market_universe(
        tickers=study.dataset_spec.tickers,
        period=study.dataset_spec.period,
    )
    runs, run_store_summary = build_portfolio_runs(
        study=study,
        closes=closes,
        dataset_period=study.dataset_spec.period,
        dataset_metadata=metadata,
        run_store=run_store,
    )
    sanity_checks = []
    total_cached_runs = run_store_summary.cached_run_count
    total_computed_runs = run_store_summary.computed_run_count
    for period in study.dataset_spec.sanity_periods:
        sanity_closes, sanity_metadata = fetch_market_universe(
            tickers=study.dataset_spec.tickers,
            period=period,
        )
        sanity_runs, sanity_run_store_summary = build_portfolio_runs(
            study=study,
            closes=sanity_closes,
            dataset_period=period,
            dataset_metadata=sanity_metadata,
            run_store=run_store,
        )
        total_cached_runs += sanity_run_store_summary.cached_run_count
        total_computed_runs += sanity_run_store_summary.computed_run_count
        sanity_checks.append(
            {
                "period": period,
                "datasetSpec": serialize_dataset_spec(study, sanity_metadata, period_override=period),
                "runStoreSummary": sanity_run_store_summary.to_payload(),
                "runs": sanity_runs,
            }
        )

    return {
        "study": serialize_study(study, metadata),
        "runs": runs,
        "comparisonSeries": build_comparison_series(runs),
        "runStoreSummary": {
            "cachedRunCount": total_cached_runs,
            "computedRunCount": total_computed_runs,
        },
        "sanityChecks": sanity_checks,
    }


def build_run_result_store(study: StudyDefinition) -> FileRunResultStore:
    root_dir = Path(study.result_store_dir)
    if not root_dir.is_absolute():
        root_dir = PROJECT_ROOT / root_dir
    return FileRunResultStore(root_dir)


def serialize_dataset_spec(
    study: StudyDefinition,
    dataset_metadata: dict[str, str],
    *,
    period_override: str | None = None,
) -> dict:
    return {
        "tickers": study.dataset_spec.tickers,
        "period": period_override or study.dataset_spec.period,
        "sanityPeriods": study.dataset_spec.sanity_periods,
        "frequency": study.dataset_spec.frequency,
        "source": dataset_metadata["source"],
        "alignedStartDate": dataset_metadata["aligned_start_date"],
        "alignedEndDate": dataset_metadata["aligned_end_date"],
        "rowCount": dataset_metadata["row_count"],
    }


def serialize_execution_model(study: StudyDefinition) -> dict:
    return {
        "entry": study.execution_model.entry,
        "commissionPct": round(study.execution_model.commission_pct, 3),
        "slippagePct": round(study.execution_model.slippage_pct, 3),
        "rebalanceFrequency": study.execution_model.rebalance_frequency,
    }


def serialize_backtest_config(study: StudyDefinition) -> dict:
    return {
        "splitRatioPct": round(study.backtest_config.split_ratio * 100, 1),
        "initialCapital": round(study.backtest_config.initial_capital, 2),
        "maxInvestmentPct": round(study.backtest_config.max_investment_ratio * 100, 1),
        "benchmark": study.backtest_config.benchmark,
    }


def serialize_study(study: StudyDefinition, dataset_metadata: dict[str, str]) -> dict:
    strategies_by_key: dict[str, dict] = {}
    models_by_key: dict[str, dict] = {}
    for candidate_definition in study.candidate_definitions:
        strategies_by_key[candidate_definition.strategy_definition.key] = serialize_portfolio_strategy_definition(
            candidate_definition.strategy_definition
        )
        models_by_key[candidate_definition.model_definition.key] = serialize_portfolio_model_definition(
            candidate_definition.model_definition
        )

    return {
        "id": study.study_id,
        "title": study.title,
        "question": study.question,
        "datasetSpec": serialize_dataset_spec(study, dataset_metadata),
        "executionModel": serialize_execution_model(study),
        "backtestConfig": serialize_backtest_config(study),
        "portfolioState": serialize_portfolio_state(study.portfolio_state),
        "portfolioModels": list(models_by_key.values()),
        "strategyDefinitions": list(strategies_by_key.values()),
        "candidateDefinitions": [
            serialize_portfolio_candidate_definition(candidate_definition)
            for candidate_definition in study.candidate_definitions
        ],
    }


def build_portfolio_runs(
    *,
    study: StudyDefinition,
    closes,
    dataset_period: str,
    dataset_metadata: dict[str, str],
    run_store: FileRunResultStore,
) -> tuple[list[dict], RunStoreSummary]:
    serialized_execution_model = serialize_execution_model(study)
    serialized_backtest_config = serialize_backtest_config(study)
    serialized_portfolio_state = serialize_portfolio_state(study.portfolio_state)
    dataset_spec = {
        "tickers": study.dataset_spec.tickers,
        "period": dataset_period,
        "frequency": study.dataset_spec.frequency,
    }
    runs: list[dict] = []
    cached_run_count = 0
    computed_run_count = 0

    for candidate_definition in study.candidate_definitions:
        candidate = serialize_portfolio_candidate_definition(candidate_definition)
        run_definition = build_run_definition(
            candidate=candidate,
            dataset_spec=dataset_spec,
            execution_model=serialized_execution_model,
            backtest_config=serialized_backtest_config,
            portfolio_state=serialized_portfolio_state,
            dataset_metadata=dataset_metadata,
        )
        cached_run = run_store.load(run_definition)
        if cached_run is not None:
            cached_run_count += 1
            runs.append(cached_run)
            continue

        run = compare_portfolio_candidate(
            closes=closes,
            candidate_definition=candidate_definition,
            initial_capital=study.backtest_config.initial_capital,
            split_ratio=study.backtest_config.split_ratio,
            transaction_cost=study.execution_model.commission_pct / 100,
            max_investment_ratio=study.backtest_config.max_investment_ratio,
            rebalance_frequency=study.execution_model.rebalance_frequency,
            portfolio_state=study.portfolio_state,
        )
        run_store.save(run_definition, run)
        computed_run_count += 1
        runs.append(run)

    return runs, RunStoreSummary(
        cached_run_count=cached_run_count,
        computed_run_count=computed_run_count,
    )


def build_comparison_series(runs: list[dict]) -> list[dict]:
    rows_by_date: dict[str, dict] = {}

    for run in runs:
        model_key = run["key"]
        for point in run["series"]:
            row = rows_by_date.setdefault(
                point["date"],
                {
                    "date": point["date"],
                    "benchmarkEquity": point["benchmarkEquity"],
                },
            )
            row["benchmarkEquity"] = point["benchmarkEquity"]
            row[model_key] = point["portfolioEquity"]

    return [rows_by_date[key] for key in sorted(rows_by_date.keys())]
