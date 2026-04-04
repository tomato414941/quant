from __future__ import annotations

from dataclasses import replace
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
from app.study_models import BacktestConfig, ConditionVariant, ExecutionModelConfig, StudyDefinition


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_dashboard_payload(
    study: StudyDefinition,
    *,
    fetch_market_universe_bundle,
) -> dict:
    run_store = build_run_result_store(study)
    market_bundle, metadata = fetch_market_universe_bundle(
        tickers=study.dataset_spec.tickers,
        period=study.dataset_spec.period,
    )
    runs, run_store_summary = build_portfolio_runs(
        study=study,
        closes=market_bundle["closes"],
        volumes=market_bundle["volumes"],
        dataset_period=study.dataset_spec.period,
        dataset_metadata=metadata,
        run_store=run_store,
    )
    sanity_checks = []
    total_cached_runs = run_store_summary.cached_run_count
    total_computed_runs = run_store_summary.computed_run_count
    for period in study.dataset_spec.sanity_periods:
        sanity_bundle, sanity_metadata = fetch_market_universe_bundle(
            tickers=study.dataset_spec.tickers,
            period=period,
        )
        sanity_runs, sanity_run_store_summary = build_portfolio_runs(
            study=study,
            closes=sanity_bundle["closes"],
            volumes=sanity_bundle["volumes"],
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


def build_condition_sweep_payload(
    study: StudyDefinition,
    *,
    fetch_market_universe_bundle,
) -> dict:
    run_store = build_run_result_store(study)
    market_bundle, metadata = fetch_market_universe_bundle(
        tickers=study.dataset_spec.tickers,
        period=study.dataset_spec.period,
    )
    results, run_store_summary = build_condition_sweep_runs(
        study=study,
        closes=market_bundle["closes"],
        volumes=market_bundle["volumes"],
        dataset_period=study.dataset_spec.period,
        dataset_metadata=metadata,
        run_store=run_store,
    )
    return {
        "study": serialize_study(study, metadata),
        "conditionVariants": [
            serialize_condition_variant(condition_variant)
            for condition_variant in study.condition_variants
        ],
        "resultCount": len(results),
        "runStoreSummary": run_store_summary.to_payload(),
        "results": results,
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


def serialize_execution_model(execution_model: ExecutionModelConfig) -> dict:
    return {
        "key": execution_model.key,
        "label": execution_model.label,
        "entry": execution_model.entry,
        "commissionPct": round(execution_model.commission_pct, 3),
        "slippagePct": round(execution_model.slippage_pct, 3),
        "rebalanceFrequency": execution_model.rebalance_frequency,
    }


def serialize_backtest_config(backtest_config: BacktestConfig) -> dict:
    return {
        "splitRatioPct": round(backtest_config.split_ratio * 100, 1),
        "initialCapital": round(backtest_config.initial_capital, 2),
        "maxInvestmentPct": round(backtest_config.max_investment_ratio * 100, 1),
        "benchmark": backtest_config.benchmark,
        "maxWeightPct": round(backtest_config.max_weight * 100, 1)
        if backtest_config.max_weight is not None
        else None,
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
        "executionVariants": [
            serialize_execution_model(execution_model)
            for execution_model in study.execution_variants
        ],
        "backtestConfig": serialize_backtest_config(study.backtest_config),
        "portfolioState": serialize_portfolio_state(study.portfolio_state),
        "portfolioModels": list(models_by_key.values()),
        "strategyDefinitions": list(strategies_by_key.values()),
        "candidateDefinitions": [
            serialize_portfolio_candidate_definition(candidate_definition)
            for candidate_definition in study.candidate_definitions
        ],
        "conditionVariants": [
            serialize_condition_variant(condition_variant)
            for condition_variant in study.condition_variants
        ],
    }


def serialize_condition_variant(condition_variant: ConditionVariant) -> dict:
    return {
        "key": condition_variant.key,
        "label": condition_variant.label,
        "commissionPct": round(condition_variant.commission_pct, 3),
        "maxInvestmentPct": round(condition_variant.max_investment_ratio * 100, 1),
        "maxWeightPct": round(condition_variant.max_weight * 100, 1)
        if condition_variant.max_weight is not None
        else None,
    }


def build_portfolio_runs(
    *,
    study: StudyDefinition,
    closes,
    volumes,
    dataset_period: str,
    dataset_metadata: dict[str, str],
    run_store: FileRunResultStore,
) -> tuple[list[dict], RunStoreSummary]:
    serialized_backtest_config = serialize_backtest_config(study.backtest_config)
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
        for execution_variant in study.execution_variants:
            serialized_execution_model = serialize_execution_model(execution_variant)
            run_definition = build_run_definition(
                run_kind="dashboard",
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
                volumes=volumes,
                candidate_definition=candidate_definition,
                initial_capital=study.backtest_config.initial_capital,
                split_ratio=study.backtest_config.split_ratio,
                transaction_cost=execution_variant.commission_pct / 100,
                max_investment_ratio=study.backtest_config.max_investment_ratio,
                max_weight=study.backtest_config.max_weight,
                rebalance_frequency=execution_variant.rebalance_frequency,
                portfolio_state=study.portfolio_state,
            )
            run["key"] = f"{candidate_definition.key}__{execution_variant.key}"
            run["executionModel"] = serialized_execution_model
            run_store.save(run_definition, run)
            computed_run_count += 1
            runs.append(run)

    return runs, RunStoreSummary(
        cached_run_count=cached_run_count,
        computed_run_count=computed_run_count,
    )


def build_condition_sweep_runs(
    *,
    study: StudyDefinition,
    closes,
    volumes,
    dataset_period: str,
    dataset_metadata: dict[str, str],
    run_store: FileRunResultStore,
) -> tuple[list[dict], RunStoreSummary]:
    serialized_portfolio_state = serialize_portfolio_state(study.portfolio_state)
    dataset_spec = {
        "tickers": study.dataset_spec.tickers,
        "period": dataset_period,
        "frequency": study.dataset_spec.frequency,
    }
    results: list[dict] = []
    cached_run_count = 0
    computed_run_count = 0

    for candidate_definition in study.candidate_definitions:
        candidate = serialize_portfolio_candidate_definition(candidate_definition)
        for execution_variant in study.execution_variants:
            for condition_variant in study.condition_variants:
                effective_execution = replace(
                    execution_variant,
                    commission_pct=condition_variant.commission_pct,
                )
                effective_backtest = replace(
                    study.backtest_config,
                    max_investment_ratio=condition_variant.max_investment_ratio,
                    max_weight=condition_variant.max_weight,
                )
                serialized_execution_model = serialize_execution_model(effective_execution)
                serialized_backtest_config = serialize_backtest_config(effective_backtest)
                serialized_condition_variant = serialize_condition_variant(condition_variant)
                run_definition = build_run_definition(
                    run_kind="condition_sweep",
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
                    results.append(cached_run)
                    continue

                run = compare_portfolio_candidate(
                    closes=closes,
                    volumes=volumes,
                    candidate_definition=candidate_definition,
                    initial_capital=effective_backtest.initial_capital,
                    split_ratio=effective_backtest.split_ratio,
                    transaction_cost=effective_execution.commission_pct / 100,
                    max_investment_ratio=effective_backtest.max_investment_ratio,
                    max_weight=effective_backtest.max_weight,
                    rebalance_frequency=effective_execution.rebalance_frequency,
                    portfolio_state=study.portfolio_state,
                )
                compact_run = compact_condition_sweep_run(
                    run=run,
                    key=f"{candidate_definition.key}__{execution_variant.key}__{condition_variant.key}",
                    execution_model=serialized_execution_model,
                    condition_variant=serialized_condition_variant,
                )
                run_store.save(run_definition, compact_run)
                computed_run_count += 1
                results.append(compact_run)

    results.sort(
        key=lambda row: (
            row["summary"]["sharpeRatio"],
            row["summary"]["totalReturnPct"],
            -row["summary"]["maxDrawdownPct"],
        ),
        reverse=True,
    )
    return results, RunStoreSummary(
        cached_run_count=cached_run_count,
        computed_run_count=computed_run_count,
    )


def compact_condition_sweep_run(
    *,
    run: dict,
    key: str,
    execution_model: dict,
    condition_variant: dict,
) -> dict:
    return {
        "key": key,
        "strategy": run["strategy"],
        "portfolioModel": run["portfolioModel"],
        "executionModel": execution_model,
        "conditionVariant": condition_variant,
        "weights": run["weights"],
        "selectedAssets": run["selectedAssets"],
        "summary": run["summary"],
        "benchmark": run["benchmark"],
        "splitAnalysis": run["splitAnalysis"],
    }


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
