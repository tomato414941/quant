from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd

from app.portfolio import (
    StrategyDefinition,
    build_alignment_policy_spec,
    build_default_availability_policy,
    build_asset_ranking_specs_from_strategy_definitions,
    build_investment_universe_spec,
    build_observation_spec,
    build_portfolio_model_spec,
    build_portfolio_state,
    build_risk_controls_spec,
    build_selection_spec,
    build_strategy_definition,
    build_strategy_data_source_spec,
    build_strategy_execution_plan_spec,
    build_strategy_feature_definition_spec,
    build_strategy_signal_spec,
    compute_predictor_panel,
    deserialize_predictor_panel,
    evaluate_asset_ranking_spec,
    evaluate_predictor_spec,
    get_strategy_definition_signal_execution_contexts,
    serialize_asset_ranking_spec,
    serialize_predictor_spec,
    serialize_predictor_panel,
    serialize_portfolio_state,
    serialize_strategy_definition as serialize_canonical_strategy_definition,
)
from app.domain import RunContext
from app.engine import run_strategy_backtest
from app.spec import build_strategy_blueprint_from_definition
from app.predictor_registry import REGISTERED_PREDICTOR_SPECS_BY_KEY
from app.comparison_models import (
    ComparisonSpec,
    ConditionVariant,
    CostModelSpec,
    EvaluationSettings,
    EvaluationSpec,
    ExecutionAssumptionsSpec,
    MarketSliceSpec,
    RunSpec,
    SelectionPolicy,
    scale_cost_model_spec,
)
from app.run_store import FileRunResultStore, RunStoreSummary, build_run_fingerprint, build_run_spec
from app.diagnostics_service import build_evaluation_diagnostic_events
from app.instrument_registry import build_instrument_diagnostics
from app.timeframe_models import (
    DEFAULT_DAILY_TIMEFRAME,
    DEFAULT_MONTHLY_TIMEFRAME,
    DEFAULT_WEEKLY_TIMEFRAME,
    TimeframeSpec,
)

from app.comparison_market_context import (
    build_run_result_store,
    collect_required_market_fields,
    collect_strategy_predictor_source_definitions,
    collect_strategy_predictor_specs,
    fetch_market_data_by_timeframe,
    resolve_strategy_market_data_timeframe,
)
from app.comparison_run_builders import evaluate_strategy_definition_run_with_blueprint
from app.comparison_serialization import (
    serialize_comparison,
    serialize_evaluation,
    serialize_execution_assumptions,
    serialize_predictor_spec,
    serialize_strategy_definition_payload,
    serialize_timeframe,
)


def build_walk_forward_comparison_payload(
    comparison: ComparisonSpec,
    *,
    fetch_market_universe_bundle,
    start_year: int = 2020,
    end_year: int = 2025,
) -> dict:
    original_candidate_definitions = list(comparison.candidate_strategies)
    original_reference_definitions = list(comparison.reference_strategies)
    run_store = build_run_result_store(comparison)
    strategy_definitions = original_candidate_definitions + original_reference_definitions
    predictor_specs = collect_strategy_predictor_specs(strategy_definitions)
    (
        market_bundles_by_timeframe,
        metadata_by_timeframe,
        _comparison_tickers,
        comparison_timeframes,
    ) = fetch_market_data_by_timeframe(
        comparison,
        period=comparison.run_spec.market_slice.period,
        predictor_specs=predictor_specs,
        strategy_definitions=strategy_definitions,
        fetch_market_universe_bundle=fetch_market_universe_bundle,
    )
    windows = build_walk_forward_windows(
        comparison,
        start_year=start_year,
        end_year=end_year,
    )
    window_payloads: list[dict] = []
    candidate_runs_by_window: list[dict] = []
    reference_runs_by_window: list[dict] = []
    total_cached_runs = 0
    total_computed_runs = 0

    for window in windows:
        predictor_runs, predictor_panels_by_key, predictor_summary = build_walk_forward_predictor_runs(
            comparison=comparison,
            predictor_specs=predictor_specs,
            market_bundles_by_timeframe=market_bundles_by_timeframe,
            market_data_period=comparison.run_spec.market_slice.period,
            metadata_by_timeframe=metadata_by_timeframe,
            run_store=run_store,
            window=window,
        )
        candidate_runs, candidate_summary = build_walk_forward_strategy_runs(
            comparison=comparison,
            strategy_definitions=original_candidate_definitions,
            market_bundles_by_timeframe=market_bundles_by_timeframe,
            market_data_period=comparison.run_spec.market_slice.period,
            metadata_by_timeframe=metadata_by_timeframe,
            run_store=run_store,
            window=window,
            predictor_panels_by_key=predictor_panels_by_key,
        )
        reference_runs, reference_summary = build_walk_forward_strategy_runs(
            comparison=comparison,
            strategy_definitions=original_reference_definitions,
            market_bundles_by_timeframe=market_bundles_by_timeframe,
            market_data_period=comparison.run_spec.market_slice.period,
            metadata_by_timeframe=metadata_by_timeframe,
            run_store=run_store,
            window=window,
            predictor_panels_by_key=predictor_panels_by_key,
        )
        window_cached_runs = (
            predictor_summary.cached_run_count
            + candidate_summary.cached_run_count
            + reference_summary.cached_run_count
        )
        window_computed_runs = (
            predictor_summary.computed_run_count
            + candidate_summary.computed_run_count
            + reference_summary.computed_run_count
        )
        total_cached_runs += window_cached_runs
        total_computed_runs += window_computed_runs
        candidate_runs_by_window.append({"window": window, "runs": candidate_runs})
        reference_runs_by_window.append({"window": window, "runs": reference_runs})
        window_payloads.append(
            {
                **window,
                "runStoreSummary": {
                    "cachedRunCount": window_cached_runs,
                    "computedRunCount": window_computed_runs,
                },
                "predictorRunCount": len(predictor_runs),
                "candidateRunCount": len(candidate_runs),
                "referenceRunCount": len(reference_runs),
            }
        )

    candidate_results = aggregate_walk_forward_runs(candidate_runs_by_window)
    reference_results = aggregate_walk_forward_runs(reference_runs_by_window)

    return {
        "kind": "walk_forward_comparison",
        "schemaVersion": "v1",
        "comparison": serialize_comparison(
            comparison,
            metadata_by_timeframe,
            comparison_timeframes,
            candidate_strategy_definitions=original_candidate_definitions,
            reference_strategy_definitions=original_reference_definitions,
        ),
        "walkForward": {
            "kind": "walk_forward_yearly",
            "startYear": start_year,
            "endYear": end_year,
            "windowCount": len(windows),
            "rankingPolicy": {
                "primaryMetric": "average_test_sharpe_ratio",
                "secondaryMetric": "minimum_test_sharpe_ratio",
                "tertiaryMetric": "average_test_total_return",
                "tieBreaker": "average_test_max_drawdown",
            },
            "windows": window_payloads,
        },
        "resultCount": len(candidate_results),
        "runStoreSummary": {
            "cachedRunCount": total_cached_runs,
            "computedRunCount": total_computed_runs,
        },
        "candidateResults": candidate_results,
        "referenceResults": reference_results,
    }


def build_walk_forward_windows(
    comparison: ComparisonSpec,
    *,
    start_year: int,
    end_year: int,
) -> list[dict]:
    if start_year > end_year:
        raise ValueError("walk-forward start year must be <= end year.")
    windows = []
    train_start_date = comparison.run_spec.market_slice.start_date
    for year in range(start_year, end_year + 1):
        test_start = pd.Timestamp(year=year, month=1, day=1)
        train_end = test_start - pd.Timedelta(days=1)
        windows.append(
            {
                "kind": "walk_forward_year",
                "year": year,
                "trainStartDate": train_start_date,
                "trainEndDate": train_end.date().isoformat(),
                "testStartDate": test_start.date().isoformat(),
                "testEndDate": f"{year}-12-31",
            }
        )
    return windows


def slice_market_bundle_until(market_bundle: dict, end_date: str) -> dict:
    return {
        "closes": slice_market_frame_until(market_bundle["closes"], end_date),
        "volumes": slice_market_frame_until(market_bundle.get("volumes"), end_date),
    }


def slice_market_frame_until(frame, end_date: str):
    if frame is None:
        return None
    mask = pd.to_datetime(frame.index) <= pd.Timestamp(end_date)
    return frame.loc[mask].copy()


def format_market_index_date(value) -> str:
    return pd.Timestamp(value).date().isoformat()


def build_window_dataset_metadata(dataset_metadata: dict, market_bundle: dict) -> dict:
    closes = market_bundle["closes"]
    if closes.empty:
        raise ValueError("walk-forward window has no market data rows.")
    return {
        **dataset_metadata,
        "aligned_start_date": format_market_index_date(closes.index[0]),
        "aligned_end_date": format_market_index_date(closes.index[-1]),
        "row_count": len(closes),
    }


def compute_walk_forward_split_ratio(closes, test_start_date: str) -> float:
    returns = closes.pct_change(fill_method=None).iloc[1:].dropna(how="all")
    if len(returns) < 6:
        raise ValueError("At least 6 return rows are required for a walk-forward window.")
    return_dates = pd.to_datetime(returns.index)
    split_index = int((return_dates < pd.Timestamp(test_start_date)).sum())
    if split_index < 3:
        raise ValueError("walk-forward window does not have enough train return rows.")
    if len(returns) - split_index < 3:
        raise ValueError("walk-forward window does not have enough test return rows.")
    return split_index / len(returns)


def build_walk_forward_evaluation(
    comparison: ComparisonSpec,
    metadata_by_timeframe: dict[str, dict[str, object]],
    timeframes: list[TimeframeSpec],
    *,
    fields: list[str],
    period_override: str,
    window: dict,
    strategy_definitions: list | None = None,
) -> dict:
    evaluation = serialize_evaluation(
        comparison,
        metadata_by_timeframe,
        timeframes,
        fields=fields,
        period_override=period_override,
        strategy_definitions=strategy_definitions,
    )
    evaluation_settings = dict(evaluation.get("evaluationSettings", {}))
    evaluation_settings["walkForwardWindow"] = window
    evaluation["evaluationSettings"] = evaluation_settings
    evaluation["walkForwardWindow"] = window
    return evaluation


def build_walk_forward_market_slice(
    *,
    market_data_period: str,
    timeframe: TimeframeSpec,
    fields: list[str],
    window: dict,
) -> dict:
    return {
        "period": market_data_period,
        "timeframe": serialize_timeframe(timeframe),
        "fields": fields,
        "walkForwardWindow": window,
    }


def build_walk_forward_predictor_runs(
    *,
    comparison: ComparisonSpec,
    predictor_specs: list,
    market_bundles_by_timeframe: dict[str, dict],
    market_data_period: str,
    metadata_by_timeframe: dict[str, dict[str, object]],
    run_store: FileRunResultStore,
    window: dict,
) -> tuple[list[dict], dict[str, object], RunStoreSummary]:
    results: list[dict] = []
    predictor_panels_by_key: dict[str, object] = {}
    cached_run_count = 0
    computed_run_count = 0
    strategy_definitions = comparison.candidate_strategies + comparison.reference_strategies
    required_fields = collect_required_market_fields(
        comparison,
        predictor_specs,
        strategy_definitions=strategy_definitions,
    )
    serialized_execution_assumptions = serialize_execution_assumptions(comparison)
    predictor_source_definitions = collect_strategy_predictor_source_definitions(strategy_definitions)

    for predictor_spec in predictor_specs:
        timeframe_key = predictor_spec.timeframe.key
        market_bundle = slice_market_bundle_until(
            market_bundles_by_timeframe[timeframe_key],
            window["testEndDate"],
        )
        dataset_metadata = build_window_dataset_metadata(
            metadata_by_timeframe[timeframe_key],
            market_bundle,
        )
        closes = market_bundle["closes"][list(predictor_spec.signal_spec.observation_spec.tickers)]
        volumes = (
            market_bundle["volumes"][list(predictor_spec.signal_spec.observation_spec.tickers)]
            if market_bundle["volumes"] is not None
            else None
        )
        returns = closes.pct_change(fill_method=None).iloc[1:].dropna(how="all")
        aligned_volumes = volumes.loc[returns.index] if volumes is not None else None
        split_ratio = compute_walk_forward_split_ratio(closes, window["testStartDate"])
        source_strategy_definition = predictor_source_definitions.get(predictor_spec.key)
        if source_strategy_definition is None:
            raise ValueError(f"No source strategy definition found for predictor: {predictor_spec.key}")
        serialized_predictor_spec = serialize_predictor_spec(predictor_spec)
        run_spec = build_run_spec(
            run_kind="predictor_run",
            strategy_definition=serialize_strategy_definition_payload(source_strategy_definition),
            evaluation_subject={"kind": "predictor", "predictor": serialized_predictor_spec},
            market_slice=build_walk_forward_market_slice(
                market_data_period=market_data_period,
                timeframe=predictor_spec.timeframe,
                fields=required_fields,
                window=window,
            ),
            evaluation=build_walk_forward_evaluation(
                comparison,
                {timeframe_key: dataset_metadata},
                [predictor_spec.timeframe],
                fields=required_fields,
                period_override=market_data_period,
                window=window,
            ),
            execution_assumptions=serialized_execution_assumptions,
            portfolio_state=serialize_portfolio_state(comparison.run_spec.portfolio_state),
            capital_base=comparison.run_spec.capital_base,
        )
        cached_run = run_store.load(run_spec)
        if cached_run is not None:
            cached_run_count += 1
            results.append(cached_run)
            predictor_panels_by_key[predictor_spec.key] = deserialize_predictor_panel(
                cached_run.get("predictorSeries")
            )
            continue

        predictor_panel = compute_predictor_panel(
            returns=returns,
            volumes=aligned_volumes,
            predictor_spec=predictor_spec,
            bars_per_year=predictor_spec.timeframe.bars_per_year,
        )
        result = evaluate_predictor_spec(
            returns=returns,
            volumes=aligned_volumes,
            split_ratio=split_ratio,
            predictor_spec=predictor_spec,
            bars_per_year=predictor_spec.timeframe.bars_per_year,
            predictor_panel=predictor_panel,
        )
        run_store.save(run_spec, result)
        computed_run_count += 1
        results.append(result)
        predictor_panels_by_key[predictor_spec.key] = deserialize_predictor_panel(
            result.get("predictorSeries")
        )

    return results, predictor_panels_by_key, RunStoreSummary(
        cached_run_count=cached_run_count,
        computed_run_count=computed_run_count,
    )


def build_walk_forward_strategy_runs(
    *,
    comparison: ComparisonSpec,
    strategy_definitions: list[StrategyDefinition],
    market_bundles_by_timeframe: dict[str, dict],
    market_data_period: str,
    metadata_by_timeframe: dict[str, dict[str, object]],
    run_store: FileRunResultStore,
    window: dict,
    predictor_panels_by_key: dict[str, object] | None = None,
) -> tuple[list[dict], RunStoreSummary]:
    serialized_execution_assumptions = serialize_execution_assumptions(comparison)
    predictor_specs = collect_strategy_predictor_specs(strategy_definitions)
    required_fields = collect_required_market_fields(
        comparison,
        predictor_specs,
        strategy_definitions=strategy_definitions,
    )
    runs: list[dict] = []
    cached_run_count = 0
    computed_run_count = 0

    for strategy_definition in strategy_definitions:
        market_data_timeframe = resolve_strategy_market_data_timeframe(strategy_definition)
        timeframe_key = market_data_timeframe.key
        market_bundle = slice_market_bundle_until(
            market_bundles_by_timeframe[timeframe_key],
            window["testEndDate"],
        )
        dataset_metadata = build_window_dataset_metadata(
            metadata_by_timeframe[timeframe_key],
            market_bundle,
        )
        serialized_strategy_definition = serialize_strategy_definition_payload(strategy_definition)
        _selection_contexts, predictor_context = get_strategy_definition_signal_execution_contexts(strategy_definition)
        predictor_panel = None
        if predictor_context is not None and predictor_panels_by_key is not None:
            predictor_panel = predictor_panels_by_key.get(str(predictor_context["predictorKey"]))
        split_ratio = compute_walk_forward_split_ratio(
            market_bundle["closes"],
            window["testStartDate"],
        )
        run_spec = build_run_spec(
            run_kind="strategy_run",
            market_slice=build_walk_forward_market_slice(
                market_data_period=market_data_period,
                timeframe=market_data_timeframe,
                fields=required_fields,
                window=window,
            ),
            evaluation=build_walk_forward_evaluation(
                comparison,
                {timeframe_key: dataset_metadata},
                [market_data_timeframe],
                fields=required_fields,
                period_override=market_data_period,
                window=window,
                strategy_definitions=[strategy_definition],
            ),
            execution_assumptions=serialized_execution_assumptions,
            portfolio_state=serialize_portfolio_state(comparison.run_spec.portfolio_state),
            capital_base=comparison.run_spec.capital_base,
            strategy_definition=serialized_strategy_definition,
        )
        cached_run = run_store.load(run_spec)
        if cached_run is not None:
            cached_run_count += 1
            runs.append(cached_run)
            continue

        run = evaluate_strategy_definition_run_with_blueprint(
            closes=market_bundle["closes"],
            volumes=market_bundle["volumes"],
            strategy_definition=strategy_definition,
            bars_per_year=market_data_timeframe.bars_per_year,
            initial_capital=comparison.run_spec.capital_base,
            split_ratio=split_ratio,
            execution_assumptions=serialized_execution_assumptions,
            portfolio_state=comparison.run_spec.portfolio_state,
            predictor_panel=predictor_panel,
        )
        run_store.save(run_spec, run)
        computed_run_count += 1
        runs.append(run)

    return runs, RunStoreSummary(
        cached_run_count=cached_run_count,
        computed_run_count=computed_run_count,
    )


def aggregate_walk_forward_runs(runs_by_window: list[dict]) -> list[dict]:
    grouped: dict[str, dict] = {}
    for window_runs in runs_by_window:
        window = window_runs["window"]
        for run in window_runs["runs"]:
            group = grouped.setdefault(
                run["key"],
                {
                    "kind": "walk_forward_strategy_result",
                    "schemaVersion": "v1",
                    "strategyKey": run["key"],
                    "strategyLabel": run["strategy"]["label"],
                    "strategy": run["strategy"],
                    "windows": [],
                },
            )
            test_summary = run["splitAnalysis"]["test"]["portfolio"]
            train_summary = run["splitAnalysis"]["train"]["portfolio"]
            train_split = run["splitAnalysis"]["train"]
            test_split = run["splitAnalysis"]["test"]
            group["windows"].append(
                {
                    "year": window["year"],
                    "testStartDate": window["testStartDate"],
                    "testEndDate": window["testEndDate"],
                    "test": test_summary,
                    "train": train_summary,
                    "weights": run.get("weights", []),
                    "selectedAssets": run.get("selectedAssets", []),
                    "executionTrace": run.get("executionTrace", []),
                    "executionDecisionSummary": run.get("decisionSummary", {}),
                    "testAvailability": summarize_run_series_availability(
                        run.get("series", []),
                        start_date=test_split.get("startDate"),
                        end_date=test_split.get("endDate"),
                    ),
                    "trainAvailability": summarize_run_series_availability(
                        run.get("series", []),
                        start_date=train_split.get("startDate"),
                        end_date=train_split.get("endDate"),
                    ),
                }
            )

    results = [summarize_walk_forward_group(group) for group in grouped.values()]
    return sort_walk_forward_results(results)


def summarize_run_series_availability(
    series: list[dict],
    *,
    start_date: str | None,
    end_date: str | None,
) -> dict[str, object]:
    empty_summary = {
        "barCount": 0,
        "minAvailableAssetCount": 0,
        "maxAvailableAssetCount": 0,
        "minEligibleAssetCount": 0,
        "maxEligibleAssetCount": 0,
        "newlyEligibleAssetCount": 0,
        "removedAssetCount": 0,
        "newlyEligibleAssets": [],
        "removedAssets": [],
    }
    if not series or start_date is None or end_date is None:
        return empty_summary

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    rows = [
        row for row in series
        if start <= pd.Timestamp(row["date"]) <= end
    ]
    if not rows:
        return empty_summary

    available_counts = [int(row.get("availableAssetCount", 0)) for row in rows]
    eligible_counts = [int(row.get("eligibleAssetCount", 0)) for row in rows]
    newly_eligible_assets = sorted({
        asset
        for row in rows
        for asset in row.get("newlyEligibleAssets", [])
    })
    removed_assets = sorted({
        asset
        for row in rows
        for asset in row.get("removedAssets", [])
    })
    return {
        "barCount": len(rows),
        "minAvailableAssetCount": min(available_counts),
        "maxAvailableAssetCount": max(available_counts),
        "minEligibleAssetCount": min(eligible_counts),
        "maxEligibleAssetCount": max(eligible_counts),
        "newlyEligibleAssetCount": len(newly_eligible_assets),
        "removedAssetCount": len(removed_assets),
        "newlyEligibleAssets": newly_eligible_assets,
        "removedAssets": removed_assets,
    }


def summarize_walk_forward_group(group: dict) -> dict:
    windows = sorted(group["windows"], key=lambda window: window["year"])
    test_summaries = [window["test"] for window in windows]
    test_availability_summaries = [window["testAvailability"] for window in windows]
    window_count = len(test_summaries)

    def average_metric(metric_key: str) -> float:
        return round(
            sum(float(summary[metric_key]) for summary in test_summaries) / window_count,
            6,
        )

    return {
        **group,
        "windows": windows,
        "windowCount": window_count,
        "averageSharpeRatio": average_metric("sharpeRatio"),
        "minimumSharpeRatio": round(
            min(float(summary["sharpeRatio"]) for summary in test_summaries),
            6,
        ),
        "positiveReturnWindowCount": sum(
            1 for summary in test_summaries
            if float(summary["totalReturnPct"]) > 0
        ),
        "averageTotalReturnPct": average_metric("totalReturnPct"),
        "averageMaxDrawdownPct": average_metric("maxDrawdownPct"),
        "averageTurnoverPct": average_metric("turnoverPct"),
        "executionDecisionSummary": summarize_execution_decision_summaries(
            [window.get("executionDecisionSummary", {}) for window in windows]
        ),
        "minTestEligibleAssetCount": min(
            int(summary["minEligibleAssetCount"])
            for summary in test_availability_summaries
        ),
        "maxTestEligibleAssetCount": max(
            int(summary["maxEligibleAssetCount"])
            for summary in test_availability_summaries
        ),
        "testNewlyEligibleAssetCount": len({
            asset
            for summary in test_availability_summaries
            for asset in summary.get("newlyEligibleAssets", [])
        }),
        "testRemovedAssetCount": len({
            asset
            for summary in test_availability_summaries
            for asset in summary.get("removedAssets", [])
        }),
    }


def summarize_execution_decision_summaries(summaries: list[dict]) -> dict:
    clean_summaries = [summary for summary in summaries if summary]
    decision_count = sum(int(summary.get("decisionCount", 0)) for summary in clean_summaries)
    rebalance_count = sum(int(summary.get("rebalanceCount", 0)) for summary in clean_summaries)
    no_trade_count = sum(int(summary.get("noTradeCount", 0)) for summary in clean_summaries)
    edge_hit_count = sum(int(summary.get("edgeHitCount", 0)) for summary in clean_summaries)
    edge_hit_sample_count = sum(int(summary.get("edgeHitSampleCount", 0)) for summary in clean_summaries)

    return {
        "decisionCount": decision_count,
        "rebalanceCount": rebalance_count,
        "noTradeCount": no_trade_count,
        "policyCounts": merge_count_maps(clean_summaries, "policyCounts"),
        "reasonCounts": merge_count_maps(clean_summaries, "reasonCounts"),
        "edgeSourceCounts": merge_count_maps(clean_summaries, "edgeSourceCounts"),
        "averageTurnoverPct": weighted_average_summary_metric(
            clean_summaries,
            "averageTurnoverPct",
        ),
        "averageEstimatedCostPct": weighted_average_summary_metric(
            clean_summaries,
            "averageEstimatedCostPct",
        ),
        "averageEstimatedEdgePct": weighted_average_summary_metric(
            clean_summaries,
            "averageEstimatedEdgePct",
        ),
        "averageRealizedEdgePct": weighted_average_summary_metric(
            clean_summaries,
            "averageRealizedEdgePct",
        ),
        "averageRealizedEdgeAfterCostPct": weighted_average_summary_metric(
            clean_summaries,
            "averageRealizedEdgeAfterCostPct",
        ),
        "averageConfidence": weighted_average_summary_metric(
            clean_summaries,
            "averageConfidence",
        ),
        "edgeHitCount": edge_hit_count,
        "edgeHitSampleCount": edge_hit_sample_count,
        "edgeHitRate": None if edge_hit_sample_count == 0 else round(edge_hit_count / edge_hit_sample_count, 6),
        "estimatedVsRealizedEdgeCorrelation": weighted_average_summary_metric_by_count(
            clean_summaries,
            "estimatedVsRealizedEdgeCorrelation",
            "edgeHitSampleCount",
        ),
        "estimatedEdgePctDistribution": merge_summary_distributions(
            clean_summaries,
            "estimatedEdgePctDistribution",
        ),
        "estimatedCostPctDistribution": merge_summary_distributions(
            clean_summaries,
            "estimatedCostPctDistribution",
        ),
        "estimatedEdgeAfterCostPctDistribution": merge_summary_distributions(
            clean_summaries,
            "estimatedEdgeAfterCostPctDistribution",
        ),
        "realizedEdgePctDistribution": merge_summary_distributions(
            clean_summaries,
            "realizedEdgePctDistribution",
        ),
        "realizedEdgeAfterCostPctDistribution": merge_summary_distributions(
            clean_summaries,
            "realizedEdgeAfterCostPctDistribution",
        ),
        "confidenceDistribution": merge_summary_distributions(
            clean_summaries,
            "confidenceDistribution",
        ),
    }


def merge_count_maps(summaries: list[dict], key: str) -> dict:
    counts: dict[str, int] = {}
    for summary in summaries:
        for name, value in (summary.get(key) or {}).items():
            counts[str(name)] = counts.get(str(name), 0) + int(value)
    return counts


def weighted_average_summary_metric(summaries: list[dict], key: str) -> float | None:
    return weighted_average_summary_metric_by_count(summaries, key, "decisionCount")


def weighted_average_summary_metric_by_count(summaries: list[dict], key: str, weight_key: str) -> float | None:
    weighted_total = 0.0
    total_weight = 0
    for summary in summaries:
        value = summary.get(key)
        weight = int(summary.get(weight_key, 0))
        if value is None or weight <= 0:
            continue
        weighted_total += float(value) * weight
        total_weight += weight
    if total_weight == 0:
        return None
    return round(weighted_total / total_weight, 6)


def empty_summary_distribution() -> dict:
    return {
        "count": 0,
        "minimum": None,
        "median": None,
        "maximum": None,
    }


def merge_summary_distributions(summaries: list[dict], key: str) -> dict:
    distributions = [
        summary.get(key)
        for summary in summaries
        if summary.get(key) and int(summary.get(key, {}).get("count", 0)) > 0
    ]
    if not distributions:
        return empty_summary_distribution()

    total_count = sum(int(distribution["count"]) for distribution in distributions)
    weighted_median_total = sum(
        float(distribution["median"]) * int(distribution["count"])
        for distribution in distributions
        if distribution.get("median") is not None
    )
    median_weight = sum(
        int(distribution["count"])
        for distribution in distributions
        if distribution.get("median") is not None
    )
    return {
        "count": total_count,
        "minimum": round(min(float(distribution["minimum"]) for distribution in distributions), 6),
        "median": None if median_weight == 0 else round(weighted_median_total / median_weight, 6),
        "maximum": round(max(float(distribution["maximum"]) for distribution in distributions), 6),
    }


def sort_walk_forward_results(results: list[dict]) -> list[dict]:
    return sorted(
        results,
        key=lambda result: (
            -float(result["averageSharpeRatio"]),
            -float(result["minimumSharpeRatio"]),
            -float(result["averageTotalReturnPct"]),
            float(result["averageMaxDrawdownPct"]),
            result["strategyKey"],
        ),
    )
