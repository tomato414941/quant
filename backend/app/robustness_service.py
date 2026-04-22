from __future__ import annotations

import copy
import math
import time
from dataclasses import replace
from typing import Callable

from app.comparison_models import ComparisonSpec, ConditionVariant, scale_cost_model_spec
from app.comparison_service import (
    build_walk_forward_comparison_payload,
    summarize_execution_decision_summaries,
)
from app.diagnostics_service import (
    build_evaluation_diagnostic_events,
    has_invalidating_diagnostic,
    representative_diagnostic_event,
    summarize_diagnostic_events,
)
from app.portfolio import build_risk_controls_spec


ROBUSTNESS_PERIODS = (
    {
        "key": "2015_2025",
        "label": "2015-2025",
        "startDate": "2015-01-01",
        "endDate": "2025-12-31",
        "walkForwardStartYear": 2020,
        "walkForwardEndYear": 2025,
    },
    {
        "key": "2018_2025",
        "label": "2018-2025",
        "startDate": "2018-01-01",
        "endDate": "2025-12-31",
        "walkForwardStartYear": 2021,
        "walkForwardEndYear": 2025,
    },
    {
        "key": "2020_2025",
        "label": "2020-2025",
        "startDate": "2020-01-01",
        "endDate": "2025-12-31",
        "walkForwardStartYear": 2023,
        "walkForwardEndYear": 2025,
    },
)
ROBUSTNESS_UNIVERSES = ("crypto_included", "no_crypto", "btc_only")
ROBUSTNESS_COST_MULTIPLIERS = (1.0, 2.0, 3.0)
ROBUSTNESS_MAX_WEIGHTS = (0.25, 0.35, 0.45)
ROBUSTNESS_MAX_INVESTMENT_RATIO = 1.0
ROBUSTNESS_BASELINE_KEY = "ref-fu-eq-cash"
WEIGHT_EPSILON_PCT = 1e-9
ROBUSTNESS_PROFILE_KEYS = ("smoke", "quick", "standard")
ROBUSTNESS_SMOKE_PERIOD_KEYS = ("2020_2025",)
ROBUSTNESS_SMOKE_UNIVERSES = ("crypto_included",)
ROBUSTNESS_SMOKE_COST_MULTIPLIERS = (1.0,)
ROBUSTNESS_SMOKE_MAX_WEIGHTS = (0.35,)
ROBUSTNESS_QUICK_PERIOD_KEYS = ("2020_2025",)
ROBUSTNESS_QUICK_UNIVERSES = ("crypto_included", "no_crypto")
ROBUSTNESS_QUICK_COST_MULTIPLIERS = (1.0, 3.0)
ROBUSTNESS_QUICK_MAX_WEIGHTS = (0.35,)

DECISION_PRIORITY = {
    "PASS": 0,
    "WATCH": 1,
    "FAIL": 2,
    "INVALID": 3,
}


def build_robustness_summary_payload(
    comparison: ComparisonSpec,
    *,
    fetch_market_universe_bundle,
    apply_universe_variant: Callable[[ComparisonSpec, str], ComparisonSpec],
    profile: str = "quick",
    period_keys: tuple[str, ...] | None = None,
    universes: tuple[str, ...] | None = None,
    cost_multipliers: tuple[float, ...] | None = None,
    max_weights: tuple[float, ...] | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    started_at = time.perf_counter()
    scenarios = build_robustness_scenarios(
        profile=profile,
        period_keys=period_keys,
        universes=universes,
        cost_multipliers=cost_multipliers,
        max_weights=max_weights,
    )
    scenario_results = []
    strategy_groups: dict[str, dict] = {}
    total_cached_runs = 0
    total_computed_runs = 0

    for scenario_index, scenario in enumerate(scenarios, start=1):
        if progress_callback:
            progress_callback({
                "kind": "scenario_started",
                "scenarioIndex": scenario_index,
                "scenarioCount": len(scenarios),
                "scenario": scenario,
            })
        scenario_started_at = time.perf_counter()
        scenario_comparison = build_scenario_comparison(
            comparison,
            scenario=scenario,
            apply_universe_variant=apply_universe_variant,
        )
        payload = build_walk_forward_comparison_payload(
            scenario_comparison,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
            start_year=scenario["walkForwardStartYear"],
            end_year=scenario["walkForwardEndYear"],
        )
        run_store_summary = payload["runStoreSummary"]
        total_cached_runs += int(run_store_summary["cachedRunCount"])
        total_computed_runs += int(run_store_summary["computedRunCount"])
        scenario_elapsed_seconds = round(time.perf_counter() - scenario_started_at, 3)

        ranked_results = payload["candidateResults"] + payload.get("referenceResults", [])
        diagnostics = collect_scenario_diagnostics(payload)
        scenario_record = {
            "kind": "robustness_scenario_result",
            "scenario": scenario,
            "runStoreSummary": run_store_summary,
            "elapsedSeconds": scenario_elapsed_seconds,
            "diagnostics": diagnostics,
            "resultCount": len(ranked_results),
            "results": [],
        }
        for rank, result in enumerate(ranked_results, start=1):
            result_record = build_scenario_strategy_result(result, rank=rank, diagnostics=diagnostics)
            scenario_record["results"].append(result_record)
            group = strategy_groups.setdefault(
                result_record["strategyKey"],
                {
                    "strategyKey": result_record["strategyKey"],
                    "strategyLabel": result_record["strategyLabel"],
                    "scenarioResults": [],
                },
            )
            group["scenarioResults"].append({**result_record, "scenario": scenario})
        scenario_results.append(scenario_record)
        if progress_callback:
            progress_callback({
                "kind": "scenario_completed",
                "scenarioIndex": scenario_index,
                "scenarioCount": len(scenarios),
                "scenario": scenario,
                "runStoreSummary": run_store_summary,
                "elapsedSeconds": scenario_elapsed_seconds,
            })

    strategy_results = [
        summarize_strategy_robustness(group, scenario_count=len(scenarios))
        for group in strategy_groups.values()
    ]
    strategy_results = sort_robustness_results(strategy_results)
    baseline_result = find_strategy_result(strategy_results, ROBUSTNESS_BASELINE_KEY)
    if baseline_result is not None:
        strategy_results = attach_delta_vs_baseline(strategy_results, baseline_result)
    matrix = build_robustness_matrix(scenarios)

    return {
        "kind": "robustness_summary",
        "schemaVersion": "v1",
        "profile": profile,
        "baselineKey": ROBUSTNESS_BASELINE_KEY,
        "baselineAvailable": baseline_result is not None,
        "scenarioCount": len(scenarios),
        "elapsedSeconds": round(time.perf_counter() - started_at, 3),
        "runStoreSummary": {
            "cachedRunCount": total_cached_runs,
            "computedRunCount": total_computed_runs,
        },
        "decisionSummary": build_decision_summary(strategy_results),
        "matrix": matrix,
        "strategyResults": strategy_results,
        "scenarioResults": scenario_results,
    }


def build_robustness_scenarios(
    *,
    profile: str = "quick",
    period_keys: tuple[str, ...] | None = None,
    universes: tuple[str, ...] | None = None,
    cost_multipliers: tuple[float, ...] | None = None,
    max_weights: tuple[float, ...] | None = None,
) -> list[dict]:
    periods, selected_universes, selected_costs, selected_weights = resolve_robustness_matrix(
        profile=profile,
        period_keys=period_keys,
        universes=universes,
        cost_multipliers=cost_multipliers,
        max_weights=max_weights,
    )
    scenarios = []
    for period in periods:
        for universe in selected_universes:
            for cost_multiplier in selected_costs:
                for max_weight in selected_weights:
                    scenarios.append(
                        {
                            "key": (
                                f"{period['key']}__{universe}"
                                f"__cost_x{format_key_number(cost_multiplier)}"
                                f"__cap_{int(max_weight * 100)}"
                            ),
                            "period": period["label"],
                            "periodKey": period["key"],
                            "startDate": period["startDate"],
                            "endDate": period["endDate"],
                            "walkForwardStartYear": period["walkForwardStartYear"],
                            "walkForwardEndYear": period["walkForwardEndYear"],
                            "universe": universe,
                            "costMultiplier": cost_multiplier,
                            "maxInvestmentRatio": ROBUSTNESS_MAX_INVESTMENT_RATIO,
                            "maxWeight": max_weight,
                            "maxWeightPct": round(max_weight * 100, 1),
                        }
                    )
    return scenarios


def resolve_robustness_matrix(
    *,
    profile: str,
    period_keys: tuple[str, ...] | None,
    universes: tuple[str, ...] | None,
    cost_multipliers: tuple[float, ...] | None,
    max_weights: tuple[float, ...] | None,
) -> tuple[tuple[dict, ...], tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
    if profile not in ROBUSTNESS_PROFILE_KEYS:
        raise ValueError(f"Unknown robustness profile: {profile}")

    if profile == "standard":
        base_period_keys = tuple(period["key"] for period in ROBUSTNESS_PERIODS)
        base_universes = ROBUSTNESS_UNIVERSES
        base_costs = ROBUSTNESS_COST_MULTIPLIERS
        base_weights = ROBUSTNESS_MAX_WEIGHTS
    elif profile == "quick":
        base_period_keys = ROBUSTNESS_QUICK_PERIOD_KEYS
        base_universes = ROBUSTNESS_QUICK_UNIVERSES
        base_costs = ROBUSTNESS_QUICK_COST_MULTIPLIERS
        base_weights = ROBUSTNESS_QUICK_MAX_WEIGHTS
    else:
        base_period_keys = ROBUSTNESS_SMOKE_PERIOD_KEYS
        base_universes = ROBUSTNESS_SMOKE_UNIVERSES
        base_costs = ROBUSTNESS_SMOKE_COST_MULTIPLIERS
        base_weights = ROBUSTNESS_SMOKE_MAX_WEIGHTS

    selected_period_keys = period_keys or base_period_keys
    selected_universes = universes or base_universes
    selected_costs = cost_multipliers or base_costs
    selected_weights = max_weights or base_weights

    period_by_key = {period["key"]: period for period in ROBUSTNESS_PERIODS}
    periods = tuple(period_by_key[key] for key in selected_period_keys)
    return periods, selected_universes, selected_costs, selected_weights


def build_robustness_matrix(scenarios: list[dict]) -> dict:
    period_by_key = {
        scenario["periodKey"]: {
            "key": scenario["periodKey"],
            "label": scenario["period"],
            "startDate": scenario["startDate"],
            "endDate": scenario["endDate"],
            "walkForwardStartYear": scenario["walkForwardStartYear"],
            "walkForwardEndYear": scenario["walkForwardEndYear"],
        }
        for scenario in scenarios
    }
    period_order = {period["key"]: index for index, period in enumerate(ROBUSTNESS_PERIODS)}
    return {
        "periods": sorted(period_by_key.values(), key=lambda period: period_order[period["key"]]),
        "universes": sorted({scenario["universe"] for scenario in scenarios}),
        "costMultipliers": sorted({float(scenario["costMultiplier"]) for scenario in scenarios}),
        "maxWeightPcts": sorted({round(float(scenario["maxWeight"]) * 100, 1) for scenario in scenarios}),
        "maxInvestmentPct": round(ROBUSTNESS_MAX_INVESTMENT_RATIO * 100, 1),
    }


def build_decision_summary(strategy_results: list[dict]) -> dict:
    counts = {decision: 0 for decision in DECISION_PRIORITY}
    for result in strategy_results:
        counts[result["decision"]] += 1
    return {
        "strategyCount": len(strategy_results),
        "counts": counts,
    }


def find_strategy_result(results: list[dict], strategy_key: str) -> dict | None:
    return next(
        (result for result in results if result["strategyKey"] == strategy_key),
        None,
    )


def attach_delta_vs_baseline(results: list[dict], baseline: dict) -> list[dict]:
    return [
        {
            **result,
            "deltaVsBaseline": build_delta_vs_baseline(result, baseline),
        }
        for result in results
    ]


def build_delta_vs_baseline(result: dict, baseline: dict) -> dict:
    return {
        "averageSharpeRatio": metric_delta(result, baseline, "averageSharpeRatio"),
        "worstSharpeRatio": metric_delta(result, baseline, "worstSharpeRatio"),
        "averageTotalReturnPct": metric_delta(result, baseline, "averageTotalReturnPct"),
        "worstMaxDrawdownPct": metric_delta(result, baseline, "worstMaxDrawdownPct"),
        "averageTurnoverPct": metric_delta(result, baseline, "averageTurnoverPct"),
        "maxTurnoverPct": metric_delta(result, baseline, "maxTurnoverPct"),
        "cryptoSensitivity": metric_delta(result, baseline, "cryptoSensitivity"),
        "costSensitivity": metric_delta(result, baseline, "costSensitivity"),
    }


def metric_delta(result: dict, baseline: dict, key: str) -> float:
    return round(float(result[key]) - float(baseline[key]), 6)


def format_key_number(value: float) -> str:
    return str(value).replace(".", "_")


def build_scenario_comparison(
    comparison: ComparisonSpec,
    *,
    scenario: dict,
    apply_universe_variant: Callable[[ComparisonSpec, str], ComparisonSpec],
) -> ComparisonSpec:
    scenario_comparison = copy.deepcopy(comparison)
    scenario_comparison.comparison_id = f"{comparison.comparison_id}__robust__{scenario['key']}"
    scenario_comparison.run_spec.market_slice = replace(
        scenario_comparison.run_spec.market_slice,
        start_date=scenario["startDate"],
        end_date=scenario["endDate"],
    )
    scenario_comparison.run_spec.execution_assumptions = replace(
        scenario_comparison.run_spec.execution_assumptions,
        cost_model=scale_cost_model_spec(
            scenario_comparison.run_spec.execution_assumptions.cost_model,
            float(scenario["costMultiplier"]),
        ),
    )
    risk_controls = build_risk_controls_spec(
        max_investment_ratio=float(scenario["maxInvestmentRatio"]),
        max_weight=float(scenario["maxWeight"]),
    )
    scenario_comparison.candidate_strategies = [
        replace(strategy, risk_controls=risk_controls)
        for strategy in scenario_comparison.candidate_strategies
    ]
    scenario_comparison.reference_strategies = [
        replace(strategy, risk_controls=risk_controls)
        for strategy in scenario_comparison.reference_strategies
    ]
    condition_variant = ConditionVariant(
        key=(
            f"robust_cost_x{format_key_number(float(scenario['costMultiplier']))}"
            f"__invest_{int(float(scenario['maxInvestmentRatio']) * 100)}"
            f"__cap_{int(float(scenario['maxWeight']) * 100)}"
        ),
        label=(
            f"Robustness cost x{float(scenario['costMultiplier']):.1f} / "
            f"investment {int(float(scenario['maxInvestmentRatio']) * 100)}% / "
            f"cap {float(scenario['maxWeight']) * 100:.0f}%"
        ),
        cost_multiplier=float(scenario["costMultiplier"]),
        max_investment_ratio=float(scenario["maxInvestmentRatio"]),
        max_weight=float(scenario["maxWeight"]),
    )
    scenario_comparison.condition_variants = [condition_variant]
    return apply_universe_variant(scenario_comparison, str(scenario["universe"]))


def collect_scenario_diagnostics(payload: dict) -> dict:
    evaluation = payload.get("comparison", {}).get("runSpec", {}).get("evaluation", {})
    availability_diagnostics = evaluation.get("availabilityDiagnostics") or {}
    instrument_diagnostics = evaluation.get("instrumentDiagnostics") or {}
    actionable_warnings = availability_diagnostics.get("actionableWarnings") or []
    calendar_boundary_warnings = availability_diagnostics.get("calendarBoundaryWarnings") or []
    asset_lifecycle_warnings = availability_diagnostics.get("assetLifecycleWarnings") or []
    unknown_symbols = instrument_diagnostics.get("unknownSymbols") or []
    diagnostic_events = list(
        evaluation.get("diagnosticEvents")
        or build_evaluation_diagnostic_events(
            availability_diagnostics=availability_diagnostics,
            instrument_diagnostics=instrument_diagnostics,
        )
    )
    diagnostic_summary = summarize_diagnostic_events(diagnostic_events)
    representative_event = representative_diagnostic_event(diagnostic_events)
    return {
        "actionableWarningCount": len(actionable_warnings),
        "calendarBoundaryWarningCount": len(calendar_boundary_warnings),
        "assetLifecycleWarningCount": len(asset_lifecycle_warnings),
        "mixedMarketCalendar": bool(instrument_diagnostics.get("mixedMarketCalendar")),
        "unknownSymbols": list(unknown_symbols),
        "flags": diagnostic_summary["flags"],
        "diagnosticSummary": diagnostic_summary,
        "diagnosticEvents": diagnostic_events,
        "representativeDiagnostic": representative_event,
        "actionableWarnings": list(actionable_warnings),
        "calendarBoundaryWarnings": list(calendar_boundary_warnings),
        "assetLifecycleWarnings": list(asset_lifecycle_warnings),
    }


PORTFOLIO_DRILLDOWN_METRIC_KEYS = (
    "sharpeRatio",
    "totalReturnPct",
    "maxDrawdownPct",
    "turnoverPct",
)
AVAILABILITY_DRILLDOWN_KEYS = (
    "barCount",
    "minAvailableAssetCount",
    "maxAvailableAssetCount",
    "minEligibleAssetCount",
    "maxEligibleAssetCount",
    "newlyEligibleAssetCount",
    "removedAssetCount",
    "newlyEligibleAssets",
    "removedAssets",
)


def project_weight_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "asset": str(row["asset"]),
            "weightPct": float(row["weightPct"]),
        }
        for row in rows
        if "asset" in row and "weightPct" in row
    ]


def summarize_weight_diversification(weights: list[dict], selected_assets: list[str]) -> dict:
    asset_weights = sorted(
        (
            float(row["weightPct"])
            for row in weights
            if row["asset"] != "CASH" and float(row["weightPct"]) > WEIGHT_EPSILON_PCT
        ),
        reverse=True,
    )
    top5_weight = sum(asset_weights[:5])
    return {
        "holdingCount": len(asset_weights),
        "selectedAssetCount": len(selected_assets),
        "top5WeightPct": round(top5_weight, 6),
        "maxAssetWeightPct": round(max(asset_weights), 6) if asset_weights else 0.0,
    }


def summarize_weight_exposure(weights: list[dict]) -> dict:
    raw_cash_weight = sum(
        float(row["weightPct"])
        for row in weights
        if row["asset"] == "CASH"
    )
    raw_invested_weight = sum(
        float(row["weightPct"])
        for row in weights
        if row["asset"] != "CASH" and float(row["weightPct"]) > WEIGHT_EPSILON_PCT
    )
    invested_weight = min(100.0, raw_invested_weight)
    cash_weight = (
        raw_cash_weight
        if raw_cash_weight > 0
        else max(0.0, 100.0 - invested_weight)
    )
    return {
        "investedWeightPct": round(invested_weight, 6),
        "cashWeightPct": round(cash_weight, 6),
    }


def summarize_window_diversification(windows: list[dict]) -> dict:
    summaries = [
        window.get("diversificationSummary", {})
        for window in windows
        if window.get("diversificationSummary")
    ]
    if not summaries:
        return empty_diversification_summary()
    return {
        "averageHoldingCount": round(average([float(summary["holdingCount"]) for summary in summaries]), 6),
        "minimumHoldingCount": int(min(int(summary["holdingCount"]) for summary in summaries)),
        "averageSelectedAssetCount": round(average([float(summary["selectedAssetCount"]) for summary in summaries]), 6),
        "averageTop5WeightPct": round(average([float(summary["top5WeightPct"]) for summary in summaries]), 6),
        "maximumTop5WeightPct": round(max(float(summary["top5WeightPct"]) for summary in summaries), 6),
        "maximumSingleAssetWeightPct": round(max(float(summary["maxAssetWeightPct"]) for summary in summaries), 6),
    }


def summarize_window_exposure(windows: list[dict]) -> dict:
    summaries = [
        window.get("exposureSummary", {})
        for window in windows
        if window.get("exposureSummary")
    ]
    if not summaries:
        return empty_exposure_summary()
    return {
        "averageInvestedWeightPct": round(average([float(summary["investedWeightPct"]) for summary in summaries]), 6),
        "averageCashWeightPct": round(average([float(summary["cashWeightPct"]) for summary in summaries]), 6),
        "minimumCashWeightPct": round(min(float(summary["cashWeightPct"]) for summary in summaries), 6),
        "maximumCashWeightPct": round(max(float(summary["cashWeightPct"]) for summary in summaries), 6),
    }


def summarize_strategy_diversification(scenario_results: list[dict]) -> dict:
    summaries = [
        result.get("diversificationSummary", {})
        for result in scenario_results
        if result.get("diversificationSummary")
    ]
    if not summaries:
        return empty_diversification_summary()
    return {
        "averageHoldingCount": round(average([float(summary["averageHoldingCount"]) for summary in summaries]), 6),
        "minimumHoldingCount": int(min(int(summary["minimumHoldingCount"]) for summary in summaries)),
        "averageSelectedAssetCount": round(average([float(summary["averageSelectedAssetCount"]) for summary in summaries]), 6),
        "averageTop5WeightPct": round(average([float(summary["averageTop5WeightPct"]) for summary in summaries]), 6),
        "maximumTop5WeightPct": round(max(float(summary["maximumTop5WeightPct"]) for summary in summaries), 6),
        "maximumSingleAssetWeightPct": round(max(float(summary["maximumSingleAssetWeightPct"]) for summary in summaries), 6),
    }


def summarize_strategy_exposure(scenario_results: list[dict]) -> dict:
    summaries = [
        result.get("exposureSummary", {})
        for result in scenario_results
        if result.get("exposureSummary")
    ]
    if not summaries:
        return empty_exposure_summary()
    return {
        "averageInvestedWeightPct": round(average([float(summary["averageInvestedWeightPct"]) for summary in summaries]), 6),
        "averageCashWeightPct": round(average([float(summary["averageCashWeightPct"]) for summary in summaries]), 6),
        "minimumCashWeightPct": round(min(float(summary["minimumCashWeightPct"]) for summary in summaries), 6),
        "maximumCashWeightPct": round(max(float(summary["maximumCashWeightPct"]) for summary in summaries), 6),
    }


def empty_diversification_summary() -> dict:
    return {
        "averageHoldingCount": 0.0,
        "minimumHoldingCount": 0,
        "averageSelectedAssetCount": 0.0,
        "averageTop5WeightPct": 0.0,
        "maximumTop5WeightPct": 0.0,
        "maximumSingleAssetWeightPct": 0.0,
    }


def empty_exposure_summary() -> dict:
    return {
        "averageInvestedWeightPct": 0.0,
        "averageCashWeightPct": 0.0,
        "minimumCashWeightPct": 0.0,
        "maximumCashWeightPct": 0.0,
    }

def build_window_drilldowns(result: dict) -> list[dict]:
    return [
        build_window_drilldown(window)
        for window in result.get("windows", [])
    ]


def build_window_drilldown(window: dict) -> dict:
    weights = project_weight_rows(window.get("weights", []))
    selected_assets = [str(asset) for asset in window.get("selectedAssets", [])]
    return {
        "year": int(window["year"]),
        "testStartDate": window["testStartDate"],
        "testEndDate": window["testEndDate"],
        "test": project_portfolio_drilldown_metrics(window.get("test", {})),
        "train": project_portfolio_drilldown_metrics(window.get("train", {})),
        "weights": weights,
        "selectedAssets": selected_assets,
        "diversificationSummary": summarize_weight_diversification(weights, selected_assets),
        "exposureSummary": summarize_weight_exposure(weights),
        "executionDecisionSummary": project_execution_decision_summary(
            window.get("executionDecisionSummary", {})
        ),
        "testAvailability": project_availability_drilldown(window.get("testAvailability", {})),
        "trainAvailability": project_availability_drilldown(window.get("trainAvailability", {})),
    }


def project_portfolio_drilldown_metrics(summary: dict) -> dict:
    return {
        key: float(summary[key])
        for key in PORTFOLIO_DRILLDOWN_METRIC_KEYS
        if key in summary
    }


def project_availability_drilldown(summary: dict) -> dict:
    return {
        key: summary[key]
        for key in AVAILABILITY_DRILLDOWN_KEYS
        if key in summary
    }


def project_execution_decision_summary(summary: dict) -> dict:
    if not summary:
        return summarize_execution_decision_summaries([])
    return {
        "decisionCount": int(summary.get("decisionCount", 0)),
        "rebalanceCount": int(summary.get("rebalanceCount", 0)),
        "noTradeCount": int(summary.get("noTradeCount", 0)),
        "policyCounts": {
            str(key): int(value)
            for key, value in (summary.get("policyCounts") or {}).items()
        },
        "reasonCounts": {
            str(key): int(value)
            for key, value in (summary.get("reasonCounts") or {}).items()
        },
        "averageTurnoverPct": optional_float(summary.get("averageTurnoverPct")),
        "averageEstimatedCostPct": optional_float(summary.get("averageEstimatedCostPct")),
        "averageEstimatedEdgePct": optional_float(summary.get("averageEstimatedEdgePct")),
        "averageConfidence": optional_float(summary.get("averageConfidence")),
        "estimatedEdgePctDistribution": project_number_distribution(
            summary.get("estimatedEdgePctDistribution")
        ),
        "estimatedCostPctDistribution": project_number_distribution(
            summary.get("estimatedCostPctDistribution")
        ),
        "estimatedEdgeAfterCostPctDistribution": project_number_distribution(
            summary.get("estimatedEdgeAfterCostPctDistribution")
        ),
        "confidenceDistribution": project_number_distribution(
            summary.get("confidenceDistribution")
        ),
    }


def project_number_distribution(distribution: object) -> dict:
    if not isinstance(distribution, dict):
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "maximum": None,
        }
    return {
        "count": int(distribution.get("count", 0)),
        "minimum": optional_float(distribution.get("minimum")),
        "median": optional_float(distribution.get("median")),
        "maximum": optional_float(distribution.get("maximum")),
    }


def optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)

def build_worst_window_summary(windows: list[dict]) -> dict:
    if not windows:
        return {}
    return min(windows, key=window_sort_key)


def build_strategy_worst_window_summary(results: list[dict]) -> dict:
    candidates = [
        (result, window)
        for result in results
        for window in result.get("windows", [])
    ]
    if not candidates:
        return {}
    worst_result, worst_window = min(
        candidates,
        key=lambda candidate: (
            *window_sort_key(candidate[1]),
            candidate[0]["scenario"]["key"],
        ),
    )
    scenario = worst_result["scenario"]
    return {
        "scenarioKey": scenario["key"],
        "period": scenario["period"],
        "universe": scenario["universe"],
        "costMultiplier": float(scenario["costMultiplier"]),
        "maxWeightPct": float(scenario["maxWeightPct"]),
        "rank": int(worst_result["rank"]),
        "window": worst_window,
    }


def window_sort_key(window: dict) -> tuple[float, float, float, int]:
    test_metrics = window.get("test", {})
    return (
        float(test_metrics["sharpeRatio"]),
        float(test_metrics["totalReturnPct"]),
        float(test_metrics["maxDrawdownPct"]),
        int(window["year"]),
    )


def build_scenario_strategy_result(result: dict, *, rank: int, diagnostics: dict) -> dict:
    windows = build_window_drilldowns(result)
    return {
        "strategyKey": result["strategyKey"],
        "strategyLabel": result["strategyLabel"],
        "rank": rank,
        "averageSharpeRatio": float(result["averageSharpeRatio"]),
        "minimumSharpeRatio": float(result["minimumSharpeRatio"]),
        "averageTotalReturnPct": float(result["averageTotalReturnPct"]),
        "averageMaxDrawdownPct": float(result["averageMaxDrawdownPct"]),
        "averageTurnoverPct": float(result["averageTurnoverPct"]),
        "positiveReturnWindowCount": int(result["positiveReturnWindowCount"]),
        "windowCount": int(result["windowCount"]),
        "diagnostics": diagnostics,
        "diversificationSummary": summarize_window_diversification(windows),
        "exposureSummary": summarize_window_exposure(windows),
        "executionDecisionSummary": project_execution_decision_summary(
            result.get("executionDecisionSummary", {})
        ),
        "windows": windows,
        "worstWindow": build_worst_window_summary(windows),
    }


def summarize_strategy_robustness(group: dict, *, scenario_count: int) -> dict:
    results = group["scenarioResults"]
    sharpe_values = [float(result["averageSharpeRatio"]) for result in results]
    min_sharpe_values = [float(result["minimumSharpeRatio"]) for result in results]
    return_values = [float(result["averageTotalReturnPct"]) for result in results]
    drawdown_values = [float(result["averageMaxDrawdownPct"]) for result in results]
    turnover_values = [float(result["averageTurnoverPct"]) for result in results]
    top5_count = sum(1 for result in results if int(result["rank"]) <= 5)
    positive_count = sum(
        1 for result in results
        if int(result["positiveReturnWindowCount"]) == int(result["windowCount"])
    )
    diagnostic_flags = sorted({
        flag
        for result in results
        for flag in result["diagnostics"].get("flags", [])
    })
    crypto_sensitivity = compute_crypto_sensitivity(results)
    cost_sensitivity = compute_cost_sensitivity(results)
    diagnostic_events = [
        event
        for result in results
        for event in result["diagnostics"].get("diagnosticEvents", [])
    ]
    diagnostic_summary = summarize_diagnostic_events(diagnostic_events)
    representative_event = representative_diagnostic_event(diagnostic_events)
    decision_result = build_robustness_decision(
        scenario_count=scenario_count,
        worst_sharpe=min(min_sharpe_values),
        top5_count=top5_count,
        diagnostic_flags=diagnostic_flags,
        crypto_sensitivity=crypto_sensitivity,
        cost_sensitivity=cost_sensitivity,
        diagnostic_events=diagnostic_events,
    )
    return {
        "kind": "robustness_strategy_result",
        "strategyKey": group["strategyKey"],
        "strategyLabel": group["strategyLabel"],
        "decision": decision_result["decision"],
        "decisionReasons": decision_result["reasons"],
        "scenarioCount": len(results),
        "averageSharpeRatio": round(average(sharpe_values), 6),
        "worstSharpeRatio": round(min(min_sharpe_values), 6),
        "sharpeStdDev": round(population_std_dev(sharpe_values), 6),
        "averageTotalReturnPct": round(average(return_values), 6),
        "worstMaxDrawdownPct": round(min(drawdown_values), 6),
        "averageTurnoverPct": round(average(turnover_values), 6),
        "maxTurnoverPct": round(max(turnover_values), 6),
        "top5ScenarioCount": top5_count,
        "positiveScenarioCount": positive_count,
        "cryptoSensitivity": round(crypto_sensitivity, 6),
        "costSensitivity": round(cost_sensitivity, 6),
        "diagnosticFlags": diagnostic_flags,
        "diagnosticSummary": diagnostic_summary,
        "representativeDiagnostic": representative_event,
        "diversificationSummary": summarize_strategy_diversification(results),
        "exposureSummary": summarize_strategy_exposure(results),
        "executionDecisionSummary": summarize_execution_decision_summaries(
            [result.get("executionDecisionSummary", {}) for result in results]
        ),
        "worstScenario": build_worst_scenario_summary(results),
        "worstWindow": build_strategy_worst_window_summary(results),
        "scenarioResults": results,
    }


def build_worst_scenario_summary(results: list[dict]) -> dict:
    worst_result = min(
        results,
        key=lambda result: (
            float(result["minimumSharpeRatio"]),
            float(result["averageSharpeRatio"]),
            result["scenario"]["key"],
        ),
    )
    return {
        "scenarioKey": worst_result["scenario"]["key"],
        "period": worst_result["scenario"]["period"],
        "universe": worst_result["scenario"]["universe"],
        "costMultiplier": float(worst_result["scenario"]["costMultiplier"]),
        "maxWeightPct": float(worst_result["scenario"]["maxWeightPct"]),
        "averageSharpeRatio": round(float(worst_result["averageSharpeRatio"]), 6),
        "minimumSharpeRatio": round(float(worst_result["minimumSharpeRatio"]), 6),
        "rank": int(worst_result["rank"]),
    }


def compute_crypto_sensitivity(results: list[dict]) -> float:
    crypto_values = [
        float(result["averageSharpeRatio"])
        for result in results
        if result["scenario"]["universe"] == "crypto_included"
    ]
    no_crypto_values = [
        float(result["averageSharpeRatio"])
        for result in results
        if result["scenario"]["universe"] == "no_crypto"
    ]
    if not crypto_values or not no_crypto_values:
        return 0.0
    return average(no_crypto_values) - average(crypto_values)


def compute_cost_sensitivity(results: list[dict]) -> float:
    low_cost_values = [
        float(result["averageSharpeRatio"])
        for result in results
        if float(result["scenario"]["costMultiplier"]) == 1.0
    ]
    high_cost_values = [
        float(result["averageSharpeRatio"])
        for result in results
        if float(result["scenario"]["costMultiplier"]) == 3.0
    ]
    if not low_cost_values or not high_cost_values:
        return 0.0
    return average(high_cost_values) - average(low_cost_values)


def classify_robustness_decision(
    *,
    scenario_count: int,
    worst_sharpe: float,
    top5_count: int,
    diagnostic_flags: list[str],
    crypto_sensitivity: float,
    cost_sensitivity: float,
) -> str:
    return build_robustness_decision(
        scenario_count=scenario_count,
        worst_sharpe=worst_sharpe,
        top5_count=top5_count,
        diagnostic_flags=diagnostic_flags,
        crypto_sensitivity=crypto_sensitivity,
        cost_sensitivity=cost_sensitivity,
    )["decision"]


def build_robustness_decision(
    *,
    scenario_count: int,
    worst_sharpe: float,
    top5_count: int,
    diagnostic_flags: list[str] | None,
    crypto_sensitivity: float,
    cost_sensitivity: float,
    diagnostic_events: list[dict[str, object]] | None = None,
) -> dict:
    diagnostic_flags = diagnostic_flags or []
    diagnostic_events = diagnostic_events or []
    top5_ratio = top5_count / scenario_count if scenario_count else 0.0
    reasons = []
    if "mixed calendar" in diagnostic_flags:
        reasons.append("mixed calendar diagnostic only")
    has_invalid_availability = (
        "actionable availability warning" in diagnostic_flags
        or has_invalidating_diagnostic(diagnostic_events, category="availability")
    )
    has_invalid_instrument = (
        "unknown symbols" in diagnostic_flags
        or has_invalidating_diagnostic(diagnostic_events, category="instrument")
    )
    if has_invalid_availability or has_invalid_instrument:
        if has_invalid_availability:
            reasons.append("actionable availability warning")
        if has_invalid_instrument:
            reasons.append("unknown symbols")
        return {"decision": "INVALID", "reasons": reasons}
    if worst_sharpe < 0.0 or top5_ratio < 0.25:
        if worst_sharpe < 0.0:
            reasons.append("worst Sharpe below 0.0")
        if top5_ratio < 0.25:
            reasons.append("top-5 scenario ratio below 25%")
        return {"decision": "FAIL", "reasons": reasons}
    if cost_sensitivity <= -0.30 or crypto_sensitivity <= -0.30:
        if cost_sensitivity <= -0.30:
            reasons.append("cost sensitivity below -0.30")
        if crypto_sensitivity <= -0.30:
            reasons.append("crypto sensitivity below -0.30")
        return {"decision": "WATCH", "reasons": reasons}
    if not reasons:
        reasons.append("passed robustness thresholds")
    return {"decision": "PASS", "reasons": reasons}


def sort_robustness_results(results: list[dict]) -> list[dict]:
    return sorted(
        results,
        key=lambda result: (
            DECISION_PRIORITY[result["decision"]],
            -float(result["worstSharpeRatio"]),
            -int(result["top5ScenarioCount"]),
            -float(result["averageSharpeRatio"]),
            result["strategyKey"],
        ),
    )


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def population_std_dev(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = average(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
