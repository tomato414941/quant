from __future__ import annotations

import copy
import math
import time
from dataclasses import replace
from typing import Callable

from app.comparison_models import ComparisonSpec, ConditionVariant, scale_cost_model_spec
from app.comparison_service import build_walk_forward_comparison_payload
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
    matrix = build_robustness_matrix(scenarios)

    return {
        "kind": "robustness_summary",
        "schemaVersion": "v1",
        "profile": profile,
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


def build_scenario_strategy_result(result: dict, *, rank: int, diagnostics: dict) -> dict:
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
        "worstScenario": build_worst_scenario_summary(results),
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
