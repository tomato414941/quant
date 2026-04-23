from __future__ import annotations

import copy
import math
from dataclasses import replace

from app.comparison_models import ComparisonSpec
from app.comparison_service import build_walk_forward_comparison_payload
from app.portfolio import (
    build_evaluator_strategy_spec,
    build_execution_policy_spec,
    build_strategy_definition_from_evaluator_strategy_spec,
)
from app.portfolio_domain import (
    StrategyDefinition,
    freeze_strategy_parameter_value,
    thaw_strategy_parameter_value,
)
from app.strategy_presets import EQUAL_WEIGHT, FULL_UNIVERSE


EDGE_ATTRIBUTION_COMPONENT_ORDER = (
    "cash",
    "universe_equal_weight",
    "selection_pure_equal_weight",
    "selection_tilt_equal_weight",
    "selection_model_no_tilt",
    "strategy_full",
)
EDGE_ATTRIBUTION_BASELINE_KEY = "universe_equal_weight"
EDGE_ATTRIBUTION_DELTA_METRICS = (
    "averageSharpeRatio",
    "minimumSharpeRatio",
    "averageTotalReturnPct",
    "averageCagrPct",
    "averageMaxDrawdownPct",
    "averageTurnoverPct",
    "averageFinalValueIndex",
)
FULL_UNIVERSE_SELECTION_STRATEGY_TYPES = {
    "full_universe",
    "full_universe_momentum_tilt",
    "full_universe_momentum_low_vol_tilt",
    "full_universe_momentum_macro_tilt",
}


def build_edge_attribution_payload(
    comparison: ComparisonSpec,
    *,
    fetch_market_universe_bundle,
    strategy_key: str,
    period: str,
    universe: str,
    start_year: int = 2020,
    end_year: int = 2025,
) -> dict:
    base_strategy = find_strategy_definition(comparison, strategy_key)
    attribution_comparison = build_edge_attribution_comparison(
        comparison,
        base_strategy=base_strategy,
        period=period,
    )
    walk_forward_payload = build_walk_forward_comparison_payload(
        attribution_comparison,
        fetch_market_universe_bundle=fetch_market_universe_bundle,
        start_year=start_year,
        end_year=end_year,
    )
    results_by_key = {
        result["strategyKey"]: result
        for result in walk_forward_payload.get("candidateResults", [])
    }
    component_results = build_component_results(
        base_strategy=base_strategy,
        walk_forward_payload=walk_forward_payload,
        results_by_key=results_by_key,
    )
    baseline = next(
        component
        for component in component_results
        if component["componentKey"] == EDGE_ATTRIBUTION_BASELINE_KEY
    )
    cash = next(component for component in component_results if component["componentKey"] == "cash")
    for component in component_results:
        component["deltaVsBaseline"] = build_metric_delta(component, baseline)
        component["deltaVsCash"] = build_metric_delta(component, cash)

    return {
        "kind": "edge_attribution",
        "schemaVersion": "v2",
        "strategyKey": base_strategy.key,
        "strategyLabel": base_strategy.label,
        "period": period,
        "universe": universe,
        "baselineKey": EDGE_ATTRIBUTION_BASELINE_KEY,
        "decisionPolicyKey": resolve_strategy_decision_policy_key(base_strategy),
        "componentApplicability": build_component_applicability(base_strategy),
        "walkForward": walk_forward_payload["walkForward"],
        "runStoreSummary": walk_forward_payload["runStoreSummary"],
        "components": component_results,
        "effectSummary": build_effect_summary(component_results),
        "diagnosis": build_edge_attribution_diagnosis(component_results),
    }


def find_strategy_definition(comparison: ComparisonSpec, strategy_key: str) -> StrategyDefinition:
    for strategy in comparison.candidate_strategies + comparison.reference_strategies:
        if strategy.key == strategy_key:
            return strategy
    raise ValueError(f"Unknown strategy key: {strategy_key}")


def build_edge_attribution_comparison(
    comparison: ComparisonSpec,
    *,
    base_strategy: StrategyDefinition,
    period: str,
) -> ComparisonSpec:
    attribution_comparison = copy.deepcopy(comparison)
    attribution_comparison.comparison_id = f"{comparison.comparison_id}__edge_attribution__{base_strategy.key}"
    attribution_comparison.title = "Edge attribution"
    attribution_comparison.question = "対象戦略の成績を cash / universe / selection / tilt / model / full に分解する"
    attribution_comparison.run_spec.market_slice.period = period
    attribution_comparison.candidate_strategies = build_edge_attribution_strategy_variants(base_strategy)
    attribution_comparison.reference_strategies = []
    return attribution_comparison


def build_edge_attribution_strategy_variants(base_strategy: StrategyDefinition) -> list[StrategyDefinition]:
    variants = [build_universe_equal_weight_variant(base_strategy)]
    if strategy_has_pure_selection_stage(base_strategy):
        variants.append(build_selection_pure_equal_weight_variant(base_strategy))
    if strategy_has_tilt_stage(base_strategy):
        variants.append(build_selection_tilt_equal_weight_variant(base_strategy))
    variants.append(build_selection_model_no_tilt_variant(base_strategy))
    variants.append(build_strategy_full_variant(base_strategy))
    return variants


def build_universe_equal_weight_variant(base_strategy: StrategyDefinition) -> StrategyDefinition:
    primary_signal = get_primary_selection_signal(base_strategy)
    execution_policy = build_execution_policy_spec(
        key=base_strategy.execution_plan.key,
        label=base_strategy.execution_plan.label,
        entry="train_once_then_periodic_rebalance",
        rebalance_schedule=base_strategy.execution_plan.rebalance_schedule,
    )
    strategy = build_evaluator_strategy_spec(
        strategy_id=build_component_strategy_key(base_strategy, "universe_equal_weight"),
        version=base_strategy.version,
        label=f"{base_strategy.label} | universe equal weight",
        hypothesis=base_strategy.hypothesis,
        description="Full eligible universe equal-weight baseline with the target strategy schedule and risk controls.",
        timeframe=primary_signal.signal_timeframe,
        investment_universe=base_strategy.investment_universe,
        selection=FULL_UNIVERSE,
        portfolio_model=EQUAL_WEIGHT,
        execution_policy=execution_policy,
        risk_controls=base_strategy.risk_controls,
        decision_schedule=base_strategy.execution_plan.decision_schedule,
        execution_mode="direct_signal_timeframe",
    )
    return build_strategy_definition_from_evaluator_strategy_spec(strategy)


def build_selection_pure_equal_weight_variant(base_strategy: StrategyDefinition) -> StrategyDefinition:
    return replace(
        base_strategy,
        strategy_id=build_component_strategy_key(base_strategy, "selection_pure_equal_weight"),
        label=f"{base_strategy.label} | selection pure equal weight",
        description="Target selection signal without tilt, using equal-weight portfolio construction.",
        signals=strip_tilt_from_selection_signals(get_selection_signals(base_strategy)),
        portfolio_model=EQUAL_WEIGHT,
    )


def build_selection_tilt_equal_weight_variant(base_strategy: StrategyDefinition) -> StrategyDefinition:
    return replace(
        base_strategy,
        strategy_id=build_component_strategy_key(base_strategy, "selection_tilt_equal_weight"),
        label=f"{base_strategy.label} | selection tilt equal weight",
        description="Target selection signal with tilt, using equal-weight portfolio construction.",
        signals=get_selection_signals(base_strategy),
        portfolio_model=EQUAL_WEIGHT,
    )


def build_selection_model_no_tilt_variant(base_strategy: StrategyDefinition) -> StrategyDefinition:
    return replace(
        base_strategy,
        strategy_id=build_component_strategy_key(base_strategy, "selection_model_no_tilt"),
        label=f"{base_strategy.label} | selection model no tilt",
        description="Target selection signal without tilt, using the target portfolio model.",
        signals=strip_tilt_from_selection_signals(get_selection_signals(base_strategy)),
        portfolio_model=base_strategy.portfolio_model,
    )


def build_strategy_full_variant(base_strategy: StrategyDefinition) -> StrategyDefinition:
    return replace(
        base_strategy,
        strategy_id=build_component_strategy_key(base_strategy, "strategy_full"),
        label=f"{base_strategy.label} | full strategy",
        description=base_strategy.description,
    )


def get_primary_selection_signal(strategy: StrategyDefinition):
    return get_selection_signals(strategy)[0]


def get_selection_signals(strategy: StrategyDefinition) -> tuple:
    selection_signals = tuple(
        signal for signal in strategy.signals
        if signal.source_kind == "selection_signal"
    )
    if not selection_signals:
        raise ValueError("Strategy must contain at least one selection signal.")
    return selection_signals


def resolve_strategy_decision_policy_key(strategy: StrategyDefinition) -> str:
    return str(dict(strategy.extensions).get("decision_policy", "cost_aware_no_trade"))


def get_selection_signal_parameters(signal) -> dict[str, object]:
    return {
        str(key): thaw_strategy_parameter_value(value)
        for key, value in signal.signal_parameters
    }


def strategy_has_pure_selection_stage(strategy: StrategyDefinition) -> bool:
    strategy_type = str(
        get_selection_signal_parameters(get_primary_selection_signal(strategy)).get("strategyType")
    )
    return strategy_type not in FULL_UNIVERSE_SELECTION_STRATEGY_TYPES


def strategy_has_tilt_stage(strategy: StrategyDefinition) -> bool:
    for signal in get_selection_signals(strategy):
        score_parameters = get_selection_signal_parameters(signal).get("scoreParameters")
        if not isinstance(score_parameters, dict):
            continue
        tilt_strength = float(score_parameters.get("tilt_strength", 0.0) or 0.0)
        tilt_shape = float(score_parameters.get("tilt_shape", 0.0) or 0.0)
        if abs(tilt_strength) > 1e-9 or abs(tilt_shape) > 1e-9:
            return True
    return False


def build_component_applicability(base_strategy: StrategyDefinition) -> list[dict[str, object]]:
    pure_selection_applicable = strategy_has_pure_selection_stage(base_strategy)
    tilt_applicable = strategy_has_tilt_stage(base_strategy)
    return [
        build_component_applicability_entry("cash", applicable=True),
        build_component_applicability_entry("universe_equal_weight", applicable=True),
        build_component_applicability_entry(
            "selection_pure_equal_weight",
            applicable=pure_selection_applicable,
            omitted_reason=None if pure_selection_applicable else "full_universe_selection_has_no_filter_stage",
        ),
        build_component_applicability_entry(
            "selection_tilt_equal_weight",
            applicable=tilt_applicable,
            omitted_reason=None if tilt_applicable else "selection_signal_has_no_tilt_stage",
        ),
        build_component_applicability_entry("selection_model_no_tilt", applicable=True),
        build_component_applicability_entry("strategy_full", applicable=True),
    ]


def build_component_applicability_entry(
    component_key: str,
    *,
    applicable: bool,
    omitted_reason: str | None = None,
) -> dict[str, object]:
    return {
        "componentKey": component_key,
        "label": "Cash" if component_key == "cash" else component_label(component_key),
        "applicable": applicable,
        "omittedReason": omitted_reason,
    }


def strip_tilt_from_selection_signals(signals: tuple) -> tuple:
    return tuple(strip_tilt_from_selection_signal(signal) for signal in signals)


def strip_tilt_from_selection_signal(signal):
    signal_parameters = {
        str(key): thaw_strategy_parameter_value(value)
        for key, value in signal.signal_parameters
    }
    score_parameters = signal_parameters.get("scoreParameters")
    if not isinstance(score_parameters, dict):
        return signal
    if "tilt_strength" not in score_parameters and "tilt_shape" not in score_parameters:
        return signal
    score_parameters = dict(score_parameters)
    score_parameters["tilt_strength"] = 0.0
    score_parameters["tilt_shape"] = 0.0
    signal_parameters["scoreParameters"] = score_parameters
    return replace(
        signal,
        signal_parameters=tuple(
            (str(key), freeze_strategy_parameter_value(value))
            for key, value in sorted(signal_parameters.items())
        ),
    )


def build_component_strategy_key(base_strategy: StrategyDefinition, component_key: str) -> str:
    return f"edge__{base_strategy.key}__{component_key}"


def build_component_results(
    *,
    base_strategy: StrategyDefinition,
    walk_forward_payload: dict,
    results_by_key: dict[str, dict],
) -> list[dict]:
    components = [build_cash_component(walk_forward_payload)]
    for component_key in EDGE_ATTRIBUTION_COMPONENT_ORDER:
        if component_key == "cash":
            continue
        strategy_key = build_component_strategy_key(base_strategy, component_key)
        result = results_by_key.get(strategy_key)
        if result is None:
            continue
        components.append(build_strategy_component(component_key, result))
    return components


def build_cash_component(walk_forward_payload: dict) -> dict:
    windows = [
        {
            "year": window["year"],
            "testStartDate": window["testStartDate"],
            "testEndDate": window["testEndDate"],
            "test": build_zero_metrics(),
            "executionTraceSummary": empty_execution_trace_summary(),
        }
        for window in walk_forward_payload["walkForward"]["windows"]
    ]
    window_count = len(windows)
    return {
        "componentKey": "cash",
        "strategyKey": "cash",
        "label": "Cash",
        "summary": {
            "averageSharpeRatio": 0.0,
            "minimumSharpeRatio": 0.0,
            "averageTotalReturnPct": 0.0,
            "averageCagrPct": 0.0,
            "averageMaxDrawdownPct": 0.0,
            "averageTurnoverPct": 0.0,
            "averageFinalValueIndex": 100.0,
            "positiveReturnWindowCount": 0,
            "windowCount": window_count,
        },
        "allocationSummary": empty_allocation_summary(),
        "executionDecisionSummary": empty_execution_summary(),
        "executionTraceSummary": empty_execution_trace_summary(),
        "windows": windows,
    }


def build_zero_metrics() -> dict:
    return {
        "totalReturnPct": 0.0,
        "cagrPct": 0.0,
        "sharpeRatio": 0.0,
        "maxDrawdownPct": 0.0,
        "turnoverPct": 0.0,
    }


def build_strategy_component(component_key: str, result: dict) -> dict:
    windows = build_component_windows(result.get("windows", []))
    return {
        "componentKey": component_key,
        "strategyKey": result["strategyKey"],
        "label": component_label(component_key),
        "strategyLabel": result["strategyLabel"],
        "applicable": True,
        "summary": build_component_summary(result),
        "allocationSummary": build_component_allocation_summary(windows),
        "executionDecisionSummary": result.get("executionDecisionSummary", {}),
        "executionTraceSummary": build_execution_trace_summary(
            [
                event
                for window in windows
                for event in window.get("executionTrace", [])
            ]
        ),
        "windows": windows,
    }


def build_component_windows(windows: list[dict]) -> list[dict]:
    return [
        {
            **window,
            "executionTraceSummary": build_execution_trace_summary(window.get("executionTrace", [])),
        }
        for window in windows
    ]


def component_label(component_key: str) -> str:
    labels = {
        "universe_equal_weight": "Universe equal weight",
        "selection_pure_equal_weight": "Selection pure equal weight",
        "selection_tilt_equal_weight": "Selection tilt equal weight",
        "selection_model_no_tilt": "Selection model no tilt",
        "strategy_full": "Strategy full",
    }
    return labels[component_key]


def build_component_summary(result: dict) -> dict:
    windows = result.get("windows", [])
    return {
        "averageSharpeRatio": result["averageSharpeRatio"],
        "minimumSharpeRatio": result["minimumSharpeRatio"],
        "averageTotalReturnPct": result["averageTotalReturnPct"],
        "averageCagrPct": average_window_metric(windows, "cagrPct"),
        "averageMaxDrawdownPct": result["averageMaxDrawdownPct"],
        "averageTurnoverPct": result["averageTurnoverPct"],
        "averageFinalValueIndex": round(100 + float(result["averageTotalReturnPct"]), 6),
        "positiveReturnWindowCount": result["positiveReturnWindowCount"],
        "windowCount": result["windowCount"],
    }


def average_window_metric(windows: list[dict], metric_key: str) -> float:
    if not windows:
        return 0.0
    return round(
        sum(float(window["test"][metric_key]) for window in windows) / len(windows),
        6,
    )


def build_metric_delta(component: dict, baseline: dict) -> dict:
    component_summary = component["summary"]
    baseline_summary = baseline["summary"]
    return {
        metric: round(float(component_summary[metric]) - float(baseline_summary[metric]), 6)
        for metric in EDGE_ATTRIBUTION_DELTA_METRICS
    }


def build_optional_metric_delta(component: dict | None, baseline: dict | None) -> dict | None:
    if component is None or baseline is None:
        return None
    return build_metric_delta(component, baseline)


def optional_metric_value(delta: dict | None, metric: str) -> float | None:
    if delta is None:
        return None
    return delta.get(metric)


def build_effect_summary(components: list[dict]) -> dict:
    components_by_key = {
        component["componentKey"]: component
        for component in components
    }
    cash = components_by_key["cash"]
    universe = components_by_key["universe_equal_weight"]
    pure_selection = components_by_key.get("selection_pure_equal_weight")
    tilt_selection = components_by_key.get("selection_tilt_equal_weight")
    model_no_tilt = components_by_key["selection_model_no_tilt"]
    full = components_by_key["strategy_full"]
    selection_baseline = pure_selection or universe
    return {
        "selectionBaselineKey": selection_baseline["componentKey"],
        "selectionBaselineLabel": component_label(selection_baseline["componentKey"]),
        "pureSelectionEffectVsUniverse": build_optional_metric_delta(pure_selection, universe),
        "tiltEffectVsPureSelection": build_optional_metric_delta(tilt_selection, pure_selection),
        "tiltEffectVsUniverse": build_optional_metric_delta(tilt_selection, universe),
        "portfolioModelEffectVsPureSelection": build_optional_metric_delta(model_no_tilt, pure_selection),
        "portfolioModelEffectVsUniverse": build_metric_delta(model_no_tilt, universe),
        "fullEffectVsTiltSelection": build_optional_metric_delta(full, tilt_selection),
        "fullEffectVsModelNoTilt": build_metric_delta(full, model_no_tilt),
        "fullEffectVsSelectionBaseline": build_metric_delta(full, selection_baseline),
        "fullEffectVsUniverse": build_metric_delta(full, universe),
        "fullEffectVsCash": build_metric_delta(full, cash),
    }



def empty_allocation_summary() -> dict:
    return {
        "averageHoldingCount": 0.0,
        "minimumHoldingCount": 0,
        "averageSelectedAssetCount": 0.0,
        "averageTop5WeightPct": 0.0,
        "maximumTop5WeightPct": 0.0,
        "maximumSingleAssetWeightPct": 0.0,
        "averageInvestedWeightPct": 0.0,
        "averageCashWeightPct": 100.0,
    }


def empty_execution_summary() -> dict:
    return {
        "decisionCount": 0,
        "rebalanceCount": 0,
        "noTradeCount": 0,
        "averageTurnoverPct": None,
        "averageEstimatedCostPct": None,
        "edgeHitRate": None,
    }


def empty_execution_trace_summary() -> dict:
    return {
        "eventCount": 0,
        "decisionEventCount": 0,
        "rebalanceEventCount": 0,
        "forcedUniverseChangeEventCount": 0,
        "tradeCount": 0,
        "noTradeCount": 0,
        "eventTypeCounts": {},
        "decisionActionCounts": {},
        "decisionReasonCounts": {},
        "edgeSourceCounts": {},
        "averageTurnoverPct": None,
        "averageEstimatedCostPct": None,
        "averageEstimatedEdgePct": None,
        "averageConfidence": None,
        "averageSelectedAssetCount": None,
        "averageAvailableAssetCount": None,
        "averageEligibleAssetCount": None,
    }


def build_execution_trace_summary(events: list[dict]) -> dict:
    if not events:
        return empty_execution_trace_summary()

    event_types = [event.get("eventType") for event in events]
    decision_actions = [event.get("decisionAction") for event in events]
    return {
        "eventCount": len(events),
        "decisionEventCount": sum(1 for event_type in event_types if event_type == "decision"),
        "rebalanceEventCount": sum(1 for event_type in event_types if event_type == "rebalance"),
        "forcedUniverseChangeEventCount": sum(
            1 for event_type in event_types if event_type == "forced_universe_change"
        ),
        "tradeCount": sum(
            1 for action in decision_actions if action in {"rebalance", "forced_rebalance"}
        ),
        "noTradeCount": sum(1 for action in decision_actions if action == "no_trade"),
        "eventTypeCounts": build_count_map(event_types),
        "decisionActionCounts": build_count_map(decision_actions),
        "decisionReasonCounts": build_count_map(event.get("decisionReason") for event in events),
        "edgeSourceCounts": build_count_map(event.get("edgeSource") for event in events),
        "averageTurnoverPct": average_optional_number(event.get("turnoverPct") for event in events),
        "averageEstimatedCostPct": average_optional_number(event.get("estimatedCostPct") for event in events),
        "averageEstimatedEdgePct": average_optional_number(event.get("estimatedEdgePct") for event in events),
        "averageConfidence": average_optional_number(event.get("averageConfidence") for event in events),
        "averageSelectedAssetCount": average_optional_number(
            len(event.get("selectedAssets") or []) for event in events
        ),
        "averageAvailableAssetCount": average_optional_number(
            event.get("availableAssetCount") for event in events
        ),
        "averageEligibleAssetCount": average_optional_number(
            event.get("eligibleAssetCount") for event in events
        ),
    }


def build_count_map(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def average_optional_number(values) -> float | None:
    numeric_values = []
    for value in values:
        if value is None:
            continue
        numeric_value = float(value)
        if math.isfinite(numeric_value):
            numeric_values.append(numeric_value)
    if not numeric_values:
        return None
    return round(sum(numeric_values) / len(numeric_values), 6)


def build_component_allocation_summary(windows: list[dict]) -> dict:
    summaries = [
        build_window_allocation_summary(window)
        for window in windows
        if window.get("weights") is not None
    ]
    if not summaries:
        return empty_allocation_summary()
    return {
        "averageHoldingCount": round(average([summary["holdingCount"] for summary in summaries]), 6),
        "minimumHoldingCount": int(min(summary["holdingCount"] for summary in summaries)),
        "averageSelectedAssetCount": round(average([summary["selectedAssetCount"] for summary in summaries]), 6),
        "averageTop5WeightPct": round(average([summary["top5WeightPct"] for summary in summaries]), 6),
        "maximumTop5WeightPct": round(max(summary["top5WeightPct"] for summary in summaries), 6),
        "maximumSingleAssetWeightPct": round(max(summary["maxAssetWeightPct"] for summary in summaries), 6),
        "averageInvestedWeightPct": round(average([summary["investedWeightPct"] for summary in summaries]), 6),
        "averageCashWeightPct": round(average([summary["cashWeightPct"] for summary in summaries]), 6),
    }


def build_window_allocation_summary(window: dict) -> dict:
    weights = list(window.get("weights") or [])
    selected_assets = list(window.get("selectedAssets") or [])
    asset_weights = sorted(
        (
            float(row["weightPct"])
            for row in weights
            if row.get("asset") != "CASH" and float(row.get("weightPct", 0.0)) > 0.0
        ),
        reverse=True,
    )
    invested_weight = min(100.0, sum(asset_weights))
    cash_weight = sum(
        float(row["weightPct"])
        for row in weights
        if row.get("asset") == "CASH"
    )
    if cash_weight <= 0:
        cash_weight = max(0.0, 100.0 - invested_weight)
    return {
        "holdingCount": len(asset_weights),
        "selectedAssetCount": len(selected_assets),
        "top5WeightPct": round(sum(asset_weights[:5]), 6),
        "maxAssetWeightPct": round(max(asset_weights), 6) if asset_weights else 0.0,
        "investedWeightPct": round(invested_weight, 6),
        "cashWeightPct": round(cash_weight, 6),
    }


def average(values: list[float]) -> float:
    return sum(float(value) for value in values) / len(values)


def build_edge_attribution_diagnosis(components: list[dict]) -> dict:
    components_by_key = {
        component["componentKey"]: component
        for component in components
    }
    universe = components_by_key["universe_equal_weight"]
    pure_selection = components_by_key.get("selection_pure_equal_weight")
    tilt_selection = components_by_key.get("selection_tilt_equal_weight")
    model_no_tilt = components_by_key["selection_model_no_tilt"]
    full = components_by_key["strategy_full"]
    has_pure_selection_stage = pure_selection is not None
    has_tilt_stage = tilt_selection is not None
    selection_baseline = pure_selection or universe
    pure_selection_effect = build_optional_metric_delta(pure_selection, universe)
    tilt_effect = build_optional_metric_delta(tilt_selection, pure_selection or universe)
    model_effect = build_metric_delta(model_no_tilt, selection_baseline)
    full_vs_tilt_effect = build_optional_metric_delta(full, tilt_selection)
    full_vs_model_effect = build_metric_delta(full, model_no_tilt)
    full_vs_pure_effect = build_optional_metric_delta(full, pure_selection)
    full_vs_selection_baseline_effect = build_metric_delta(full, selection_baseline)
    full_trace_summary = full.get("executionTraceSummary") or empty_execution_trace_summary()
    policy_counts = full.get("executionDecisionSummary", {}).get("policyCounts", {}) or {}
    has_decision_no_trade_path = "cost_aware_no_trade" in policy_counts or int(
        full_trace_summary.get("noTradeCount", 0)
    ) > 0
    cost_increase = optional_delta(
        full.get("executionDecisionSummary", {}).get("averageEstimatedCostPct"),
        selection_baseline.get("executionDecisionSummary", {}).get("averageEstimatedCostPct"),
    )
    turnover_increase = full_vs_selection_baseline_effect["averageTurnoverPct"]
    holding_count_change = round(
        float(full["allocationSummary"]["averageHoldingCount"])
        - float(selection_baseline["allocationSummary"]["averageHoldingCount"]),
        6,
    )
    max_weight_change = round(
        float(full["allocationSummary"]["maximumSingleAssetWeightPct"])
        - float(selection_baseline["allocationSummary"]["maximumSingleAssetWeightPct"]),
        6,
    )
    likely_causes = build_likely_causes(
        has_pure_selection_stage=has_pure_selection_stage,
        pure_selection_effect=pure_selection_effect,
        tilt_effect=tilt_effect,
        model_effect=model_effect,
        full_vs_tilt_effect=full_vs_tilt_effect,
        full_vs_model_effect=full_vs_model_effect,
        turnover_increase=turnover_increase,
        cost_increase=cost_increase,
        holding_count_change=holding_count_change,
        max_weight_change=max_weight_change,
        full_trace_summary=full_trace_summary,
    )
    return {
        "primaryFinding": build_primary_finding(
            has_pure_selection_stage=has_pure_selection_stage,
            has_tilt_stage=has_tilt_stage,
            pure_selection_effect=pure_selection_effect,
            tilt_effect=tilt_effect,
            model_effect=model_effect,
            full_vs_selection_baseline_effect=full_vs_selection_baseline_effect,
        ),
        "hasPureSelectionStage": has_pure_selection_stage,
        "hasTiltStage": has_tilt_stage,
        "hasDecisionNoTradePath": has_decision_no_trade_path,
        "selectionBaselineComponentKey": selection_baseline["componentKey"],
        "selectionBaselineLabel": component_label(selection_baseline["componentKey"]),
        "pureSelectionEffectReturnPct": optional_metric_value(pure_selection_effect, "averageTotalReturnPct"),
        "pureSelectionEffectSharpe": optional_metric_value(pure_selection_effect, "averageSharpeRatio"),
        "tiltEffectReturnPct": optional_metric_value(tilt_effect, "averageTotalReturnPct"),
        "tiltEffectSharpe": optional_metric_value(tilt_effect, "averageSharpeRatio"),
        "portfolioModelEffectReturnPct": model_effect["averageTotalReturnPct"],
        "portfolioModelEffectSharpe": model_effect["averageSharpeRatio"],
        "fullVsPureSelectionEffectReturnPct": optional_metric_value(full_vs_pure_effect, "averageTotalReturnPct"),
        "fullVsPureSelectionEffectSharpe": optional_metric_value(full_vs_pure_effect, "averageSharpeRatio"),
        "fullVsSelectionBaselineEffectReturnPct": full_vs_selection_baseline_effect["averageTotalReturnPct"],
        "fullVsSelectionBaselineEffectSharpe": full_vs_selection_baseline_effect["averageSharpeRatio"],
        "fullVsTiltSelectionEffectReturnPct": optional_metric_value(full_vs_tilt_effect, "averageTotalReturnPct"),
        "fullVsModelNoTiltEffectReturnPct": full_vs_model_effect["averageTotalReturnPct"],
        "selectionEffectReturnPct": optional_metric_value(pure_selection_effect, "averageTotalReturnPct"),
        "selectionEffectSharpe": optional_metric_value(pure_selection_effect, "averageSharpeRatio"),
        "portfolioAndExecutionEffectReturnPct": full_vs_selection_baseline_effect["averageTotalReturnPct"],
        "portfolioAndExecutionEffectSharpe": full_vs_selection_baseline_effect["averageSharpeRatio"],
        "turnoverIncreasePct": turnover_increase,
        "estimatedCostIncreasePct": cost_increase,
        "fullTraceNoTradeCount": int(full_trace_summary.get("noTradeCount", 0)),
        "fullTraceForcedUniverseChangeCount": int(
            full_trace_summary.get("forcedUniverseChangeEventCount", 0)
        ),
        "fullTraceAverageTurnoverPct": full_trace_summary.get("averageTurnoverPct"),
        "fullTraceAverageEstimatedCostPct": full_trace_summary.get("averageEstimatedCostPct"),
        "holdingCountChange": holding_count_change,
        "maxAssetWeightChangePct": max_weight_change,
        "likelyCauses": likely_causes,
    }


def optional_delta(left: object, right: object) -> float | None:
    if left is None or right is None:
        return None
    return round(float(left) - float(right), 6)


def build_primary_finding(
    *,
    has_pure_selection_stage: bool,
    has_tilt_stage: bool,
    pure_selection_effect: dict | None,
    tilt_effect: dict | None,
    model_effect: dict,
    full_vs_selection_baseline_effect: dict,
) -> str:
    if not has_pure_selection_stage:
        return "strategy_has_no_pure_selection_stage"
    if pure_selection_effect is not None and pure_selection_effect["averageTotalReturnPct"] <= 0:
        return "pure_selection_did_not_add_return"
    if has_tilt_stage and tilt_effect is not None and tilt_effect["averageTotalReturnPct"] < 0 and model_effect["averageTotalReturnPct"] < 0:
        return "selection_helped_but_tilt_and_portfolio_model_dragged"
    if has_tilt_stage and tilt_effect is not None and tilt_effect["averageTotalReturnPct"] < 0:
        return "selection_helped_but_tilt_dragged"
    if model_effect["averageTotalReturnPct"] < 0:
        return "selection_helped_but_portfolio_model_dragged"
    if full_vs_selection_baseline_effect["averageTotalReturnPct"] < 0:
        return "selection_helped_but_full_execution_dragged"
    if has_tilt_stage:
        return "selection_tilt_and_portfolio_helped"
    return "selection_and_portfolio_helped"


def build_likely_causes(
    *,
    has_pure_selection_stage: bool,
    pure_selection_effect: dict | None,
    tilt_effect: dict | None,
    model_effect: dict,
    full_vs_tilt_effect: dict | None,
    full_vs_model_effect: dict,
    turnover_increase: float,
    cost_increase: float | None,
    holding_count_change: float,
    max_weight_change: float,
    full_trace_summary: dict,
) -> list[str]:
    causes = []
    if not has_pure_selection_stage:
        causes.append("selection_stage_absent")
    elif pure_selection_effect is not None and pure_selection_effect["averageTotalReturnPct"] <= 0:
        causes.append("selection_signal_drag")
    if tilt_effect is not None and tilt_effect["averageTotalReturnPct"] < 0:
        causes.append("tilt_drag")
    if model_effect["averageTotalReturnPct"] < 0:
        causes.append("portfolio_model_drag")
    if full_vs_model_effect["averageTotalReturnPct"] < 0 and (
        full_vs_tilt_effect is None or full_vs_tilt_effect["averageTotalReturnPct"] < 0
    ):
        causes.append("combined_full_strategy_drag")
    if turnover_increase > 25.0:
        causes.append("turnover_drag")
    if cost_increase is not None and cost_increase > 0.005:
        causes.append("estimated_cost_drag")
    if holding_count_change < -2.0 or max_weight_change > 10.0:
        causes.append("concentration_change")
    if int(full_trace_summary.get("noTradeCount", 0)) > 0:
        causes.append("execution_skipped_trades")
    if int(full_trace_summary.get("forcedUniverseChangeEventCount", 0)) > 0:
        causes.append("forced_universe_changes")
    trace_cost = full_trace_summary.get("averageEstimatedCostPct")
    if trace_cost is not None and float(trace_cost) > 0.0:
        causes.append("execution_cost_visible_in_trace")
    if not causes:
        causes.append("no_single_obvious_drag")
    return causes
