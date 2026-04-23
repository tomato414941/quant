from app.edge_attribution_service import (
    build_edge_attribution_strategy_variants,
    build_execution_trace_summary,
)
from app.main import DEFAULT_COMPARISON_SPEC
from app.portfolio import get_strategy_signal_execution_contexts
from app.portfolio_domain import (
    build_executable_evaluator_strategy_spec_from_definition,
    thaw_strategy_parameter_value,
)


def get_selection_strategy_type(strategy) -> str:
    selection_signal = next(
        signal for signal in strategy.signals
        if signal.source_kind == "selection_signal"
    )
    return str(dict(selection_signal.signal_parameters).get("strategyType"))


def find_full_universe_tilt_strategy():
    for strategy in DEFAULT_COMPARISON_SPEC.candidate_strategies:
        strategy_type = get_selection_strategy_type(strategy)
        score_parameters = get_score_parameters(strategy)
        if strategy_type.startswith("full_universe") and float(score_parameters.get("tilt_strength", 0.0)) > 0.0:
            return strategy
    raise AssertionError("No full-universe tilted strategy found.")


def find_filtered_strategy():
    for strategy in DEFAULT_COMPARISON_SPEC.candidate_strategies:
        strategy_type = get_selection_strategy_type(strategy)
        if not strategy_type.startswith("full_universe"):
            return strategy
    raise AssertionError("No filtered strategy found.")


def get_score_parameters(strategy) -> dict:
    selection_signal = next(
        signal for signal in strategy.signals
        if signal.source_kind == "selection_signal"
    )
    score_parameters = thaw_strategy_parameter_value(
        dict(selection_signal.signal_parameters).get("scoreParameters")
    )
    return dict(score_parameters)


def get_runtime_score_parameters(strategy) -> dict:
    executable_strategy = build_executable_evaluator_strategy_spec_from_definition(strategy)
    selection_contexts, _ = get_strategy_signal_execution_contexts(executable_strategy)
    assert len(selection_contexts) == 1
    selection = selection_contexts[0]["selection"]
    return dict(selection.ranking_signal.score_parameters)


def test_edge_attribution_variants_omit_pure_selection_for_full_universe_tilt() -> None:
    base_strategy = find_full_universe_tilt_strategy()

    variants = build_edge_attribution_strategy_variants(base_strategy)
    variants_by_key = {variant.strategy_id.split("__")[-1]: variant for variant in variants}

    assert list(variants_by_key) == [
        "universe_equal_weight",
        "selection_tilt_equal_weight",
        "selection_model_no_tilt",
        "strategy_full",
    ]
    assert "selection_pure_equal_weight" not in variants_by_key
    assert variants_by_key["selection_tilt_equal_weight"].portfolio_model.model_type == "equal_weight"
    assert variants_by_key["selection_model_no_tilt"].portfolio_model == base_strategy.portfolio_model
    assert variants_by_key["strategy_full"].portfolio_model == base_strategy.portfolio_model

    tilt_params = get_score_parameters(variants_by_key["selection_tilt_equal_weight"])
    model_params = get_score_parameters(variants_by_key["selection_model_no_tilt"])
    full_params = get_score_parameters(variants_by_key["strategy_full"])

    assert tilt_params["tilt_strength"] > 0.0
    assert "tilt_shape" in tilt_params
    assert model_params["tilt_strength"] == 0.0
    assert model_params["tilt_shape"] == 0.0
    assert full_params == get_score_parameters(base_strategy)


def test_edge_attribution_filtered_variants_keep_pure_selection_stage() -> None:
    base_strategy = find_filtered_strategy()

    variants = build_edge_attribution_strategy_variants(base_strategy)
    variants_by_key = {variant.strategy_id.split("__")[-1]: variant for variant in variants}

    assert list(variants_by_key) == [
        "universe_equal_weight",
        "selection_pure_equal_weight",
        "selection_model_no_tilt",
        "strategy_full",
    ]
    assert "selection_tilt_equal_weight" not in variants_by_key
    assert variants_by_key["selection_pure_equal_weight"].portfolio_model.model_type == "equal_weight"
    assert variants_by_key["selection_model_no_tilt"].portfolio_model == base_strategy.portfolio_model

    pure_params = get_runtime_score_parameters(variants_by_key["selection_pure_equal_weight"])
    model_params = get_runtime_score_parameters(variants_by_key["selection_model_no_tilt"])
    full_params = get_runtime_score_parameters(variants_by_key["strategy_full"])

    assert pure_params == get_runtime_score_parameters(base_strategy)
    assert model_params == get_runtime_score_parameters(base_strategy)
    assert full_params == get_runtime_score_parameters(base_strategy)


def test_build_execution_trace_summary_counts_runtime_events() -> None:
    summary = build_execution_trace_summary([
        {
            "eventType": "decision",
            "decisionAction": "rebalance",
            "decisionReason": "scheduled_rebalance",
            "edgeSource": "signal_return_proxy",
            "turnoverPct": 10.0,
            "estimatedCostPct": 0.2,
            "estimatedEdgePct": 1.2,
            "averageConfidence": 0.6,
            "selectedAssets": ["AAA", "BBB"],
            "availableAssetCount": 4,
            "eligibleAssetCount": 3,
        },
        {
            "eventType": "rebalance",
            "decisionAction": "rebalance",
            "decisionReason": "scheduled_rebalance",
            "turnoverPct": 8.0,
            "estimatedCostPct": 0.1,
            "selectedAssets": ["AAA"],
            "availableAssetCount": 4,
            "eligibleAssetCount": 3,
        },
        {
            "eventType": "decision",
            "decisionAction": "no_trade",
            "decisionReason": "edge_below_cost",
            "edgeSource": "signal_return_proxy",
            "turnoverPct": 0.0,
            "estimatedCostPct": 0.0,
            "estimatedEdgePct": -0.5,
            "averageConfidence": 0.4,
            "selectedAssets": [],
            "availableAssetCount": 3,
            "eligibleAssetCount": 2,
        },
        {
            "eventType": "forced_universe_change",
            "decisionAction": "forced_rebalance",
            "decisionReason": "asset_unavailable",
            "turnoverPct": 5.0,
            "estimatedCostPct": 0.05,
            "selectedAssets": ["BBB"],
            "availableAssetCount": 2,
            "eligibleAssetCount": 1,
        },
    ])

    assert summary["eventCount"] == 4
    assert summary["decisionEventCount"] == 2
    assert summary["rebalanceEventCount"] == 1
    assert summary["forcedUniverseChangeEventCount"] == 1
    assert summary["tradeCount"] == 3
    assert summary["noTradeCount"] == 1
    assert summary["eventTypeCounts"] == {
        "decision": 2,
        "forced_universe_change": 1,
        "rebalance": 1,
    }
    assert summary["decisionActionCounts"] == {
        "forced_rebalance": 1,
        "no_trade": 1,
        "rebalance": 2,
    }
    assert summary["decisionReasonCounts"]["edge_below_cost"] == 1
    assert summary["edgeSourceCounts"] == {"signal_return_proxy": 2}
    assert summary["averageTurnoverPct"] == 5.75
    assert summary["averageEstimatedCostPct"] == 0.0875
    assert summary["averageEstimatedEdgePct"] == 0.35
    assert summary["averageConfidence"] == 0.5
    assert summary["averageSelectedAssetCount"] == 1.0
    assert summary["averageAvailableAssetCount"] == 3.25
    assert summary["averageEligibleAssetCount"] == 2.25


def test_build_execution_trace_summary_handles_empty_trace() -> None:
    summary = build_execution_trace_summary([])

    assert summary["eventCount"] == 0
    assert summary["eventTypeCounts"] == {}
    assert summary["averageTurnoverPct"] is None
