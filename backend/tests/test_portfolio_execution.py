import json
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd

from app.portfolio import (
    COST_AWARE_NO_TRADE_DECISION_POLICY,
    SIGNAL_RETURN_PROXY_EDGE_SOURCE,
    PREDICTION_FEATURE_NAMES,
    build_observation_spec,
    build_prediction_combiner_spec,
    build_derived_feature_spec,
    build_feature_spec,
    build_feature_input_spec,
    build_prediction_output_spec,
    build_alignment_policy_spec,
    build_decision_use_spec,
    build_prediction_engine_spec,
    build_prediction_learner_spec,
    build_predicted_quantity_spec,
    build_prediction_signal_source_spec,
    build_signal_spec,
    build_prediction_target_spec,
    build_predictor_spec,
    build_predictor_use_spec,
    build_ranking_feature_recipe_spec,
    build_strategy_definition_from_evaluator_strategy_spec,
    build_strategy_signal_execution_contexts_from_definition,
    build_strategy_definition,
    build_strategy_execution_plan_spec,
    build_strategy_signal_spec,
    build_training_spec,
    build_investment_universe_spec,
    build_portfolio_model_spec,
    build_portfolio_state,
    build_strategy_data_source_spec,
    build_strategy_feature_definition_spec,
    build_executable_evaluator_strategy_spec_from_definition,
    build_execution_policy_spec,
    build_risk_controls_spec,
    build_selection_spec,
    build_evaluator_strategy_spec,
    compare_portfolio_runs,
    compute_portfolio_allocation,
    compute_predictor_panel,
    convert_window_spec_to_bars,
    compute_trade_cost,
    compute_strategy_score_series,
    build_strategy_forecast_snapshot,
    evaluate_predictor_spec,
    get_direct_execution_strategy_definition_compatibility_issues,
    get_strategy_definition_signal_execution_contexts,
    get_strategy_signal_execution_contexts,
    is_direct_execution_compatible_strategy_definition,
    prepare_strategy_market_data,
    prepare_strategy_predictor_panel,
    prepare_strategy_signal_data,
    extract_predictor_signal_payload,
    prepare_signal_component_data,
    resample_market_frame_to_timeframe,
    resample_returns_frame_to_timeframe,
    resolve_decision_policy_kind,
    resolve_decision_schedule,
    resolve_strategy_market_data_timeframe_key,
    serialize_strategy_definition,
    serialize_evaluator_strategy_spec,
    serialize_strategy_signal_spec,
    summarize_portfolio_decision_events,
    select_assets,
    should_rebalance,
)
from app.strategy_definition_builder import (
    ExecutionVariantDefinition,
    PortfolioModelVariantDefinition,
    PredictorDefinitionDefinition,
    PredictorVariantDefinition,
    SelectionDefinitionDefinition,
    SelectionVariantDefinition,
    build_predictor_strategy_definition,
    build_predictor_strategy_definition_product,
    build_selection_strategy_definition,
    build_selection_strategy_definition_product,
)
from app.strategy_candidate_baselines import (
    BASELINE_CANDIDATE_DEFINITIONS,
)
from app.strategy_candidate_filtered import (
    FILTERED_CANDIDATE_DEFINITIONS,
)
from app.strategy_candidate_predictors import (
    PREDICTOR_CANDIDATE_DEFINITIONS,
)
from app.strategy_candidate_full_universe import (
    EXPERIMENTAL_FULL_UNIVERSE_CANDIDATE_DEFINITIONS,
    FULL_UNIVERSE_CANDIDATE_DEFINITIONS,
)
from app.strategy_candidate_timeframes import (
    TIMEFRAME_VARIANT_CANDIDATE_DEFINITIONS,
)
from app.strategy_candidate_universe_variants import (
    UNIVERSE_VARIANT_CANDIDATE_DEFINITIONS,
)
from app.strategy_catalog import CANONICAL_CANDIDATE_DEFINITIONS
from app.strategy_presets import FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M
from app.timeframe_models import DEFAULT_DAILY_TIMEFRAME, DEFAULT_MONTHLY_TIMEFRAME, DEFAULT_WEEKLY_TIMEFRAME


def make_execution_assumptions() -> dict:
    return {
        "kind": "close_execution_assumptions",
        "label": "終値約定",
        "parameters": {
            "fillPrice": "close",
        },
        "costModel": {
            "kind": "flat_cost",
            "parameters": {"commissionPct": 0.1, "slippagePct": 0.0},
            "perAssetOverrides": {},
        },
    }


def make_strategy(
    strategy_type: str,
    model_type: str,
    *,
    max_investment_ratio: float = 1.0,
    max_weight: float | None = None,
    score_parameters: dict[str, float | str | bool] | None = None,
    predictor_use=None,
    decision_policy: str = "direct_score_to_weight",
):
    return build_evaluator_strategy_spec(
        investment_universe=build_investment_universe_spec(
            tickers=[
                "SPY",
                "QQQ",
                "IWM",
                "EFA",
                "EEM",
                "TLT",
                "IEF",
                "LQD",
                "HYG",
                "GLD",
                "BTC-USD",
                "ETH-USD",
            ],
            key="test_universe",
            label="Test universe",
        ),
        selection=build_selection_spec(
            strategy_type,
            score_parameters=score_parameters,
        ),
        portfolio_model=build_portfolio_model_spec(model_type),
        risk_controls=build_risk_controls_spec(
            max_investment_ratio=max_investment_ratio,
            max_weight=max_weight,
        ),
        predictor_use=predictor_use,
        extensions={"decision_policy": decision_policy},
    )


def make_predictor_spec(
    strategy,
    *,
    predictor_key: str,
    label: str,
    horizon_bars: int,
    min_train_samples: int = 3,
    signal_source_feature_key: str | None = None,
    combiner_kind: str = "learner_only",
):
    signal_source_spec = None
    learner_spec = build_prediction_learner_spec(
        key=f"learner__{predictor_key}",
        label=f"{label} linear learner",
        kind="linear_regression",
    )
    if signal_source_feature_key is not None:
        signal_source_spec = build_prediction_signal_source_spec(
            key=f"signal-source__{predictor_key}",
            label=f"{label} signal source",
            kind="derived_feature",
            feature_key=signal_source_feature_key,
        )
    if combiner_kind == "weighted_blend":
        combiner_spec = build_prediction_combiner_spec(
            key=f"combiner__{predictor_key}",
            label=f"{label} weighted blend",
            kind="weighted_blend",
            signal_source_weight=0.7,
            learner_weight=0.3,
        )
    else:
        combiner_spec = build_prediction_combiner_spec(
            key=f"combiner__{predictor_key}",
            label=f"{label} learner only",
            kind="learner_only",
        )
    return build_predictor_spec(
        key=predictor_key,
        label=label,
        description=strategy.description,
        timeframe=strategy.timeframe,
        signal_spec=build_signal_spec(
            key=f"signal__{predictor_key}",
            label=f"{label} signal",
            observation_spec=build_observation_spec(
                key=f"observation__{predictor_key}",
                label=f"{label} observation",
                tickers=strategy.investment_universe.tickers,
                fields=strategy.selection.ranking_signal.feature_inputs,
            ),
            entity_kind="asset_set",
            entity_identifiers=strategy.investment_universe.tickers,
            output_spec=build_prediction_output_spec(
                key=f"output__{predictor_key}",
                label=f"{label} output",
                kind="score",
            ),
            decision_use_spec=build_decision_use_spec(strategy.selection),
        ),
        predicted_quantity_spec=build_predicted_quantity_spec(
            key=f"quantity__{predictor_key}",
            label=f"{label} quantity",
            kind="return",
        ),
        target_spec=build_prediction_target_spec(
            key=f"{predictor_key}__target",
            label=label,
            horizon_spec={"unit": "bars", "value": horizon_bars},
            baseline="cross_sectional_mean",
            transform="identity",
        ),
        feature_spec=build_feature_spec(
            key=f"features__{predictor_key}",
            label=f"{label} features",
            feature_inputs=tuple(
                build_feature_input_spec(key=feature_input)
                for feature_input in strategy.selection.ranking_signal.feature_inputs
            ),
            derived_features=tuple(
                build_derived_feature_spec(key=feature_key)
                for feature_key in PREDICTION_FEATURE_NAMES
            ),
            ranking_feature_recipe=build_ranking_feature_recipe_spec(strategy.selection),
        ),
        engine_spec=build_prediction_engine_spec(
            key=f"engine__{predictor_key}",
            label=f"{label} linear",
            signal_source_spec=signal_source_spec,
            learner_spec=learner_spec,
            combiner_spec=combiner_spec,
        ),
        training_spec=build_training_spec(
            key=f"training__{predictor_key}",
            label=f"{label} training",
            fit_mode="expanding",
            min_train_samples=min_train_samples,
        ),
    )


def test_compare_portfolio_runs_applies_rebalance_on_next_bar() -> None:
    closes = pd.DataFrame(
        {
            "AAA": [100, 100, 100, 100, 100, 100, 100],
            "BBB": [100, 100, 100, 100, 100, 200, 200],
        },
        index=pd.date_range("2025-01-01", periods=7, freq="D"),
    )
    strategy = build_evaluator_strategy_spec(
        investment_universe=build_investment_universe_spec(
            tickers=list(closes.columns),
            key="next_bar_execution_universe",
            label="Next bar execution universe",
        ),
        selection=build_selection_spec("full_universe"),
        portfolio_model=build_portfolio_model_spec("equal_weight"),
        execution_policy=build_execution_policy_spec(
            key="every_bar",
            label="毎バー",
            entry="train_once_then_periodic_rebalance",
            rebalance_schedule="every_bar",
        ),
        risk_controls=build_risk_controls_spec(max_investment_ratio=1.0),
        decision_schedule="every_bar",
    )

    allocation_calls = 0

    def fake_compute_portfolio_allocation(**kwargs):
        nonlocal allocation_calls
        allocation_calls += 1
        if allocation_calls == 1:
            return ["AAA"], np.asarray([1.0, 0.0], dtype="float64")
        return ["BBB"], np.asarray([0.0, 1.0], dtype="float64")

    with patch("app.portfolio.compute_portfolio_allocation", side_effect=fake_compute_portfolio_allocation):
        run = compare_portfolio_runs(
            closes=closes,
            volumes=None,
            strategies=[strategy],
            initial_capital=1000.0,
            split_ratio=0.5,
            transaction_cost=0.0,
            portfolio_state=build_portfolio_state(current_weights={}, cash_weight=1.0),
        )[0]

    assert run["splitAnalysis"]["test"]["portfolio"]["totalReturnPct"] == 0.0
    assert any(event["eventType"] == "decision" for event in run["executionTrace"])
    assert any(event["eventType"] == "rebalance" for event in run["executionTrace"])
    trace_event = run["executionTrace"][0]
    assert "targetWeights" in trace_event
    assert "executedWeights" in trace_event
    assert trace_event["phase"] in {"train", "test"}


def test_compare_portfolio_runs_uses_explicit_decision_schedule() -> None:
    closes = pd.DataFrame(
        {
            "AAA": [100, 110, 121, 133, 146, 146, 146, 146],
            "BBB": [100, 108, 116, 125, 135, 148, 163, 179],
            "CCC": [100, 107, 114, 122, 130, 139, 149, 160],
            "DDD": [100, 90, 81, 73, 95, 124, 161, 209],
        },
        index=pd.date_range("2025-01-01", periods=8, freq="D"),
    )
    volumes = pd.DataFrame(
        {ticker: [1_000_000 + idx * 10_000 for idx in range(len(closes))] for ticker in closes.columns},
        index=closes.index,
    )
    hold_strategy = build_evaluator_strategy_spec(
        strategy_id="decision_schedule__hold",
        investment_universe=build_investment_universe_spec(
            tickers=list(closes.columns),
            key="decision_schedule_universe",
            label="Decision schedule universe",
        ),
        selection=build_selection_spec(
            "momentum_top3",
            score_parameters={"windowSpec": {"unit": "days", "value": 2}},
        ),
        portfolio_model=build_portfolio_model_spec("equal_weight"),
        execution_policy=build_execution_policy_spec(
            key="every_bar",
            label="毎バー",
            entry="train_once_then_periodic_rebalance",
            rebalance_schedule="every_bar",
        ),
        risk_controls=build_risk_controls_spec(max_investment_ratio=1.0),
        decision_schedule="hold",
    )
    adaptive_strategy = build_evaluator_strategy_spec(
        strategy_id="decision_schedule__every_bar",
        investment_universe=hold_strategy.investment_universe,
        selection=hold_strategy.selection,
        portfolio_model=hold_strategy.portfolio_model,
        execution_policy=hold_strategy.execution_policy,
        risk_controls=hold_strategy.risk_controls,
        decision_schedule="every_bar",
    )

    runs = compare_portfolio_runs(
        closes=closes,
        volumes=volumes,
        strategies=[hold_strategy, adaptive_strategy],
        initial_capital=1000.0,
        split_ratio=0.5,
        transaction_cost=0.001,
        bars_per_year=252.0,
    )

    hold_run, adaptive_run = runs
    assert "DDD" not in hold_run["selectedAssets"]
    assert "DDD" in adaptive_run["selectedAssets"]
    assert adaptive_run["summary"]["turnoverPct"] > hold_run["summary"]["turnoverPct"]


def test_summarize_portfolio_decision_events_reports_decision_distributions() -> None:
    summary = summarize_portfolio_decision_events([
        {
            "policy": "cost_aware_no_trade",
            "action": "no_trade",
            "reason": "edge_below_cost",
            "turnoverPct": 10.0,
            "estimatedCostPct": 0.2,
            "estimatedEdgePct": 0.1,
            "edgeSource": SIGNAL_RETURN_PROXY_EDGE_SOURCE,
            "realizedEdgePct": -0.2,
            "realizedEdgeAfterCostPct": -0.4,
            "edgeHit": False,
            "averageConfidence": 0.5,
        },
        {
            "policy": "cost_aware_no_trade",
            "action": "rebalance",
            "reason": "edge_after_cost",
            "turnoverPct": 20.0,
            "estimatedCostPct": 0.3,
            "estimatedEdgePct": 0.7,
            "edgeSource": SIGNAL_RETURN_PROXY_EDGE_SOURCE,
            "realizedEdgePct": 0.5,
            "realizedEdgeAfterCostPct": 0.2,
            "edgeHit": True,
            "averageConfidence": 0.9,
        },
        {
            "policy": "direct_score_to_weight",
            "action": "rebalance",
            "reason": "direct_policy",
            "turnoverPct": 30.0,
            "estimatedCostPct": 0.1,
            "estimatedEdgePct": None,
            "realizedEdgePct": None,
            "realizedEdgeAfterCostPct": None,
            "edgeHit": None,
            "averageConfidence": None,
        },
    ])

    assert summary["estimatedEdgePctDistribution"] == {
        "count": 2,
        "minimum": 0.1,
        "median": 0.4,
        "maximum": 0.7,
    }
    assert summary["estimatedCostPctDistribution"] == {
        "count": 3,
        "minimum": 0.1,
        "median": 0.2,
        "maximum": 0.3,
    }
    assert summary["estimatedEdgeAfterCostPctDistribution"] == {
        "count": 2,
        "minimum": -0.1,
        "median": 0.15,
        "maximum": 0.4,
    }
    assert summary["realizedEdgePctDistribution"] == {
        "count": 2,
        "minimum": -0.2,
        "median": 0.15,
        "maximum": 0.5,
    }
    assert summary["realizedEdgeAfterCostPctDistribution"] == {
        "count": 2,
        "minimum": -0.4,
        "median": -0.1,
        "maximum": 0.2,
    }
    assert summary["confidenceDistribution"] == {
        "count": 2,
        "minimum": 0.5,
        "median": 0.7,
        "maximum": 0.9,
    }
    assert summary["edgeSourceCounts"] == {SIGNAL_RETURN_PROXY_EDGE_SOURCE: 2}
    assert summary["edgeHitCount"] == 1
    assert summary["edgeHitSampleCount"] == 2
    assert summary["edgeHitRate"] == 0.5
    assert summary["estimatedVsRealizedEdgeCorrelation"] == 1.0


def test_cost_aware_decision_policy_can_skip_rebalance_when_edge_is_below_cost() -> None:
    closes = pd.DataFrame(
        {
            "AAA": [100, 101, 102, 103, 104, 105, 106, 107],
            "BBB": [100, 99, 98, 97, 96, 95, 94, 93],
            "CCC": [100, 100, 101, 101, 102, 102, 103, 103],
        },
        index=pd.date_range("2025-01-01", periods=8, freq="D"),
    )
    strategy = build_evaluator_strategy_spec(
        strategy_id="cost_aware_no_trade_test",
        investment_universe=build_investment_universe_spec(
            tickers=list(closes.columns),
            key="cost_aware_universe",
            label="Cost aware universe",
        ),
        selection=build_selection_spec(
            "momentum_top3",
            score_parameters={"windowSpec": {"unit": "bars", "value": 3}},
        ),
        portfolio_model=build_portfolio_model_spec("equal_weight"),
        execution_policy=build_execution_policy_spec(
            key="every_bar",
            label="毎バー",
            entry="train_once_then_periodic_rebalance",
            rebalance_schedule="every_bar",
        ),
        risk_controls=build_risk_controls_spec(max_investment_ratio=1.0),
        decision_schedule="every_bar",
        extensions={"decision_policy": COST_AWARE_NO_TRADE_DECISION_POLICY},
    )

    run = compare_portfolio_runs(
        closes=closes,
        volumes=None,
        strategies=[strategy],
        initial_capital=1000.0,
        split_ratio=0.5,
        transaction_cost=0.2,
        portfolio_state=build_portfolio_state(current_weights={}, cash_weight=1.0),
    )[0]

    decision_summary = run["decisionSummary"]
    assert decision_summary["policyCounts"][COST_AWARE_NO_TRADE_DECISION_POLICY] >= 1
    assert decision_summary["noTradeCount"] >= 1
    assert "edge_below_cost" in decision_summary["reasonCounts"]
    assert decision_summary["edgeSourceCounts"][SIGNAL_RETURN_PROXY_EDGE_SOURCE] >= 1
    assert "realizedEdgePctDistribution" in decision_summary
    assert "edgeHitRate" in decision_summary
    no_trade_events = [event for event in run["executionTrace"] if event["decisionAction"] == "no_trade"]
    assert no_trade_events
    assert no_trade_events[0]["decisionReason"] == "edge_below_cost"
    assert no_trade_events[0]["estimatedEdgePct"] is not None
    assert run["strategy"]["components"]["optional"]["decisionPolicy"]["key"] == COST_AWARE_NO_TRADE_DECISION_POLICY


def test_default_decision_policy_is_cost_aware_no_trade() -> None:
    strategy = build_evaluator_strategy_spec(
        strategy_id="default_decision_policy_test",
        investment_universe=build_investment_universe_spec(
            tickers=["AAA", "BBB", "CCC"],
            key="default_decision_policy_universe",
            label="Default decision policy universe",
        ),
        selection=build_selection_spec(
            "momentum_top3",
            score_parameters={"windowSpec": {"unit": "bars", "value": 3}},
        ),
        portfolio_model=build_portfolio_model_spec("equal_weight"),
        execution_policy=build_execution_policy_spec(
            key="every_bar",
            label="毎バー",
            entry="train_once_then_periodic_rebalance",
            rebalance_schedule="every_bar",
        ),
        risk_controls=build_risk_controls_spec(max_investment_ratio=1.0),
        decision_schedule="every_bar",
    )

    assert resolve_decision_policy_kind(strategy) == COST_AWARE_NO_TRADE_DECISION_POLICY


def test_compute_trade_cost_increases_when_liquidity_is_lower() -> None:
    weight_delta = pd.Series([0.2, 0.2], index=["SPY", "BTC-USD"]).to_numpy(dtype="float64")
    linear_cost_rates = pd.Series([0.0003, 0.0025], index=["SPY", "BTC-USD"]).to_numpy(dtype="float64")
    impact_cost_rates = pd.Series([0.0002, 0.0025], index=["SPY", "BTC-USD"]).to_numpy(dtype="float64")
    price_snapshot = pd.Series({"SPY": 500.0, "BTC-USD": 60_000.0}, dtype="float64")

    high_liquidity_volumes = pd.DataFrame(
        {
            "SPY": [10_000_000.0, 11_000_000.0, 9_500_000.0],
            "BTC-USD": [3_000.0, 3_200.0, 3_100.0],
        }
    )
    low_liquidity_volumes = pd.DataFrame(
        {
            "SPY": [2_000_000.0, 2_100_000.0, 1_900_000.0],
            "BTC-USD": [300.0, 320.0, 310.0],
        }
    )

    high_liquidity_cost = compute_trade_cost(
        weight_delta=weight_delta,
        linear_cost_rates=linear_cost_rates,
        impact_cost_rates=impact_cost_rates,
        portfolio_equity=10_000.0,
        price_snapshot=price_snapshot,
        volume_history=high_liquidity_volumes,
        adv_window_bars=20,
        min_adv_notional=1_000_000.0,
    )
    low_liquidity_cost = compute_trade_cost(
        weight_delta=weight_delta,
        linear_cost_rates=linear_cost_rates,
        impact_cost_rates=impact_cost_rates,
        portfolio_equity=10_000.0,
        price_snapshot=price_snapshot,
        volume_history=low_liquidity_volumes,
        adv_window_bars=20,
        min_adv_notional=1_000_000.0,
    )

    assert low_liquidity_cost > high_liquidity_cost


def test_should_rebalance_supports_every_bar_schedule() -> None:
    assert should_rebalance("2025-01-01", "2025-01-02", "every_bar") is True
    assert should_rebalance("2025-01-01", "2025-01-01", "every_bar") is False


def test_should_rebalance_supports_week_end_schedule() -> None:
    assert should_rebalance("2025-01-03", "2025-01-06", "week_end") is True
    assert should_rebalance("2025-01-02", "2025-01-03", "week_end") is False
