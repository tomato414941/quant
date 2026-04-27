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
from app.portfolio_runs import compare_portfolio_runs as compare_portfolio_runs_with_dynamic_allocation
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


def test_compare_portfolio_runs_uses_explicit_signal_execution_contexts() -> None:
    closes = pd.DataFrame(
        {
            "AAA": [100, 102, 104, 103, 105, 107, 108, 110],
            "BBB": [100, 101, 102, 103, 104, 105, 106, 107],
            "CCC": [100, 99, 101, 100, 102, 101, 103, 104],
        },
        index=pd.date_range("2024-01-05", periods=8, freq="W-FRI"),
    )
    strategy = build_evaluator_strategy_spec(
        strategy_id="explicit_signal_execution_contexts",
        timeframe=DEFAULT_WEEKLY_TIMEFRAME,
        investment_universe=build_investment_universe_spec(
            tickers=["AAA", "BBB", "CCC"],
            key="explicit_signal_execution_contexts_universe",
            label="Explicit signal execution contexts universe",
        ),
        selection=build_selection_spec("full_universe"),
        portfolio_model=build_portfolio_model_spec("equal_weight"),
        execution_policy=build_execution_policy_spec(
            key="month_end",
            label="月次",
            entry="train_once_then_periodic_rebalance",
            rebalance_schedule="month_end",
        ),
        risk_controls=build_risk_controls_spec(max_investment_ratio=1.0),
    )
    signal_execution_contexts = get_strategy_definition_signal_execution_contexts(strategy)

    with patch(
        "app.portfolio.get_strategy_signal_execution_contexts",
        side_effect=AssertionError("explicit contexts should avoid strategy re-extraction"),
    ):
        runs = compare_portfolio_runs(
            closes=closes,
            volumes=None,
            strategies=[strategy],
            initial_capital=10000,
            split_ratio=0.6,
            transaction_cost=0.001,
            bars_per_year=DEFAULT_WEEKLY_TIMEFRAME.bars_per_year,
            strategy_signal_execution_contexts_by_key={strategy.key: signal_execution_contexts},
        )

    assert len(runs) == 1
    assert runs[0]["key"] == strategy.key


def test_compare_portfolio_runs_does_not_apply_test_initial_weights_to_train() -> None:
    closes = pd.DataFrame(
        {
            "AAA": [100, 110, 120, 130, 140, 150, 160],
            "BBB": [100, 100, 100, 100, 100, 100, 100],
        },
        index=pd.date_range("2025-01-01", periods=7, freq="D"),
    )
    strategy = build_evaluator_strategy_spec(
        investment_universe=build_investment_universe_spec(
            tickers=list(closes.columns),
            key="no_lookahead_train_universe",
            label="No lookahead train universe",
        ),
        selection=build_selection_spec("full_universe"),
        portfolio_model=build_portfolio_model_spec("equal_weight"),
        risk_controls=build_risk_controls_spec(max_investment_ratio=1.0),
    )

    run = compare_portfolio_runs(
        closes=closes,
        volumes=None,
        strategies=[strategy],
        initial_capital=1000.0,
        split_ratio=0.5,
        transaction_cost=0.0,
        portfolio_state=build_portfolio_state(current_weights={}, cash_weight=1.0),
    )[0]

    assert run["splitAnalysis"]["train"]["portfolio"]["totalReturnPct"] == 0.0
    assert run["summary"] == run["splitAnalysis"]["test"]["portfolio"]


def test_compare_portfolio_runs_adds_allocation_fallback_only_to_decision_trace() -> None:
    closes = pd.DataFrame(
        {
            "AAA": [100, 101, 102, 103, 104, 105, 106],
            "BBB": [100, 100, 101, 101, 102, 102, 103],
        },
        index=pd.date_range("2025-01-01", periods=7, freq="D"),
    )
    strategy = build_evaluator_strategy_spec(
        investment_universe=build_investment_universe_spec(
            tickers=list(closes.columns),
            key="allocation_fallback_trace_universe",
            label="Allocation fallback trace universe",
        ),
        selection=build_selection_spec("full_universe"),
        portfolio_model=build_portfolio_model_spec("equal_weight"),
        execution_policy=build_execution_policy_spec(
            key="every_bar",
            label="Every bar",
            entry="train_once_then_periodic_rebalance",
            rebalance_schedule="every_bar",
        ),
        risk_controls=build_risk_controls_spec(max_investment_ratio=1.0),
        extensions={"decision_policy": "direct_score_to_weight"},
    )
    fallback_metadata = {
        "modelType": "risk_budgeting",
        "reason": "optimizer_exception",
        "fallback": "equal_weight",
        "exceptionType": "RuntimeError",
        "message": "boom",
    }

    def allocate_without_metadata(**kwargs):
        return ["AAA", "BBB"], np.asarray([0.5, 0.5], dtype="float64")

    def allocate_with_metadata(**kwargs):
        return ["AAA", "BBB"], np.asarray([0.5, 0.5], dtype="float64"), fallback_metadata

    run_without_metadata = compare_portfolio_runs_with_dynamic_allocation(
        closes=closes,
        volumes=None,
        strategies=[strategy],
        initial_capital=1000.0,
        split_ratio=0.5,
        transaction_cost=0.0,
        dynamic_allocation_fn=allocate_without_metadata,
    )[0]
    run_with_metadata = compare_portfolio_runs_with_dynamic_allocation(
        closes=closes,
        volumes=None,
        strategies=[strategy],
        initial_capital=1000.0,
        split_ratio=0.5,
        transaction_cost=0.0,
        dynamic_allocation_fn=allocate_with_metadata,
    )[0]

    assert run_with_metadata["weights"] == run_without_metadata["weights"]
    assert run_with_metadata["selectedAssets"] == run_without_metadata["selectedAssets"]
    assert run_with_metadata["summary"] == run_without_metadata["summary"]
    assert run_with_metadata["series"] == run_without_metadata["series"]
    decision_events = [
        event for event in run_with_metadata["executionTrace"] if event["eventType"] == "decision"
    ]
    assert decision_events
    assert all(event["allocationFallback"] == fallback_metadata for event in decision_events)
    assert run_with_metadata["allocationFallbackCount"] == len(decision_events)
    assert run_with_metadata["allocationFallbackRate"] == 1.0
    assert run_with_metadata["diagnosticEventCount"] == 0
    assert run_with_metadata["availabilityWarningCount"] == 0
    assert run_without_metadata["allocationFallbackCount"] == 0
    assert run_without_metadata["allocationFallbackRate"] == 0.0
    assert all("allocationFallback" not in event for event in run_with_metadata["decisionEvents"])
    assert all(
        "allocationFallback" not in event
        for event in run_without_metadata["executionTrace"]
        if event["eventType"] == "decision"
    )


def test_compare_portfolio_runs_returns_strategy_combinations() -> None:
    closes = pd.DataFrame(
        {
            "SPY": [100, 101, 103, 102, 104, 106, 107],
            "QQQ": [100, 103, 105, 107, 108, 110, 112],
            "IWM": [100, 99, 100, 101, 103, 102, 104],
            "EFA": [100, 101, 102, 103, 104, 105, 106],
            "EEM": [100, 98, 99, 100, 101, 102, 103],
            "TLT": [100, 100, 99, 100, 101, 102, 101],
            "IEF": [100, 100, 100, 100, 101, 101, 102],
            "LQD": [100, 101, 101, 102, 103, 103, 104],
            "HYG": [100, 101, 102, 103, 104, 104, 105],
            "GLD": [100, 101, 100, 102, 103, 104, 105],
            "BTC-USD": [100, 105, 107, 110, 112, 115, 118],
            "ETH-USD": [100, 106, 108, 112, 115, 119, 122],
        },
        index=[
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
            "2025-01-04",
            "2025-01-05",
            "2025-01-06",
            "2025-01-07",
        ],
    )

    payload = compare_portfolio_runs(
        closes=closes,
        volumes=None,
        strategies=[
            make_strategy("full_universe", "equal_weight", max_investment_ratio=0.8),
            make_strategy("full_universe", "risk_budgeting", max_investment_ratio=0.8),
            make_strategy("full_universe", "minimum_variance", max_investment_ratio=0.8),
            make_strategy(
                "full_universe",
                "hierarchical_risk_parity",
                max_investment_ratio=0.8,
            ),
            make_strategy("momentum_top3", "equal_weight", max_investment_ratio=0.8),
            make_strategy("momentum_top3", "risk_budgeting", max_investment_ratio=0.8),
            make_strategy(
                "momentum_top3",
                "minimum_variance",
                max_investment_ratio=0.8,
            ),
            make_strategy(
                "momentum_top3",
                "hierarchical_risk_parity",
                max_investment_ratio=0.8,
            ),
        ],
        initial_capital=10_000,
        split_ratio=0.6,
        execution_assumptions=make_execution_assumptions(),
        transaction_cost=0.001,
        portfolio_state=build_portfolio_state(
            current_weights={
                "SPY": 0.1,
                "QQQ": 0.1,
                "IWM": 0.1,
                "EFA": 0.1,
                "EEM": 0.1,
                "TLT": 0.1,
                "IEF": 0.1,
                "LQD": 0.1,
            },
            cash_weight=0.2,
        ),
    )

    assert [row["key"] for row in payload] == [
        "full_universe__equal_weight",
        "full_universe__risk_budgeting",
        "full_universe__minimum_variance",
        "full_universe__hierarchical_risk_parity",
        "momentum_top3__equal_weight",
        "momentum_top3__risk_budgeting",
        "momentum_top3__minimum_variance",
        "momentum_top3__hierarchical_risk_parity",
    ]
    assert any(row["asset"] == "CASH" and row["weightPct"] == 20.0 for row in payload[0]["weights"])
    assert payload[0]["summary"] == payload[0]["splitAnalysis"]["test"]["portfolio"]
    assert payload[0]["summary"]["turnoverPct"] > 0.0
    assert len(payload[0]["series"]) == 6


def test_dual_momentum_can_fall_back_to_cash() -> None:
    closes = pd.DataFrame(
        {
            "SPY": [100, 99, 98, 97, 96, 95, 94],
            "QQQ": [100, 99, 98, 97, 96, 95, 94],
            "TLT": [100, 99, 98, 97, 96, 95, 94],
        },
        index=[
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
            "2025-01-04",
            "2025-01-05",
            "2025-01-06",
            "2025-01-07",
        ],
    )

    payload = compare_portfolio_runs(
        closes=closes,
        volumes=None,
        strategies=[
            make_strategy(
                "dual_momentum_top3",
                "hierarchical_risk_parity",
                max_investment_ratio=0.85,
            )
        ],
        initial_capital=10_000,
        split_ratio=0.6,
        execution_assumptions=make_execution_assumptions(),
        transaction_cost=0.001,
        portfolio_state=build_portfolio_state(current_weights={}, cash_weight=1.0),
    )

    assert payload[0]["selectedAssets"] == []
    assert payload[0]["summary"]["totalReturnPct"] == 0.0
    assert payload[0]["weights"][0]["asset"] == "CASH"
    assert payload[0]["weights"][0]["weightPct"] == 100.0


def test_compare_portfolio_runs_respects_max_weight_cap() -> None:
    closes = pd.DataFrame(
        {
            "SPY": [100, 101, 102, 103, 104, 105, 106],
            "QQQ": [100, 102, 103, 105, 106, 108, 109],
            "TLT": [100, 100, 101, 101, 102, 102, 103],
            "GLD": [100, 101, 101, 102, 103, 103, 104],
        },
        index=[
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
            "2025-01-04",
            "2025-01-05",
            "2025-01-06",
            "2025-01-07",
        ],
    )

    payload = compare_portfolio_runs(
        closes=closes,
        volumes=None,
        strategies=[
            make_strategy(
                "full_universe",
                "equal_weight",
                max_weight=0.2,
            )
        ],
        initial_capital=10_000,
        split_ratio=0.6,
        execution_assumptions=make_execution_assumptions(),
        transaction_cost=0.001,
        portfolio_state=build_portfolio_state(current_weights={}, cash_weight=1.0),
    )

    asset_weights = [row["weightPct"] for row in payload[0]["weights"] if row["asset"] != "CASH"]
    cash_rows = [row for row in payload[0]["weights"] if row["asset"] == "CASH"]

    assert max(asset_weights) == 20.0
    assert cash_rows[0]["weightPct"] == 20.0


def test_compare_portfolio_runs_supports_single_asset_benchmark_with_cash_reserve() -> None:
    closes = pd.DataFrame(
        {"SPY": [100, 101, 102, 103, 104, 105, 106]},
        index=pd.date_range("2025-01-01", periods=7, freq="D"),
    )
    strategy = build_evaluator_strategy_spec(
        strategy_id="spy_cash_benchmark",
        investment_universe=build_investment_universe_spec(
            tickers=["SPY"],
            key="spy_only",
            label="SPY only",
        ),
        selection=build_selection_spec("full_universe"),
        portfolio_model=build_portfolio_model_spec("equal_weight"),
        risk_controls=build_risk_controls_spec(max_investment_ratio=0.85),
    )

    payload = compare_portfolio_runs(
        closes=closes,
        volumes=None,
        strategies=[strategy],
        initial_capital=10_000,
        split_ratio=0.6,
        execution_assumptions=make_execution_assumptions(),
        transaction_cost=0.001,
        portfolio_state=build_portfolio_state(current_weights={}, cash_weight=1.0),
    )

    assert payload[0]["selectedAssets"] == ["SPY"]
    assert payload[0]["weights"] == [
        {"asset": "SPY", "weightPct": 85.0},
        {"asset": "CASH", "weightPct": 15.0},
    ]


def test_compare_portfolio_runs_supports_asset_specific_linear_cost() -> None:
    closes = pd.DataFrame(
        {
            "SPY": [100, 101, 102, 103, 104, 105, 106],
            "QQQ": [100, 101, 102, 103, 104, 105, 106],
        },
        index=[
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
            "2025-01-04",
            "2025-01-05",
            "2025-01-06",
            "2025-01-07",
        ],
    )

    strategy = build_evaluator_strategy_spec(
        investment_universe=build_investment_universe_spec(
            tickers=["SPY", "QQQ"],
            key="test_universe_small",
            label="Test universe small",
        ),
        selection=build_selection_spec("full_universe"),
        portfolio_model=build_portfolio_model_spec("equal_weight"),
        risk_controls=build_risk_controls_spec(max_investment_ratio=1.0),
    )

    flat_payload = compare_portfolio_runs(
        closes=closes,
        volumes=None,
        strategies=[strategy],
        initial_capital=10_000,
        split_ratio=0.6,
        execution_assumptions={
            "kind": "close_execution_assumptions",
            "label": "終値約定",
            "parameters": {"entry": "hold", "rebalanceSchedule": "hold"},
            "costModel": {
                "kind": "flat_cost",
                "parameters": {"commissionPct": 0.1, "slippagePct": 0.0},
                "perAssetOverrides": {},
            },
        },
        cost_model={
            "kind": "flat_cost",
            "parameters": {"commissionPct": 0.1, "slippagePct": 0.0},
            "perAssetOverrides": {},
        },
        portfolio_state=build_portfolio_state(current_weights={}, cash_weight=1.0),
    )
    asset_specific_payload = compare_portfolio_runs(
        closes=closes,
        volumes=None,
        strategies=[strategy],
        initial_capital=10_000,
        split_ratio=0.6,
        execution_assumptions={
            "kind": "close_execution_assumptions",
            "label": "終値約定",
            "parameters": {"entry": "hold", "rebalanceSchedule": "hold"},
            "costModel": {
                "kind": "asset_specific_linear_cost",
                "parameters": {"commissionPct": 0.1, "slippagePct": 0.0},
                "perAssetOverrides": {
                    "SPY": {"commissionPct": 0.5, "slippagePct": 0.0},
                },
            },
        },
        cost_model={
            "kind": "asset_specific_linear_cost",
            "parameters": {"commissionPct": 0.1, "slippagePct": 0.0},
            "perAssetOverrides": {
                "SPY": {"commissionPct": 0.5, "slippagePct": 0.0},
            },
        },
        portfolio_state=build_portfolio_state(current_weights={}, cash_weight=1.0),
    )

    assert (
        asset_specific_payload[0]["summary"]["totalReturnPct"]
        < flat_payload[0]["summary"]["totalReturnPct"]
    )


def test_compare_portfolio_runs_tracks_dynamic_asset_eligibility() -> None:
    closes = pd.DataFrame(
        {
            "AAA": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            "BBB": [100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0, 103.5, 104.0],
            "LATE": [None, None, None, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0],
        },
        index=[
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
            "2025-01-04",
            "2025-01-05",
            "2025-01-06",
            "2025-01-07",
            "2025-01-08",
            "2025-01-09",
        ],
    )
    strategy = build_evaluator_strategy_spec(
        strategy_id="dynamic_asset_eligibility",
        investment_universe=build_investment_universe_spec(
            tickers=list(closes.columns),
            key="dynamic_asset_eligibility_universe",
            label="Dynamic asset eligibility universe",
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
    )

    run = compare_portfolio_runs(
        closes=closes,
        volumes=None,
        strategies=[strategy],
        initial_capital=1000.0,
        split_ratio=0.5,
        transaction_cost=0.0,
        bars_per_year=252.0,
        availability_policy={
            "kind": "asset_availability_policy",
            "minHistoryBars": 2,
            "maxStaleBars": 5,
            "delistedAssetPolicy": "liquidate_to_cash",
        },
    )[0]

    assert run["series"][0]["date"] == "2025-01-02"
    assert min(point["eligibleAssetCount"] for point in run["series"]) == 2
    assert max(point["eligibleAssetCount"] for point in run["series"]) == 3
    assert any("LATE" in point["newlyEligibleAssets"] for point in run["series"])
    assert run["availabilitySummary"]["minEligibleAssetCount"] == 2
    assert run["availabilitySummary"]["maxEligibleAssetCount"] == 3


def test_compare_portfolio_runs_traces_forced_universe_change() -> None:
    closes = pd.DataFrame(
        {
            "AAA": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
            "GONE": [100.0, 101.0, 102.0, 103.0, None, None, None, None],
        },
        index=pd.date_range("2025-01-01", periods=8, freq="D"),
    )
    strategy = build_evaluator_strategy_spec(
        strategy_id="forced_universe_change_trace",
        investment_universe=build_investment_universe_spec(
            tickers=list(closes.columns),
            key="forced_universe_change_universe",
            label="Forced universe change universe",
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
    )

    run = compare_portfolio_runs(
        closes=closes,
        volumes=None,
        strategies=[strategy],
        initial_capital=1000.0,
        split_ratio=0.5,
        transaction_cost=0.0,
        bars_per_year=252.0,
        portfolio_state=build_portfolio_state(current_weights={"GONE": 1.0}, cash_weight=0.0),
        availability_policy={
            "kind": "asset_availability_policy",
            "minHistoryBars": 1,
            "maxStaleBars": 1,
            "delistedAssetPolicy": "liquidate_to_cash",
        },
    )[0]

    forced_events = [event for event in run["executionTrace"] if event["eventType"] == "forced_universe_change"]
    assert forced_events
    assert forced_events[0]["decisionReason"] == "asset_unavailable"
    assert forced_events[0]["decisionAction"] == "forced_rebalance"
    assert any(row["asset"] == "GONE" and row["weightPct"] == 0.0 for row in forced_events[0]["executedWeights"])
