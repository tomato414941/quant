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


def test_build_strategy_forecast_snapshot_wraps_ranking_score() -> None:
    returns = pd.DataFrame(
        {
            "AAA": [0.01, 0.02, 0.03, 0.01],
            "BBB": [0.01, -0.01, 0.0, 0.01],
            "CCC": [-0.01, -0.02, -0.01, 0.0],
        },
        index=pd.date_range("2025-01-01", periods=4, freq="D"),
    )
    strategy = build_evaluator_strategy_spec(
        investment_universe=build_investment_universe_spec(
            tickers=list(returns.columns),
            key="forecast_universe",
            label="Forecast universe",
        ),
        selection=build_selection_spec(
            "momentum_top3",
            score_parameters={"windowSpec": {"unit": "bars", "value": 3}},
        ),
        portfolio_model=build_portfolio_model_spec("equal_weight"),
        risk_controls=build_risk_controls_spec(max_investment_ratio=1.0),
    )

    forecast = build_strategy_forecast_snapshot(
        returns,
        None,
        strategy,
        bars_per_year=252.0,
        current_date="2025-01-04",
    )

    assert forecast is not None
    assert forecast.as_of_date == "2025-01-04"
    assert set(forecast.score.index) == set(returns.columns)
    assert forecast.expected_return_proxy is not None
    assert forecast.edge_source == SIGNAL_RETURN_PROXY_EDGE_SOURCE
    assert forecast.percentile_rank["AAA"] > forecast.percentile_rank["CCC"]
    assert forecast.confidence["AAA"] == 1.0


def test_mean_risk_utility_does_not_use_uncalibrated_signal_proxy_for_allocation() -> None:
    returns = pd.DataFrame(
        {
            "AAA": [0.01, 0.02, 0.01, 0.03, 0.02, 0.01],
            "BBB": [0.00, 0.01, 0.00, 0.01, 0.00, 0.01],
            "CCC": [-0.01, -0.02, -0.01, -0.02, -0.01, -0.02],
        },
        index=pd.date_range("2025-01-01", periods=6, freq="D"),
    )
    strategy = build_evaluator_strategy_spec(
        strategy_id="mean_risk_utility_proxy_guard_test",
        investment_universe=build_investment_universe_spec(
            tickers=list(returns.columns),
            key="mean_risk_utility_proxy_guard_universe",
            label="Mean risk utility proxy guard universe",
        ),
        selection=build_selection_spec(
            "momentum_top3",
            score_parameters={"windowSpec": {"unit": "bars", "value": 3}},
        ),
        portfolio_model=build_portfolio_model_spec("mean_risk_utility"),
        execution_policy=build_execution_policy_spec(
            key="every_bar",
            label="毎バー",
            entry="train_once_then_periodic_rebalance",
            rebalance_schedule="every_bar",
        ),
        risk_controls=build_risk_controls_spec(max_investment_ratio=1.0),
    )

    with patch("app.portfolio.compute_expected_return_proxy", side_effect=AssertionError("proxy used")):
        selected_assets, weights = compute_portfolio_allocation(
            history_returns=returns,
            volume_history=None,
            strategy=strategy,
            portfolio_model=build_portfolio_model_spec("mean_risk_utility"),
            bars_per_year=252.0,
            universe_columns=returns.columns,
            max_investment_ratio=1.0,
            max_weight=None,
            previous_weights=None,
            transaction_cost=0.001,
            current_date="2025-01-06",
            predictor_panel=None,
        )

    assert selected_assets
    assert np.isclose(weights.sum(), 1.0)


def test_prediction_supplement_changes_momentum_scores() -> None:
    closes = pd.DataFrame(
        {
            "SPY": [100, 101, 103, 102, 104, 106, 107, 108, 109],
            "QQQ": [100, 103, 105, 107, 108, 110, 112, 113, 115],
            "TLT": [100, 100, 99, 100, 101, 102, 101, 101, 102],
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
    volumes = pd.DataFrame(
        {
            "SPY": [1_000_000, 1_020_000, 1_010_000, 1_030_000, 1_040_000, 1_050_000, 1_045_000, 1_060_000, 1_070_000],
            "QQQ": [900_000, 940_000, 960_000, 970_000, 990_000, 1_000_000, 1_010_000, 1_030_000, 1_050_000],
            "TLT": [800_000, 805_000, 810_000, 815_000, 820_000, 825_000, 830_000, 835_000, 840_000],
        },
        index=closes.index,
    )
    returns = closes.pct_change().dropna()
    volume_history = volumes.loc[returns.index]

    baseline_strategy = make_strategy(
        "full_universe_momentum_tilt",
        "hierarchical_risk_parity",
        score_parameters={"tilt_strength": 0.35, "tilt_shape": 1.0, "windowSpec": {"unit": "bars", "value": 3}},
    )
    supplemented_strategy = make_strategy(
        "full_universe_momentum_tilt",
        "hierarchical_risk_parity",
        score_parameters={
            "tilt_strength": 0.35,
            "tilt_shape": 1.0,
            "windowSpec": {"unit": "bars", "value": 3},
        },
        predictor_use=build_predictor_use_spec(
            predictor_key="pred-test-supplement-2bar",
            signal_weight=0.8,
            predictor_weight=0.2,
        ),
    )
    predictor_panel = pd.DataFrame(
        {
            "SPY": [0.8],
            "QQQ": [0.1],
            "TLT": [-0.7],
        },
        index=[returns.index[-1]],
    )

    baseline_scores = compute_strategy_score_series(
        returns,
        volume_history,
        baseline_strategy,
        bars_per_year=252,
    )
    supplemented_scores = compute_strategy_score_series(
        returns,
        volume_history,
        supplemented_strategy,
        bars_per_year=252,
        current_date=str(returns.index[-1]),
        predictor_panel=predictor_panel,
    )

    assert baseline_scores is not None
    assert supplemented_scores is not None
    assert not baseline_scores.equals(supplemented_scores)


def test_compute_strategy_score_series_uses_explicit_predictor_context() -> None:
    closes = pd.DataFrame(
        {
            "SPY": [100, 101, 103, 102, 104, 106],
            "QQQ": [100, 103, 105, 107, 108, 110],
            "TLT": [100, 100, 99, 100, 101, 102],
        },
        index=pd.date_range("2025-01-01", periods=6, freq="D"),
    )
    returns = closes.pct_change().dropna()
    strategy = make_strategy(
        "full_universe_momentum_tilt",
        "hierarchical_risk_parity",
        score_parameters={
            "tilt_strength": 0.35,
            "tilt_shape": 1.0,
            "windowSpec": {"unit": "bars", "value": 3},
        },
        predictor_use=build_predictor_use_spec(
            predictor_key="pred-test-explicit-context-2bar",
            signal_weight=0.8,
            predictor_weight=0.2,
        ),
    )
    selection_contexts, predictor_context = get_strategy_signal_execution_contexts(strategy)
    predictor_panel = pd.DataFrame(
        {
            "SPY": [0.8],
            "QQQ": [0.1],
            "TLT": [-0.7],
        },
        index=[returns.index[-1]],
    )
    assert predictor_context is not None

    with patch(
        "app.portfolio.get_strategy_signal_execution_contexts",
        side_effect=AssertionError("explicit score contexts should avoid strategy re-extraction"),
    ):
        scores = compute_strategy_score_series(
            returns,
            None,
            strategy,
            bars_per_year=252,
            current_date=str(returns.index[-1]),
            predictor_panel=predictor_panel,
            selection_contexts=selection_contexts,
            predictor_context=predictor_context,
        )

    assert scores is not None
    assert scores["SPY"] > scores["TLT"]


def test_compute_strategy_score_series_uses_latest_predictor_snapshot_before_current_date() -> None:
    closes = pd.DataFrame(
        {
            "SPY": [100, 101, 103, 102, 104, 106],
            "QQQ": [100, 103, 105, 107, 108, 110],
            "TLT": [100, 100, 99, 100, 101, 102],
        },
        index=pd.date_range("2025-01-01", periods=6, freq="D"),
    )
    returns = closes.pct_change().dropna()
    strategy = make_strategy(
        "full_universe_momentum_tilt",
        "hierarchical_risk_parity",
        score_parameters={
            "tilt_strength": 0.35,
            "tilt_shape": 1.0,
            "windowSpec": {"unit": "bars", "value": 3},
        },
        predictor_use=build_predictor_use_spec(
            predictor_key="pred-test-asof-2bar",
            signal_weight=0.8,
            predictor_weight=0.2,
        ),
    )
    predictor_panel = pd.DataFrame(
        {
            "SPY": [0.8],
            "QQQ": [0.1],
            "TLT": [-0.7],
        },
        index=[returns.index[-2]],
    )

    exact_scores = compute_strategy_score_series(
        returns,
        None,
        strategy,
        bars_per_year=252,
        current_date=str(returns.index[-2]),
        predictor_panel=predictor_panel,
    )
    asof_scores = compute_strategy_score_series(
        returns,
        None,
        strategy,
        bars_per_year=252,
        current_date=str(returns.index[-1]),
        predictor_panel=predictor_panel,
    )

    assert exact_scores is not None
    assert asof_scores is not None
    pd.testing.assert_series_equal(asof_scores.sort_index(), exact_scores.sort_index())


def test_predictor_evaluation_includes_predictor_series() -> None:
    closes = pd.DataFrame(
        {
            "SPY": [100, 101, 103, 102, 104, 106, 107, 108, 109],
            "QQQ": [100, 103, 105, 107, 108, 110, 112, 113, 115],
            "TLT": [100, 100, 99, 100, 101, 102, 101, 101, 102],
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
    returns = closes.pct_change().dropna()
    strategy = make_strategy(
        "full_universe_momentum_tilt",
        "hierarchical_risk_parity",
        score_parameters={
            "tilt_strength": 0.35,
            "tilt_shape": 1.0,
            "windowSpec": {"unit": "bars", "value": 3},
        },
        predictor_use=build_predictor_use_spec(
            predictor_key="pred-test-eval-2bar",
            signal_weight=0.8,
            predictor_weight=0.2,
        ),
    )
    predictor_spec = make_predictor_spec(
        strategy,
        predictor_key="pred-test-eval-2bar",
        label="2bar補助予測",
        horizon_bars=2,
    )

    result = evaluate_predictor_spec(
        returns=returns,
        volumes=None,
        split_ratio=0.6,
        predictor_spec=predictor_spec,
        bars_per_year=252,
    )

    assert result["predictorSpec"]["kind"] == "predictor_spec"
    assert "predictorSeries" in result
    assert result["predictorSeries"]["assets"]


def test_strategy_run_can_reuse_predictor_panel() -> None:
    closes = pd.DataFrame(
        {
            "SPY": [100, 101, 103, 102, 104, 106, 107, 108, 109],
            "QQQ": [100, 103, 105, 107, 108, 110, 112, 113, 115],
            "TLT": [100, 100, 99, 100, 101, 102, 101, 101, 102],
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
    returns = closes.pct_change().dropna()
    volumes = pd.DataFrame(
        {
            "SPY": [1_000_000, 1_020_000, 1_010_000, 1_030_000, 1_040_000, 1_050_000, 1_045_000, 1_060_000, 1_070_000],
            "QQQ": [900_000, 940_000, 960_000, 970_000, 990_000, 1_000_000, 1_010_000, 1_030_000, 1_050_000],
            "TLT": [800_000, 805_000, 810_000, 815_000, 820_000, 825_000, 830_000, 835_000, 840_000],
        },
        index=closes.index,
    )
    strategy = make_strategy(
        "full_universe_momentum_tilt",
        "hierarchical_risk_parity",
        score_parameters={
            "tilt_strength": 0.35,
            "tilt_shape": 1.0,
            "windowSpec": {"unit": "bars", "value": 3},
        },
        predictor_use=build_predictor_use_spec(
            predictor_key="pred-test-reuse-2bar",
            signal_weight=0.8,
            predictor_weight=0.2,
        ),
    )
    predictor_spec = make_predictor_spec(
        strategy,
        predictor_key="pred-test-reuse-2bar",
        label="2bar補助予測",
        horizon_bars=2,
    )

    predictor_panel = compute_predictor_panel(
        returns=returns,
        volumes=volumes.loc[returns.index],
        predictor_spec=predictor_spec,
        bars_per_year=252,
    )

    inline_payload = compare_portfolio_runs(
        closes=closes,
        volumes=volumes,
        strategies=[strategy],
        initial_capital=10_000,
        split_ratio=0.6,
        execution_assumptions=make_execution_assumptions(),
        transaction_cost=0.001,
        portfolio_state=build_portfolio_state(current_weights={}, cash_weight=1.0),
    )
    reused_payload = compare_portfolio_runs(
        closes=closes,
        volumes=volumes,
        strategies=[strategy],
        initial_capital=10_000,
        split_ratio=0.6,
        execution_assumptions=make_execution_assumptions(),
        transaction_cost=0.001,
        portfolio_state=build_portfolio_state(current_weights={}, cash_weight=1.0),
        predictor_panels_by_strategy={strategy.key: predictor_panel},
    )

    assert reused_payload[0]["summary"] == inline_payload[0]["summary"]


def test_predictor_panel_supports_momentum_signal_source_blend() -> None:
    closes = pd.DataFrame(
        {
            "SPY": [100, 101, 103, 102, 104, 106, 107, 108, 109],
            "QQQ": [100, 103, 105, 107, 108, 110, 112, 113, 115],
            "TLT": [100, 100, 99, 100, 101, 102, 101, 101, 102],
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
    volumes = pd.DataFrame(
        {
            "SPY": [1_000_000, 1_020_000, 1_010_000, 1_030_000, 1_040_000, 1_050_000, 1_045_000, 1_060_000, 1_070_000],
            "QQQ": [900_000, 940_000, 960_000, 970_000, 990_000, 1_000_000, 1_010_000, 1_030_000, 1_050_000],
            "TLT": [800_000, 805_000, 810_000, 815_000, 820_000, 825_000, 830_000, 835_000, 840_000],
        },
        index=closes.index,
    )
    returns = closes.pct_change().dropna()
    strategy = make_strategy(
        "full_universe_momentum_tilt",
        "hierarchical_risk_parity",
        score_parameters={
            "tilt_strength": 0.35,
            "tilt_shape": 1.0,
            "windowSpec": {"unit": "bars", "value": 3},
        },
    )
    predictor_spec = make_predictor_spec(
        strategy,
        predictor_key="pred-test-momentum-blend-2bar",
        label="2barモメンタム補助予測",
        horizon_bars=2,
        signal_source_feature_key="momentum",
        combiner_kind="weighted_blend",
    )

    predictor_panel = compute_predictor_panel(
        returns=returns,
        volumes=volumes.loc[returns.index],
        predictor_spec=predictor_spec,
        bars_per_year=252,
    )

    assert predictor_panel.notna().any().any()
