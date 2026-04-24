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


def test_resample_market_frame_to_timeframe_supports_alignment_methods() -> None:
    frame = pd.DataFrame(
        {"SPY": [100.0, 101.0, 102.0, 103.0]},
        index=pd.to_datetime(["2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09"]),
    )

    end_of_period = resample_market_frame_to_timeframe(
        frame,
        target_timeframe_key="1w",
        value_kind="close",
        alignment_method="end_of_period",
    )
    calendar_resample = resample_market_frame_to_timeframe(
        frame,
        target_timeframe_key="1w",
        value_kind="close",
        alignment_method="calendar_resample",
    )
    asof_last = resample_market_frame_to_timeframe(
        frame,
        target_timeframe_key="1w",
        value_kind="close",
        alignment_method="asof_last",
    )

    assert str(end_of_period.index[-1].date()) == "2025-01-09"
    assert str(calendar_resample.index[-1].date()) == "2025-01-10"
    assert str(asof_last.index[-1].date()) == "2025-01-10"
    assert float(end_of_period.iloc[-1, 0]) == 103.0
    assert float(calendar_resample.iloc[-1, 0]) == 103.0


def test_prepare_signal_component_data_uses_alignment_policy_for_signal_returns() -> None:
    closes = pd.DataFrame(
        {"SPY": [100.0, 101.0, 102.0, 103.0, 104.0]},
        index=pd.to_datetime(["2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09", "2025-01-13"]),
    )
    returns = closes.pct_change().dropna()
    volumes = pd.DataFrame(
        {"SPY": [10.0, 11.0, 12.0, 13.0]},
        index=returns.index,
    )

    end_of_period_returns, end_of_period_volumes, end_of_period_bars = prepare_signal_component_data(
        history_returns=returns,
        volume_history=volumes,
        data_timeframe_key="1d",
        signal_timeframe_key="1w",
        alignment_policy={"method": "end_of_period"},
    )
    calendar_returns, calendar_volumes, calendar_bars = prepare_signal_component_data(
        history_returns=returns,
        volume_history=volumes,
        data_timeframe_key="1d",
        signal_timeframe_key="1w",
        alignment_policy={"method": "calendar_resample"},
    )

    assert end_of_period_bars == DEFAULT_WEEKLY_TIMEFRAME.bars_per_year
    assert calendar_bars == DEFAULT_WEEKLY_TIMEFRAME.bars_per_year
    assert str(end_of_period_returns.index[-1].date()) == "2025-01-13"
    assert str(calendar_returns.index[-1].date()) == "2025-01-17"
    assert str(end_of_period_volumes.index[0].date()) == "2025-01-13"
    assert str(calendar_volumes.index[0].date()) == "2025-01-17"


def test_prepare_strategy_market_data_resamples_daily_source_to_weekly_signal_timeframe() -> None:
    closes = pd.DataFrame(
        {
            "AAA": [100, 101, 102, 103, 104, 105, 106],
            "BBB": [100, 100, 101, 101, 102, 103, 103],
        },
        index=pd.date_range("2025-01-01", periods=7, freq="D"),
    )
    volumes = pd.DataFrame(
        {
            "AAA": [10, 11, 12, 13, 14, 15, 16],
            "BBB": [20, 21, 22, 23, 24, 25, 26],
        },
        index=closes.index,
    )
    strategy = build_evaluator_strategy_spec(
        strategy_id="weekly_signal_from_daily_source",
        timeframe=DEFAULT_WEEKLY_TIMEFRAME,
        investment_universe=build_investment_universe_spec(
            tickers=["AAA", "BBB"],
            key="weekly_signal_source_universe",
            label="Weekly signal source universe",
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
        signal_execution_contexts=[
            {
                "selectionKey": "full_universe",
                "strategyType": "full_universe",
                "label": "full_universe",
                "description": "full_universe",
                "scoreParameters": {},
                "weight": 1.0,
                "dataTimeframe": "1d",
                "signalTimeframe": "1w",
                "alignmentPolicy": None,
            }
        ],
    )

    prepared_closes, prepared_volumes = prepare_strategy_market_data(
        closes=closes,
        volumes=volumes,
        strategy=strategy,
    )

    assert list(prepared_closes.index) == [pd.Timestamp("2025-01-03"), pd.Timestamp("2025-01-07")]
    assert prepared_closes.loc[pd.Timestamp("2025-01-03"), "AAA"] == 102
    assert prepared_closes.loc[pd.Timestamp("2025-01-07"), "AAA"] == 106
    assert prepared_volumes.loc[pd.Timestamp("2025-01-03"), "AAA"] == 33
    assert prepared_volumes.loc[pd.Timestamp("2025-01-07"), "AAA"] == 58


def test_compare_portfolio_runs_keeps_daily_performance_with_weekly_signal_timeframe() -> None:
    closes = pd.DataFrame(
        {
            "AAA": [100 + idx for idx in range(24)],
            "BBB": [100 + (idx // 2) for idx in range(24)],
            "CCC": [100 - 4 + idx for idx in range(24)],
        },
        index=pd.date_range("2025-01-01", periods=24, freq="D"),
    )
    volumes = pd.DataFrame(
        {ticker: [1_000_000 + idx * 10_000 for idx in range(len(closes))] for ticker in closes.columns},
        index=closes.index,
    )
    weekly_signal_strategy = build_evaluator_strategy_spec(
        strategy_id="daily_performance_weekly_signal",
        timeframe=DEFAULT_WEEKLY_TIMEFRAME,
        investment_universe=build_investment_universe_spec(
            tickers=list(closes.columns),
            key="daily_performance_weekly_signal_universe",
            label="Daily performance weekly signal universe",
        ),
        selection=build_selection_spec(
            "momentum_top3",
            score_parameters={"windowSpec": {"unit": "weeks", "value": 1}},
        ),
        portfolio_model=build_portfolio_model_spec("equal_weight"),
        execution_policy=build_execution_policy_spec(
            key="month_end",
            label="月次",
            entry="train_once_then_periodic_rebalance",
            rebalance_schedule="month_end",
        ),
        risk_controls=build_risk_controls_spec(max_investment_ratio=1.0),
        signal_execution_contexts=[
            {
                "selectionKey": "momentum_top3",
                "strategyType": "momentum_top3",
                "label": "momentum_top3",
                "description": "momentum_top3",
                "scoreParameters": {"windowSpec": {"unit": "weeks", "value": 1}},
                "weight": 1.0,
                "dataTimeframe": "1d",
                "signalTimeframe": "1w",
                "alignmentPolicy": None,
            }
        ],
        decision_schedule="every_bar",
    )

    run = compare_portfolio_runs(
        closes=closes,
        volumes=volumes,
        strategies=[weekly_signal_strategy],
        initial_capital=1000.0,
        split_ratio=0.5,
        transaction_cost=0.001,
        bars_per_year=252.0,
    )[0]

    assert len(run["series"]) == len(closes.pct_change().dropna())
    assert run["splitAnalysis"]["test"]["barCount"] > 0
    assert run["strategy"]["components"]["core"]["dataResolution"]["key"] == "1w"


def test_positive_momentum_universe_selects_only_positive_assets() -> None:
    returns = pd.DataFrame(
        {
            "SPY": [0.01, 0.01, 0.01],
            "QQQ": [0.02, 0.02, 0.02],
            "TLT": [-0.01, -0.01, -0.01],
        }
    )
    selection = build_selection_spec(
        "positive_momentum_universe",
        score_parameters={"windowSpec": {"unit": "bars", "value": 3}},
    )

    selected_assets = select_assets(
        returns,
        None,
        selection,
        bars_per_year=252,
    )

    assert selected_assets == ["QQQ", "SPY"]


def test_positive_momentum_universe_can_fall_back_to_cash() -> None:
    returns = pd.DataFrame(
        {
            "SPY": [-0.01, -0.01, -0.01],
            "QQQ": [-0.02, -0.02, -0.02],
            "TLT": [-0.005, -0.005, -0.005],
        }
    )
    selection = build_selection_spec(
        "positive_momentum_universe",
        score_parameters={"windowSpec": {"unit": "bars", "value": 3}},
    )

    selected_assets = select_assets(
        returns,
        None,
        selection,
        bars_per_year=252,
    )

    assert selected_assets == []


def test_positive_trend_short_reversal_prefers_pullbacks_in_positive_trends() -> None:
    returns = pd.DataFrame(
        {
            "SPY": [0.02, 0.02, 0.02, 0.02, -0.02, -0.02],
            "QQQ": [0.01, 0.01, 0.01, 0.01, -0.01, -0.01],
            "TLT": [0.005, 0.005, 0.005, 0.005, 0.005, 0.005],
            "GLD": [-0.01, -0.01, -0.01, -0.01, -0.01, -0.01],
        }
    )
    selection = build_selection_spec(
        "positive_trend_short_reversal",
        score_parameters={
            "trendWindowSpec": {"unit": "bars", "value": 6},
            "reversalWindowSpec": {"unit": "bars", "value": 2},
            "assetCount": 2,
        },
    )

    selected_assets = select_assets(
        returns,
        None,
        selection,
        bars_per_year=252,
    )

    assert selected_assets == ["SPY", "QQQ"]


def test_risk_regime_positive_momentum_limits_to_defensive_assets_when_risk_proxy_is_negative() -> None:
    returns = pd.DataFrame(
        {
            "SPY": [0.01, 0.01, -0.03, -0.03, -0.03],
            "QQQ": [0.01, 0.01, -0.04, -0.04, -0.04],
            "TLT": [0.01, 0.01, 0.01, 0.01, 0.01],
            "GLD": [0.005, 0.005, 0.005, 0.005, 0.005],
            "UUP": [0.003, 0.003, 0.003, 0.003, 0.003],
            "BTC-USD": [0.04, 0.04, 0.04, 0.04, 0.04],
        }
    )
    selection = build_selection_spec(
        "risk_regime_positive_momentum",
        score_parameters={
            "windowSpec": {"unit": "bars", "value": 5},
            "riskWindowSpec": {"unit": "bars", "value": 3},
            "riskProxyAssets": ("SPY", "QQQ"),
            "defensiveAssetClasses": ("bond_etf", "commodity_etf", "currency_etf"),
        },
    )

    selected_assets = select_assets(
        returns,
        None,
        selection,
        bars_per_year=252,
    )

    assert selected_assets == ["TLT", "GLD", "UUP"]


def test_trailing_momentum_low_vol_strategy_prefers_recent_winners() -> None:
    closes = pd.DataFrame(
        {
            "SPY": [100, 102, 104, 106, 108, 110, 112],
            "QQQ": [100, 101, 102, 103, 104, 105, 106],
            "TLT": [100, 99, 98, 97, 96, 95, 94],
            "GLD": [100, 100.5, 101, 101.5, 102, 102.5, 103],
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
        strategies=[make_strategy("trailing_momentum_low_vol_universe", "hierarchical_risk_parity")],
        initial_capital=10_000,
        split_ratio=0.6,
        execution_assumptions=make_execution_assumptions(),
        transaction_cost=0.001,
        portfolio_state=build_portfolio_state(current_weights={}, cash_weight=1.0),
    )

    assert sorted(payload[0]["selectedAssets"]) == ["GLD", "QQQ"]


def test_full_universe_momentum_tilt_overweights_stronger_assets() -> None:
    closes = pd.DataFrame(
        {
            "SPY": [100, 103, 106, 109, 112, 115, 118],
            "QQQ": [100, 102, 104, 106, 108, 110, 112],
            "TLT": [100, 99.5, 99, 98.5, 98, 97.5, 97],
            "GLD": [100, 100.2, 100.4, 100.6, 100.8, 101.0, 101.2],
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

    baseline_payload = compare_portfolio_runs(
        closes=closes,
        volumes=None,
        strategies=[make_strategy("full_universe", "hierarchical_risk_parity")],
        initial_capital=10_000,
        split_ratio=0.6,
        execution_assumptions=make_execution_assumptions(),
        transaction_cost=0.001,
        portfolio_state=build_portfolio_state(current_weights={}, cash_weight=1.0),
    )
    tilted_payload = compare_portfolio_runs(
        closes=closes,
        volumes=None,
        strategies=[make_strategy("full_universe_momentum_tilt", "hierarchical_risk_parity")],
        initial_capital=10_000,
        split_ratio=0.6,
        execution_assumptions=make_execution_assumptions(),
        transaction_cost=0.001,
        portfolio_state=build_portfolio_state(current_weights={}, cash_weight=1.0),
    )

    baseline_weight_map = {
        row["asset"]: row["weightPct"] for row in baseline_payload[0]["weights"] if row["asset"] != "CASH"
    }
    tilted_weight_map = {
        row["asset"]: row["weightPct"] for row in tilted_payload[0]["weights"] if row["asset"] != "CASH"
    }
    assert tilted_weight_map["TLT"] < baseline_weight_map["TLT"]


def test_momentum_window_spec_changes_ranking_scores() -> None:
    closes = pd.DataFrame(
        {
            "SPY": [100, 104, 108, 112, 116, 120, 124],
            "QQQ": [100, 90, 85, 84, 95, 105, 115],
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
    returns = closes.pct_change().dropna()
    short_window_strategy = build_selection_spec(
        "full_universe_momentum_tilt",
        score_parameters={"tilt_strength": 0.35, "tilt_shape": 1.0, "windowSpec": {"unit": "bars", "value": 2}},
    )
    long_window_strategy = build_selection_spec(
        "full_universe_momentum_tilt",
        score_parameters={"tilt_strength": 0.35, "tilt_shape": 1.0, "windowSpec": {"unit": "bars", "value": 6}},
    )

    short_scores = compute_strategy_score_series(
        returns,
        None,
        short_window_strategy,
        bars_per_year=252,
    )
    long_scores = compute_strategy_score_series(
        returns,
        None,
        long_window_strategy,
        bars_per_year=252,
    )

    assert short_scores is not None
    assert long_scores is not None
    assert short_scores["QQQ"] > short_scores["SPY"]
    assert long_scores["SPY"] > long_scores["QQQ"]


def test_select_assets_uses_explicit_selection_contexts_for_momentum_top3() -> None:
    closes = pd.DataFrame(
        {
            "AAA": [100, 101, 102, 103, 104, 105, 106, 107],
            "BBB": [100, 102, 103, 104, 105, 106, 107, 108],
            "CCC": [100, 99, 98, 100, 103, 107, 112, 118],
            "DDD": [100, 101, 101, 102, 102, 103, 103, 104],
        },
        index=pd.date_range("2025-01-01", periods=8, freq="D"),
    )
    returns = closes.pct_change().dropna()
    strategy = build_evaluator_strategy_spec(
        strategy_id="explicit_selection_assets_top3",
        investment_universe=build_investment_universe_spec(
            tickers=list(closes.columns),
            key="explicit_selection_assets_top3_universe",
            label="Explicit selection assets top3 universe",
        ),
        selection=build_selection_spec(
            "momentum_top3",
            score_parameters={"windowSpec": {"unit": "bars", "value": 2}},
        ),
        portfolio_model=build_portfolio_model_spec("equal_weight"),
        execution_policy=build_execution_policy_spec(
            key="every_bar",
            label="毎バー",
            entry="train_once_then_periodic_rebalance",
            rebalance_schedule="every_bar",
        ),
        risk_controls=build_risk_controls_spec(max_investment_ratio=1.0),
        signal_execution_contexts=[
            {
                "selectionKey": "momentum_top3",
                "strategyType": "momentum_top3",
                "label": "momentum_top3",
                "description": "momentum_top3",
                "scoreParameters": {"windowSpec": {"unit": "bars", "value": 2}},
                "weight": 0.4,
                "dataTimeframe": "1d",
                "signalTimeframe": "1d",
                "alignmentPolicy": None,
            },
            {
                "selectionKey": "secondary_momo4",
                "strategyType": "momentum_top3",
                "label": "Secondary momentum",
                "description": "Secondary momentum",
                "scoreParameters": {"windowSpec": {"unit": "bars", "value": 4}},
                "weight": 0.6,
                "dataTimeframe": "1d",
                "signalTimeframe": "1d",
                "alignmentPolicy": None,
            },
        ],
    )
    selection_contexts, _ = get_strategy_signal_execution_contexts(strategy)

    with patch(
        "app.portfolio.get_strategy_signal_execution_contexts",
        side_effect=AssertionError("explicit selection contexts should avoid strategy re-extraction"),
    ):
        selected_assets = select_assets(
            returns,
            None,
            strategy,
            bars_per_year=252.0,
            selection_contexts=selection_contexts,
        )

    assert "CCC" in selected_assets


def test_compare_portfolio_runs_blends_additional_selection_signals_for_momentum_top3() -> None:
    closes = pd.DataFrame(
        {
            "AAA": [100, 101, 102, 103, 104, 105, 106, 107],
            "BBB": [100, 102, 103, 104, 105, 106, 107, 108],
            "CCC": [100, 99, 98, 100, 103, 107, 112, 118],
            "DDD": [100, 101, 101, 102, 102, 103, 103, 104],
        },
        index=pd.date_range("2025-01-01", periods=8, freq="D"),
    )
    strategy = build_evaluator_strategy_spec(
        strategy_id="multi_selection_top3",
        investment_universe=build_investment_universe_spec(
            tickers=list(closes.columns),
            key="multi_selection_top3_universe",
            label="Multi selection top3 universe",
        ),
        selection=build_selection_spec(
            "momentum_top3",
            score_parameters={"windowSpec": {"unit": "bars", "value": 2}},
        ),
        portfolio_model=build_portfolio_model_spec("equal_weight"),
        execution_policy=build_execution_policy_spec(
            key="every_bar",
            label="毎バー",
            entry="train_once_then_periodic_rebalance",
            rebalance_schedule="every_bar",
        ),
        risk_controls=build_risk_controls_spec(max_investment_ratio=1.0),
        signal_execution_contexts=[
            {
                "selectionKey": "momentum_top3",
                "strategyType": "momentum_top3",
                "label": "momentum_top3",
                "description": "momentum_top3",
                "scoreParameters": {"windowSpec": {"unit": "bars", "value": 2}},
                "weight": 0.4,
                "dataTimeframe": "1d",
                "signalTimeframe": "1d",
                "alignmentPolicy": None,
            },
            {
                "selectionKey": "secondary_momo4",
                "strategyType": "momentum_top3",
                "label": "Secondary momentum",
                "description": "Secondary momentum",
                "scoreParameters": {"windowSpec": {"unit": "bars", "value": 4}},
                "weight": 0.6,
                "dataTimeframe": "1d",
                "signalTimeframe": "1d",
                "alignmentPolicy": None,
            },
        ],
    )

    run = compare_portfolio_runs(
        closes=closes,
        volumes=None,
        strategies=[strategy],
        initial_capital=1000.0,
        split_ratio=0.5,
        transaction_cost=0.001,
        bars_per_year=252.0,
    )[0]

    assert "CCC" in run["selectedAssets"]


def test_compute_strategy_score_series_blends_explicit_additional_selection_signals() -> None:
    closes = pd.DataFrame(
        {
            "SPY": [100, 101, 103, 102, 104, 106, 108, 109],
            "QQQ": [100, 104, 105, 107, 109, 112, 114, 116],
            "TLT": [100, 99, 98, 99, 100, 101, 102, 103],
        },
        index=pd.date_range("2025-01-01", periods=8, freq="D"),
    )
    returns = closes.pct_change().dropna()
    primary_strategy = make_strategy(
        "full_universe_momentum_tilt",
        "hierarchical_risk_parity",
        score_parameters={
            "tilt_strength": 0.35,
            "tilt_shape": 1.0,
            "windowSpec": {"unit": "bars", "value": 3},
        },
    )
    blended_strategy = build_evaluator_strategy_spec(
        strategy_id="multi_selection_blend",
        timeframe=primary_strategy.timeframe,
        investment_universe=primary_strategy.investment_universe,
        selection=primary_strategy.selection,
        portfolio_model=primary_strategy.portfolio_model,
        execution_policy=primary_strategy.execution_policy,
        risk_controls=primary_strategy.risk_controls,
        signal_execution_contexts=[
            {
                "selectionKey": primary_strategy.selection.key,
                "strategyType": primary_strategy.selection.strategy_type,
                "label": primary_strategy.selection.label,
                "description": primary_strategy.selection.description,
                "scoreParameters": {
                    "tilt_strength": 0.35,
                    "tilt_shape": 1.0,
                    "windowSpec": {"unit": "bars", "value": 3},
                },
                "weight": 0.6,
                "dataTimeframe": primary_strategy.timeframe.key,
                "signalTimeframe": primary_strategy.timeframe.key,
                "alignmentPolicy": None,
            },
            {
                "selectionKey": "secondary_momo2",
                "strategyType": "full_universe_momentum_tilt",
                "label": "Secondary momentum",
                "description": "Secondary momentum",
                "scoreParameters": {
                    "tilt_strength": 0.35,
                    "tilt_shape": 1.0,
                    "windowSpec": {"unit": "bars", "value": 2},
                },
                "weight": 0.4,
                "dataTimeframe": primary_strategy.timeframe.key,
                "signalTimeframe": primary_strategy.timeframe.key,
                "alignmentPolicy": None,
            },
        ],
    )

    primary_scores = compute_strategy_score_series(
        returns,
        None,
        primary_strategy,
        bars_per_year=252,
    )
    blended_scores = compute_strategy_score_series(
        returns,
        None,
        blended_strategy,
        bars_per_year=252,
    )

    assert primary_scores is not None
    assert blended_scores is not None
    assert not primary_scores.equals(blended_scores)


def test_prepare_strategy_signal_data_uses_explicit_selection_contexts() -> None:
    closes = pd.DataFrame(
        {
            "AAA": [100, 102, 104, 103, 105, 107, 108],
            "BBB": [100, 101, 102, 103, 104, 105, 106],
        },
        index=pd.date_range("2025-01-01", periods=7, freq="D"),
    )
    returns = closes.pct_change().dropna()
    selection_contexts = [
        {
            "source_kind": "selection_signal",
            "selection": build_selection_spec("full_universe"),
            "weight": 1.0,
            "data_timeframe_key": "1d",
            "signal_timeframe_key": "1w",
            "alignment_policy": None,
        }
    ]

    with patch(
        "app.portfolio.get_strategy_signal_execution_contexts",
        side_effect=AssertionError("explicit selection contexts should avoid strategy re-extraction"),
    ):
        signal_returns, signal_volumes, signal_bars_per_year = prepare_strategy_signal_data(
            history_returns=returns,
            volume_history=None,
            selection_contexts=selection_contexts,
        )

    assert signal_volumes is None
    assert signal_bars_per_year == DEFAULT_WEEKLY_TIMEFRAME.bars_per_year
    assert list(signal_returns.index) == [pd.Timestamp("2025-01-07")]


def test_prepare_strategy_predictor_panel_uses_explicit_predictor_context() -> None:
    predictor_panel = pd.DataFrame(
        {
            "AAA": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "BBB": [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        },
        index=pd.date_range("2025-01-01", periods=7, freq="D"),
    )
    strategy = build_evaluator_strategy_spec(
        strategy_id="explicit_predictor_signal_data",
        timeframe=DEFAULT_WEEKLY_TIMEFRAME,
        investment_universe=build_investment_universe_spec(
            tickers=["AAA", "BBB"],
            key="explicit_predictor_signal_data_universe",
            label="Explicit predictor signal data universe",
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
        predictor_use=build_predictor_use_spec(
            predictor_key="pred-explicit-predictor-context",
            signal_weight=0.6,
            predictor_weight=0.4,
        ),
        predictor_signal_execution_context={
            "predictorKey": "pred-explicit-predictor-context",
            "signalWeight": 0.6,
            "predictorWeight": 0.4,
            "dataTimeframe": "1d",
            "signalTimeframe": "1w",
            "alignmentPolicy": None,
        },
    )
    _, predictor_context = get_strategy_signal_execution_contexts(strategy)
    assert predictor_context is not None

    with patch(
        "app.portfolio.get_strategy_signal_execution_contexts",
        side_effect=AssertionError("explicit predictor context should avoid strategy re-extraction"),
    ):
        prepared_panel = prepare_strategy_predictor_panel(
            predictor_panel=predictor_panel,
            predictor_context=predictor_context,
        )

    assert prepared_panel is not None
    assert list(prepared_panel.index) == [pd.Timestamp("2025-01-03"), pd.Timestamp("2025-01-07")]
    assert prepared_panel.loc[pd.Timestamp("2025-01-07"), "AAA"] == 0.7


def test_prepare_strategy_predictor_panel_resamples_daily_source_to_weekly_signal_timeframe() -> None:
    predictor_panel = pd.DataFrame(
        {
            "AAA": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "BBB": [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        },
        index=pd.date_range("2025-01-01", periods=7, freq="D"),
    )
    strategy = build_evaluator_strategy_spec(
        strategy_id="weekly_predictor_from_daily_source",
        timeframe=DEFAULT_WEEKLY_TIMEFRAME,
        investment_universe=build_investment_universe_spec(
            tickers=["AAA", "BBB"],
            key="weekly_predictor_source_universe",
            label="Weekly predictor source universe",
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
        predictor_use=build_predictor_use_spec(
            predictor_key="pred-weekly-predictor-source",
            signal_weight=0.6,
            predictor_weight=0.4,
        ),
        signal_execution_contexts=[
            {
                "selectionKey": "full_universe",
                "strategyType": "full_universe",
                "label": "full_universe",
                "description": "full_universe",
                "scoreParameters": {},
                "weight": 1.0,
                "dataTimeframe": "1d",
                "signalTimeframe": "1w",
                "alignmentPolicy": None,
            }
        ],
        predictor_signal_execution_context={
            "predictorKey": "pred-weekly-predictor-source",
            "signalWeight": 0.6,
            "predictorWeight": 0.4,
            "dataTimeframe": "1d",
            "signalTimeframe": "1w",
            "alignmentPolicy": None,
        },
    )

    prepared_panel = prepare_strategy_predictor_panel(
        predictor_panel=predictor_panel,
        strategy=strategy,
    )

    calendar_strategy = build_evaluator_strategy_spec(
        strategy_id="weekly_predictor_from_daily_source_calendar",
        timeframe=DEFAULT_WEEKLY_TIMEFRAME,
        investment_universe=strategy.investment_universe,
        selection=strategy.selection,
        portfolio_model=strategy.portfolio_model,
        execution_policy=strategy.execution_policy,
        risk_controls=strategy.risk_controls,
        predictor_use=strategy.predictor_use,
        signal_execution_contexts=[
            {
                "selectionKey": "full_universe",
                "strategyType": "full_universe",
                "label": "full_universe",
                "description": "full_universe",
                "scoreParameters": {},
                "weight": 1.0,
                "dataTimeframe": "1d",
                "signalTimeframe": "1w",
                "alignmentPolicy": None,
            }
        ],
        predictor_signal_execution_context={
            "predictorKey": "pred-weekly-predictor-source",
            "signalWeight": 0.6,
            "predictorWeight": 0.4,
            "dataTimeframe": "1d",
            "signalTimeframe": "1w",
            "alignmentPolicy": {"method": "calendar_resample"},
        },
    )
    calendar_panel = prepare_strategy_predictor_panel(
        predictor_panel=predictor_panel,
        strategy=calendar_strategy,
    )

    assert prepared_panel is not None
    assert list(prepared_panel.index) == [pd.Timestamp("2025-01-03"), pd.Timestamp("2025-01-07")]
    assert prepared_panel.loc[pd.Timestamp("2025-01-03"), "AAA"] == 0.3
    assert prepared_panel.loc[pd.Timestamp("2025-01-07"), "AAA"] == 0.7
    assert calendar_panel is not None
    assert list(calendar_panel.index) == [pd.Timestamp("2025-01-03"), pd.Timestamp("2025-01-10")]
    assert calendar_panel.loc[pd.Timestamp("2025-01-03"), "AAA"] == 0.3
    assert calendar_panel.loc[pd.Timestamp("2025-01-10"), "AAA"] == 0.7


def test_convert_window_spec_to_bars_supports_duration_units() -> None:
    assert convert_window_spec_to_bars({"unit": "bars", "value": 8}, bars_per_year=252) == 8
    assert convert_window_spec_to_bars({"unit": "days", "value": 5}, bars_per_year=252) == 5
    assert convert_window_spec_to_bars({"unit": "weeks", "value": 2}, bars_per_year=252) == 10
    assert convert_window_spec_to_bars({"unit": "months", "value": 3}, bars_per_year=252) == 63
    assert convert_window_spec_to_bars({"unit": "years", "value": 1}, bars_per_year=252) == 252
