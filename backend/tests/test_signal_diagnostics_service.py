import numpy as np
import pandas as pd

from app import signal_diagnostics_service
from app.portfolio import (
    build_evaluator_strategy_spec,
    build_execution_policy_spec,
    build_investment_universe_spec,
    build_portfolio_model_spec,
    build_risk_controls_spec,
    build_selection_spec,
    build_strategy_definition_from_evaluator_strategy_spec,
)


def build_test_strategy_definition():
    strategy = build_evaluator_strategy_spec(
        strategy_id="signal_diagnostics_test_strategy",
        investment_universe=build_investment_universe_spec(
            tickers=["AAA", "BBB", "CCC", "DDD"],
            key="signal_diagnostics_test_universe",
            label="Signal diagnostics test universe",
        ),
        selection=build_selection_spec(
            "momentum_top3",
            score_parameters={"windowSpec": {"unit": "bars", "value": 3}},
        ),
        portfolio_model=build_portfolio_model_spec("equal_weight"),
        execution_policy=build_execution_policy_spec(
            key="every_bar",
            label="Every bar",
            entry="train_once_then_periodic_rebalance",
            rebalance_schedule="every_bar",
        ),
        risk_controls=build_risk_controls_spec(max_investment_ratio=1.0),
    )
    return build_strategy_definition_from_evaluator_strategy_spec(strategy)


def test_build_strategy_signal_diagnostics_reports_positive_signal_relationship() -> None:
    index = pd.date_range("2025-01-01", periods=40, freq="D")
    closes = pd.DataFrame(
        {
            "AAA": [100 * (1.020 ** row) for row in range(len(index))],
            "BBB": [100 * (1.010 ** row) for row in range(len(index))],
            "CCC": [100 * (0.995 ** row) for row in range(len(index))],
            "DDD": [100 * (0.990 ** row) for row in range(len(index))],
        },
        index=index,
    )

    result = signal_diagnostics_service.build_strategy_signal_diagnostics(
        strategy_definition=build_test_strategy_definition(),
        closes=closes,
        volumes=None,
        horizons=(1, 5),
        bucket_count=2,
    )

    one_day = result["horizonResults"][0]
    five_day = result["horizonResults"][1]
    assert one_day["horizon"] == "1d"
    assert one_day["sampleCount"] > 0
    assert one_day["rankIc"] > 0.9
    assert one_day["topMinusBottomForwardReturnPct"] > 0
    assert one_day["hitRate"] == 1.0
    assert five_day["horizon"] == "5d"
    assert five_day["topMinusBottomForwardReturnPct"] > one_day["topMinusBottomForwardReturnPct"]


def test_build_strategy_signal_diagnostics_handles_constant_scores() -> None:
    index = pd.date_range("2025-01-01", periods=12, freq="D")
    closes = pd.DataFrame(
        {ticker: np.full(len(index), 100.0) for ticker in ["AAA", "BBB", "CCC", "DDD"]},
        index=index,
    )

    result = signal_diagnostics_service.build_strategy_signal_diagnostics(
        strategy_definition=build_test_strategy_definition(),
        closes=closes,
        volumes=None,
        horizons=(1,),
        bucket_count=2,
    )

    horizon = result["horizonResults"][0]
    assert horizon["sampleCount"] == 0
    assert horizon["rankIc"] is None
    assert horizon["topMinusBottomForwardReturnPct"] is None
    assert horizon["hitRate"] is None


def test_parse_signal_horizon_label_requires_day_suffix() -> None:
    assert signal_diagnostics_service.parse_signal_horizon_label("21d") == 21
