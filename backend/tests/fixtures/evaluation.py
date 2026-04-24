import pandas as pd

from app.portfolio import (
    build_evaluator_strategy_spec,
    build_execution_policy_spec,
    build_investment_universe_spec,
    build_portfolio_model_spec,
    build_risk_controls_spec,
    build_selection_spec,
)


REGRESSION_TICKERS = ("SPY", "QQQ", "TLT", "GLD", "LATE", "GONE")


def build_regression_market_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.date_range("2025-01-01", periods=16, freq="D")
    closes = pd.DataFrame(
        {
            "SPY": [100, 101, 102, 103, 104, 106, 107, 108, 110, 111, 112, 113, 115, 116, 117, 118],
            "QQQ": [100, 102, 105, 107, 110, 114, 116, 119, 123, 125, 128, 132, 135, 139, 142, 146],
            "TLT": [100, 100, 99, 100, 99, 98, 98, 97, 97, 96, 96, 95, 95, 94, 94, 93],
            "GLD": [
                100,
                100.2,
                100.5,
                100.7,
                101.0,
                101.2,
                101.5,
                101.8,
                102.0,
                102.3,
                102.5,
                102.8,
                103.0,
                103.3,
                103.5,
                103.8,
            ],
            "LATE": [None, None, None, None, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61],
            "GONE": [100, 101, 102, 103, 104, 105, 106, 107, None, None, None, None, None, None, None, None],
        },
        index=index,
    )
    volumes = pd.DataFrame(
        {
            "SPY": [1_000_000 + index * 8_000 for index in range(len(index))],
            "QQQ": [900_000 + index * 12_000 for index in range(len(index))],
            "TLT": [800_000 + index * 2_000 for index in range(len(index))],
            "GLD": [700_000 + index * 3_000 for index in range(len(index))],
            "LATE": [
                0,
                0,
                0,
                0,
                500_000,
                520_000,
                540_000,
                560_000,
                580_000,
                600_000,
                620_000,
                640_000,
                660_000,
                680_000,
                700_000,
                720_000,
            ],
            "GONE": [450_000, 455_000, 460_000, 465_000, 470_000, 475_000, 480_000, 485_000, 0, 0, 0, 0, 0, 0, 0, 0],
        },
        index=index,
    )
    return closes, volumes


def build_regression_execution_assumptions() -> dict:
    return {
        "kind": "close_execution_assumptions",
        "label": "終値約定",
        "parameters": {"fillPrice": "close"},
        "costModel": {
            "kind": "flat_cost",
            "parameters": {"commissionPct": 0.1, "slippagePct": 0.0},
            "perAssetOverrides": {},
        },
    }


def build_regression_strategy(
    *,
    strategy_id: str,
    strategy_type: str,
    model_type: str,
    score_parameters: dict[str, object] | None = None,
    max_investment_ratio: float = 1.0,
    decision_policy: str = "direct_score_to_weight",
):
    return build_evaluator_strategy_spec(
        strategy_id=strategy_id,
        investment_universe=build_investment_universe_spec(
            tickers=list(REGRESSION_TICKERS),
            key="regression_universe",
            label="Regression universe",
        ),
        selection=build_selection_spec(
            strategy_type,
            score_parameters=score_parameters,
        ),
        portfolio_model=build_portfolio_model_spec(model_type),
        execution_policy=build_execution_policy_spec(
            key="every_bar",
            label="毎バー",
            entry="train_once_then_periodic_rebalance",
            rebalance_schedule="every_bar",
        ),
        risk_controls=build_risk_controls_spec(max_investment_ratio=max_investment_ratio),
        decision_schedule="every_bar",
        extensions={"decision_policy": decision_policy},
    )
