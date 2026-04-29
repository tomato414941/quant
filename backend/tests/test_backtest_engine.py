import numpy as np
import pandas as pd
import pytest

from app.backtest import run_equal_weight_full_period_backtest


def test_full_period_equal_weight_backtest_invests_from_start() -> None:
    closes = pd.DataFrame(
        {
            "AAA": [100.0, 110.0, 121.0, 133.1],
            "BBB": [100.0, 100.0, 100.0, 100.0],
        },
        index=pd.date_range("2025-01-01", periods=4, freq="D"),
    )

    result = run_equal_weight_full_period_backtest(
        closes=closes,
        initial_capital=1000.0,
        transaction_cost=0.0,
        rebalance_schedule="hold",
    )

    assert result["kind"] == "full_period_backtest"
    assert result["firstInvestedDate"] == "2025-01-02 00:00:00"
    assert result["events"][0]["eventType"] == "rebalance"
    assert result["events"][0]["turnoverPct"] == 100.0
    assert result["events"][0]["executedWeights"] == [
        {"asset": "AAA", "weightPct": 50.0},
        {"asset": "BBB", "weightPct": 50.0},
    ]
    assert result["series"][0]["portfolioReturnPct"] == 5.0
    assert result["series"][0]["portfolioEquity"] == 1050.0
    assert result["summary"]["totalReturnPct"] > 0.0
    assert result["summary"]["turnoverPct"] == 100.0


def test_full_period_equal_weight_backtest_rejects_missing_held_return() -> None:
    closes = pd.DataFrame(
        {
            "AAA": [100.0, np.nan, 101.0],
            "BBB": [100.0, 100.0, 100.0],
        },
        index=pd.date_range("2025-01-01", periods=3, freq="D"),
    )

    with pytest.raises(ValueError, match="Missing return for held assets.*AAA"):
        run_equal_weight_full_period_backtest(
            closes=closes,
            initial_capital=1000.0,
            transaction_cost=0.0,
        )
