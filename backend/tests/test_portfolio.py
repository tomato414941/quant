import pandas as pd

from app.portfolio import (
    build_portfolio_state,
    build_portfolio_model_definition,
    build_portfolio_strategy_definition,
    compare_portfolio_runs,
)


def test_compare_portfolio_runs_returns_strategy_model_combinations() -> None:
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
        strategy_definitions=[
            build_portfolio_strategy_definition("full_universe"),
            build_portfolio_strategy_definition("momentum_top3"),
        ],
        model_definitions=[
            build_portfolio_model_definition("equal_weight"),
            build_portfolio_model_definition("risk_budgeting"),
            build_portfolio_model_definition("minimum_variance"),
            build_portfolio_model_definition("hierarchical_risk_parity"),
        ],
        initial_capital=10_000,
        split_ratio=0.6,
        transaction_cost=0.001,
        max_investment_ratio=0.8,
        rebalance_frequency="monthly",
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
    assert payload[0]["summary"]["turnoverPct"] >= 80.0
    assert len(payload[0]["series"]) == 6
