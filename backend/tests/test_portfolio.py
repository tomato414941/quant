import pandas as pd

from app.portfolio import (
    build_portfolio_candidate_definition,
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
        volumes=None,
        candidate_definitions=[
            build_portfolio_candidate_definition(
                build_portfolio_strategy_definition("full_universe"),
                build_portfolio_model_definition("equal_weight"),
            ),
            build_portfolio_candidate_definition(
                build_portfolio_strategy_definition("full_universe"),
                build_portfolio_model_definition("risk_budgeting"),
            ),
            build_portfolio_candidate_definition(
                build_portfolio_strategy_definition("full_universe"),
                build_portfolio_model_definition("minimum_variance"),
            ),
            build_portfolio_candidate_definition(
                build_portfolio_strategy_definition("full_universe"),
                build_portfolio_model_definition("hierarchical_risk_parity"),
            ),
            build_portfolio_candidate_definition(
                build_portfolio_strategy_definition("momentum_top3"),
                build_portfolio_model_definition("equal_weight"),
            ),
            build_portfolio_candidate_definition(
                build_portfolio_strategy_definition("momentum_top3"),
                build_portfolio_model_definition("risk_budgeting"),
            ),
            build_portfolio_candidate_definition(
                build_portfolio_strategy_definition("momentum_top3"),
                build_portfolio_model_definition("minimum_variance"),
            ),
            build_portfolio_candidate_definition(
                build_portfolio_strategy_definition("momentum_top3"),
                build_portfolio_model_definition("hierarchical_risk_parity"),
            ),
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
        candidate_definitions=[
            build_portfolio_candidate_definition(
                build_portfolio_strategy_definition("dual_momentum_top3"),
                build_portfolio_model_definition("hierarchical_risk_parity"),
            )
        ],
        initial_capital=10_000,
        split_ratio=0.6,
        transaction_cost=0.001,
        max_investment_ratio=0.85,
        rebalance_frequency="monthly",
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
        candidate_definitions=[
            build_portfolio_candidate_definition(
                build_portfolio_strategy_definition("full_universe"),
                build_portfolio_model_definition("equal_weight"),
            )
        ],
        initial_capital=10_000,
        split_ratio=0.6,
        transaction_cost=0.001,
        max_investment_ratio=1.0,
        max_weight=0.2,
        rebalance_frequency="monthly",
        portfolio_state=build_portfolio_state(current_weights={}, cash_weight=1.0),
    )

    asset_weights = [row["weightPct"] for row in payload[0]["weights"] if row["asset"] != "CASH"]
    cash_rows = [row for row in payload[0]["weights"] if row["asset"] == "CASH"]

    assert max(asset_weights) == 20.0
    assert cash_rows[0]["weightPct"] == 20.0


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
        candidate_definitions=[
            build_portfolio_candidate_definition(
                build_portfolio_strategy_definition("trailing_momentum_low_vol_universe"),
                build_portfolio_model_definition("hierarchical_risk_parity"),
            )
        ],
        initial_capital=10_000,
        split_ratio=0.6,
        transaction_cost=0.001,
        max_investment_ratio=1.0,
        rebalance_frequency="annual",
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
        candidate_definitions=[
            build_portfolio_candidate_definition(
                build_portfolio_strategy_definition("full_universe"),
                build_portfolio_model_definition("hierarchical_risk_parity"),
            )
        ],
        initial_capital=10_000,
        split_ratio=0.6,
        transaction_cost=0.001,
        max_investment_ratio=1.0,
        rebalance_frequency="annual",
        portfolio_state=build_portfolio_state(current_weights={}, cash_weight=1.0),
    )
    tilted_payload = compare_portfolio_runs(
        closes=closes,
        volumes=None,
        candidate_definitions=[
            build_portfolio_candidate_definition(
                build_portfolio_strategy_definition("full_universe_momentum_tilt"),
                build_portfolio_model_definition("hierarchical_risk_parity"),
            )
        ],
        initial_capital=10_000,
        split_ratio=0.6,
        transaction_cost=0.001,
        max_investment_ratio=1.0,
        rebalance_frequency="annual",
        portfolio_state=build_portfolio_state(current_weights={}, cash_weight=1.0),
    )

    baseline_weight_map = {
        row["asset"]: row["weightPct"] for row in baseline_payload[0]["weights"] if row["asset"] != "CASH"
    }
    tilted_weight_map = {
        row["asset"]: row["weightPct"] for row in tilted_payload[0]["weights"] if row["asset"] != "CASH"
    }
    assert tilted_weight_map["TLT"] < baseline_weight_map["TLT"]
