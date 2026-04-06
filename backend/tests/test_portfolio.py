import pandas as pd

from app.portfolio import (
    build_investment_universe_spec,
    build_portfolio_model_spec,
    build_portfolio_state,
    build_risk_controls_spec,
    build_selection_spec,
    build_strategy_spec,
    compare_portfolio_runs,
    compute_trade_cost,
    compute_strategy_score_series,
    should_rebalance,
)


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
    score_parameters: dict[str, float] | None = None,
):
    return build_strategy_spec(
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
    assert payload[0]["summary"]["turnoverPct"] >= 80.0
    assert len(payload[0]["series"]) == 6


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
        adv_window_days=20,
        min_adv_notional=1_000_000.0,
    )
    low_liquidity_cost = compute_trade_cost(
        weight_delta=weight_delta,
        linear_cost_rates=linear_cost_rates,
        impact_cost_rates=impact_cost_rates,
        portfolio_equity=10_000.0,
        price_snapshot=price_snapshot,
        volume_history=low_liquidity_volumes,
        adv_window_days=20,
        min_adv_notional=1_000_000.0,
    )

    assert low_liquidity_cost > high_liquidity_cost


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


def test_momentum_window_days_changes_ranking_scores() -> None:
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
        score_parameters={"tilt_strength": 0.35, "tilt_shape": 1.0, "window_days": 3},
    )
    long_window_strategy = build_selection_spec(
        "full_universe_momentum_tilt",
        score_parameters={"tilt_strength": 0.35, "tilt_shape": 1.0, "window_days": 7},
    )

    short_scores = compute_strategy_score_series(returns, None, short_window_strategy)
    long_scores = compute_strategy_score_series(returns, None, long_window_strategy)

    assert short_scores is not None
    assert long_scores is not None
    assert short_scores["QQQ"] > short_scores["SPY"]
    assert long_scores["SPY"] > long_scores["QQQ"]


def test_should_rebalance_supports_daily_frequency() -> None:
    assert should_rebalance("2025-01-01", "2025-01-02", "daily") is True
    assert should_rebalance("2025-01-01", "2025-01-01", "daily") is False


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

    strategy = build_strategy_spec(
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
            "parameters": {"entry": "hold", "rebalanceFrequency": "hold"},
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
            "parameters": {"entry": "hold", "rebalanceFrequency": "hold"},
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
