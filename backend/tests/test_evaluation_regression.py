from app.portfolio import build_portfolio_state, compare_portfolio_runs
from tests.fixtures.evaluation import (
    build_regression_execution_assumptions,
    build_regression_market_data,
    build_regression_strategy,
)


REGRESSION_AVAILABILITY_POLICY = {
    "kind": "asset_availability_policy",
    "minHistoryBars": 2,
    "maxStaleBars": 2,
    "delistedAssetPolicy": "liquidate_to_cash",
}


def run_regression_strategy(
    *,
    strategy_type: str,
    model_type: str,
    score_parameters: dict[str, object] | None = None,
    portfolio_state=None,
):
    closes, volumes = build_regression_market_data()
    strategy = build_regression_strategy(
        strategy_id=f"regression_{strategy_type}_{model_type}",
        strategy_type=strategy_type,
        model_type=model_type,
        score_parameters=score_parameters,
    )
    return compare_portfolio_runs(
        closes=closes,
        volumes=volumes,
        strategies=[strategy],
        initial_capital=10_000,
        split_ratio=0.5,
        bars_per_year=252.0,
        execution_assumptions=build_regression_execution_assumptions(),
        portfolio_state=portfolio_state or build_portfolio_state(current_weights={}, cash_weight=1.0),
        availability_policy=REGRESSION_AVAILABILITY_POLICY,
    )[0]


def assert_stable_common_summary(run: dict, *, total_return_pct: float, turnover_pct: float) -> None:
    assert run["summary"]["totalReturnPct"] == total_return_pct
    assert run["summary"]["maxDrawdownPct"] == 0.0
    assert run["summary"]["turnoverPct"] == turnover_pct
    assert run["availabilitySummary"] == {
        "minAvailableAssetCount": 5,
        "maxAvailableAssetCount": 6,
        "minEligibleAssetCount": 5,
        "maxEligibleAssetCount": 6,
        "newlyEligibleAssetCount": 6,
        "removedAssetCount": 1,
    }


def assert_stable_direct_decisions(
    run: dict,
    *,
    average_turnover_pct: float,
    average_realized_edge_pct: float,
) -> None:
    assert run["decisionSummary"]["decisionCount"] == 7
    assert run["decisionSummary"]["rebalanceCount"] == 7
    assert run["decisionSummary"]["noTradeCount"] == 0
    assert run["decisionSummary"]["policyCounts"] == {"direct_score_to_weight": 7}
    assert run["decisionSummary"]["reasonCounts"] == {"direct_policy": 7}
    assert run["decisionSummary"]["averageTurnoverPct"] == average_turnover_pct
    assert run["decisionSummary"]["averageRealizedEdgePct"] == average_realized_edge_pct


def test_regression_full_universe_equal_weight_result_is_stable() -> None:
    run = run_regression_strategy(strategy_type="full_universe", model_type="equal_weight")

    assert run["selectedAssets"] == ["SPY", "QQQ", "TLT", "GLD", "LATE"]
    assert run["weights"] == [
        {"asset": "SPY", "weightPct": 20.0},
        {"asset": "QQQ", "weightPct": 20.0},
        {"asset": "TLT", "weightPct": 20.0},
        {"asset": "GLD", "weightPct": 20.0},
        {"asset": "LATE", "weightPct": 20.0},
        {"asset": "GONE", "weightPct": 0.0},
    ]
    assert_stable_common_summary(run, total_return_pct=8.12, turnover_pct=100.0)
    assert_stable_direct_decisions(run, average_turnover_pct=16.67, average_realized_edge_pct=0.199)
    assert [event["eventType"] for event in run["executionTrace"]] == ["decision", "rebalance"] * 7


def test_regression_momentum_top3_result_is_stable() -> None:
    run = run_regression_strategy(
        strategy_type="momentum_top3",
        model_type="equal_weight",
        score_parameters={"windowSpec": {"unit": "bars", "value": 4}},
    )

    assert run["selectedAssets"] == ["QQQ", "LATE", "SPY"]
    assert run["weights"] == [
        {"asset": "SPY", "weightPct": 33.33},
        {"asset": "QQQ", "weightPct": 33.33},
        {"asset": "LATE", "weightPct": 33.33},
        {"asset": "TLT", "weightPct": 0.0},
        {"asset": "GLD", "weightPct": 0.0},
        {"asset": "GONE", "weightPct": 0.0},
    ]
    assert_stable_common_summary(run, total_return_pct=15.46, turnover_pct=100.0)
    assert_stable_direct_decisions(run, average_turnover_pct=14.29, average_realized_edge_pct=0.3381)
    assert [event["eventType"] for event in run["executionTrace"]] == ["decision", "rebalance"] * 7


def test_regression_dynamic_availability_trace_is_stable() -> None:
    run = run_regression_strategy(
        strategy_type="full_universe",
        model_type="equal_weight",
        portfolio_state=build_portfolio_state(current_weights={"GONE": 1.0}, cash_weight=0.0),
    )

    assert run["selectedAssets"] == ["SPY", "QQQ", "TLT", "GLD", "LATE"]
    assert_stable_common_summary(run, total_return_pct=8.02, turnover_pct=200.0)
    assert_stable_direct_decisions(run, average_turnover_pct=16.67, average_realized_edge_pct=0.199)
    assert [event["eventType"] for event in run["executionTrace"]][:3] == [
        "forced_universe_change",
        "decision",
        "rebalance",
    ]
    assert run["executionTrace"][0]["decisionReason"] == "asset_unavailable"
    assert run["executionTrace"][0]["turnoverPct"] == 100.0
    assert [row["removedAssets"] for row in run["series"]].count(["GONE"]) == 1
