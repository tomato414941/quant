from app.strategy import (
    PricePoint,
    build_strategy_definition,
    compare_periods,
    compare_strategies,
    compare_tickers,
    run_backtest,
    run_split_backtest,
)


def test_backtest_returns_series_and_summary() -> None:
    prices = [
        PricePoint(date="2025-01-01", close=100),
        PricePoint(date="2025-01-02", close=95),
        PricePoint(date="2025-01-03", close=97),
        PricePoint(date="2025-01-04", close=98),
    ]
    result = run_backtest(
        prices,
        strategy_definition=build_strategy_definition(
            engine="mean_reversion",
            threshold=0.03,
            holding_days=1,
        ),
        initial_capital=10_000,
    )

    assert len(result["series"]) == 4
    assert result["summary"]["strategy"]["tradeCount"] > 0
    assert result["summary"]["config"]["strategyId"] == "mean_reversion"
    assert result["summary"]["config"]["strategyLabel"] == "逆張り 3.0% / 1日"


def test_backtest_supports_momentum_strategy() -> None:
    prices = [
        PricePoint(date="2025-01-01", close=100),
        PricePoint(date="2025-01-02", close=105),
        PricePoint(date="2025-01-03", close=108),
        PricePoint(date="2025-01-04", close=109),
    ]

    result = run_backtest(
        prices,
        strategy_definition=build_strategy_definition(
            engine="momentum",
            threshold=0.03,
            holding_days=1,
        ),
        initial_capital=10_000,
    )

    assert result["summary"]["strategy"]["tradeCount"] == 1
    assert result["series"][2]["signal"] is True


def test_backtest_rejects_invalid_threshold() -> None:
    prices = [
        PricePoint(date="2025-01-01", close=100),
        PricePoint(date="2025-01-02", close=95),
        PricePoint(date="2025-01-03", close=97),
    ]

    try:
        run_backtest(
            prices,
            strategy_definition=build_strategy_definition(
                engine="mean_reversion",
                threshold=1.5,
                holding_days=1,
            ),
            initial_capital=10_000,
        )
    except ValueError as exc:
        assert "Threshold" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_backtest_applies_transaction_cost() -> None:
    prices = [
        PricePoint(date="2025-01-01", close=100),
        PricePoint(date="2025-01-02", close=95),
        PricePoint(date="2025-01-03", close=100),
    ]

    strategy_definition = build_strategy_definition(
        engine="mean_reversion",
        threshold=0.03,
        holding_days=1,
    )

    without_cost = run_backtest(
        prices,
        strategy_definition=strategy_definition,
        initial_capital=10_000,
        transaction_cost=0.0,
    )
    with_cost = run_backtest(
        prices,
        strategy_definition=strategy_definition,
        initial_capital=10_000,
        transaction_cost=0.001,
    )

    assert with_cost["summary"]["strategy"]["totalReturnPct"] < without_cost["summary"]["strategy"]["totalReturnPct"]


def test_split_backtest_returns_train_and_test() -> None:
    prices = [
        PricePoint(date=f"2025-01-{day:02d}", close=100 + day)
        for day in range(1, 11)
    ]

    result = run_split_backtest(
        prices=prices,
        strategy_definition=build_strategy_definition(
            engine="momentum",
            threshold=0.02,
            holding_days=1,
        ),
        initial_capital=10_000,
        split_ratio=0.6,
    )

    assert result["config"]["splitRatioPct"] == 60.0
    assert result["train"]["dayCount"] == 6
    assert result["test"]["dayCount"] == 4


def test_compare_tickers_returns_sorted_results() -> None:
    datasets = [
        {
            "ticker": "AAA",
            "period": "1y",
            "prices": [
                PricePoint(date="2025-01-01", close=100),
                PricePoint(date="2025-01-02", close=95),
                PricePoint(date="2025-01-03", close=100),
            ],
        },
        {
            "ticker": "BBB",
            "period": "1y",
            "prices": [
                PricePoint(date="2025-01-01", close=100),
                PricePoint(date="2025-01-02", close=99),
                PricePoint(date="2025-01-03", close=98),
            ],
        },
    ]

    payload = compare_tickers(
        datasets=datasets,
        strategy_definition=build_strategy_definition(
            engine="mean_reversion",
            threshold=0.03,
            holding_days=1,
        ),
        initial_capital=10_000,
        transaction_cost=0.0,
    )

    assert payload["results"][0]["ticker"] == "AAA"


def test_compare_periods_returns_all_requested_periods() -> None:
    datasets = [
        {
            "ticker": "SPY",
            "period": "6mo",
            "prices": [
                PricePoint(date="2025-01-01", close=100),
                PricePoint(date="2025-01-02", close=95),
                PricePoint(date="2025-01-03", close=99),
            ],
        },
        {
            "ticker": "SPY",
            "period": "1y",
            "prices": [
                PricePoint(date="2025-01-01", close=100),
                PricePoint(date="2025-01-02", close=104),
                PricePoint(date="2025-01-03", close=108),
            ],
        },
    ]

    payload = compare_periods(
        datasets=datasets,
        strategy_definition=build_strategy_definition(
            engine="momentum",
            threshold=0.03,
            holding_days=1,
        ),
        initial_capital=10_000,
        transaction_cost=0.0,
    )

    assert [row["period"] for row in payload["results"]] == ["6mo", "1y"]
    assert payload["config"]["strategyId"] == "momentum"


def test_compare_strategies_returns_all_definitions() -> None:
    prices = [
        PricePoint(date="2025-01-01", close=100),
        PricePoint(date="2025-01-02", close=95),
        PricePoint(date="2025-01-03", close=100),
        PricePoint(date="2025-01-04", close=104),
        PricePoint(date="2025-01-05", close=102),
        PricePoint(date="2025-01-06", close=106),
    ]

    payload = compare_strategies(
        prices=prices,
        strategy_definitions=[
            build_strategy_definition(
                engine="mean_reversion",
                threshold=0.03,
                holding_days=1,
                key="mr_3_1",
            ),
            build_strategy_definition(
                engine="momentum",
                threshold=0.03,
                holding_days=1,
                key="mo_3_1",
            ),
        ],
        initial_capital=10_000,
        transaction_cost=0.0,
        split_ratio=0.6,
    )

    assert [row["definition"]["key"] for row in payload] == ["mr_3_1", "mo_3_1"]
    assert all("series" in row for row in payload)
