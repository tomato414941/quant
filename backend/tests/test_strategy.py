from app.strategy import PricePoint, run_backtest


def test_backtest_returns_series_and_summary() -> None:
    prices = [
        PricePoint(date="2025-01-01", close=100),
        PricePoint(date="2025-01-02", close=95),
        PricePoint(date="2025-01-03", close=97),
        PricePoint(date="2025-01-04", close=98),
    ]
    result = run_backtest(prices, threshold=0.03, initial_capital=10_000)

    assert len(result["series"]) == 4
    assert result["summary"]["strategy"]["tradeCount"] > 0
    assert "benchmark" in result["summary"]
    assert result["summary"]["config"]["holdingDays"] == 1


def test_backtest_rejects_invalid_threshold() -> None:
    prices = [
        PricePoint(date="2025-01-01", close=100),
        PricePoint(date="2025-01-02", close=95),
        PricePoint(date="2025-01-03", close=97),
    ]

    try:
        run_backtest(prices, threshold=1.5, initial_capital=10_000)
    except ValueError as exc:
        assert "Threshold" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_backtest_supports_multiple_holding_days() -> None:
    prices = [
        PricePoint(date="2025-01-01", close=100),
        PricePoint(date="2025-01-02", close=95),
        PricePoint(date="2025-01-03", close=96),
        PricePoint(date="2025-01-04", close=98),
    ]

    result = run_backtest(prices, threshold=0.03, initial_capital=10_000, holding_days=2)

    assert result["series"][2]["position"] == 1
    assert result["series"][3]["position"] == 1
    assert result["summary"]["strategy"]["tradeCount"] == 1
