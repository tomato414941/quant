from app.strategy import PricePoint, generate_demo_prices, parse_uploaded_prices, run_backtest


def test_demo_backtest_returns_series_and_summary() -> None:
    result = run_backtest(generate_demo_prices(), threshold=0.03, initial_capital=10_000)

    assert len(result["series"]) == 260
    assert result["summary"]["strategy"]["tradeCount"] > 0
    assert "benchmark" in result["summary"]


def test_parse_uploaded_prices_accepts_common_columns() -> None:
    contents = b"Date,Close\n2025-01-01,100\n2025-01-02,95\n2025-01-03,97\n"

    prices = parse_uploaded_prices(contents)

    assert prices == [
        PricePoint(date="2025-01-01", close=100.0),
        PricePoint(date="2025-01-02", close=95.0),
        PricePoint(date="2025-01-03", close=97.0),
    ]


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
