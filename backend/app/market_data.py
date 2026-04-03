from __future__ import annotations

import yfinance as yf

from app.strategy import PricePoint


SUPPORTED_PERIODS = {"6mo", "1y", "2y", "5y", "10y", "max"}


def fetch_market_prices(ticker: str, period: str) -> tuple[list[PricePoint], dict[str, str]]:
    normalized_ticker = ticker.strip().upper()
    if not normalized_ticker:
        raise ValueError("Ticker is required.")
    if period not in SUPPORTED_PERIODS:
        raise ValueError("Unsupported period.")

    data = yf.download(
        tickers=normalized_ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if data.empty:
        raise ValueError("No market data was returned for the requested ticker.")

    closes = data.get("Close")
    if closes is None or closes.empty:
        raise ValueError("Close prices are missing from the market data response.")
    if getattr(closes, "ndim", 1) > 1:
        closes = closes.iloc[:, 0]

    prices = [
        PricePoint(date=format_market_date(index), close=float(value))
        for index, value in closes.dropna().items()
    ]
    if len(prices) < 3:
        raise ValueError("At least 3 rows are required for a backtest.")

    return (
        prices,
        {
            "ticker": normalized_ticker,
            "period": period,
            "source": "Yahoo Finance via yfinance",
        },
    )


def format_market_date(raw_value) -> str:
    if hasattr(raw_value, "strftime"):
        return raw_value.strftime("%Y-%m-%d")
    return str(raw_value)
