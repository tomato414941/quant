from __future__ import annotations

import pandas as pd
import yfinance as yf

from app.strategy import PricePoint


SUPPORTED_PERIODS = {"6mo", "1y", "2y", "3y", "5y", "10y", "max"}


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


def fetch_market_universe(
    tickers: list[str],
    period: str,
) -> tuple[pd.DataFrame, dict[str, str | list[str]]]:
    normalized_tickers = [ticker.strip().upper() for ticker in tickers if ticker.strip()]
    unique_tickers = list(dict.fromkeys(normalized_tickers))
    if not unique_tickers:
        raise ValueError("At least one ticker is required.")

    series_by_ticker: dict[str, pd.Series] = {}
    source = "Yahoo Finance via yfinance"

    for ticker in unique_tickers:
        prices, metadata = fetch_market_prices(ticker=ticker, period=period)
        source = metadata["source"]
        series_by_ticker[ticker] = pd.Series(
            data=[point.close for point in prices],
            index=[point.date for point in prices],
            name=ticker,
            dtype="float64",
        )

    closes = pd.concat(series_by_ticker.values(), axis=1, join="inner").sort_index()
    closes = closes.dropna()
    if len(closes) < 3:
        raise ValueError("At least 3 aligned rows are required for a portfolio backtest.")

    return closes, {
        "tickers": unique_tickers,
        "period": period,
        "source": source,
        "aligned_start_date": str(closes.index[0]),
        "aligned_end_date": str(closes.index[-1]),
        "row_count": len(closes),
    }


def format_market_date(raw_value) -> str:
    if hasattr(raw_value, "strftime"):
        return raw_value.strftime("%Y-%m-%d")
    return str(raw_value)
