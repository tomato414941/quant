from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch_market_universe(
    tickers: list[str],
    period: str,
) -> tuple[pd.DataFrame, dict[str, str | list[str]]]:
    bundle, metadata = fetch_market_universe_bundle(tickers=tickers, period=period)
    return bundle["closes"], metadata


def fetch_market_universe_bundle(
    tickers: list[str],
    period: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, str | list[str]]]:
    normalized_tickers = [ticker.strip().upper() for ticker in tickers if ticker.strip()]
    unique_tickers = list(dict.fromkeys(normalized_tickers))
    if not unique_tickers:
        raise ValueError("At least one ticker is required.")

    close_series_by_ticker: dict[str, pd.Series] = {}
    volume_series_by_ticker: dict[str, pd.Series] = {}
    source = "Yahoo Finance via yfinance"

    for ticker in unique_tickers:
        data = yf.download(
            tickers=ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if data.empty:
            raise ValueError("No market data was returned for the requested ticker.")
        close_data = data.get("Close")
        if close_data is None or close_data.empty:
            raise ValueError("Close prices are missing from the market data response.")
        volume_data = data.get("Volume")
        if getattr(close_data, "ndim", 1) > 1:
            close_data = close_data.iloc[:, 0]
        if volume_data is None or volume_data.empty:
            volume_data = pd.Series(0.0, index=close_data.index)
        elif getattr(volume_data, "ndim", 1) > 1:
            volume_data = volume_data.iloc[:, 0]

        source = "Yahoo Finance via yfinance"
        close_series_by_ticker[ticker] = pd.Series(
            data=[float(value) for value in close_data.dropna().values],
            index=[format_market_date(index) for index in close_data.dropna().index],
            name=ticker,
            dtype="float64",
        )
        volume_series_by_ticker[ticker] = pd.Series(
            data=[float(value) for value in volume_data.reindex(close_data.dropna().index).fillna(0.0).values],
            index=[format_market_date(index) for index in close_data.dropna().index],
            name=ticker,
            dtype="float64",
        )

    closes = pd.concat(close_series_by_ticker.values(), axis=1, join="inner").sort_index()
    closes = closes.dropna()
    if len(closes) < 3:
        raise ValueError("At least 3 aligned rows are required for a portfolio backtest.")
    volumes = pd.concat(volume_series_by_ticker.values(), axis=1, join="inner").sort_index()
    volumes = volumes.reindex(closes.index).fillna(0.0)

    metadata = {
        "tickers": unique_tickers,
        "period": period,
        "source": source,
        "aligned_start_date": str(closes.index[0]),
        "aligned_end_date": str(closes.index[-1]),
        "row_count": len(closes),
    }
    return {"closes": closes, "volumes": volumes}, metadata


def format_market_date(raw_value) -> str:
    if hasattr(raw_value, "strftime"):
        return raw_value.strftime("%Y-%m-%d")
    return str(raw_value)
