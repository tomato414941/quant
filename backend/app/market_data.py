from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Protocol

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class MarketDataRequest:
    tickers: tuple[str, ...]
    period: str
    timeframe: str = "1d"


class MarketDataProvider(Protocol):
    def fetch_bundle(
        self,
        request: MarketDataRequest,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, str | list[str]]]: ...


class YFinanceMarketDataProvider:
    source_label = "Yahoo Finance via yfinance"
    max_attempts = 3
    retry_delay_seconds = 0.5

    def fetch_bundle(
        self,
        request: MarketDataRequest,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, str | list[str]]]:
        unique_tickers = normalize_tickers(request.tickers)
        close_series_by_ticker: dict[str, pd.Series] = {}
        volume_series_by_ticker: dict[str, pd.Series] = {}
        failed_tickers: dict[str, str] = {}

        for ticker in unique_tickers:
            data, failure_reason = self._download_ticker(
                ticker=ticker,
                period=request.period,
                timeframe=request.timeframe,
            )
            if data is None:
                failed_tickers[ticker] = failure_reason or "unknown_download_failure"
                continue
            close_data = data["Close"]
            volume_data = data["Volume"]

            clean_close_data = close_data.dropna()
            if clean_close_data.empty:
                failed_tickers[ticker] = "empty_close_series"
                continue
            close_series_by_ticker[ticker] = pd.Series(
                data=[float(value) for value in clean_close_data.values],
                index=[format_market_date(index) for index in clean_close_data.index],
                name=ticker,
                dtype="float64",
            )
            volume_series_by_ticker[ticker] = pd.Series(
                data=[
                    float(value)
                    for value in volume_data.reindex(clean_close_data.index).fillna(0.0).values
                ],
                index=[format_market_date(index) for index in clean_close_data.index],
                name=ticker,
                dtype="float64",
            )

        if len(close_series_by_ticker) < 2:
            failed_details = ", ".join(
                f"{ticker}: {reason}" for ticker, reason in failed_tickers.items()
            ) or "unknown"
            raise ValueError(
                "At least two tickers with valid market data are required. "
                f"Failures: {failed_details}"
            )

        closes = pd.concat(close_series_by_ticker.values(), axis=1, join="inner").sort_index()
        closes = closes.dropna()
        if len(closes) < 3:
            raise ValueError(
                "At least 3 aligned rows are required for a portfolio backtest after "
                f"provider filtering. Available tickers: {list(close_series_by_ticker)}"
            )
        volumes = pd.concat(volume_series_by_ticker.values(), axis=1, join="inner").sort_index()
        volumes = volumes.reindex(closes.index).fillna(0.0)

        metadata = {
            "tickers": list(closes.columns),
            "requested_tickers": unique_tickers,
            "failed_tickers": failed_tickers,
            "period": request.period,
            "timeframe": request.timeframe,
            "source": self.source_label,
            "aligned_start_date": str(closes.index[0]),
            "aligned_end_date": str(closes.index[-1]),
            "row_count": len(closes),
        }
        return {"closes": closes, "volumes": volumes}, metadata

    def _download_ticker(
        self,
        *,
        ticker: str,
        period: str,
        timeframe: str,
    ) -> tuple[pd.DataFrame | None, str | None]:
        last_failure_reason: str | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                data = yf.download(
                    tickers=ticker,
                    period=period,
                    interval=timeframe,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
            except Exception as exc:
                last_failure_reason = f"download_exception:{type(exc).__name__}"
            else:
                normalized, failure_reason = self._normalize_download_frame(data)
                if normalized is not None:
                    return normalized, None
                last_failure_reason = failure_reason

            if attempt < self.max_attempts:
                time.sleep(self.retry_delay_seconds)

        return None, last_failure_reason

    def _normalize_download_frame(
        self,
        data,
    ) -> tuple[pd.DataFrame | None, str | None]:
        if not isinstance(data, pd.DataFrame):
            return None, f"unexpected_payload:{type(data).__name__}"
        if data.empty:
            return None, "empty_frame"

        close_data = data.get("Close")
        if close_data is None or close_data.empty:
            return None, "missing_close"
        volume_data = data.get("Volume")

        if getattr(close_data, "ndim", 1) > 1:
            close_data = close_data.iloc[:, 0]
        if volume_data is None or volume_data.empty:
            volume_data = pd.Series(0.0, index=close_data.index)
        elif getattr(volume_data, "ndim", 1) > 1:
            volume_data = volume_data.iloc[:, 0]

        normalized = pd.DataFrame({"Close": close_data, "Volume": volume_data})
        return normalized, None


DEFAULT_MARKET_DATA_PROVIDER: MarketDataProvider = YFinanceMarketDataProvider()


def normalize_tickers(tickers: tuple[str, ...] | list[str]) -> list[str]:
    normalized_tickers = [ticker.strip().upper() for ticker in tickers if ticker.strip()]
    unique_tickers = list(dict.fromkeys(normalized_tickers))
    if not unique_tickers:
        raise ValueError("At least one ticker is required.")
    return unique_tickers


def fetch_market_universe(
    tickers: list[str],
    period: str,
    timeframe: str = "1d",
) -> tuple[pd.DataFrame, dict[str, str | list[str]]]:
    bundle, metadata = fetch_market_universe_bundle(
        tickers=tickers,
        period=period,
        timeframe=timeframe,
    )
    return bundle["closes"], metadata


def fetch_market_universe_bundle(
    tickers: list[str],
    period: str,
    timeframe: str = "1d",
) -> tuple[dict[str, pd.DataFrame], dict[str, str | list[str]]]:
    request = MarketDataRequest(
        tickers=tuple(tickers),
        period=period,
        timeframe=timeframe,
    )
    return DEFAULT_MARKET_DATA_PROVIDER.fetch_bundle(request)


def format_market_date(raw_value) -> str:
    if hasattr(raw_value, "strftime"):
        return raw_value.strftime("%Y-%m-%d")
    return str(raw_value)
