from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Protocol

import pandas as pd
import yfinance as yf


DEFAULT_MAX_STALE_BARS = 5
DEFAULT_ADJUSTMENT_POLICY = "auto_adjust"
DEFAULT_MARKET_SNAPSHOT_STORAGE_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "market_snapshots"
)


@dataclass(frozen=True)
class MarketDataRequest:
    tickers: tuple[str, ...]
    period: str
    timeframe: str = "1d"
    start_date: str | None = None
    end_date: str | None = None
    max_stale_bars: int = DEFAULT_MAX_STALE_BARS


class MarketDataProvider(Protocol):
    def fetch_bundle(
        self,
        request: MarketDataRequest,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, object]]: ...


class YFinanceMarketDataProvider:
    source_label = "Yahoo Finance via yfinance"
    max_attempts = 3
    retry_delay_seconds = 0.5

    def fetch_bundle(
        self,
        request: MarketDataRequest,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
        unique_tickers = normalize_tickers(request.tickers)
        close_series_by_ticker: dict[str, pd.Series] = {}
        volume_series_by_ticker: dict[str, pd.Series] = {}
        failed_tickers: dict[str, str] = {}
        asset_availability: dict[str, dict[str, object]] = {}

        for ticker in unique_tickers:
            data, failure_reason = self._download_ticker(
                ticker=ticker,
                period=request.period,
                timeframe=request.timeframe,
                start_date=request.start_date,
                end_date=request.end_date,
            )
            if data is None:
                failed_reason = failure_reason or "unknown_download_failure"
                failed_tickers[ticker] = failed_reason
                asset_availability[ticker] = {
                    "requested": True,
                    "available": False,
                    "failedReason": failed_reason,
                    "firstValidDate": None,
                    "lastValidDate": None,
                    "validRowCount": 0,
                }
                continue
            close_data = data["Close"]
            volume_data = data["Volume"]

            clean_close_data = close_data.dropna()
            if clean_close_data.empty:
                failed_tickers[ticker] = "empty_close_series"
                asset_availability[ticker] = {
                    "requested": True,
                    "available": False,
                    "failedReason": "empty_close_series",
                    "firstValidDate": None,
                    "lastValidDate": None,
                    "validRowCount": 0,
                }
                continue
            close_series = pd.Series(
                data=[float(value) for value in clean_close_data.values],
                index=[format_market_date(index) for index in clean_close_data.index],
                name=ticker,
                dtype="float64",
            )
            close_series_by_ticker[ticker] = close_series
            asset_availability[ticker] = {
                "requested": True,
                "available": True,
                "failedReason": None,
                "firstValidDate": str(close_series.index[0]),
                "lastValidDate": str(close_series.index[-1]),
                "validRowCount": int(close_series.count()),
            }
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

        aligned_index = build_union_market_index(close_series_by_ticker.values())
        closes = pd.concat(
            [
                align_series_to_index(series, aligned_index, limit=request.max_stale_bars)
                for series in close_series_by_ticker.values()
            ],
            axis=1,
        ).sort_index()
        closes = closes.dropna(how="all")
        if len(closes) < 3:
            raise ValueError(
                "At least 3 aligned rows are required for a portfolio backtest after "
                f"provider filtering. Available tickers: {list(close_series_by_ticker)}"
            )
        volumes = pd.concat(
            [
                series.reindex(closes.index)
                for series in volume_series_by_ticker.values()
            ],
            axis=1,
        ).sort_index()
        volumes = volumes.reindex(closes.index).fillna(0.0)

        metadata = {
            "tickers": list(closes.columns),
            "requested_tickers": unique_tickers,
            "failed_tickers": failed_tickers,
            "assetAvailability": asset_availability,
            "period": request.period,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "timeframe": request.timeframe,
            "source": self.source_label,
            "aligned_start_date": str(closes.index[0]),
            "aligned_end_date": str(closes.index[-1]),
            "row_count": len(closes),
        }
        metadata["datasetSnapshot"] = build_dataset_snapshot_metadata(
            source=self.source_label,
            timeframe=request.timeframe,
            period=request.period,
            start_date=request.start_date,
            end_date=request.end_date,
            requested_tickers=unique_tickers,
            available_tickers=list(closes.columns),
            row_count=len(closes),
            adjustment_policy=DEFAULT_ADJUSTMENT_POLICY,
        )
        return {"closes": closes, "volumes": volumes}, metadata

    def _download_ticker(
        self,
        *,
        ticker: str,
        period: str,
        timeframe: str,
        start_date: str | None,
        end_date: str | None,
    ) -> tuple[pd.DataFrame | None, str | None]:
        last_failure_reason: str | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                download_kwargs = {
                    "tickers": ticker,
                    "interval": timeframe,
                    "auto_adjust": True,
                    "progress": False,
                    "threads": False,
                }
                if start_date is not None or end_date is not None:
                    download_kwargs["start"] = start_date
                    download_kwargs["end"] = exclusive_yfinance_end_date(end_date) if end_date is not None else None
                else:
                    download_kwargs["period"] = period
                data = yf.download(**download_kwargs)
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


def build_union_market_index(series_values) -> pd.Index:
    union_index = None
    for series in series_values:
        union_index = series.index if union_index is None else union_index.union(series.index)
    if union_index is None:
        return pd.Index([])
    return union_index.sort_values()


def align_series_to_index(series: pd.Series, target_index, limit: int | None = None) -> pd.Series:
    combined_index = series.index.union(target_index)
    return series.reindex(combined_index).sort_index().ffill(limit=limit).reindex(target_index)


def fetch_market_universe(
    tickers: list[str],
    period: str,
    timeframe: str = "1d",
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    bundle, metadata = fetch_market_universe_bundle(
        tickers=tickers,
        period=period,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
    )
    return bundle["closes"], metadata


def fetch_market_universe_bundle(
    tickers: list[str],
    period: str,
    timeframe: str = "1d",
    start_date: str | None = None,
    end_date: str | None = None,
    max_stale_bars: int = DEFAULT_MAX_STALE_BARS,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    request = MarketDataRequest(
        tickers=tuple(tickers),
        period=period,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        max_stale_bars=max_stale_bars,
    )
    return DEFAULT_MARKET_DATA_PROVIDER.fetch_bundle(request)


def build_dataset_snapshot_metadata(
    *,
    source: str,
    timeframe: str,
    period: str,
    start_date: str | None,
    end_date: str | None,
    requested_tickers: list[str],
    available_tickers: list[str],
    row_count: int,
    adjustment_policy: str = DEFAULT_ADJUSTMENT_POLICY,
    created_at_utc: str | None = None,
) -> dict[str, object]:
    created_at = created_at_utc or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    payload: dict[str, object] = {
        "source": source,
        "createdAtUtc": created_at,
        "timeframe": timeframe,
        "period": period,
        "start": start_date,
        "end": end_date,
        "requestedTickers": list(requested_tickers),
        "availableTickers": list(available_tickers),
        "rowCount": int(row_count),
        "adjustmentPolicy": adjustment_policy,
    }
    fingerprint = build_dataset_snapshot_fingerprint(payload)
    payload["fingerprint"] = fingerprint
    payload["snapshotId"] = fingerprint
    return payload


def write_market_data_snapshot(
    bundle: dict[str, pd.DataFrame],
    metadata: dict[str, object],
    storage_dir: Path | str = DEFAULT_MARKET_SNAPSHOT_STORAGE_DIR,
) -> Path:
    snapshot = metadata.get("datasetSnapshot")
    if not isinstance(snapshot, dict):
        raise ValueError("metadata must include datasetSnapshot metadata.")

    snapshot_id = snapshot.get("snapshotId")
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise ValueError("datasetSnapshot must include a non-empty snapshotId.")
    if snapshot_id != Path(snapshot_id).name:
        raise ValueError("datasetSnapshot snapshotId must be a single path segment.")

    closes = bundle.get("closes")
    volumes = bundle.get("volumes")
    if not isinstance(closes, pd.DataFrame):
        raise ValueError("bundle must include a closes DataFrame.")
    if not isinstance(volumes, pd.DataFrame):
        raise ValueError("bundle must include a volumes DataFrame.")

    snapshot_path = Path(storage_dir) / snapshot_id
    snapshot_path.mkdir(parents=True, exist_ok=True)

    files = {
        "closes": "closes.csv",
        "volumes": "volumes.csv",
    }
    closes.to_csv(snapshot_path / files["closes"], index_label="date")
    volumes.to_csv(snapshot_path / files["volumes"], index_label="date")

    manifest = {
        "schemaVersion": 1,
        "datasetSnapshot": snapshot,
        "files": files,
    }
    with (snapshot_path / "manifest.json").open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")

    return snapshot_path


def build_dataset_snapshot_fingerprint(snapshot: dict[str, object]) -> str:
    fingerprint_payload = {
        key: value
        for key, value in snapshot.items()
        if key not in {"createdAtUtc", "fingerprint", "snapshotId"}
    }
    encoded = json.dumps(
        fingerprint_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def exclusive_yfinance_end_date(end_date: str | None) -> str | None:
    if end_date is None:
        return None
    parsed_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    return (parsed_date + timedelta(days=1)).isoformat()


def format_market_date(raw_value) -> str:
    if hasattr(raw_value, "strftime"):
        return raw_value.strftime("%Y-%m-%d")
    return str(raw_value)
