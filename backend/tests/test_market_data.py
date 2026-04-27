import pandas as pd
import pytest

from app.market_data import (
    MarketDataRequest,
    YFinanceMarketDataProvider,
    build_dataset_snapshot_metadata,
)


def build_download_frame(values: list[float]) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=len(values), freq="D")
    return pd.DataFrame(
        {
            "Close": values,
            "Volume": [1000.0] * len(values),
        },
        index=index,
    )


def test_yfinance_provider_allows_partial_failures(monkeypatch) -> None:
    provider = YFinanceMarketDataProvider()

    def fake_download(*, tickers, **kwargs):
        if tickers == "EWJ":
            raise TypeError("'Response' object is not subscriptable")
        if tickers == "SPY":
            return build_download_frame([100.0, 101.0, 102.0, 103.0])
        if tickers == "QQQ":
            return build_download_frame([200.0, 202.0, 204.0, 206.0])
        raise AssertionError(f"unexpected ticker: {tickers}")

    monkeypatch.setattr("app.market_data.yf.download", fake_download)

    bundle, metadata = provider.fetch_bundle(
        MarketDataRequest(
            tickers=("SPY", "EWJ", "QQQ"),
            period="10y",
            timeframe="1d",
        )
    )

    assert list(bundle["closes"].columns) == ["SPY", "QQQ"]
    assert metadata["tickers"] == ["SPY", "QQQ"]
    assert metadata["requested_tickers"] == ["SPY", "EWJ", "QQQ"]
    assert metadata["failed_tickers"] == {"EWJ": "download_exception:TypeError"}
    assert metadata["datasetSnapshot"]["source"] == "Yahoo Finance via yfinance"
    assert metadata["datasetSnapshot"]["requestedTickers"] == ["SPY", "EWJ", "QQQ"]
    assert metadata["datasetSnapshot"]["availableTickers"] == ["SPY", "QQQ"]


def test_yfinance_provider_raises_when_too_few_tickers_survive(monkeypatch) -> None:
    provider = YFinanceMarketDataProvider()

    def fake_download(*, tickers, **kwargs):
        if tickers == "SPY":
            return build_download_frame([100.0, 101.0, 102.0, 103.0])
        raise TypeError("network failure")

    monkeypatch.setattr("app.market_data.yf.download", fake_download)

    with pytest.raises(ValueError, match="At least two tickers"):
        provider.fetch_bundle(
            MarketDataRequest(
                tickers=("SPY", "EWJ"),
                period="10y",
                timeframe="1d",
            )
        )


def test_yfinance_provider_uses_explicit_date_range(monkeypatch) -> None:
    provider = YFinanceMarketDataProvider()
    calls = []

    def fake_download(*, tickers, **kwargs):
        calls.append({"tickers": tickers, **kwargs})
        return build_download_frame([100.0, 101.0, 102.0, 103.0])

    monkeypatch.setattr("app.market_data.yf.download", fake_download)

    _bundle, metadata = provider.fetch_bundle(
        MarketDataRequest(
            tickers=("SPY", "QQQ"),
            period="2015_2025",
            timeframe="1d",
            start_date="2015-01-01",
            end_date="2025-12-31",
        )
    )

    assert all(call["start"] == "2015-01-01" for call in calls)
    assert all(call["end"] == "2026-01-01" for call in calls)
    assert all("period" not in call for call in calls)
    assert metadata["period"] == "2015_2025"
    assert metadata["start_date"] == "2015-01-01"
    assert metadata["end_date"] == "2025-12-31"


def test_yfinance_provider_aligns_to_union_calendar(monkeypatch) -> None:
    provider = YFinanceMarketDataProvider()

    def fake_download(*, tickers, **kwargs):
        if tickers == "SPY":
            return pd.DataFrame(
                {"Close": [100.0, 101.0, 102.0], "Volume": [1000.0, 1100.0, 1200.0]},
                index=pd.to_datetime(["2025-01-03", "2025-01-10", "2025-01-17"]),
            )
        if tickers == "BTC-USD":
            return pd.DataFrame(
                {"Close": [200.0, 202.0, 204.0], "Volume": [2000.0, 2100.0, 2200.0]},
                index=pd.to_datetime(["2025-01-02", "2025-01-09", "2025-01-16"]),
            )
        raise AssertionError(f"unexpected ticker: {tickers}")

    monkeypatch.setattr("app.market_data.yf.download", fake_download)

    bundle, metadata = provider.fetch_bundle(
        MarketDataRequest(
            tickers=("SPY", "BTC-USD"),
            period="2015_2025",
            timeframe="1wk",
            start_date="2015-01-01",
            end_date="2025-12-31",
        )
    )

    assert list(bundle["closes"].index) == [
        "2025-01-02",
        "2025-01-03",
        "2025-01-09",
        "2025-01-10",
        "2025-01-16",
        "2025-01-17",
    ]
    assert list(bundle["closes"].columns) == ["SPY", "BTC-USD"]
    assert pd.isna(bundle["closes"].loc["2025-01-02", "SPY"])
    assert float(bundle["closes"].loc["2025-01-10", "BTC-USD"]) == 202.0
    assert metadata["assetAvailability"]["SPY"]["firstValidDate"] == "2025-01-03"
    assert metadata["assetAvailability"]["BTC-USD"]["firstValidDate"] == "2025-01-02"
    assert metadata["aligned_start_date"] == "2025-01-02"
    assert metadata["aligned_end_date"] == "2025-01-17"


def test_yfinance_provider_keeps_rows_before_late_asset_starts(monkeypatch) -> None:
    provider = YFinanceMarketDataProvider()

    def fake_download(*, tickers, **kwargs):
        if tickers == "SPY":
            return pd.DataFrame(
                {
                    "Close": [100.0, 101.0, 102.0, 103.0],
                    "Volume": [1000.0, 1100.0, 1200.0, 1300.0],
                },
                index=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]),
            )
        if tickers == "QQQ":
            return pd.DataFrame(
                {
                    "Close": [200.0, 202.0, 204.0, 206.0],
                    "Volume": [2000.0, 2100.0, 2200.0, 2300.0],
                },
                index=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]),
            )
        if tickers == "ETH-USD":
            return pd.DataFrame(
                {"Close": [300.0, 303.0], "Volume": [3000.0, 3300.0]},
                index=pd.to_datetime(["2025-01-03", "2025-01-04"]),
            )
        raise AssertionError(f"unexpected ticker: {tickers}")

    monkeypatch.setattr("app.market_data.yf.download", fake_download)

    bundle, metadata = provider.fetch_bundle(
        MarketDataRequest(
            tickers=("SPY", "QQQ", "ETH-USD"),
            period="2015_2025",
            timeframe="1d",
            start_date="2025-01-01",
            end_date="2025-01-04",
        )
    )

    assert list(bundle["closes"].index) == [
        "2025-01-01",
        "2025-01-02",
        "2025-01-03",
        "2025-01-04",
    ]
    assert pd.isna(bundle["closes"].loc["2025-01-01", "ETH-USD"])
    assert pd.isna(bundle["closes"].loc["2025-01-02", "ETH-USD"])
    assert float(bundle["closes"].loc["2025-01-03", "ETH-USD"]) == 300.0
    assert metadata["assetAvailability"]["ETH-USD"]["firstValidDate"] == "2025-01-03"
    assert metadata["aligned_start_date"] == "2025-01-01"


def test_dataset_snapshot_fingerprint_excludes_created_at_utc() -> None:
    first_snapshot = build_dataset_snapshot_metadata(
        source="toy",
        timeframe="1d",
        period="toy_period",
        start_date="2025-01-01",
        end_date="2025-01-07",
        requested_tickers=["SPY", "QQQ", "EWJ"],
        available_tickers=["SPY", "QQQ"],
        row_count=7,
        adjustment_policy="toy_adjusted",
        created_at_utc="2026-01-01T00:00:00Z",
    )
    second_snapshot = build_dataset_snapshot_metadata(
        source="toy",
        timeframe="1d",
        period="toy_period",
        start_date="2025-01-01",
        end_date="2025-01-07",
        requested_tickers=["SPY", "QQQ", "EWJ"],
        available_tickers=["SPY", "QQQ"],
        row_count=7,
        adjustment_policy="toy_adjusted",
        created_at_utc="2026-01-02T00:00:00Z",
    )

    assert first_snapshot["createdAtUtc"] != second_snapshot["createdAtUtc"]
    assert first_snapshot["fingerprint"] == second_snapshot["fingerprint"]
    assert first_snapshot["snapshotId"] == first_snapshot["fingerprint"]
