import pandas as pd
import pytest

from app.market_data import MarketDataRequest, YFinanceMarketDataProvider


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


def test_yfinance_provider_aligns_to_base_calendar(monkeypatch) -> None:
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

    assert list(bundle["closes"].index) == ["2025-01-03", "2025-01-10", "2025-01-17"]
    assert list(bundle["closes"].columns) == ["SPY", "BTC-USD"]
    assert float(bundle["closes"].loc["2025-01-10", "BTC-USD"]) == 202.0
    assert metadata["aligned_start_date"] == "2025-01-03"
    assert metadata["aligned_end_date"] == "2025-01-17"
