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
