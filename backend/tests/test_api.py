import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.strategy import PricePoint


client = TestClient(app)


def fake_fetch_market_prices(ticker: str, period: str) -> tuple[list[PricePoint], dict[str, str]]:
    if ticker == "QQQ":
        return (
            [
                PricePoint(date="2025-01-01", close=100),
                PricePoint(date="2025-01-02", close=104),
                PricePoint(date="2025-01-03", close=108),
                PricePoint(date="2025-01-04", close=109),
                PricePoint(date="2025-01-05", close=110),
            ],
            {"ticker": ticker, "period": period, "source": "test"},
        )

    if period == "6mo":
        prices = [
            PricePoint(date="2025-01-01", close=100),
            PricePoint(date="2025-01-02", close=95),
            PricePoint(date="2025-01-03", close=97),
            PricePoint(date="2025-01-04", close=98),
            PricePoint(date="2025-01-05", close=96),
        ]
    else:
        prices = [
            PricePoint(date="2025-01-01", close=100),
            PricePoint(date="2025-01-02", close=95),
            PricePoint(date="2025-01-03", close=97),
            PricePoint(date="2025-01-04", close=98),
            PricePoint(date="2025-01-05", close=96),
            PricePoint(date="2025-01-06", close=99),
        ]

    return prices, {"ticker": ticker, "period": period, "source": "test"}


def fake_fetch_market_universe(tickers: list[str], period: str) -> tuple[pd.DataFrame, dict]:
    if period == "3y":
        frame = pd.DataFrame(
            {
                "SPY": [100, 102, 104, 103, 105, 107, 108],
                "QQQ": [100, 104, 107, 109, 111, 114, 116],
                "IWM": [100, 101, 102, 102, 103, 104, 105],
                "EFA": [100, 101, 103, 104, 105, 106, 108],
                "EEM": [100, 99, 101, 102, 104, 105, 106],
                "EWJ": [100, 100, 101, 102, 103, 104, 105],
                "EWZ": [100, 98, 100, 103, 105, 106, 108],
                "VNQ": [100, 101, 103, 102, 104, 105, 106],
                "TLT": [100, 99, 98, 99, 100, 101, 102],
                "IEF": [100, 100, 100, 101, 101, 102, 102],
                "LQD": [100, 100, 101, 102, 102, 103, 104],
                "HYG": [100, 101, 102, 103, 104, 105, 106],
                "TIP": [100, 100, 101, 101, 102, 103, 104],
                "GLD": [100, 100, 101, 102, 102, 103, 104],
                "SLV": [100, 101, 103, 104, 105, 107, 108],
                "DBC": [100, 101, 100, 102, 103, 104, 105],
                "USO": [100, 103, 101, 104, 106, 108, 109],
                "UUP": [100, 99, 99, 100, 101, 101, 102],
                "BTC-USD": [100, 106, 108, 111, 113, 117, 119],
                "ETH-USD": [100, 107, 109, 114, 118, 121, 124],
            },
            index=[
                "2025-01-01",
                "2025-01-02",
                "2025-01-03",
                "2025-01-04",
                "2025-01-05",
                "2025-01-06",
                "2025-01-07",
            ],
        )
        aligned = frame[tickers]
        return aligned, {
            "tickers": tickers,
            "period": period,
            "source": "test",
            "aligned_start_date": aligned.index[0],
            "aligned_end_date": aligned.index[-1],
            "row_count": len(aligned),
        }

    frame = pd.DataFrame(
        {
            "SPY": [100, 101, 103, 102, 104, 106, 107],
            "QQQ": [100, 103, 105, 107, 108, 110, 112],
            "IWM": [100, 99, 100, 101, 103, 102, 104],
            "EFA": [100, 101, 102, 103, 104, 105, 106],
            "EEM": [100, 98, 99, 100, 101, 102, 103],
            "EWJ": [100, 100, 101, 102, 102, 103, 104],
            "EWZ": [100, 97, 98, 100, 102, 104, 105],
            "VNQ": [100, 101, 102, 101, 103, 104, 105],
            "TLT": [100, 100, 99, 100, 101, 102, 101],
            "IEF": [100, 100, 100, 100, 101, 101, 102],
            "LQD": [100, 100, 101, 101, 102, 102, 103],
            "HYG": [100, 101, 102, 102, 103, 104, 105],
            "TIP": [100, 100, 100, 101, 101, 102, 103],
            "GLD": [100, 101, 100, 102, 103, 104, 105],
            "SLV": [100, 101, 102, 103, 104, 105, 106],
            "DBC": [100, 99, 100, 101, 102, 103, 104],
            "USO": [100, 101, 100, 102, 104, 105, 107],
            "UUP": [100, 99, 99, 100, 100, 101, 102],
            "BTC-USD": [100, 105, 107, 110, 112, 115, 118],
            "ETH-USD": [100, 106, 108, 112, 115, 119, 122],
        },
        index=[
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
            "2025-01-04",
            "2025-01-05",
            "2025-01-06",
            "2025-01-07",
        ],
    )
    aligned = frame[tickers]
    return aligned, {
        "tickers": tickers,
        "period": period,
        "source": "test",
        "aligned_start_date": aligned.index[0],
        "aligned_end_date": aligned.index[-1],
        "row_count": len(aligned),
    }


def test_healthcheck() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_dashboard_endpoint(monkeypatch) -> None:
    monkeypatch.setattr("app.main.fetch_market_universe", fake_fetch_market_universe)

    response = client.get("/api/dashboard")

    assert response.status_code == 200
    payload = response.json()
    assert payload["study"]["id"] == "etf_portfolio_models_10y"
    assert payload["study"]["datasetSpec"]["source"] == "test"
    assert payload["study"]["datasetSpec"]["period"] == "10y"
    assert payload["study"]["datasetSpec"]["sanityPeriods"] == ["3y"]
    assert payload["study"]["datasetSpec"]["alignedStartDate"] == "2025-01-01"
    assert payload["study"]["datasetSpec"]["alignedEndDate"] == "2025-01-07"
    assert payload["study"]["datasetSpec"]["rowCount"] == 7
    assert payload["study"]["executionModel"]["commissionPct"] == 0.1
    assert payload["study"]["executionModel"]["rebalanceFrequency"] == "monthly"
    assert payload["study"]["backtestConfig"]["maxInvestmentPct"] == 85.0
    assert len(payload["study"]["strategyDefinitions"]) == 2
    assert len(payload["study"]["datasetSpec"]["tickers"]) == 20
    assert len(payload["study"]["portfolioModels"]) == 4
    assert payload["runs"][0]["splitAnalysis"]["config"]["splitRatioPct"] == 70.0
    assert payload["runs"][0]["strategy"]["label"] == "全資産"
    assert payload["runs"][0]["portfolioModel"]["label"] == "等金額配分"
    assert payload["comparisonSeries"][0]["date"] == "2025-01-02"
    assert len(payload["sanityChecks"]) == 1
    assert payload["sanityChecks"][0]["period"] == "3y"
    assert payload["sanityChecks"][0]["datasetSpec"]["alignedStartDate"] == "2025-01-01"
    assert len(payload["sanityChecks"][0]["runs"]) == 8


def test_market_backtest_endpoint(monkeypatch) -> None:
    monkeypatch.setattr("app.main.fetch_market_prices", fake_fetch_market_prices)

    response = client.get(
        "/api/backtest",
        params={
            "ticker": "SPY",
            "period": "2y",
            "threshold": 0.03,
            "initial_capital": 5000,
            "holding_days": 2,
            "split_ratio": 0.6,
            "strategy": "mean_reversion",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["config"]["initialCapital"] == 5000
    assert payload["summary"]["config"]["holdingDays"] == 2
    assert payload["summary"]["config"]["strategyDefinition"]["engine"] == "mean_reversion"
    assert payload["splitAnalysis"]["train"]["dayCount"] == 3
    assert payload["dataset"]["ticker"] == "SPY"


def test_market_grid_search_endpoint(monkeypatch) -> None:
    monkeypatch.setattr("app.main.fetch_market_prices", fake_fetch_market_prices)

    response = client.get(
        "/api/grid-search",
        params={
            "ticker": "SPY",
            "period": "2y",
            "threshold_values": "2,3,4",
            "holding_days_values": "1,2",
            "initial_capital": 5000,
            "strategy": "momentum",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["config"]["strategyId"] == "momentum"
    assert payload["dataset"]["source"] == "test"
    assert len(payload["results"]) == 6


def test_ticker_compare_endpoint(monkeypatch) -> None:
    monkeypatch.setattr("app.main.fetch_market_prices", fake_fetch_market_prices)

    response = client.get(
        "/api/ticker-compare",
        params={
            "tickers": "SPY,QQQ",
            "period": "2y",
            "threshold": 0.03,
            "holding_days": 1,
            "initial_capital": 5000,
            "transaction_cost": 0.001,
            "strategy": "momentum",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["config"]["transactionCostPct"] == 0.1
    assert payload["config"]["strategyId"] == "momentum"
    assert payload["config"]["strategyDefinition"]["engine"] == "momentum"
    assert len(payload["results"]) == 2


def test_period_compare_endpoint(monkeypatch) -> None:
    monkeypatch.setattr("app.main.fetch_market_prices", fake_fetch_market_prices)

    response = client.get(
        "/api/period-compare",
        params={
            "ticker": "SPY",
            "periods": "6mo,1y",
            "threshold": 0.03,
            "holding_days": 1,
            "initial_capital": 5000,
            "strategy": "mean_reversion",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset"]["ticker"] == "SPY"
    assert [row["period"] for row in payload["results"]] == ["6mo", "1y"]
    assert payload["config"]["strategyDefinition"]["engine"] == "mean_reversion"
