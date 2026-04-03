from fastapi.testclient import TestClient

from app.main import app
from app.strategy import PricePoint


client = TestClient(app)


def fake_fetch_market_prices(ticker: str, period: str) -> tuple[list[PricePoint], dict[str, str]]:
    if ticker == "QQQ":
        return (
            [
                PricePoint(date="2025-01-01", close=100),
                PricePoint(date="2025-01-02", close=94),
                PricePoint(date="2025-01-03", close=99),
                PricePoint(date="2025-01-04", close=101),
                PricePoint(date="2025-01-05", close=100),
            ],
            {"ticker": ticker, "period": period, "source": "test"},
        )

    return (
        [
            PricePoint(date="2025-01-01", close=100),
            PricePoint(date="2025-01-02", close=95),
            PricePoint(date="2025-01-03", close=97),
            PricePoint(date="2025-01-04", close=98),
            PricePoint(date="2025-01-05", close=96),
        ],
        {"ticker": ticker, "period": period, "source": "test"},
    )


def test_healthcheck() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_market_backtest_endpoint(monkeypatch) -> None:
    monkeypatch.setattr("app.main.fetch_market_prices", fake_fetch_market_prices)

    response = client.get(
        "/api/backtest",
        params={"ticker": "SPY", "period": "2y", "threshold": 0.03, "initial_capital": 5000, "holding_days": 2},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["config"]["initialCapital"] == 5000
    assert payload["summary"]["config"]["holdingDays"] == 2
    assert payload["dataset"]["ticker"] == "SPY"
    assert len(payload["series"]) == 5


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
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["config"]["initialCapital"] == 5000
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
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["config"]["transactionCostPct"] == 0.1
    assert len(payload["results"]) == 2
