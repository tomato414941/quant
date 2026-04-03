from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_healthcheck() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_demo_backtest_endpoint() -> None:
    response = client.get("/api/backtest/demo", params={"threshold": 0.03, "initial_capital": 5000})

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["config"]["initialCapital"] == 5000
    assert len(payload["series"]) == 260
