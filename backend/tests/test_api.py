import copy

import pandas as pd
from fastapi.testclient import TestClient

from app import main as main_module
from app.main import app
from app.study_models import ConditionVariant


client = TestClient(app)


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


def fake_fetch_market_universe_bundle(tickers: list[str], period: str) -> tuple[dict, dict]:
    closes, metadata = fake_fetch_market_universe(tickers, period)
    volumes = pd.DataFrame(
        {
            ticker: [1_000_000 + index * 10_000 + offset * 1_000 for index in range(len(closes))]
            for offset, ticker in enumerate(closes.columns)
        },
        index=closes.index,
    )
    return {"closes": closes, "volumes": volumes}, metadata


def test_healthcheck() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_dashboard_endpoint(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.fetch_market_universe_bundle", fake_fetch_market_universe_bundle)
    config = copy.deepcopy(main_module.DEFAULT_DASHBOARD_CONFIG)
    config.result_store_dir = str(tmp_path / "run_results")
    monkeypatch.setattr(main_module, "DEFAULT_DASHBOARD_CONFIG", config)

    response = client.get("/api/dashboard")

    assert response.status_code == 200
    payload = response.json()
    assert payload["study"]["id"] == "etf_portfolio_models_10y"
    assert payload["study"]["selectionPolicy"]["primaryMetric"] == "sharpe_ratio"
    assert payload["study"]["evaluationContext"]["datasetContext"]["source"] == "test"
    assert payload["study"]["evaluationContext"]["kind"] == "evaluation_context"
    assert payload["study"]["evaluationContext"]["schemaVersion"] == "v1"
    assert payload["study"]["evaluationContext"]["datasetContext"]["period"] == "10y"
    assert payload["study"]["evaluationContext"]["datasetContext"]["sanityPeriods"] == ["3y"]
    assert payload["study"]["evaluationContext"]["datasetContext"]["alignedStartDate"] == "2025-01-01"
    assert payload["study"]["evaluationContext"]["datasetContext"]["alignedEndDate"] == "2025-01-07"
    assert payload["study"]["evaluationContext"]["datasetContext"]["rowCount"] == 7
    assert payload["study"]["strategyDefinitions"][0]["components"]["core"]["dataResolution"]["label"] == "daily"
    assert payload["study"]["strategyDefinitions"][0]["kind"] == "strategy_definition"
    assert payload["study"]["strategyDefinitions"][0]["schemaVersion"] == "v1"
    assert payload["study"]["evaluationContext"]["costAssumptions"]["commissionPct"] == 0.05
    assert payload["study"]["evaluationContext"]["evaluationSettings"]["splitRatioPct"] == 70.0
    assert payload["study"]["initialPortfolioState"]["weights"][0]["asset"] == "CASH"
    assert payload["study"]["initialPortfolioState"]["weights"][0]["weightPct"] == 15.0
    assert len(payload["study"]["strategyDefinitions"]) == 25
    assert payload["study"]["strategyDefinitions"][0]["components"]["core"]["investmentUniverse"]["label"]
    assert payload["study"]["strategyDefinitions"][0]["components"]["optional"]["assetRankingModel"] is None
    assert "filterRules" in payload["study"]["strategyDefinitions"][0]["components"]["optional"]
    assert (
        payload["study"]["strategyDefinitions"][5]["components"]["optional"]["assetRankingModel"]["parameters"]["windowDays"]
        == 252.0
    )
    assert (
        payload["study"]["strategyDefinitions"][5]["components"]["optional"]["tiltRule"]["parameters"]["strength"]
        == 0.35
    )
    assert payload["study"]["marketUniverse"]["assetCount"] == 20
    assert len(payload["study"]["marketUniverse"]["tickers"]) == 20
    assert payload["runs"][0]["splitAnalysis"]["config"]["splitRatioPct"] == 70.0
    assert payload["runs"][0]["kind"] == "run_result"
    assert payload["runs"][0]["schemaVersion"] == "v1"
    assert payload["runs"][0]["strategy"]["components"]["core"]["investmentUniverse"]["label"] == "20資産マルチアセット"
    assert payload["runs"][0]["strategy"]["components"]["core"]["portfolioModel"]["label"] == "等金額配分"
    assert payload["comparisonSeries"][0]["date"] == "2025-01-02"
    assert payload["runs"][0]["strategy"]["components"]["core"]["executionPolicy"]["label"] == "年次"
    assert payload["runStoreSummary"]["cachedRunCount"] == 0
    assert payload["runStoreSummary"]["computedRunCount"] == 50
    assert len(payload["sanityChecks"]) == 1
    assert payload["sanityChecks"][0]["period"] == "3y"
    assert payload["sanityChecks"][0]["evaluationContext"]["datasetContext"]["alignedStartDate"] == "2025-01-01"
    assert payload["sanityChecks"][0]["runStoreSummary"]["cachedRunCount"] == 0
    assert payload["sanityChecks"][0]["runStoreSummary"]["computedRunCount"] == 25
    assert len(payload["sanityChecks"][0]["runs"]) == 25

    second_response = client.get("/api/dashboard")

    assert second_response.status_code == 200
    second_payload = second_response.json()
    assert second_payload["runStoreSummary"]["cachedRunCount"] == 50
    assert second_payload["runStoreSummary"]["computedRunCount"] == 0
    assert second_payload["sanityChecks"][0]["runStoreSummary"]["cachedRunCount"] == 25
    assert second_payload["sanityChecks"][0]["runStoreSummary"]["computedRunCount"] == 0


def test_dashboard_reuses_existing_runs_when_strategy_added(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.fetch_market_universe_bundle", fake_fetch_market_universe_bundle)
    base_config = copy.deepcopy(main_module.DEFAULT_DASHBOARD_CONFIG)
    config = copy.deepcopy(base_config)
    config.result_store_dir = str(tmp_path / "run_results")
    config.dataset_spec.sanity_periods = []
    config.strategy_definitions = [config.strategy_definitions[0]]
    monkeypatch.setattr(main_module, "DEFAULT_DASHBOARD_CONFIG", config)

    first_response = client.get("/api/dashboard")

    assert first_response.status_code == 200
    first_payload = first_response.json()
    assert len(first_payload["runs"]) == 1
    assert first_payload["runStoreSummary"]["cachedRunCount"] == 0
    assert first_payload["runStoreSummary"]["computedRunCount"] == 1

    config.strategy_definitions.append(copy.deepcopy(base_config.strategy_definitions[1]))

    second_response = client.get("/api/dashboard")

    assert second_response.status_code == 200
    second_payload = second_response.json()
    assert len(second_payload["runs"]) == 2
    assert second_payload["runStoreSummary"]["cachedRunCount"] == 1
    assert second_payload["runStoreSummary"]["computedRunCount"] == 1


def test_condition_sweep_reuses_existing_runs_when_condition_added(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.fetch_market_universe_bundle", fake_fetch_market_universe_bundle)
    config = copy.deepcopy(main_module.DEFAULT_DASHBOARD_CONFIG)
    config.result_store_dir = str(tmp_path / "run_results")
    config.dataset_spec.sanity_periods = []
    config.strategy_definitions = [config.strategy_definitions[0]]
    config.condition_variants = [
        ConditionVariant(
            key="baseline",
            label="手数料 0.10% / 投資 85% / 上限なし",
            commission_pct=0.1,
            max_investment_ratio=0.85,
            max_weight=None,
        ),
        ConditionVariant(
            key="cash_80",
            label="手数料 0.10% / 投資 80% / 上限なし",
            commission_pct=0.1,
            max_investment_ratio=0.8,
            max_weight=None,
        ),
    ]
    monkeypatch.setattr(main_module, "DEFAULT_DASHBOARD_CONFIG", config)

    first_response = client.get("/api/condition-sweep")

    assert first_response.status_code == 200
    first_payload = first_response.json()
    assert first_payload["resultCount"] == 2
    assert first_payload["runStoreSummary"]["cachedRunCount"] == 0
    assert first_payload["runStoreSummary"]["computedRunCount"] == 2
    assert len(first_payload["conditionVariants"]) == 2
    assert first_payload["results"][0]["conditionVariant"]["label"]

    second_response = client.get("/api/condition-sweep")

    assert second_response.status_code == 200
    second_payload = second_response.json()
    assert second_payload["runStoreSummary"]["cachedRunCount"] == 2
    assert second_payload["runStoreSummary"]["computedRunCount"] == 0

    config.condition_variants.append(
        ConditionVariant(
            key="cap_45",
            label="手数料 0.10% / 投資 85% / 45%上限",
            commission_pct=0.1,
            max_investment_ratio=0.85,
            max_weight=0.45,
        )
    )

    third_response = client.get("/api/condition-sweep")

    assert third_response.status_code == 200
    third_payload = third_response.json()
    assert third_payload["resultCount"] == 3
    assert third_payload["runStoreSummary"]["cachedRunCount"] == 2
    assert third_payload["runStoreSummary"]["computedRunCount"] == 1


def test_run_catalog_endpoint(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.fetch_market_universe_bundle", fake_fetch_market_universe_bundle)
    config = copy.deepcopy(main_module.DEFAULT_DASHBOARD_CONFIG)
    config.result_store_dir = str(tmp_path / "run_results")
    monkeypatch.setattr(main_module, "DEFAULT_DASHBOARD_CONFIG", config)

    dashboard_response = client.get("/api/dashboard")
    assert dashboard_response.status_code == 200

    response = client.get("/api/run-catalog", params={"limit": 5, "run_kind": "dashboard"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["studyId"] == "etf_portfolio_models_10y"
    assert payload["limit"] == 5
    assert payload["runKind"] == "dashboard"
    assert payload["recordCount"] == 5
    assert payload["records"][0]["strategyLabel"]
    assert payload["records"][0]["investmentUniverseLabel"]
    assert payload["records"][0]["portfolioModelLabel"]
    assert payload["records"][0]["commissionPct"] is not None
    assert payload["records"][0]["sharpeRatio"] is not None
    assert payload["records"][0]["generationMethod"] is None


def test_generate_parameter_sweep_runs_endpoint(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.fetch_market_universe_bundle", fake_fetch_market_universe_bundle)
    config = copy.deepcopy(main_module.DEFAULT_DASHBOARD_CONFIG)
    config.result_store_dir = str(tmp_path / "run_results")
    monkeypatch.setattr(main_module, "DEFAULT_DASHBOARD_CONFIG", config)

    response = client.post("/api/runs/generate-parameter-sweep")

    assert response.status_code == 200
    payload = response.json()
    assert payload["study"]["id"] == "etf_portfolio_models_10y"
    assert payload["generation"]["method"] == "parameter_sweep"
    assert payload["generation"]["batchKey"] == "local_tilt_search_v1"
    assert payload["generation"]["spec"]["parameterGrid"]["tiltStrength"] == [0.15, 0.2, 0.25, 0.3, 0.35]
    assert payload["resultCount"] == 125
    assert payload["runStoreSummary"]["cachedRunCount"] == 0
    assert payload["runStoreSummary"]["computedRunCount"] == 125
    assert payload["results"][0]["family"]["label"]
    assert payload["results"][0]["parameterSet"]["tiltStrength"] in [0.15, 0.2, 0.25, 0.3, 0.35]
    assert payload["results"][0]["parameterSet"]["maxWeightPct"] in [40.0, 42.5, 45.0, 47.5, 50.0]
    assert payload["results"][0]["summary"]["sharpeRatio"] is not None

    second_response = client.post("/api/runs/generate-parameter-sweep")

    assert second_response.status_code == 200
    second_payload = second_response.json()
    assert second_payload["resultCount"] == 125
    assert second_payload["runStoreSummary"]["cachedRunCount"] == 125
    assert second_payload["runStoreSummary"]["computedRunCount"] == 0

    catalog_response = client.get(
        "/api/run-catalog",
        params={"generation_method": "parameter_sweep", "limit": 5},
    )

    assert catalog_response.status_code == 200
    catalog_payload = catalog_response.json()
    assert catalog_payload["generationMethod"] == "parameter_sweep"
    assert catalog_payload["recordCount"] == 5
    assert all(record["generationMethod"] == "parameter_sweep" for record in catalog_payload["records"])


def test_ranking_evaluation_endpoint(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.fetch_market_universe_bundle", fake_fetch_market_universe_bundle)
    config = copy.deepcopy(main_module.DEFAULT_DASHBOARD_CONFIG)
    config.result_store_dir = str(tmp_path / "run_results")
    monkeypatch.setattr(main_module, "DEFAULT_DASHBOARD_CONFIG", config)

    response = client.get("/api/ranking-evaluation")

    assert response.status_code == 200
    payload = response.json()
    assert payload["study"]["id"] == "etf_portfolio_models_10y"
    assert payload["resultCount"] == 12
    assert payload["runStoreSummary"]["cachedRunCount"] == 0
    assert payload["runStoreSummary"]["computedRunCount"] == 12
    assert payload["results"][0]["rankingDefinition"]["rankingModel"]["label"]
    assert payload["results"][0]["overall"]["observationCount"] >= 1
    assert payload["results"][0]["overall"]["meanTopMinusBottomPct"] is not None

    second_response = client.get("/api/ranking-evaluation")

    assert second_response.status_code == 200
    second_payload = second_response.json()
    assert second_payload["runStoreSummary"]["cachedRunCount"] == 12
    assert second_payload["runStoreSummary"]["computedRunCount"] == 0
