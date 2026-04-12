import copy
from dataclasses import replace

import pandas as pd
from fastapi.testclient import TestClient

from app import main as main_module
from app.comparison_models import ConditionVariant
from app.main import app
from app.portfolio import build_asset_ranking_specs
from app.timeframe_models import build_timeframe_spec


client = TestClient(app)


def fake_fetch_market_universe(
    tickers: list[str],
    period: str,
    timeframe: str = "1d",
) -> tuple[pd.DataFrame, dict]:
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
            "timeframe": timeframe,
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
        "timeframe": timeframe,
        "aligned_start_date": aligned.index[0],
        "aligned_end_date": aligned.index[-1],
        "row_count": len(aligned),
    }


def fake_fetch_market_universe_bundle(
    tickers: list[str],
    period: str,
    timeframe: str = "1d",
) -> tuple[dict, dict]:
    closes, metadata = fake_fetch_market_universe(tickers, period, timeframe)
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
    config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)
    config.result_store_dir = str(tmp_path / "run_results")
    monkeypatch.setattr(main_module, "DEFAULT_COMPARISON_SPEC", config)

    response = client.get("/api/dashboard")

    assert response.status_code == 200
    payload = response.json()
    expected_strategy_count = len(config.candidate_strategies)
    expected_reference_count = len(config.reference_strategies)
    expected_predictor_count = sum(
        1
        for strategy in config.candidate_strategies
        if strategy.predictor_use is not None
    )

    assert payload["comparison"]["comparisonId"] == "etf_portfolio_models_10y"
    assert payload["comparison"]["selectionPolicy"]["primaryMetric"] == "sharpe_ratio"
    assert payload["comparison"]["runSpec"]["kind"] == "comparison_run_spec"
    assert payload["comparison"]["runSpec"]["executionAssumptions"]["kind"] == "close_execution_assumptions"
    assert payload["comparison"]["runSpec"]["executionAssumptions"]["parameters"]["fillPrice"] == "close"
    assert payload["comparison"]["runSpec"]["executionAssumptions"]["costModel"]["kind"] == "asset_specific_adv_cost"
    assert (
        payload["comparison"]["runSpec"]["executionAssumptions"]["costModel"]["parameters"]["commissionPct"]
        == 0.05
    )
    assert (
        payload["comparison"]["runSpec"]["executionAssumptions"]["costModel"]["parameters"]["slippagePct"]
        == 0.02
    )
    assert (
        payload["comparison"]["runSpec"]["executionAssumptions"]["costModel"]["parameters"]["impactCoefficientPct"]
        == 0.08
    )
    assert (
        payload["comparison"]["runSpec"]["executionAssumptions"]["costModel"]["parameters"]["advWindowBars"]
        == 20.0
    )
    assert payload["comparison"]["runSpec"]["evaluation"]["kind"] == "evaluation_spec"
    assert payload["comparison"]["runSpec"]["evaluation"]["schemaVersion"] == "v1"
    assert {
        context["timeframe"]["key"]
        for context in payload["comparison"]["runSpec"]["evaluation"]["marketDataContexts"]
    } == {"1d", "1w", "1mo"}
    assert {
        context["source"]
        for context in payload["comparison"]["runSpec"]["evaluation"]["marketDataContexts"]
    } == {"test"}
    assert {
        context["period"]
        for context in payload["comparison"]["runSpec"]["evaluation"]["marketDataContexts"]
    } == {"10y"}
    assert all(context["sanityPeriods"] == ["3y"] for context in payload["comparison"]["runSpec"]["evaluation"]["marketDataContexts"])
    assert all(context["alignedStartDate"] == "2025-01-01" for context in payload["comparison"]["runSpec"]["evaluation"]["marketDataContexts"])
    assert all(context["alignedEndDate"] == "2025-01-07" for context in payload["comparison"]["runSpec"]["evaluation"]["marketDataContexts"])
    assert all(context["rowCount"] == 7 for context in payload["comparison"]["runSpec"]["evaluation"]["marketDataContexts"])
    assert {timeframe["key"] for timeframe in payload["comparison"]["runSpec"]["marketSlice"]["timeframes"]} == {"1d", "1w", "1mo"}
    assert payload["comparison"]["runSpec"]["marketSlice"]["fields"] == ["close", "volume"]
    assert (
        payload["comparison"]["runSpec"]["evaluation"]["evaluationSettings"]["splitRatioPct"]
        == 70.0
    )
    assert payload["comparison"]["runSpec"]["capitalBase"] == 10000
    assert payload["comparison"]["runSpec"]["portfolioState"]["weights"][0]["asset"] == "CASH"
    assert payload["comparison"]["runSpec"]["portfolioState"]["weights"][0]["weightPct"] == 15.0
    assert len(payload["comparison"]["candidateStrategies"]) == expected_strategy_count
    assert len(payload["comparison"]["referenceStrategies"]) == expected_reference_count
    assert payload["comparison"]["candidateStrategies"][0]["kind"] == "strategy_spec"
    assert payload["comparison"]["candidateStrategies"][0]["schemaVersion"] == "v1"
    assert payload["comparison"]["candidateStrategies"][0]["components"]["core"]["investmentUniverse"]["label"]
    assert (
        payload["comparison"]["candidateStrategies"][0]["components"]["core"]["executionPolicy"]["rebalanceSchedule"]
        == "year_end"
    )
    assert (
        payload["comparison"]["candidateStrategies"][0]["components"]["optional"]["assetRankingModel"]
        is None
    )
    assert "filterRules" in payload["comparison"]["candidateStrategies"][0]["components"]["optional"]
    assert (
        payload["comparison"]["candidateStrategies"][5]["components"]["optional"]["assetRankingModel"][
            "parameters"
        ]["windowSpec"]["unit"]
        == "months"
    )
    assert (
        payload["comparison"]["candidateStrategies"][5]["components"]["optional"]["assetRankingModel"][
            "parameters"
        ]["windowSpec"]["value"]
        == 12
    )
    assert (
        payload["comparison"]["candidateStrategies"][5]["components"]["optional"]["tiltRule"][
            "parameters"
        ]["strength"]
        == 0.35
    )
    assert any(
        strategy["components"]["optional"]["predictor"] is not None
        for strategy in payload["comparison"]["candidateStrategies"]
    )
    assert payload["comparison"]["marketUniverse"]["assetCount"] == 20
    assert len(payload["comparison"]["marketUniverse"]["tickers"]) == 20

    assert len(payload["candidateRuns"]) == expected_strategy_count
    assert len(payload["referenceRuns"]) == expected_reference_count
    assert len(payload["predictorRuns"]) == expected_predictor_count
    assert {
        run["strategy"]["components"]["core"]["dataResolution"]["key"]
        for run in payload["candidateRuns"]
    } == {"1d", "1w", "1mo"}
    assert payload["candidateRuns"][0]["splitAnalysis"]["config"]["splitRatioPct"] == 70.0
    assert payload["candidateRuns"][0]["kind"] == "run_result"
    assert payload["candidateRuns"][0]["schemaVersion"] == "v1"
    assert (
        payload["candidateRuns"][0]["strategy"]["components"]["core"]["investmentUniverse"]["label"]
        == "20資産マルチアセット"
    )
    assert payload["referenceRuns"][0]["strategy"]["label"] == "等金額買い持ち + CASH"
    assert payload["runStoreSummary"]["cachedRunCount"] == 0
    assert payload["runStoreSummary"]["computedRunCount"] == (
        (expected_strategy_count + expected_reference_count) * 2
        + expected_predictor_count * 2
    )
    assert len(payload["sanityChecks"]) == 1
    assert payload["sanityChecks"][0]["period"] == "3y"
    assert all(
        context["alignedStartDate"] == "2025-01-01"
        for context in payload["sanityChecks"][0]["evaluation"]["marketDataContexts"]
    )
    assert payload["sanityChecks"][0]["runStoreSummary"]["cachedRunCount"] == 0
    assert (
        payload["sanityChecks"][0]["runStoreSummary"]["computedRunCount"]
        == expected_strategy_count + expected_reference_count + expected_predictor_count
    )
    assert len(payload["sanityChecks"][0]["predictorRuns"]) == expected_predictor_count
    assert len(payload["sanityChecks"][0]["candidateRuns"]) == expected_strategy_count
    assert len(payload["sanityChecks"][0]["referenceRuns"]) == expected_reference_count
    assert payload["candidateRuns"][0]["splitAnalysis"]["train"]["barCount"] > 0
    assert payload["candidateRuns"][0]["splitAnalysis"]["test"]["barCount"] > 0

    second_response = client.get("/api/dashboard")

    assert second_response.status_code == 200
    second_payload = second_response.json()
    assert second_payload["runStoreSummary"]["cachedRunCount"] == (
        (expected_strategy_count + expected_reference_count) * 2
        + expected_predictor_count * 2
    )


def test_predictor_runs_endpoint(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.fetch_market_universe_bundle", fake_fetch_market_universe_bundle)
    config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)
    config.result_store_dir = str(tmp_path / "run_results")
    monkeypatch.setattr(main_module, "DEFAULT_COMPARISON_SPEC", config)

    response = client.post("/api/predictor-runs")

    assert response.status_code == 200
    payload = response.json()
    expected_predictor_count = len(main_module.REGISTERED_PREDICTOR_SPECS)
    assert payload["kind"] == "predictor_run_collection"
    assert payload["comparisonId"] == "etf_portfolio_models_10y"
    assert payload["runSpec"]["kind"] == "comparison_run_spec"
    assert len(payload["predictorSpecs"]) == expected_predictor_count
    assert len(payload["predictorRuns"]) == expected_predictor_count
    assert payload["runStoreSummary"]["cachedRunCount"] == 0
    assert payload["runStoreSummary"]["computedRunCount"] == expected_predictor_count * 2
    assert len(payload["sanityChecks"]) == 1
    assert len(payload["sanityChecks"][0]["predictorRuns"]) == expected_predictor_count

    index_response = client.get("/api/predictor-runs", params={"limit": 10})
    assert index_response.status_code == 200
    index_payload = index_response.json()
    assert index_payload["kind"] == "predictor_run_index"
    assert index_payload["totalCount"] == expected_predictor_count * 2
    assert index_payload["recordCount"] == min(10, expected_predictor_count * 2)
    assert index_payload["sortBy"] == "test_rank_ic"
    assert index_payload["records"][0]["runKind"] == "predictor_run"
    assert "trainingFitMode" in index_payload["records"][0]
    run_key = index_payload["records"][0]["runKey"]

    detail_response = client.get(f"/api/predictor-runs/{run_key}")
    assert detail_response.status_code == 200
    detail_payload = detail_response.json()
    assert detail_payload["kind"] == "predictor_run_detail"
    assert detail_payload["record"]["runKey"] == run_key
    assert detail_payload["record"]["runSpec"]["runKind"] == "predictor_run"

    filtered_response = client.get(
        "/api/predictor-runs",
        params={
            "learner_kind": "linear_regression",
            "combiner_kind": "learner_only",
            "horizon_value": 5,
            "sort_by": "test_top_minus_bottom",
        },
    )
    assert filtered_response.status_code == 200
    filtered_payload = filtered_response.json()
    assert filtered_payload["filters"]["learnerKind"] == "linear_regression"
    assert filtered_payload["filters"]["combinerKind"] == "learner_only"
    assert filtered_payload["filters"]["horizonValue"] == 5
    assert filtered_payload["sortBy"] == "test_top_minus_bottom"
    assert all(record["learnerKind"] == "linear_regression" for record in filtered_payload["records"])
    assert all(record["combinerKind"] == "learner_only" for record in filtered_payload["records"])
    assert all(record["horizonValue"] == 5 for record in filtered_payload["records"])

    second_response = client.post("/api/predictor-runs")
    assert second_response.status_code == 200
    second_payload = second_response.json()
    assert second_payload["runStoreSummary"]["cachedRunCount"] == expected_predictor_count * 2
    assert second_payload["runStoreSummary"]["computedRunCount"] == 0


def test_strategy_runs_endpoint(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.fetch_market_universe_bundle", fake_fetch_market_universe_bundle)
    config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)
    config.result_store_dir = str(tmp_path / "run_results")
    monkeypatch.setattr(main_module, "DEFAULT_COMPARISON_SPEC", config)

    response = client.post("/api/strategy-runs")

    assert response.status_code == 200
    payload = response.json()
    expected_strategy_count = len(config.candidate_strategies)
    expected_reference_count = len(config.reference_strategies)
    expected_predictor_count = sum(
        1 for strategy in config.candidate_strategies if strategy.predictor_use is not None
    )
    assert payload["kind"] == "strategy_run_collection"
    assert payload["comparisonId"] == "etf_portfolio_models_10y"
    assert payload["runSpec"]["kind"] == "comparison_run_spec"
    assert len(payload["candidateStrategies"]) == expected_strategy_count
    assert len(payload["referenceStrategies"]) == expected_reference_count
    assert len(payload["predictorRuns"]) == expected_predictor_count
    assert len(payload["candidateRuns"]) == expected_strategy_count
    assert len(payload["referenceRuns"]) == expected_reference_count
    assert payload["runStoreSummary"]["cachedRunCount"] == 0
    assert payload["runStoreSummary"]["computedRunCount"] == (
        (expected_strategy_count + expected_reference_count) * 2
        + expected_predictor_count * 2
    )
    assert len(payload["sanityChecks"]) == 1
    assert len(payload["sanityChecks"][0]["predictorRuns"]) == expected_predictor_count

    index_response = client.get("/api/strategy-runs", params={"limit": 10})
    assert index_response.status_code == 200
    index_payload = index_response.json()
    assert index_payload["kind"] == "strategy_run_index"
    assert index_payload["totalCount"] == (expected_strategy_count + expected_reference_count) * 2
    assert index_payload["recordCount"] == 10
    assert index_payload["records"][0]["runKind"] == "strategy_run"
    run_key = index_payload["records"][0]["runKey"]

    detail_response = client.get(f"/api/strategy-runs/{run_key}")
    assert detail_response.status_code == 200
    detail_payload = detail_response.json()
    assert detail_payload["kind"] == "strategy_run_detail"
    assert detail_payload["record"]["runKey"] == run_key
    assert detail_payload["record"]["runSpec"]["runKind"] == "strategy_run"

    second_response = client.post("/api/strategy-runs")
    assert second_response.status_code == 200
    second_payload = second_response.json()
    assert second_payload["runStoreSummary"]["cachedRunCount"] == (
        (expected_strategy_count + expected_reference_count) * 2
        + expected_predictor_count * 2
    )
    assert second_payload["runStoreSummary"]["computedRunCount"] == 0


def test_dashboard_endpoint_supports_mixed_strategy_timeframes(monkeypatch, tmp_path) -> None:
    fetch_calls: list[tuple[str, str]] = []

    def recording_fetch_market_universe_bundle(
        tickers: list[str],
        period: str,
        timeframe: str = "1d",
    ) -> tuple[dict, dict]:
        fetch_calls.append((period, timeframe))
        return fake_fetch_market_universe_bundle(tickers, period, timeframe)

    monkeypatch.setattr(
        "app.main.fetch_market_universe_bundle",
        recording_fetch_market_universe_bundle,
    )
    config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)
    weekly_strategy = replace(
        config.candidate_strategies[0],
        strategy_id="stg-fu-eq-weekly",
        label="全資産 × 等金額配分 × 週次データ",
        timeframe=build_timeframe_spec(
            key="1w",
            label="週次",
            yfinance_interval="1wk",
            bar_seconds=604_800,
            bars_per_year=52,
        ),
    )
    config.candidate_strategies = [config.candidate_strategies[0], weekly_strategy]
    config.reference_strategies = []
    config.result_store_dir = str(tmp_path / "run_results")
    monkeypatch.setattr(main_module, "DEFAULT_COMPARISON_SPEC", config)

    response = client.get("/api/dashboard")

    assert response.status_code == 200
    payload = response.json()
    assert {(period, timeframe) for period, timeframe in fetch_calls} == {
        ("10y", "1d"),
        ("10y", "1wk"),
        ("3y", "1d"),
        ("3y", "1wk"),
    }
    assert [timeframe["key"] for timeframe in payload["comparison"]["runSpec"]["marketSlice"]["timeframes"]] == [
        "1d",
        "1w",
    ]
    assert [context["timeframe"]["key"] for context in payload["comparison"]["runSpec"]["evaluation"]["marketDataContexts"]] == [
        "1d",
        "1w",
    ]
    assert {run["strategy"]["components"]["core"]["dataResolution"]["key"] for run in payload["candidateRuns"]} == {
        "1d",
        "1w",
    }
    second_response = client.get("/api/dashboard")
    assert second_response.status_code == 200
    second_payload = second_response.json()
    assert second_payload["runStoreSummary"]["computedRunCount"] == 0
    assert (
        second_payload["runStoreSummary"]["cachedRunCount"]
        == len(config.candidate_strategies) * 2
    )


def test_dashboard_reuses_existing_runs_when_strategy_added(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.fetch_market_universe_bundle", fake_fetch_market_universe_bundle)
    base_config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)
    config = copy.deepcopy(base_config)
    config.result_store_dir = str(tmp_path / "run_results")
    config.run_spec.market_slice.sanity_periods = []
    config.candidate_strategies = [config.candidate_strategies[0]]
    config.reference_strategies = []
    monkeypatch.setattr(main_module, "DEFAULT_COMPARISON_SPEC", config)

    first_response = client.get("/api/dashboard")

    assert first_response.status_code == 200
    first_payload = first_response.json()
    assert len(first_payload["candidateRuns"]) == 1
    assert len(first_payload["referenceRuns"]) == 0
    assert first_payload["runStoreSummary"]["cachedRunCount"] == 0
    assert first_payload["runStoreSummary"]["computedRunCount"] == 1

    config.candidate_strategies.append(copy.deepcopy(base_config.candidate_strategies[1]))

    second_response = client.get("/api/dashboard")

    assert second_response.status_code == 200
    second_payload = second_response.json()
    assert len(second_payload["candidateRuns"]) == 2
    assert second_payload["runStoreSummary"]["cachedRunCount"] == 1
    assert second_payload["runStoreSummary"]["computedRunCount"] == 1


def test_condition_sweep_reuses_existing_runs_when_condition_added(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.fetch_market_universe_bundle", fake_fetch_market_universe_bundle)
    config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)
    config.result_store_dir = str(tmp_path / "run_results")
    config.run_spec.market_slice.sanity_periods = []
    config.candidate_strategies = [config.candidate_strategies[0]]
    config.reference_strategies = []
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
    monkeypatch.setattr(main_module, "DEFAULT_COMPARISON_SPEC", config)

    first_response = client.get("/api/condition-sweep")

    assert first_response.status_code == 200
    first_payload = first_response.json()
    assert first_payload["comparison"]["comparisonId"] == "etf_portfolio_models_10y"
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
    config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)
    config.result_store_dir = str(tmp_path / "run_results")
    monkeypatch.setattr(main_module, "DEFAULT_COMPARISON_SPEC", config)

    dashboard_response = client.get("/api/dashboard")
    assert dashboard_response.status_code == 200

    response = client.get("/api/run-catalog", params={"limit": 5, "run_kind": "strategy_run"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["comparisonId"] == "etf_portfolio_models_10y"
    assert payload["limit"] == 5
    assert payload["runKind"] == "strategy_run"
    assert payload["recordCount"] == 5
    assert payload["records"][0]["strategyLabel"]
    assert payload["records"][0]["investmentUniverseLabel"]
    assert payload["records"][0]["portfolioModelLabel"]
    assert payload["records"][0]["commissionPct"] is not None
    assert payload["records"][0]["sharpeRatio"] is not None
    assert payload["records"][0]["generationMethod"] is None


def test_generate_parameter_sweep_runs_endpoint(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.fetch_market_universe_bundle", fake_fetch_market_universe_bundle)
    config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)
    config.result_store_dir = str(tmp_path / "run_results")
    monkeypatch.setattr(main_module, "DEFAULT_COMPARISON_SPEC", config)

    response = client.post("/api/runs/generate-parameter-sweep")

    assert response.status_code == 200
    payload = response.json()
    assert payload["comparison"]["comparisonId"] == "etf_portfolio_models_10y"
    assert payload["generation"]["method"] == "parameter_sweep"
    assert payload["generation"]["batchKey"] == "local_tilt_search_9m_v1"
    assert payload["generation"]["spec"]["families"][0]["windowSpec"]["unit"] == "months"
    assert payload["generation"]["spec"]["families"][0]["windowSpec"]["value"] == 9
    assert payload["generation"]["spec"]["parameterGrid"]["tiltStrength"] == [0.15, 0.2, 0.25, 0.3, 0.35]
    assert payload["resultCount"] == 125
    assert payload["runStoreSummary"]["cachedRunCount"] == 0
    assert payload["runStoreSummary"]["computedRunCount"] == 125
    assert payload["results"][0]["family"]["label"]
    assert payload["results"][0]["parameterSet"]["tiltStrength"] in [0.15, 0.2, 0.25, 0.3, 0.35]
    assert payload["results"][0]["parameterSet"]["windowSpec"]["unit"] == "months"
    assert payload["results"][0]["parameterSet"]["windowSpec"]["value"] == 9
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
    config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)
    config.result_store_dir = str(tmp_path / "run_results")
    monkeypatch.setattr(main_module, "DEFAULT_COMPARISON_SPEC", config)

    response = client.get("/api/ranking-evaluation")

    assert response.status_code == 200
    payload = response.json()
    expected_ranking_count = len(build_asset_ranking_specs(config.candidate_strategies))
    assert payload["comparison"]["comparisonId"] == "etf_portfolio_models_10y"
    assert payload["resultCount"] == expected_ranking_count
    assert payload["runStoreSummary"]["cachedRunCount"] == 0
    assert payload["runStoreSummary"]["computedRunCount"] == expected_ranking_count
    assert payload["results"][0]["rankingSpec"]["rankingModel"]["label"]
    assert payload["results"][0]["overall"]["observationCount"] >= 1
    assert payload["results"][0]["overall"]["meanTopMinusBottomPct"] is not None

    second_response = client.get("/api/ranking-evaluation")

    assert second_response.status_code == 200
    second_payload = second_response.json()
    assert second_payload["runStoreSummary"]["cachedRunCount"] == expected_ranking_count
    assert second_payload["runStoreSummary"]["computedRunCount"] == 0
