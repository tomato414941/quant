import copy
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from app import main as main_module
from app.comparison_models import ConditionVariant
from app.comparison_market_context import build_run_result_store
from app.comparison_serialization import build_availability_diagnostics
from app.main import app
from app.portfolio import (
    StrategyDefinition,
    build_alignment_policy_spec,
    build_asset_ranking_specs_from_strategy_definitions,
    build_strategy_definition_from_evaluator_strategy_spec,
    build_strategy_execution_plan_spec,
    build_strategy_signal_spec,
    get_strategy_definition_signal_execution_contexts,
)
from app.strategy_candidate_predictors import PREDICTOR_CANDIDATE_DEFINITIONS
from app.timeframe_models import DEFAULT_DAILY_TIMEFRAME, DEFAULT_MONTHLY_TIMEFRAME, DEFAULT_WEEKLY_TIMEFRAME, build_timeframe_spec


client = TestClient(app)


def normalize_strategy_definition(strategy):
    if isinstance(strategy, StrategyDefinition):
        return strategy
    return build_strategy_definition_from_evaluator_strategy_spec(strategy)


def normalize_strategy_definitions(strategies):
    return [normalize_strategy_definition(strategy) for strategy in strategies]


def strategy_definition_has_selection_type(
    strategy: StrategyDefinition,
    strategy_type: str,
) -> bool:
    return any(
        signal.source_kind == "selection_signal"
        and dict(signal.signal_parameters).get("strategyType") == strategy_type
        for signal in strategy.signals
    )


def count_predictor_specs(strategies) -> int:
    predictor_keys = set()
    for strategy in normalize_strategy_definitions(strategies):
        _selection_contexts, predictor_context = get_strategy_definition_signal_execution_contexts(strategy)
        if predictor_context is not None:
            predictor_keys.add(str(predictor_context["predictorKey"]))
    return len(predictor_keys)


def test_availability_diagnostics_classifies_calendar_lifecycle_and_actionable_risks() -> None:
    warnings = [
        {
            "kind": "aligned_start_after_requested_start",
            "timeframe": "1d",
            "requestedStartDate": "2025-01-01",
            "alignedStartDate": "2025-01-02",
            "message": "1d data starts at 2025-01-02, after requested start 2025-01-01.",
        },
        {
            "kind": "asset_available_after_aligned_start",
            "timeframe": "1d",
            "asset": "ETH-USD",
            "alignedStartDate": "2015-01-01",
            "firstValidDate": "2017-11-09",
            "message": "ETH-USD becomes available on 2017-11-09, after aligned start 2015-01-01.",
        },
        {
            "kind": "requested_asset_unavailable",
            "timeframe": "1d",
            "asset": "MISSING",
            "message": "1d data has no usable rows for MISSING.",
        },
    ]

    diagnostics = build_availability_diagnostics(warnings, {"maxStaleBars": 5})

    assert diagnostics["warningCount"] == 3
    assert diagnostics["calendarBoundaryWarningCount"] == 1
    assert diagnostics["assetLifecycleWarningCount"] == 1
    assert diagnostics["actionableWarningCount"] == 1
    assert diagnostics["calendarBoundaryWarnings"][0]["kind"] == "aligned_start_after_requested_start"
    assert diagnostics["assetLifecycleWarnings"][0]["asset"] == "ETH-USD"
    assert diagnostics["actionableWarnings"][0]["asset"] == "MISSING"


def test_comparison_service_does_not_import_evaluator_strategy_spec_dto_bridge() -> None:
    source = Path(main_module.__file__).with_name("comparison_service.py").read_text()
    forbidden_tokens = (
        "build_evaluator_strategy_spec(",
        "build_strategy_definition_from_evaluator_strategy_spec",
        "evaluate_strategy_definition_run(",
    )
    for token in forbidden_tokens:
        assert token not in source


def fake_fetch_market_universe(
    tickers: list[str],
    period: str,
    timeframe: str = "1d",
    start_date: str | None = None,
    end_date: str | None = None,
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
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[dict, dict]:
    closes, metadata = fake_fetch_market_universe(tickers, period, timeframe, start_date, end_date)
    volumes = pd.DataFrame(
        {
            ticker: [1_000_000 + index * 10_000 + offset * 1_000 for index in range(len(closes))]
            for offset, ticker in enumerate(closes.columns)
        },
        index=closes.index,
    )
    return {"closes": closes, "volumes": volumes}, metadata


def fake_fetch_market_universe_bundle_extended(
    tickers: list[str],
    period: str,
    timeframe: str = "1d",
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[dict, dict]:
    if timeframe == "1wk":
        index = pd.date_range("2024-01-05", periods=20, freq="W-FRI")
    elif timeframe == "1mo":
        index = pd.date_range("2024-01-31", periods=20, freq="ME")
    else:
        index = pd.date_range("2024-01-01", periods=84, freq="D")

    closes = pd.DataFrame(
        {
            ticker: [
                100.0
                + offset * 0.5
                + step * (1.0 + offset * 0.03)
                + ((step + offset) % 5) * 0.2
                for step in range(len(index))
            ]
            for offset, ticker in enumerate(tickers)
        },
        index=index,
    )
    volumes = pd.DataFrame(
        {
            ticker: [1_000_000 + step * 10_000 + offset * 1_000 for step in range(len(index))]
            for offset, ticker in enumerate(tickers)
        },
        index=index,
    )
    return {"closes": closes, "volumes": volumes}, {
        "tickers": tickers,
        "period": period,
        "source": "test",
        "timeframe": timeframe,
        "aligned_start_date": str(index[0].date()),
        "aligned_end_date": str(index[-1].date()),
        "row_count": len(index),
    }


def test_healthcheck() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_strategy_inventory_api() -> None:
    response = client.get("/api/strategy-inventory")

    payload = response.json()
    assert response.status_code == 200
    assert payload["kind"] == "strategy_inventory"
    assert payload["schemaVersion"] == "v1"
    assert payload["counts"]["total"] == len(payload["entries"])
    assert any(entry["strategyId"] == "stg-fu-eq" for entry in payload["entries"])


def test_strategy_inventory_api_filters_entries() -> None:
    response = client.get("/api/strategy-inventory", params={"status": "active", "priority": "high"})

    payload = response.json()
    assert response.status_code == 200
    assert payload["counts"]["total"] > 0
    assert all(entry["status"] == "active" for entry in payload["entries"])
    assert all(entry["priority"] == "high" for entry in payload["entries"])


def test_strategy_inventory_api_with_latest_runs(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.fetch_market_universe_bundle", fake_fetch_market_universe_bundle)
    config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)
    config.result_store_dir = str(tmp_path / "run_results")
    monkeypatch.setattr(main_module, "DEFAULT_COMPARISON_SPEC", config)

    strategy_runs_response = client.post("/api/strategy-runs")
    assert strategy_runs_response.status_code == 200

    response = client.get("/api/strategy-inventory", params={"status": "active", "with_latest_runs": True})

    payload = response.json()
    assert response.status_code == 200
    assert payload["withLatestRuns"] is True
    assert any(entry["runStatus"] == "available" for entry in payload["entries"])
    assert all("latestRun" in entry for entry in payload["entries"])


def test_strategy_inventory_api_with_evaluation_matrix(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.fetch_market_universe_bundle", fake_fetch_market_universe_bundle)
    config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)
    config.result_store_dir = str(tmp_path / "run_results")
    monkeypatch.setattr(main_module, "DEFAULT_COMPARISON_SPEC", config)

    strategy_runs_response = client.post("/api/strategy-runs")
    assert strategy_runs_response.status_code == 200

    response = client.get("/api/strategy-inventory", params={"status": "active", "with_evaluation_matrix": True})

    payload = response.json()
    assert response.status_code == 200
    assert payload["withEvaluationMatrix"] is True
    assert len(payload["evaluationMatrix"]) == payload["counts"]["total"]
    assert any(
        cell["runStatus"] == "available"
        for row in payload["evaluationMatrix"]
        for cell in row["profiles"]
    )


def test_strategy_inventory_api_rejects_invalid_filter() -> None:
    response = client.get("/api/strategy-inventory", params={"status": "unknown"})

    assert response.status_code == 400
    assert "Unknown strategy inventory status" in response.json()["detail"]


def test_comparison_run_spec_endpoint(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.fetch_market_universe_bundle", fake_fetch_market_universe_bundle)
    config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)
    config.result_store_dir = str(tmp_path / "run_results")
    monkeypatch.setattr(main_module, "DEFAULT_COMPARISON_SPEC", config)

    response = client.get("/api/comparison-run-spec")

    assert response.status_code == 200
    payload = response.json()
    assert payload["comparisonId"] == config.comparison_id
    assert payload["kind"] == "comparison_run_spec_payload"
    assert payload["runSpec"]["kind"] == "comparison_run_spec"
    assert payload["runSpec"]["marketSlice"]["period"] == config.run_spec.market_slice.period
    assert payload["runSpec"]["marketSlice"]["startDate"] == config.run_spec.market_slice.start_date
    assert payload["runSpec"]["marketSlice"]["endDate"] == config.run_spec.market_slice.end_date
    assert payload["comparisonFingerprint"]
    assert payload["runSpecFingerprint"]
    assert payload["selectionPolicy"]["primaryMetric"] == config.selection_policy.primary_metric
    assert payload["candidateStrategyCount"] == len(config.candidate_strategies)
    assert payload["referenceStrategyCount"] == len(config.reference_strategies)
    assert len(payload["candidateStrategies"]) == len(config.candidate_strategies)
    assert len(payload["referenceStrategies"]) == len(config.reference_strategies)
    assert payload["candidateStrategies"][0]["kind"] == "strategy_definition"
    assert payload["referenceStrategies"][0]["kind"] == "strategy_definition"
    assert payload["conditionVariants"][0]["key"]


def test_rerun_comparison_run_spec_endpoint_rejects_mismatched_fingerprint(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.fetch_market_universe_bundle", fake_fetch_market_universe_bundle)
    config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)
    config.result_store_dir = str(tmp_path / "run_results")
    monkeypatch.setattr(main_module, "DEFAULT_COMPARISON_SPEC", config)

    spec_response = client.get("/api/comparison-run-spec")
    assert spec_response.status_code == 200
    payload = spec_response.json()
    payload["comparisonFingerprint"] = "invalid"

    response = client.post("/api/comparison-run-spec/rerun", json=payload)

    assert response.status_code == 400
    assert "comparisonFingerprint does not match" in response.json()["detail"]


def test_rerun_comparison_run_spec_endpoint(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.fetch_market_universe_bundle", fake_fetch_market_universe_bundle)
    config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)
    config.result_store_dir = str(tmp_path / "run_results")
    monkeypatch.setattr(main_module, "DEFAULT_COMPARISON_SPEC", config)

    spec_response = client.get("/api/comparison-run-spec")
    assert spec_response.status_code == 200

    response = client.post("/api/comparison-run-spec/rerun", json=spec_response.json())

    assert response.status_code == 200
    payload = response.json()
    assert payload["comparison"]["comparisonId"] == config.comparison_id
    assert len(payload["candidateRuns"]) == len(config.candidate_strategies)
    assert len(payload["referenceRuns"]) == len(config.reference_strategies)
    assert payload["runStoreSummary"]["cachedRunCount"] + payload["runStoreSummary"]["computedRunCount"] > 0


def test_comparison_endpoint(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.fetch_market_universe_bundle", fake_fetch_market_universe_bundle)
    config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)
    config.result_store_dir = str(tmp_path / "run_results")
    monkeypatch.setattr(main_module, "DEFAULT_COMPARISON_SPEC", config)

    response = client.get("/api/comparison")

    assert response.status_code == 200
    payload = response.json()
    expected_strategy_count = len(config.candidate_strategies)
    expected_reference_count = len(config.reference_strategies)
    expected_predictor_count = count_predictor_specs(config.candidate_strategies)

    assert payload["comparison"]["comparisonId"] == "etf_portfolio_models_2015_2025"
    assert payload["comparison"]["selectionPolicy"]["primaryMetric"] == "sharpe_ratio"
    assert payload["comparison"]["runSpec"]["kind"] == "comparison_run_spec"
    assert payload["comparison"]["runSpec"]["executionAssumptions"]["kind"] == "close_execution_assumptions"
    assert payload["comparison"]["runSpec"]["executionAssumptions"]["parameters"]["fillPrice"] == "close"
    assert (
        payload["comparison"]["runSpec"]["executionAssumptions"]["parameters"]["costProfileKey"]
        == "retail_multi_asset_default"
    )
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
    instrument_diagnostics = payload["comparison"]["runSpec"]["evaluation"]["instrumentDiagnostics"]
    assert instrument_diagnostics["costProfileKey"] == "retail_multi_asset_default"
    assert instrument_diagnostics["mixedMarketCalendar"] is False
    assert instrument_diagnostics["marketCalendars"] == {"nyse": 18}
    assert "crypto" not in instrument_diagnostics["assetClassCounts"]
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
    } == {config.run_spec.market_slice.period}
    assert {
        context["startDate"]
        for context in payload["comparison"]["runSpec"]["evaluation"]["marketDataContexts"]
    } == {config.run_spec.market_slice.start_date}
    assert {
        context["endDate"]
        for context in payload["comparison"]["runSpec"]["evaluation"]["marketDataContexts"]
    } == {config.run_spec.market_slice.end_date}
    assert all(context["sanityPeriods"] == ["3y"] for context in payload["comparison"]["runSpec"]["evaluation"]["marketDataContexts"])
    assert all(context["alignedStartDate"] == "2025-01-01" for context in payload["comparison"]["runSpec"]["evaluation"]["marketDataContexts"])
    assert all(context["alignedEndDate"] == "2025-01-07" for context in payload["comparison"]["runSpec"]["evaluation"]["marketDataContexts"])
    assert all(context["rowCount"] == 7 for context in payload["comparison"]["runSpec"]["evaluation"]["marketDataContexts"])
    warnings = payload["comparison"]["runSpec"]["evaluation"]["warnings"]
    assert {warning["kind"] for warning in warnings} == {
        "aligned_start_after_requested_start",
        "aligned_end_before_requested_end",
    }
    start_warnings = [warning for warning in warnings if warning["kind"] == "aligned_start_after_requested_start"]
    end_warnings = [warning for warning in warnings if warning["kind"] == "aligned_end_before_requested_end"]
    assert {warning["requestedStartDate"] for warning in start_warnings} == {config.run_spec.market_slice.start_date}
    assert {warning["alignedStartDate"] for warning in start_warnings} == {"2025-01-01"}
    assert {warning["requestedEndDate"] for warning in end_warnings} == {config.run_spec.market_slice.end_date}
    assert {warning["alignedEndDate"] for warning in end_warnings} == {"2025-01-07"}
    diagnostic_events = payload["comparison"]["runSpec"]["evaluation"]["diagnosticEvents"]
    assert {event["category"] for event in diagnostic_events} == {"availability"}
    assert {event["severity"] for event in diagnostic_events} == {"invalidating"}
    diagnostics = payload["comparison"]["runSpec"]["evaluation"]["availabilityDiagnostics"]
    assert diagnostics["warningCount"] == len(warnings)
    assert diagnostics["actionableWarningCount"] == len(warnings)
    assert diagnostics["calendarBoundaryWarningCount"] == 0
    assert diagnostics["calendarBoundaryWarnings"] == []
    assert {warning["kind"] for warning in diagnostics["actionableWarnings"]} == {
        "aligned_start_after_requested_start",
        "aligned_end_before_requested_end",
    }
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
    assert payload["comparison"]["candidateStrategies"][0]["kind"] == "strategy_definition"
    assert payload["comparison"]["candidateStrategies"][0]["schemaVersion"] == "v1"
    assert payload["comparison"]["candidateStrategies"][0]["components"]["core"]["investmentUniverse"]["label"]
    assert (
        payload["comparison"]["candidateStrategies"][0]["components"]["core"]["executionPlan"]["rebalanceSchedule"]
        == "year_end"
    )
    execution_support = payload["comparison"]["candidateStrategies"][0]["executionSupport"]
    assert execution_support["directExecutionCompatible"] is True
    assert "strategySpecAdapterCompatible" not in execution_support
    assert "strategySpecAdapterIssues" not in execution_support
    assert "evaluatorAdapterCompatible" not in execution_support
    assert "evaluatorAdapterIssues" not in execution_support
    assert payload["comparison"]["candidateStrategies"][0]["components"]["optional"]["signals"][0]["sourceKind"] == "selection_signal"
    assert (
        payload["candidateRuns"][5]["strategy"]["components"]["optional"]["signals"][0]["signalParameters"]["scoreParameters"]["windowSpec"]["unit"]
        == "months"
    )
    assert (
        payload["candidateRuns"][5]["strategy"]["components"]["optional"]["signals"][0]["signalParameters"]["scoreParameters"]["windowSpec"]["value"]
        == 12
    )
    assert any(
        any(signal["sourceKind"] == "predictor_overlay" for signal in strategy["components"]["optional"]["signals"])
        for strategy in payload["comparison"]["candidateStrategies"]
    )
    assert payload["comparison"]["marketUniverse"]["assetCount"] == 18
    assert len(payload["comparison"]["marketUniverse"]["tickers"]) == 18

    assert len(payload["candidateRuns"]) == expected_strategy_count
    assert len(payload["referenceRuns"]) == expected_reference_count
    assert len(payload["predictorRuns"]) == expected_predictor_count
    assert {
        run["strategy"]["components"]["optional"]["signals"][0]["dataTimeframe"]["key"]
        for run in payload["candidateRuns"]
    } == {"1d", "1w", "1mo"}
    assert payload["candidateRuns"][0]["splitAnalysis"]["config"]["splitRatioPct"] == 70.0
    assert payload["candidateRuns"][0]["kind"] == "run_result"
    assert payload["candidateRuns"][0]["schemaVersion"] == "v1"
    assert (
        payload["candidateRuns"][0]["strategy"]["components"]["core"]["investmentUniverse"]["label"]
        == "ETF"
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

    second_response = client.get("/api/comparison")

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
    expected_predictor_count = count_predictor_specs(config.candidate_strategies + config.reference_strategies)
    assert payload["kind"] == "predictor_run_collection"
    assert payload["comparisonId"] == "etf_portfolio_models_2015_2025"
    assert payload["runSpec"]["kind"] == "comparison_run_spec"
    assert len(payload["predictorSpecs"]) == expected_predictor_count
    assert len(payload["predictorRuns"]) == expected_predictor_count
    assert payload["predictorSpecs"][0]["signalSpec"]["kind"] == "signal_spec"
    assert payload["predictorSpecs"][0]["signalSpec"]["observationSpec"]["kind"] == "observation_spec"
    assert payload["predictorSpecs"][0]["signalSpec"]["observationSpec"]["assetCount"] >= 1
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
    assert index_payload["records"][0]["logicVersion"] == "v71"
    assert index_payload["records"][0]["strategyDefinitionFingerprint"]
    assert index_payload["records"][0]["evaluationSubjectFingerprint"]
    assert index_payload["records"][0]["marketDataFingerprint"]
    assert index_payload["records"][0]["evaluationFingerprint"]
    assert "trainingFitMode" in index_payload["records"][0]
    assert "signalSourceKind" in index_payload["records"][0]
    assert "observationLabel" in index_payload["records"][0]
    assert "signalEntityKind" in index_payload["records"][0]
    assert "groupedSummaries" in index_payload
    assert "bestBySignalSource" in index_payload["groupedSummaries"]
    assert "bestBySignalSourceAndHorizon" in index_payload["groupedSummaries"]
    assert "bestBySignalSourceAndPeriod" in index_payload["groupedSummaries"]
    run_key = index_payload["records"][0]["runKey"]

    detail_response = client.get(f"/api/predictor-runs/{run_key}")
    assert detail_response.status_code == 200
    detail_payload = detail_response.json()
    assert detail_payload["kind"] == "predictor_run_detail"
    assert detail_payload["record"]["runKey"] == run_key
    assert detail_payload["record"]["runSpec"]["runKind"] == "predictor_run"
    assert detail_payload["record"]["runSpec"]["logicVersion"] == "v71"
    assert detail_payload["record"]["runSpec"]["strategyDefinition"]["kind"] == "strategy_definition"
    assert detail_payload["record"]["runSpec"]["evaluationSubject"]["kind"] == "predictor"
    assert detail_payload["record"]["runSpec"]["evaluationSubject"]["predictor"]["kind"] == "predictor_spec"
    assert "strategy" not in detail_payload["record"]["runSpec"]
    assert set(detail_payload["record"]["runSpec"]["fingerprints"].keys()) == {"strategyDefinition", "evaluationSubject", "marketData", "evaluation"}

    fingerprint_filtered_response = client.get(
        "/api/predictor-runs",
        params={
            "limit": 10,
            "strategy_definition_fingerprint": index_payload["records"][0]["strategyDefinitionFingerprint"],
        },
    )
    assert fingerprint_filtered_response.status_code == 200
    fingerprint_filtered_payload = fingerprint_filtered_response.json()
    assert (
        fingerprint_filtered_payload["filters"]["strategyDefinitionFingerprint"]
        == index_payload["records"][0]["strategyDefinitionFingerprint"]
    )
    assert fingerprint_filtered_payload["recordCount"] >= 1
    assert all(
        record["strategyDefinitionFingerprint"] == index_payload["records"][0]["strategyDefinitionFingerprint"]
        for record in fingerprint_filtered_payload["records"]
    )

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

    signal_filtered_response = client.get(
        "/api/predictor-runs",
        params={
            "signal_source_kind": "derived_feature",
            "signal_source_feature_key": "momentum",
        },
    )
    assert signal_filtered_response.status_code == 200
    signal_filtered_payload = signal_filtered_response.json()
    assert signal_filtered_payload["filters"]["signalSourceKind"] == "derived_feature"
    assert signal_filtered_payload["filters"]["signalSourceFeatureKey"] == "momentum"
    assert signal_filtered_payload["recordCount"] >= 1
    assert all(
        record["signalSourceKind"] == "derived_feature"
        for record in signal_filtered_payload["records"]
    )
    assert all(
        record["signalSourceFeatureKey"] == "momentum"
        for record in signal_filtered_payload["records"]
    )
    assert any(
        summary["signalSourceKind"] == "derived_feature"
        and summary["signalSourceFeatureKey"] == "momentum"
        for summary in signal_filtered_payload["groupedSummaries"]["bestBySignalSource"]
    )
    assert any(
        summary["signalSourceKind"] == "derived_feature"
        and summary["signalSourceFeatureKey"] == "momentum"
        and summary["horizonValue"] in {5, 10}
        for summary in signal_filtered_payload["groupedSummaries"]["bestBySignalSourceAndHorizon"]
    )

    second_response = client.post("/api/predictor-runs")
    assert second_response.status_code == 200
    second_payload = second_response.json()
    assert second_payload["runStoreSummary"]["cachedRunCount"] == expected_predictor_count * 2
    assert second_payload["runStoreSummary"]["computedRunCount"] == 0


def test_comparison_endpoint_accepts_definition_candidates(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.fetch_market_universe_bundle", fake_fetch_market_universe_bundle_extended)
    config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)
    config.result_store_dir = str(tmp_path / "run_results")
    definition = normalize_strategy_definition(config.candidate_strategies[0])
    config = replace(
        config,
        candidate_strategies=[definition, *config.candidate_strategies[1:]],
    )
    monkeypatch.setattr(main_module, "DEFAULT_COMPARISON_SPEC", config)

    response = client.get("/api/comparison")

    assert response.status_code == 200
    payload = response.json()
    assert payload["comparison"]["candidateStrategies"][0]["kind"] == "strategy_definition"
    assert payload["comparison"]["candidateStrategies"][0]["strategyId"] == definition.strategy_id
    assert payload["comparison"]["candidateStrategies"][0]["executionSupport"]["directExecutionCompatible"] is True
    signal_market_data_contexts = [
        context
        for context in payload["comparison"]["runSpec"]["evaluation"]["signalMarketDataContexts"]
        if context["strategyId"] == definition.strategy_id
    ]
    assert len(signal_market_data_contexts) == len(definition.signals)
    assert signal_market_data_contexts[0]["signalKey"] == definition.signals[0].key
    assert signal_market_data_contexts[0]["dataTimeframe"]["key"] == definition.signals[0].data_timeframe.key
    assert signal_market_data_contexts[0]["signalTimeframe"]["key"] == definition.signals[0].signal_timeframe.key
    assert payload["candidateRuns"][0]["strategy"]["strategyId"] == definition.strategy_id
    assert len(payload["candidateRuns"]) == len(config.candidate_strategies)



def test_comparison_endpoint_supports_explicit_strategy_signal_context_without_extensions(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.fetch_market_universe_bundle", fake_fetch_market_universe_bundle_extended)
    config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)
    config.result_store_dir = str(tmp_path / "run_results")
    definition = normalize_strategy_definition(config.candidate_strategies[0])
    definition = replace(
        definition,
        signals=(
            replace(
                definition.signals[0],
                data_timeframe=DEFAULT_DAILY_TIMEFRAME,
                signal_timeframe=DEFAULT_WEEKLY_TIMEFRAME,
                alignment_policy=build_alignment_policy_spec(
                    key="weekly",
                    label="Weekly",
                    method="asof_last",
                ),
            ),
            *definition.signals[1:],
        ),
        execution_plan=build_strategy_execution_plan_spec(
            key=definition.execution_plan.key,
            label=definition.execution_plan.label,
            decision_schedule="every_bar",
            rebalance_schedule=definition.execution_plan.rebalance_schedule,
        ),
        extensions=(),
    )
    config = replace(config, candidate_strategies=[definition, *config.candidate_strategies[1:]])
    monkeypatch.setattr(main_module, "DEFAULT_COMPARISON_SPEC", config)

    response = client.get("/api/comparison")

    assert response.status_code == 200
    payload = response.json()
    direct_run = payload["candidateRuns"][0]
    assert direct_run["strategy"]["components"]["optional"]["signals"][0]["signalTimeframe"]["key"] == "1w"
    assert direct_run["strategy"]["components"]["core"]["executionPlan"]["decisionSchedule"] == "every_bar"
    signal = direct_run["strategy"]["components"]["optional"]["signals"][0]
    assert signal["dataTimeframe"]["key"] == "1d"
    assert signal["signalTimeframe"]["key"] == "1w"
    market_contexts = payload["comparison"]["runSpec"]["evaluation"]["marketDataContexts"]
    assert any(context["timeframe"]["key"] == "1d" for context in market_contexts)


def test_comparison_endpoint_accepts_direct_execution_definition_candidates(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.fetch_market_universe_bundle", fake_fetch_market_universe_bundle_extended)
    config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)
    config.result_store_dir = str(tmp_path / "run_results")
    definition = normalize_strategy_definition(config.candidate_strategies[0])
    direct_definition = replace(
        definition,
        signals=[
            replace(
                definition.signals[0],
                signal_timeframe=DEFAULT_WEEKLY_TIMEFRAME,
            ),
            *definition.signals[1:],
        ],
        execution_plan=build_strategy_execution_plan_spec(
            key=definition.execution_plan.key,
            label=definition.execution_plan.label,
            decision_schedule="every_bar",
            rebalance_schedule="month_end",
        ),
    )
    config = replace(
        config,
        candidate_strategies=[direct_definition, *config.candidate_strategies[1:]],
    )
    monkeypatch.setattr(main_module, "DEFAULT_COMPARISON_SPEC", config)

    response = client.get("/api/comparison")

    assert response.status_code == 200
    payload = response.json()
    assert payload["comparison"]["candidateStrategies"][0]["executionSupport"]["directExecutionCompatible"] is True
    assert payload["comparison"]["candidateStrategies"][0]["executionSupport"]["directExecutionCompatible"] is True
    direct_signal_contexts = [
        context
        for context in payload["comparison"]["runSpec"]["evaluation"]["signalMarketDataContexts"]
        if context["strategyId"] == direct_definition.strategy_id
    ]
    assert direct_signal_contexts[0]["dataTimeframe"]["key"] == "1d"
    assert direct_signal_contexts[0]["signalTimeframe"]["key"] == "1w"
    strategy_payload = payload["candidateRuns"][0]["strategy"]
    assert strategy_payload["kind"] == "strategy_definition"
    assert strategy_payload["components"]["core"]["executionPlan"]["decisionSchedule"] == "every_bar"
    assert strategy_payload["components"]["core"]["executionPlan"]["rebalanceSchedule"] == "month_end"
    assert strategy_payload["components"]["optional"]["signals"][0]["dataTimeframe"]["key"] == "1d"
    assert strategy_payload["components"]["optional"]["signals"][0]["signalTimeframe"]["key"] == "1w"


def test_comparison_endpoint_fetches_signal_source_timeframe_for_direct_execution(monkeypatch, tmp_path) -> None:
    requested_timeframes: list[str] = []

    def tracking_fetch_market_universe_bundle(
        tickers: list[str],
        period: str,
        timeframe: str = "1d",
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        requested_timeframes.append(timeframe)
        return fake_fetch_market_universe_bundle_extended(
            tickers,
            period,
            timeframe,
            start_date,
            end_date,
        )

    monkeypatch.setattr("app.main.fetch_market_universe_bundle", tracking_fetch_market_universe_bundle)
    config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)
    config.result_store_dir = str(tmp_path / "run_results")
    definition = normalize_strategy_definition(config.candidate_strategies[0])
    direct_definition = replace(
        definition,
        signals=[
            replace(
                definition.signals[0],
                signal_timeframe=DEFAULT_WEEKLY_TIMEFRAME,
            ),
            *definition.signals[1:],
        ],
        execution_plan=build_strategy_execution_plan_spec(
            key=definition.execution_plan.key,
            label=definition.execution_plan.label,
            decision_schedule="every_bar",
            rebalance_schedule="month_end",
        ),
    )
    config = replace(
        config,
        candidate_strategies=[direct_definition, *config.candidate_strategies[1:]],
    )
    monkeypatch.setattr(main_module, "DEFAULT_COMPARISON_SPEC", config)

    response = client.get("/api/comparison")

    assert response.status_code == 200
    payload = response.json()
    market_data_context_keys = {
        context["timeframe"]["key"]
        for context in payload["comparison"]["runSpec"]["evaluation"]["marketDataContexts"]
    }
    assert "1d" in market_data_context_keys
    assert "1w" in market_data_context_keys
    assert "1d" in requested_timeframes
    assert "1wk" in requested_timeframes


def test_comparison_endpoint_accepts_direct_execution_multi_selection_definition_candidates(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.fetch_market_universe_bundle", fake_fetch_market_universe_bundle_extended)
    config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)
    config.result_store_dir = str(tmp_path / "run_results")
    definition = next(
        strategy
        for strategy in normalize_strategy_definitions(config.candidate_strategies)
        if strategy_definition_has_selection_type(strategy, "full_universe_momentum_tilt")
    )
    secondary_signal = build_strategy_signal_spec(
        key="selection__secondary",
        label="Secondary momentum signal",
        description="Secondary momentum signal",
        observation_spec=definition.signals[0].observation_spec,
        data_timeframe=definition.signals[0].data_timeframe,
        signal_timeframe=DEFAULT_WEEKLY_TIMEFRAME,
        source_kind="selection_signal",
        weight=0.4,
        signal_parameters={
            "selectionKey": "secondary_momo6",
            "strategyType": "full_universe_momentum_tilt",
            "scoreParameters": {
                "tilt_strength": 0.35,
                "tilt_shape": 1.0,
                "windowSpec": {"unit": "months", "value": 6},
            },
        },
    )
    direct_definition = replace(
        definition,
        signals=(
            replace(definition.signals[0], signal_timeframe=DEFAULT_WEEKLY_TIMEFRAME, weight=0.6),
            secondary_signal,
        ),
        execution_plan=build_strategy_execution_plan_spec(
            key=definition.execution_plan.key,
            label=definition.execution_plan.label,
            decision_schedule="every_bar",
            rebalance_schedule="month_end",
        ),
    )
    config = replace(
        config,
        candidate_strategies=[direct_definition, *config.candidate_strategies[1:]],
    )
    monkeypatch.setattr(main_module, "DEFAULT_COMPARISON_SPEC", config)

    response = client.get("/api/comparison")

    assert response.status_code == 200
    payload = response.json()
    strategy_payload = payload["candidateRuns"][0]["strategy"]
    assert payload["comparison"]["candidateStrategies"][0]["executionSupport"]["directExecutionCompatible"] is True
    selection_signals = strategy_payload["components"]["optional"]["signals"]
    assert selection_signals[1]["signalParameters"]["selectionKey"] == "secondary_momo6"
    assert selection_signals[1]["signalTimeframe"]["key"] == "1w"


def test_comparison_endpoint_returns_selection_alignment_policy_payloads_for_direct_execution_multi_selection_candidates(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.fetch_market_universe_bundle", fake_fetch_market_universe_bundle_extended)
    config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)
    config.result_store_dir = str(tmp_path / "run_results")
    definition = next(
        strategy
        for strategy in normalize_strategy_definitions(config.candidate_strategies)
        if strategy_definition_has_selection_type(strategy, "full_universe_momentum_tilt")
    )
    primary_signal = replace(
        definition.signals[0],
        signal_timeframe=DEFAULT_WEEKLY_TIMEFRAME,
        weight=0.6,
        alignment_policy=build_alignment_policy_spec(
            key="primary_asof_last",
            label="Primary as-of",
            method="asof_last",
        ),
    )
    secondary_signal = build_strategy_signal_spec(
        key="selection__secondary",
        label="Secondary momentum signal",
        description="Secondary momentum signal",
        observation_spec=definition.signals[0].observation_spec,
        data_timeframe=definition.signals[0].data_timeframe,
        signal_timeframe=DEFAULT_WEEKLY_TIMEFRAME,
        source_kind="selection_signal",
        weight=0.4,
        alignment_policy=build_alignment_policy_spec(
            key="secondary_end_of_period",
            label="Secondary end-of-period",
            method="end_of_period",
        ),
        signal_parameters={
            "selectionKey": "secondary_momo6",
            "strategyType": "full_universe_momentum_tilt",
            "scoreParameters": {
                "tilt_strength": 0.35,
                "tilt_shape": 1.0,
                "windowSpec": {"unit": "months", "value": 6},
            },
        },
    )
    direct_definition = replace(
        definition,
        signals=(primary_signal, secondary_signal),
        execution_plan=build_strategy_execution_plan_spec(
            key=definition.execution_plan.key,
            label=definition.execution_plan.label,
            decision_schedule="every_bar",
            rebalance_schedule="month_end",
        ),
    )
    config = replace(
        config,
        candidate_strategies=[direct_definition, *config.candidate_strategies[1:]],
    )
    monkeypatch.setattr(main_module, "DEFAULT_COMPARISON_SPEC", config)

    response = client.get("/api/comparison")

    assert response.status_code == 200
    payload = response.json()
    signals = payload["candidateRuns"][0]["strategy"]["components"]["optional"]["signals"]
    assert signals[0]["alignmentPolicy"]["method"] == "asof_last"
    assert signals[1]["alignmentPolicy"]["method"] == "end_of_period"


def test_comparison_endpoint_returns_predictor_alignment_policy_payload_for_direct_execution_candidates(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.fetch_market_universe_bundle", fake_fetch_market_universe_bundle_extended)
    config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)
    config.result_store_dir = str(tmp_path / "run_results")
    predictor_definition = PREDICTOR_CANDIDATE_DEFINITIONS[0]
    direct_predictor_definition = replace(
        predictor_definition,
        signals=(
            replace(predictor_definition.signals[0], signal_timeframe=DEFAULT_WEEKLY_TIMEFRAME),
            replace(
                predictor_definition.signals[1],
                signal_timeframe=DEFAULT_WEEKLY_TIMEFRAME,
                alignment_policy=build_alignment_policy_spec(
                    key="predictor_calendar_resample",
                    label="Predictor calendar resample",
                    method="calendar_resample",
                ),
            ),
        ),
        execution_plan=build_strategy_execution_plan_spec(
            key=predictor_definition.execution_plan.key,
            label=predictor_definition.execution_plan.label,
            decision_schedule="every_bar",
            rebalance_schedule="month_end",
        ),
    )
    config = replace(
        config,
        candidate_strategies=[direct_predictor_definition, *config.candidate_strategies[1:]],
    )
    monkeypatch.setattr(main_module, "DEFAULT_COMPARISON_SPEC", config)

    response = client.get("/api/comparison")

    assert response.status_code == 200
    payload = response.json()
    predictor_signal = payload["candidateRuns"][0]["strategy"]["components"]["optional"]["signals"][1]
    assert predictor_signal["alignmentPolicy"]["method"] == "calendar_resample"


def test_comparison_endpoint_accepts_direct_execution_predictor_definition_candidates_with_multi_selection(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.fetch_market_universe_bundle", fake_fetch_market_universe_bundle_extended)
    config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)
    config.result_store_dir = str(tmp_path / "run_results")
    predictor_definition = PREDICTOR_CANDIDATE_DEFINITIONS[0]
    secondary_signal = build_strategy_signal_spec(
        key="selection__secondary",
        label="Secondary momentum signal",
        description="Secondary momentum signal",
        observation_spec=predictor_definition.signals[0].observation_spec,
        data_timeframe=predictor_definition.signals[0].data_timeframe,
        signal_timeframe=DEFAULT_WEEKLY_TIMEFRAME,
        source_kind="selection_signal",
        weight=0.4,
        signal_parameters={
            "selectionKey": "secondary_momo6",
            "strategyType": "full_universe_momentum_tilt",
            "scoreParameters": {
                "tilt_strength": 0.35,
                "tilt_shape": 1.0,
                "windowSpec": {"unit": "months", "value": 6},
            },
        },
    )
    direct_predictor_definition = replace(
        predictor_definition,
        signals=(
            replace(predictor_definition.signals[0], signal_timeframe=DEFAULT_WEEKLY_TIMEFRAME, weight=0.6),
            secondary_signal,
            replace(predictor_definition.signals[1], signal_timeframe=DEFAULT_WEEKLY_TIMEFRAME),
        ),
        execution_plan=build_strategy_execution_plan_spec(
            key=predictor_definition.execution_plan.key,
            label=predictor_definition.execution_plan.label,
            decision_schedule="every_bar",
            rebalance_schedule="month_end",
        ),
    )
    config = replace(
        config,
        candidate_strategies=[direct_predictor_definition, *config.candidate_strategies[1:]],
    )
    monkeypatch.setattr(main_module, "DEFAULT_COMPARISON_SPEC", config)

    response = client.get("/api/comparison")

    assert response.status_code == 200
    payload = response.json()
    strategy_payload = payload["candidateRuns"][0]["strategy"]
    signals = strategy_payload["components"]["optional"]["signals"]
    assert signals[2]["predictorKey"] == predictor_definition.signals[1].predictor_key
    assert signals[1]["signalParameters"]["selectionKey"] == "secondary_momo6"


def test_comparison_endpoint_accepts_direct_execution_predictor_definition_candidates(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.fetch_market_universe_bundle", fake_fetch_market_universe_bundle_extended)
    config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)
    config.result_store_dir = str(tmp_path / "run_results")
    predictor_definition = PREDICTOR_CANDIDATE_DEFINITIONS[0]
    direct_predictor_definition = replace(
        predictor_definition,
        signals=[
            replace(predictor_definition.signals[0], signal_timeframe=DEFAULT_WEEKLY_TIMEFRAME),
            replace(predictor_definition.signals[1], signal_timeframe=DEFAULT_WEEKLY_TIMEFRAME),
        ],
        execution_plan=build_strategy_execution_plan_spec(
            key=predictor_definition.execution_plan.key,
            label=predictor_definition.execution_plan.label,
            decision_schedule="every_bar",
            rebalance_schedule="month_end",
        ),
    )
    config = replace(
        config,
        candidate_strategies=[direct_predictor_definition, *config.candidate_strategies[1:]],
    )
    monkeypatch.setattr(main_module, "DEFAULT_COMPARISON_SPEC", config)

    response = client.get("/api/comparison")

    assert response.status_code == 200
    payload = response.json()
    strategy_payload = payload["candidateRuns"][0]["strategy"]
    assert payload["comparison"]["candidateStrategies"][0]["executionSupport"]["directExecutionCompatible"] is True
    assert payload["comparison"]["candidateStrategies"][0]["executionSupport"]["directExecutionCompatible"] is True
    assert strategy_payload["kind"] == "strategy_definition"
    assert strategy_payload["components"]["optional"]["signals"][0]["signalTimeframe"]["key"] == "1w"
    assert strategy_payload["components"]["optional"]["signals"][1]["predictorKey"] == predictor_definition.signals[1].predictor_key
    assert strategy_payload["components"]["core"]["executionPlan"]["decisionSchedule"] == "every_bar"
    assert strategy_payload["components"]["optional"]["signals"][0]["dataTimeframe"]["key"] == "1d"


def test_comparison_endpoint_reports_incompatible_definition(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.fetch_market_universe_bundle", fake_fetch_market_universe_bundle)
    config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)
    config.result_store_dir = str(tmp_path / "run_results")
    definition = normalize_strategy_definition(config.candidate_strategies[0])
    incompatible_definition = replace(
        definition,
        signals=[
            replace(
                definition.signals[0],
                signal_timeframe=DEFAULT_MONTHLY_TIMEFRAME,
            ),
            *definition.signals[1:],
        ],
        execution_plan=build_strategy_execution_plan_spec(
            key=definition.execution_plan.key,
            label=definition.execution_plan.label,
            decision_schedule="quarter_end",
            rebalance_schedule=definition.execution_plan.rebalance_schedule,
        ),
    )
    config = replace(
        config,
        candidate_strategies=[incompatible_definition, *config.candidate_strategies[1:]],
    )
    monkeypatch.setattr(main_module, "DEFAULT_COMPARISON_SPEC", config)

    response = client.get("/api/comparison")

    assert response.status_code == 400
    assert incompatible_definition.strategy_id in response.json()["detail"]
    assert "not executable" in response.json()["detail"]
    assert "Direct execution incompatibilities" in response.json()["detail"]
    assert "decision_schedule to match rebalance_schedule or be every_bar" in response.json()["detail"]



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
    expected_predictor_count = count_predictor_specs(config.candidate_strategies)
    assert payload["kind"] == "strategy_run_collection"
    assert payload["comparisonId"] == "etf_portfolio_models_2015_2025"
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
    assert index_payload["records"][0]["logicVersion"] == "v71"
    assert index_payload["records"][0]["strategyDefinitionFingerprint"]
    assert index_payload["records"][0]["evaluationSubjectFingerprint"]
    assert index_payload["records"][0]["marketDataFingerprint"]
    assert index_payload["records"][0]["evaluationFingerprint"]
    run_key = index_payload["records"][0]["runKey"]

    detail_response = client.get(f"/api/strategy-runs/{run_key}")
    assert detail_response.status_code == 200
    detail_payload = detail_response.json()
    assert detail_payload["kind"] == "strategy_run_detail"
    assert detail_payload["record"]["runKey"] == run_key
    assert detail_payload["record"]["runSpec"]["runKind"] == "strategy_run"
    assert detail_payload["record"]["runSpec"]["logicVersion"] == "v71"
    assert set(detail_payload["record"]["runSpec"]["fingerprints"].keys()) == {"strategyDefinition", "evaluationSubject", "marketData", "evaluation"}

    fingerprint_filtered_response = client.get(
        "/api/strategy-runs",
        params={
            "limit": 10,
            "strategy_definition_fingerprint": index_payload["records"][0]["strategyDefinitionFingerprint"],
        },
    )
    assert fingerprint_filtered_response.status_code == 200
    fingerprint_filtered_payload = fingerprint_filtered_response.json()
    assert (
        fingerprint_filtered_payload["filters"]["strategyDefinitionFingerprint"]
        == index_payload["records"][0]["strategyDefinitionFingerprint"]
    )
    assert fingerprint_filtered_payload["recordCount"] >= 1
    assert all(
        record["strategyDefinitionFingerprint"] == index_payload["records"][0]["strategyDefinitionFingerprint"]
        for record in fingerprint_filtered_payload["records"]
    )

    second_response = client.post("/api/strategy-runs")
    assert second_response.status_code == 200
    second_payload = second_response.json()
    assert second_payload["runStoreSummary"]["cachedRunCount"] == (
        (expected_strategy_count + expected_reference_count) * 2
        + expected_predictor_count * 2
    )
    assert second_payload["runStoreSummary"]["computedRunCount"] == 0


def test_comparison_endpoint_supports_mixed_strategy_timeframes(monkeypatch, tmp_path) -> None:
    fetch_calls: list[tuple[str, str]] = []

    def recording_fetch_market_universe_bundle(
        tickers: list[str],
        period: str,
        timeframe: str = "1d",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> tuple[dict, dict]:
        fetch_calls.append((period, timeframe))
        return fake_fetch_market_universe_bundle(
            tickers,
            period,
            timeframe,
            start_date,
            end_date,
        )

    monkeypatch.setattr(
        "app.main.fetch_market_universe_bundle",
        recording_fetch_market_universe_bundle,
    )
    config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)
    weekly_timeframe = build_timeframe_spec(
        key="1w",
        label="週次",
        yfinance_interval="1wk",
        bar_seconds=604_800,
        bars_per_year=52,
    )
    weekly_definition = replace(
        config.candidate_strategies[0],
        strategy_id="stg-fu-eq-weekly",
        label="全資産 × 等金額配分 × 週次データ",
        signals=tuple(
            replace(signal, data_timeframe=weekly_timeframe, signal_timeframe=weekly_timeframe)
            for signal in config.candidate_strategies[0].signals
        ),
    )
    config.candidate_strategies = [config.candidate_strategies[0], weekly_definition]
    config.reference_strategies = []
    config.result_store_dir = str(tmp_path / "run_results")
    monkeypatch.setattr(main_module, "DEFAULT_COMPARISON_SPEC", config)

    response = client.get("/api/comparison")

    assert response.status_code == 200
    payload = response.json()
    assert {(period, timeframe) for period, timeframe in fetch_calls} == {
        (config.run_spec.market_slice.period, "1d"),
        (config.run_spec.market_slice.period, "1wk"),
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
    assert {run["strategy"]["components"]["optional"]["signals"][0]["dataTimeframe"]["key"] for run in payload["candidateRuns"]} == {
        "1d",
        "1w",
    }
    second_response = client.get("/api/comparison")
    assert second_response.status_code == 200
    second_payload = second_response.json()
    assert second_payload["runStoreSummary"]["computedRunCount"] == 0
    assert (
        second_payload["runStoreSummary"]["cachedRunCount"]
        == len(config.candidate_strategies) * 2
    )


def test_comparison_reuses_existing_runs_when_strategy_added(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.fetch_market_universe_bundle", fake_fetch_market_universe_bundle)
    base_config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)
    config = copy.deepcopy(base_config)
    config.result_store_dir = str(tmp_path / "run_results")
    config.run_spec.market_slice.sanity_periods = []
    config.candidate_strategies = [config.candidate_strategies[0]]
    config.reference_strategies = []
    monkeypatch.setattr(main_module, "DEFAULT_COMPARISON_SPEC", config)

    first_response = client.get("/api/comparison")

    assert first_response.status_code == 200
    first_payload = first_response.json()
    assert len(first_payload["candidateRuns"]) == 1
    assert len(first_payload["referenceRuns"]) == 0
    assert first_payload["runStoreSummary"]["cachedRunCount"] == 0
    assert first_payload["runStoreSummary"]["computedRunCount"] == 1

    config.candidate_strategies.append(copy.deepcopy(base_config.candidate_strategies[1]))

    second_response = client.get("/api/comparison")

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
            label="コスト x1.0 / 投資 85% / 上限なし",
            cost_multiplier=1.0,
            max_investment_ratio=0.85,
            max_weight=None,
        ),
        ConditionVariant(
            key="cash_80",
            label="コスト x2.0 / 投資 80% / 上限なし",
            cost_multiplier=2.0,
            max_investment_ratio=0.8,
            max_weight=None,
        ),
    ]
    monkeypatch.setattr(main_module, "DEFAULT_COMPARISON_SPEC", config)

    first_response = client.get("/api/condition-sweep")

    assert first_response.status_code == 200
    first_payload = first_response.json()
    assert first_payload["comparison"]["comparisonId"] == "etf_portfolio_models_2015_2025"
    assert first_payload["resultCount"] == 2
    assert first_payload["runStoreSummary"]["cachedRunCount"] == 0
    assert first_payload["runStoreSummary"]["computedRunCount"] == 2
    assert len(first_payload["conditionVariants"]) == 2
    assert {variant["costMultiplier"] for variant in first_payload["conditionVariants"]} == {1.0, 2.0}
    assert first_payload["results"][0]["conditionVariant"]["label"]

    second_response = client.get("/api/condition-sweep")

    assert second_response.status_code == 200
    second_payload = second_response.json()
    assert second_payload["runStoreSummary"]["cachedRunCount"] == 2
    assert second_payload["runStoreSummary"]["computedRunCount"] == 0

    config.condition_variants.append(
        ConditionVariant(
            key="cap_45",
            label="コスト x1.0 / 投資 85% / 45%上限",
            cost_multiplier=1.0,
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


def test_latest_run_catalog_endpoint(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.fetch_market_universe_bundle", fake_fetch_market_universe_bundle)
    config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)
    config.result_store_dir = str(tmp_path / "run_results")
    monkeypatch.setattr(main_module, "DEFAULT_COMPARISON_SPEC", config)

    comparison_response = client.get("/api/comparison")
    assert comparison_response.status_code == 200

    catalog_response = client.get("/api/run-catalog", params={"limit": 1, "run_kind": "strategy_run"})
    assert catalog_response.status_code == 200
    fingerprint = catalog_response.json()["records"][0]["strategyDefinitionFingerprint"]

    response = client.get(
        "/api/run-catalog/latest",
        params={
            "run_kind": "strategy_run",
            "strategy_definition_fingerprint": fingerprint,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["filters"]["strategyDefinitionFingerprint"] == fingerprint
    assert payload["record"] is not None
    assert payload["record"]["strategyDefinitionFingerprint"] == fingerprint


def test_latest_run_catalog_endpoint_requires_fingerprint(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.fetch_market_universe_bundle", fake_fetch_market_universe_bundle)
    config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)
    config.result_store_dir = str(tmp_path / "run_results")
    monkeypatch.setattr(main_module, "DEFAULT_COMPARISON_SPEC", config)

    response = client.get("/api/run-catalog/latest")

    assert response.status_code == 400
    assert "At least one fingerprint filter is required" in response.json()["detail"]


def test_run_catalog_endpoint(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.fetch_market_universe_bundle", fake_fetch_market_universe_bundle)
    config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)
    config.result_store_dir = str(tmp_path / "run_results")
    monkeypatch.setattr(main_module, "DEFAULT_COMPARISON_SPEC", config)

    comparison_response = client.get("/api/comparison")
    assert comparison_response.status_code == 200

    response = client.get("/api/run-catalog", params={"limit": 5, "run_kind": "strategy_run"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["comparisonId"] == "etf_portfolio_models_2015_2025"
    assert payload["limit"] == 5
    assert payload["runKind"] == "strategy_run"
    assert payload["recordCount"] == 5
    assert payload["records"][0]["logicVersion"] == "v71"
    assert payload["records"][0]["strategyDefinitionFingerprint"]
    assert payload["records"][0]["evaluationSubjectFingerprint"]
    assert payload["records"][0]["marketDataFingerprint"]
    assert payload["records"][0]["evaluationFingerprint"]
    assert payload["records"][0]["strategyLabel"]
    assert payload["records"][0]["investmentUniverseLabel"]
    assert payload["records"][0]["portfolioModelLabel"]
    assert payload["records"][0]["commissionPct"] is not None
    assert payload["records"][0]["sharpeRatio"] is not None
    assert payload["records"][0]["generationMethod"] is None
    strategy_definition_fingerprint = payload["records"][0]["strategyDefinitionFingerprint"]

    filtered_response = client.get(
        "/api/run-catalog",
        params={
            "limit": 5,
            "run_kind": "strategy_run",
            "strategy_definition_fingerprint": strategy_definition_fingerprint,
        },
    )

    assert filtered_response.status_code == 200
    filtered_payload = filtered_response.json()
    assert filtered_payload["filters"]["strategyDefinitionFingerprint"] == strategy_definition_fingerprint
    assert filtered_payload["recordCount"] >= 1
    assert all(
        record["strategyDefinitionFingerprint"] == strategy_definition_fingerprint
        for record in filtered_payload["records"]
    )


def test_generate_parameter_sweep_runs_endpoint(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.fetch_market_universe_bundle", fake_fetch_market_universe_bundle)
    config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)
    config.result_store_dir = str(tmp_path / "run_results")
    monkeypatch.setattr(main_module, "DEFAULT_COMPARISON_SPEC", config)

    response = client.post("/api/runs/generate-parameter-sweep")

    assert response.status_code == 200
    payload = response.json()
    assert payload["comparison"]["comparisonId"] == "etf_portfolio_models_2015_2025"
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
    expected_ranking_count = len(
        build_asset_ranking_specs_from_strategy_definitions(
            normalize_strategy_definitions(config.candidate_strategies)
        )
    )
    assert payload["comparison"]["comparisonId"] == "etf_portfolio_models_2015_2025"
    assert payload["resultCount"] == expected_ranking_count
    assert payload["runStoreSummary"]["cachedRunCount"] == 0
    assert payload["runStoreSummary"]["computedRunCount"] == expected_ranking_count
    assert payload["results"][0]["rankingSpec"]["rankingModel"]["label"]
    assert payload["results"][0]["overall"]["observationCount"] >= 1
    assert payload["results"][0]["overall"]["meanTopMinusBottomPct"] is not None

    ranking_records = build_run_result_store(config).list_records(
        run_kind="ranking_evaluation",
        limit=1,
    )
    assert len(ranking_records) == 1
    ranking_run_spec = ranking_records[0]["runSpec"]
    assert ranking_run_spec["strategyDefinition"]["kind"] == "strategy_definition"
    assert ranking_run_spec["evaluationSubject"]["kind"] == "ranking"
    assert ranking_run_spec["evaluationSubject"]["ranking"]["kind"] == "asset_ranking_spec"
    assert "strategy" not in ranking_run_spec
    assert set(ranking_run_spec["fingerprints"].keys()) == {"strategyDefinition", "evaluationSubject", "marketData", "evaluation"}

    second_response = client.get("/api/ranking-evaluation")

    assert second_response.status_code == 200
    second_payload = second_response.json()
    assert second_payload["runStoreSummary"]["cachedRunCount"] == expected_ranking_count
    assert second_payload["runStoreSummary"]["computedRunCount"] == 0
