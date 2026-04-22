import copy
import json
import math

import pandas as pd

from app import cli as cli_module
from app import main as main_module
from app.comparison_service import sort_walk_forward_results


def fake_fetch_market_universe_bundle(
    tickers: list[str],
    period: str,
    timeframe: str = "1d",
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[dict, dict]:
    if period == "3y":
        closes = pd.DataFrame(
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
    else:
        closes = pd.DataFrame(
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

    aligned_closes = closes[tickers]
    volumes = pd.DataFrame(
        {
            ticker: [1_000_000 + row_index * 10_000 for row_index in range(len(aligned_closes))]
            for ticker in aligned_closes.columns
        },
        index=aligned_closes.index,
    )
    return {
        "closes": aligned_closes,
        "volumes": volumes,
    }, {
        "tickers": tickers,
        "period": period,
        "source": "test",
        "timeframe": timeframe,
        "aligned_start_date": aligned_closes.index[0],
        "aligned_end_date": aligned_closes.index[-1],
        "row_count": len(aligned_closes),
    }



def fake_fetch_market_universe_bundle_multiyear(
    tickers: list[str],
    period: str,
    timeframe: str = "1d",
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[dict, dict]:
    if timeframe == "1mo":
        index = pd.date_range("2019-01-31", "2021-12-31", freq="ME")
    elif timeframe == "1wk":
        index = pd.date_range("2019-01-04", "2021-12-31", freq="W-FRI")
    else:
        index = pd.date_range("2019-01-01", "2021-12-31", freq="D")

    data = {}
    for ticker_index, ticker in enumerate(tickers):
        base = 90.0 + ticker_index * 3.0
        drift = 0.03 + ticker_index * 0.002
        data[ticker] = [
            base + row_index * drift + math.sin(row_index / 9 + ticker_index) * 0.5
            for row_index in range(len(index))
        ]
    closes = pd.DataFrame(data, index=index)
    volumes = pd.DataFrame(
        {
            ticker: [1_000_000 + ticker_index * 10_000 + row_index * 100 for row_index in range(len(index))]
            for ticker_index, ticker in enumerate(closes.columns)
        },
        index=index,
    )
    return {
        "closes": closes,
        "volumes": volumes,
    }, {
        "tickers": tickers,
        "period": period,
        "source": "test",
        "timeframe": timeframe,
        "aligned_start_date": str(closes.index[0].date()),
        "aligned_end_date": str(closes.index[-1].date()),
        "row_count": len(closes),
    }


def configure_cli(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_module, "fetch_market_universe_bundle", fake_fetch_market_universe_bundle)
    config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)
    config.result_store_dir = str(tmp_path / "run_results")
    monkeypatch.setattr(cli_module, "DEFAULT_COMPARISON_SPEC", config)
    return config


def test_comparison_summary_command(monkeypatch, tmp_path, capsys) -> None:
    config = configure_cli(monkeypatch, tmp_path)

    exit_code = cli_module.main(["comparison-summary", "--top", "2"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert config.comparison_id in captured.out
    assert "Top 2 candidate runs by test performance:" in captured.out
    assert "Test Sharpe" in captured.out



def configure_cli_multiyear(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_module, "fetch_market_universe_bundle", fake_fetch_market_universe_bundle_multiyear)
    config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)
    config.result_store_dir = str(tmp_path / "run_results")
    config.candidate_strategies = config.candidate_strategies[:3]
    config.reference_strategies = config.reference_strategies[:1]
    config.run_spec.market_slice.start_date = "2019-01-01"
    config.run_spec.market_slice.end_date = "2021-12-31"
    monkeypatch.setattr(cli_module, "DEFAULT_COMPARISON_SPEC", config)
    return config


def test_comparison_summary_walk_forward_command(monkeypatch, tmp_path, capsys) -> None:
    config = configure_cli_multiyear(monkeypatch, tmp_path)

    exit_code = cli_module.main([
        "comparison-summary",
        "--walk-forward",
        "--walk-forward-start-year",
        "2020",
        "--walk-forward-end-year",
        "2021",
        "--top",
        "2",
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert config.comparison_id in captured.out
    assert "Walk-forward: 2020-2021 (2 windows)" in captured.out
    assert "Top 2 candidate strategies by walk-forward test performance:" in captured.out
    assert "Avg Sharpe" in captured.out
    assert "Min Sharpe" in captured.out
    assert "Availability policy:" in captured.out
    assert "Market availability:" in captured.out
    assert "Test eligible assets" in captured.out
    assert "Warnings:" not in captured.out


def test_comparison_summary_walk_forward_command_json(monkeypatch, tmp_path, capsys) -> None:
    configure_cli_multiyear(monkeypatch, tmp_path)

    exit_code = cli_module.main([
        "comparison-summary",
        "--walk-forward",
        "--walk-forward-start-year",
        "2020",
        "--walk-forward-end-year",
        "2021",
        "--json",
    ])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["kind"] == "walk_forward_comparison"
    assert payload["walkForward"]["startYear"] == 2020
    assert payload["walkForward"]["endYear"] == 2021
    assert payload["walkForward"]["windowCount"] == 2
    assert payload["candidateResults"][0]["windowCount"] == 2
    assert "averageSharpeRatio" in payload["candidateResults"][0]
    assert "minimumSharpeRatio" in payload["candidateResults"][0]
    assert "minTestEligibleAssetCount" in payload["candidateResults"][0]
    assert "testAvailability" in payload["candidateResults"][0]["windows"][0]


def test_benchmark_decomposition_command_json(monkeypatch, tmp_path, capsys) -> None:
    configure_cli_multiyear(monkeypatch, tmp_path)

    exit_code = cli_module.main([
        "benchmark-decomposition",
        "--walk-forward-start-year",
        "2020",
        "--walk-forward-end-year",
        "2021",
        "--json",
    ])

    payload = json.loads(capsys.readouterr().out)
    benchmarks_by_key = {benchmark["strategyKey"]: benchmark for benchmark in payload["benchmarks"]}
    assert exit_code == 0
    assert payload["kind"] == "benchmark_decomposition"
    assert payload["schemaVersion"] == "v1"
    assert payload["baselineKey"] == "ref-fu-eq-cash-15"
    assert payload["walkForward"]["startYear"] == 2020
    assert payload["walkForward"]["endYear"] == 2021
    assert len(payload["benchmarks"]) == 6
    assert {
        "ref-fu-eq-cash-15",
        "ref-fu-eq-cash-0",
        "ref-fu-eq-cash-25",
        "ref-etf-eq-cash-15",
        "ref-spy-hold",
        "ref-spy-cash-15",
    } == set(benchmarks_by_key)
    assert benchmarks_by_key["ref-fu-eq-cash-15"]["summary"]["windowCount"] == 2
    assert benchmarks_by_key["ref-fu-eq-cash-15"]["deltaVsBaseline"]["averageSharpeRatio"] == 0.0
    assert any(
        row["asset"] == "CASH" and row["weightPct"] == 15.0
        for row in benchmarks_by_key["ref-fu-eq-cash-15"]["finalWeights"]
    )
    assert not any(
        row["asset"] in {"BTC-USD", "ETH-USD"}
        for row in benchmarks_by_key["ref-etf-eq-cash-15"]["finalWeights"]
    )
    assert {row["asset"] for row in benchmarks_by_key["ref-spy-hold"]["finalWeights"]} == {"SPY"}
    assert benchmarks_by_key["ref-spy-hold"]["worstWindow"] is not None


def test_benchmark_decomposition_command(monkeypatch, tmp_path, capsys) -> None:
    configure_cli_multiyear(monkeypatch, tmp_path)

    exit_code = cli_module.main([
        "benchmark-decomposition",
        "--walk-forward-start-year",
        "2020",
        "--walk-forward-end-year",
        "2021",
        "--top",
        "3",
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Benchmark decomposition" in captured.out
    assert "Walk-forward: 2020-2021 (2 windows)" in captured.out
    assert "Top 3 benchmarks by walk-forward performance:" in captured.out
    assert "Delta vs ref-fu-eq-cash-15" in captured.out


def test_comparison_summary_walk_forward_respects_universe_variant(monkeypatch, tmp_path, capsys) -> None:
    configure_cli_multiyear(monkeypatch, tmp_path)

    exit_code = cli_module.main([
        "comparison-summary",
        "--walk-forward",
        "--walk-forward-start-year",
        "2020",
        "--walk-forward-end-year",
        "2020",
        "--universe",
        "no_crypto",
        "--json",
    ])

    payload = json.loads(capsys.readouterr().out)
    tickers = payload["comparison"]["marketUniverse"]["tickers"]
    assert exit_code == 0
    assert "BTC-USD" not in tickers
    assert "ETH-USD" not in tickers


def configure_cli_robustness(monkeypatch, tmp_path):
    config = configure_cli_multiyear(monkeypatch, tmp_path)
    monkeypatch.setattr(
        cli_module.robustness_service,
        "ROBUSTNESS_PERIODS",
        (
            {
                "key": "2019_2021",
                "label": "2019-2021",
                "startDate": "2019-01-01",
                "endDate": "2021-12-31",
                "walkForwardStartYear": 2020,
                "walkForwardEndYear": 2020,
            },
        ),
    )
    monkeypatch.setattr(cli_module.robustness_service, "ROBUSTNESS_UNIVERSES", ("crypto_included",))
    monkeypatch.setattr(cli_module.robustness_service, "ROBUSTNESS_COST_MULTIPLIERS", (1.0,))
    monkeypatch.setattr(cli_module.robustness_service, "ROBUSTNESS_MAX_WEIGHTS", (0.35,))
    monkeypatch.setattr(cli_module.robustness_service, "ROBUSTNESS_SMOKE_PERIOD_KEYS", ("2019_2021",))
    monkeypatch.setattr(cli_module.robustness_service, "ROBUSTNESS_SMOKE_UNIVERSES", ("crypto_included",))
    monkeypatch.setattr(cli_module.robustness_service, "ROBUSTNESS_SMOKE_COST_MULTIPLIERS", (1.0,))
    monkeypatch.setattr(cli_module.robustness_service, "ROBUSTNESS_SMOKE_MAX_WEIGHTS", (0.35,))
    monkeypatch.setattr(cli_module.robustness_service, "ROBUSTNESS_QUICK_PERIOD_KEYS", ("2019_2021",))
    monkeypatch.setattr(cli_module.robustness_service, "ROBUSTNESS_QUICK_UNIVERSES", ("crypto_included",))
    monkeypatch.setattr(cli_module.robustness_service, "ROBUSTNESS_QUICK_COST_MULTIPLIERS", (1.0,))
    monkeypatch.setattr(cli_module.robustness_service, "ROBUSTNESS_QUICK_MAX_WEIGHTS", (0.35,))
    return config


def test_robustness_summary_command(monkeypatch, tmp_path, capsys) -> None:
    configure_cli_robustness(monkeypatch, tmp_path)

    exit_code = cli_module.main(["robustness-summary", "--top", "2"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Robustness summary: 1 scenarios (quick profile)" in captured.out
    assert "Decisions:" in captured.out
    assert "Top 2 strategies by robustness:" in captured.out
    assert "Decision" in captured.out
    assert "Reasons:" in captured.out
    assert "Worst Sharpe" in captured.out
    assert "Risks:" in captured.out
    assert "Worst scenario" in captured.out
    assert "Worst window" in captured.out


def test_robustness_summary_command_json(monkeypatch, tmp_path, capsys) -> None:
    configure_cli_robustness(monkeypatch, tmp_path)

    exit_code = cli_module.main(["robustness-summary", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["kind"] == "robustness_summary"
    assert payload["schemaVersion"] == "v1"
    assert payload["profile"] == "quick"
    assert payload["scenarioCount"] == 1
    assert "elapsedSeconds" in payload
    assert "elapsedSeconds" in payload["scenarioResults"][0]
    assert payload["decisionSummary"]["strategyCount"] == len(payload["strategyResults"])
    assert payload["matrix"]["universes"] == ["crypto_included"]
    assert payload["scenarioResults"][0]["scenario"]["period"] == "2019-2021"
    assert payload["strategyResults"][0]["decision"] in {"PASS", "WATCH", "FAIL", "INVALID"}
    assert "decisionReasons" in payload["strategyResults"][0]
    assert "diagnosticSummary" in payload["strategyResults"][0]
    assert "diagnosticEvents" in payload["scenarioResults"][0]["diagnostics"]
    assert "worstSharpeRatio" in payload["strategyResults"][0]
    assert "top5ScenarioCount" in payload["strategyResults"][0]
    assert "worstScenario" in payload["strategyResults"][0]
    assert "worstWindow" in payload["strategyResults"][0]
    assert "windows" in payload["scenarioResults"][0]["results"][0]
    assert "worstWindow" in payload["scenarioResults"][0]["results"][0]


def test_robustness_summary_command_filters_strategy_keys(monkeypatch, tmp_path, capsys) -> None:
    config = configure_cli_robustness(monkeypatch, tmp_path)
    selected_key = config.candidate_strategies[0].key

    exit_code = cli_module.main([
        "robustness-summary",
        "--json",
        "--strategy-key",
        selected_key,
    ])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert {result["strategyKey"] for result in payload["strategyResults"]} == {selected_key}
    assert {
        result["strategyKey"]
        for scenario in payload["scenarioResults"]
        for result in scenario["results"]
    } == {selected_key}


def test_robustness_summary_command_progress_and_output_json(monkeypatch, tmp_path, capsys) -> None:
    configure_cli_robustness(monkeypatch, tmp_path)
    output_path = tmp_path / "robustness-summary.json"

    exit_code = cli_module.main([
        "robustness-summary",
        "--profile",
        "smoke",
        "--progress",
        "--json",
        "--output",
        str(output_path),
    ])

    captured = capsys.readouterr()
    stdout_payload = json.loads(captured.out)
    file_payload = json.loads(output_path.read_text())
    assert exit_code == 0
    assert stdout_payload["profile"] == "smoke"
    assert stdout_payload["scenarioCount"] == 1
    assert stdout_payload["kind"] == file_payload["kind"]
    assert stdout_payload["scenarioCount"] == file_payload["scenarioCount"]
    assert "scenario 1/1 started" in captured.err
    assert "scenario 1/1 done" in captured.err
    assert "elapsed=" in captured.err


def test_render_availability_diagnostics_shows_calendar_boundary_classification() -> None:
    lines = []

    cli_module.render_availability_diagnostics(
        lines,
        {
            "actionableWarnings": [],
            "calendarBoundaryWarningCount": 2,
            "calendarBoundaryWarnings": [
                {
                    "kind": "aligned_start_after_requested_start",
                    "message": "1d data starts at 2025-01-02, after requested start 2025-01-01.",
                },
                {
                    "kind": "aligned_end_before_requested_end",
                    "message": "1d data ends at 2025-12-25, before requested end 2025-12-29.",
                },
            ],
        },
    )

    output = "\n".join(lines)
    assert "Warnings:" not in output
    assert "Calendar boundary differences:" in output
    assert "classified as non-actionable" in output
    assert "1d data ends" not in output


def test_render_availability_diagnostics_keeps_actionable_risks() -> None:
    lines = []

    cli_module.render_availability_diagnostics(
        lines,
        {
            "actionableWarnings": [
                {
                    "kind": "requested_asset_unavailable",
                    "message": "1d data has no usable rows for MISSING.",
                }
            ],
            "calendarBoundaryWarningCount": 0,
            "calendarBoundaryWarnings": [],
            "assetLifecycleWarningCount": 1,
            "assetLifecycleWarnings": [
                {
                    "kind": "asset_available_after_aligned_start",
                    "message": "ETH-USD becomes available on 2017-11-09, after aligned start 2015-01-01.",
                }
            ],
        },
    )

    output = "\n".join(lines)
    assert "Warnings:" in output
    assert "MISSING" in output
    assert "Asset lifecycle differences:" in output
    assert "ETH-USD becomes available" not in output


def test_sort_candidate_runs_uses_test_metrics() -> None:
    weak_test_run = {
        "summary": {"sharpeRatio": 9.0, "totalReturnPct": 90.0, "maxDrawdownPct": 1.0},
        "splitAnalysis": {
            "train": {"portfolio": {"sharpeRatio": 9.0, "totalReturnPct": 90.0, "maxDrawdownPct": 1.0}},
            "test": {"portfolio": {"sharpeRatio": 0.1, "totalReturnPct": 1.0, "maxDrawdownPct": 9.0}},
        },
    }
    strong_test_run = {
        "summary": {"sharpeRatio": 1.0, "totalReturnPct": 10.0, "maxDrawdownPct": 5.0},
        "splitAnalysis": {
            "train": {"portfolio": {"sharpeRatio": 1.0, "totalReturnPct": 10.0, "maxDrawdownPct": 5.0}},
            "test": {"portfolio": {"sharpeRatio": 2.0, "totalReturnPct": 5.0, "maxDrawdownPct": 4.0}},
        },
    }

    assert cli_module.sort_candidate_runs([weak_test_run, strong_test_run])[0] is strong_test_run


def test_sort_walk_forward_results_uses_minimum_sharpe_tiebreak() -> None:
    fragile = {
        "strategyKey": "fragile",
        "averageSharpeRatio": 2.0,
        "minimumSharpeRatio": -1.0,
        "averageTotalReturnPct": 20.0,
        "averageMaxDrawdownPct": 8.0,
    }
    stable = {
        "strategyKey": "stable",
        "averageSharpeRatio": 2.0,
        "minimumSharpeRatio": 1.0,
        "averageTotalReturnPct": 10.0,
        "averageMaxDrawdownPct": 4.0,
    }

    assert sort_walk_forward_results([fragile, stable])[0] is stable


def test_apply_comparison_universe_variant_removes_crypto_assets() -> None:
    config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)

    filtered = cli_module.apply_comparison_universe_variant(config, "no_crypto")

    assert filtered.comparison_id.endswith("__no_crypto")
    assert "BTC-USD" not in filtered.run_spec.portfolio_state.current_weights
    assert "ETH-USD" not in filtered.run_spec.portfolio_state.current_weights
    assert filtered.run_spec.portfolio_state.cash_weight > config.run_spec.portfolio_state.cash_weight
    assert all(
        "BTC-USD" not in strategy.investment_universe.tickers
        and "ETH-USD" not in strategy.investment_universe.tickers
        for strategy in filtered.candidate_strategies + filtered.reference_strategies
    )
    assert all(
        "BTC-USD" not in signal.observation_spec.tickers
        and "ETH-USD" not in signal.observation_spec.tickers
        for strategy in filtered.candidate_strategies + filtered.reference_strategies
        for signal in strategy.signals
    )


def test_apply_comparison_universe_variant_keeps_btc_only() -> None:
    config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)

    filtered = cli_module.apply_comparison_universe_variant(config, "btc_only")

    assert "BTC-USD" in filtered.run_spec.portfolio_state.current_weights
    assert "ETH-USD" not in filtered.run_spec.portfolio_state.current_weights
    assert any(
        "BTC-USD" in strategy.investment_universe.tickers
        for strategy in filtered.candidate_strategies + filtered.reference_strategies
    )
    assert all(
        "ETH-USD" not in strategy.investment_universe.tickers
        for strategy in filtered.candidate_strategies + filtered.reference_strategies
    )


def test_run_catalog_command_json(monkeypatch, tmp_path, capsys) -> None:
    config = configure_cli(monkeypatch, tmp_path)

    cli_module.main(["comparison-summary", "--top", "1"])
    capsys.readouterr()

    exit_code = cli_module.main(["run-catalog", "--limit", "3", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["comparisonId"] == config.comparison_id
    assert payload["recordCount"] > 0
    assert payload["records"][0]["logicVersion"] == "v66"
    assert payload["records"][0]["strategyDefinitionFingerprint"]
    assert payload["records"][0]["evaluationSubjectFingerprint"]
    assert payload["records"][0]["marketDataFingerprint"]
    assert payload["records"][0]["evaluationFingerprint"]

    fingerprint = payload["records"][0]["strategyDefinitionFingerprint"]
    exit_code = cli_module.main(
        [
            "run-catalog",
            "--limit",
            "3",
            "--strategy-definition-fingerprint",
            fingerprint,
            "--json",
        ]
    )

    captured = capsys.readouterr()
    filtered_payload = json.loads(captured.out)
    assert exit_code == 0
    assert filtered_payload["filters"]["strategyDefinitionFingerprint"] == fingerprint
    assert filtered_payload["recordCount"] >= 1
    assert all(
        record["strategyDefinitionFingerprint"] == fingerprint
        for record in filtered_payload["records"]
    )


def test_rebuild_run_index_command_json(monkeypatch, tmp_path, capsys) -> None:
    configure_cli(monkeypatch, tmp_path)

    cli_module.main(["comparison-summary", "--top", "1"])
    capsys.readouterr()

    exit_code = cli_module.main(["rebuild-run-index", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["entryCount"] >= 1


def test_comparison_run_spec_command_json(monkeypatch, tmp_path, capsys) -> None:
    config = configure_cli(monkeypatch, tmp_path)

    exit_code = cli_module.main(["comparison-run-spec", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["comparisonId"] == config.comparison_id
    assert payload["kind"] == "comparison_run_spec_payload"
    assert payload["runSpec"]["kind"] == "comparison_run_spec"
    assert payload["comparisonFingerprint"]
    assert payload["runSpecFingerprint"]
    assert payload["selectionPolicy"]["primaryMetric"] == config.selection_policy.primary_metric
    assert len(payload["candidateStrategies"]) == len(config.candidate_strategies)
    assert len(payload["referenceStrategies"]) == len(config.reference_strategies)
    assert payload["candidateStrategies"][0]["kind"] == "strategy_definition"
    assert payload["referenceStrategies"][0]["kind"] == "strategy_definition"


def test_rerun_comparison_spec_command_json(monkeypatch, tmp_path, capsys) -> None:
    config = configure_cli(monkeypatch, tmp_path)
    spec_file = tmp_path / "comparison-run-spec.json"

    exit_code = cli_module.main(["comparison-run-spec", "--json"])
    captured = capsys.readouterr()
    assert exit_code == 0
    spec_file.write_text(captured.out)

    exit_code = cli_module.main(["rerun-comparison-spec", str(spec_file), "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["comparison"]["comparisonId"] == config.comparison_id
    assert len(payload["candidateRuns"]) == len(config.candidate_strategies)
    assert len(payload["referenceRuns"]) == len(config.reference_strategies)
    assert payload["runStoreSummary"]["cachedRunCount"] + payload["runStoreSummary"]["computedRunCount"] > 0


def test_latest_run_command_json(monkeypatch, tmp_path, capsys) -> None:
    configure_cli(monkeypatch, tmp_path)

    cli_module.main(["comparison-summary", "--top", "1"])
    capsys.readouterr()

    cli_module.main(["run-catalog", "--limit", "1", "--json"])
    catalog_payload = json.loads(capsys.readouterr().out)
    fingerprint = catalog_payload["records"][0]["strategyDefinitionFingerprint"]

    exit_code = cli_module.main([
        "latest-run",
        "--run-kind",
        "strategy_run",
        "--strategy-definition-fingerprint",
        fingerprint,
        "--json",
    ])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["filters"]["strategyDefinitionFingerprint"] == fingerprint
    assert payload["record"] is not None
    assert payload["record"]["strategyDefinitionFingerprint"] == fingerprint


def test_latest_run_command_requires_fingerprint(monkeypatch, tmp_path) -> None:
    configure_cli(monkeypatch, tmp_path)

    try:
        cli_module.main(["latest-run", "--json"])
    except ValueError as exc:
        assert "At least one fingerprint filter is required" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_rerun_comparison_spec_command_rejects_mismatched_fingerprint(monkeypatch, tmp_path, capsys) -> None:
    configure_cli(monkeypatch, tmp_path)
    spec_file = tmp_path / "comparison-run-spec-invalid.json"

    exit_code = cli_module.main(["comparison-run-spec", "--json"])
    captured = capsys.readouterr()
    assert exit_code == 0

    payload = json.loads(captured.out)
    payload["runSpecFingerprint"] = "invalid"
    spec_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    try:
        cli_module.main(["rerun-comparison-spec", str(spec_file), "--json"])
    except ValueError as exc:
        assert "runSpecFingerprint does not match" in str(exc)
    else:
        raise AssertionError("expected ValueError")
