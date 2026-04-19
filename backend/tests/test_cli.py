import copy
import json

import pandas as pd

from app import cli as cli_module
from app import main as main_module


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
    assert payload["records"][0]["logicVersion"] == "v60"
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
