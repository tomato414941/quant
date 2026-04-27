from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path

import pytest

from app.run_store import FileRunResultStore, RUN_STORE_INDEX_FILENAME, build_run_spec


def make_run_spec(*, run_kind: str, strategy_label: str, fingerprint_seed: str) -> dict:
    return build_run_spec(
        run_kind=run_kind,
        strategy_definition={"label": strategy_label, "strategyId": strategy_label, "seed": fingerprint_seed},
        market_slice={"period": "10y", "timeframe": {"key": "1d"}},
        evaluation={
            "evaluationSettings": {"splitRatioPct": 70.0},
            "marketDataContexts": [{"timeframe": "1d"}],
            "signalMarketDataContexts": [],
        },
        execution_assumptions={"costModel": {"kind": "pct", "parameters": {"commissionPct": 0.1}}},
        portfolio_state={"kind": "portfolio_state", "positions": []},
        capital_base=100000.0,
    )


def test_run_store_writes_index_and_filters_records(tmp_path: Path) -> None:
    store = FileRunResultStore(tmp_path)
    first_run_spec = make_run_spec(
        run_kind="strategy_run",
        strategy_label="alpha",
        fingerprint_seed="alpha",
    )
    second_run_spec = make_run_spec(
        run_kind="strategy_run",
        strategy_label="beta",
        fingerprint_seed="beta",
    )

    store.save(first_run_spec, {"summary": {"sharpeRatio": 1.0}})
    store.save(second_run_spec, {"summary": {"sharpeRatio": 2.0}})

    index_path = tmp_path / RUN_STORE_INDEX_FILENAME
    assert index_path.exists()

    filtered_records = store.list_records(
        run_kind="strategy_run",
        strategy_definition_fingerprint=second_run_spec["fingerprints"]["strategyDefinition"],
    )

    assert len(filtered_records) == 1
    assert filtered_records[0]["runSpec"]["strategyDefinition"]["label"] == "beta"
    assert "strategy" not in filtered_records[0]["runSpec"]
    assert filtered_records[0]["runSpec"]["fingerprints"]["evaluationSubject"]


def test_run_store_rebuilds_index_from_saved_runs(tmp_path: Path) -> None:
    store = FileRunResultStore(tmp_path)
    run_spec = make_run_spec(
        run_kind="predictor_run",
        strategy_label="predictor-alpha",
        fingerprint_seed="predictor-alpha",
    )
    store.save(run_spec, {"summary": {"sharpeRatio": 0.5}})

    index_path = tmp_path / RUN_STORE_INDEX_FILENAME
    index_path.unlink()

    rebuilt_records = store.list_records(run_kind="predictor_run")

    assert len(rebuilt_records) == 1
    assert rebuilt_records[0]["runSpec"]["runKind"] == "predictor_run"
    assert index_path.exists()


def test_run_store_warns_when_rebuilding_corrupt_index(tmp_path: Path, caplog) -> None:
    store = FileRunResultStore(tmp_path)
    run_spec = make_run_spec(
        run_kind="strategy_run",
        strategy_label="alpha",
        fingerprint_seed="alpha",
    )
    store.save(run_spec, {"summary": {"sharpeRatio": 1.0}})
    index_path = tmp_path / RUN_STORE_INDEX_FILENAME
    index_path.write_text("{invalid json", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="app.run_store"):
        records = store.list_records(run_kind="strategy_run")

    assert len(records) == 1
    assert "Rebuilding run store index" in caplog.text
    assert str(index_path) in caplog.text


def test_run_store_concurrent_save_does_not_lose_index_updates(tmp_path: Path) -> None:
    run_specs = [
        make_run_spec(
            run_kind="strategy_run",
            strategy_label=f"strategy-{index}",
            fingerprint_seed=f"strategy-{index}",
        )
        for index in range(24)
    ]

    def save_run(index: int) -> None:
        store = FileRunResultStore(tmp_path)
        store.save(run_specs[index], {"summary": {"sharpeRatio": index}})

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(save_run, range(len(run_specs))))

    index_payload = json.loads((tmp_path / RUN_STORE_INDEX_FILENAME).read_text(encoding="utf-8"))

    assert len(index_payload["entries"]) == len(run_specs)
    assert {entry["strategyDefinitionFingerprint"] for entry in index_payload["entries"]} == {
        run_spec["fingerprints"]["strategyDefinition"] for run_spec in run_specs
    }


def test_run_store_concurrent_save_same_run_key_does_not_duplicate_index(tmp_path: Path) -> None:
    run_spec = make_run_spec(
        run_kind="strategy_run",
        strategy_label="shared",
        fingerprint_seed="shared",
    )

    def save_run(index: int) -> None:
        store = FileRunResultStore(tmp_path)
        store.save(run_spec, {"summary": {"sharpeRatio": index}})

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(save_run, range(24)))

    index_payload = json.loads((tmp_path / RUN_STORE_INDEX_FILENAME).read_text(encoding="utf-8"))
    run_keys = [entry["runKey"] for entry in index_payload["entries"]]

    assert len(run_keys) == 1
    assert len(set(run_keys)) == 1


def test_run_store_atomic_index_write_failure_keeps_existing_index(tmp_path: Path, monkeypatch) -> None:
    store = FileRunResultStore(tmp_path)
    first_run_spec = make_run_spec(
        run_kind="strategy_run",
        strategy_label="alpha",
        fingerprint_seed="alpha",
    )
    second_run_spec = make_run_spec(
        run_kind="strategy_run",
        strategy_label="beta",
        fingerprint_seed="beta",
    )
    store.save(first_run_spec, {"summary": {"sharpeRatio": 1.0}})
    index_path = tmp_path / RUN_STORE_INDEX_FILENAME
    original_index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    original_replace = Path.replace

    def fail_index_replace(self: Path, target: Path) -> Path:
        if target == index_path:
            raise OSError("simulated index replace failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_index_replace)

    with pytest.raises(OSError, match="simulated index replace failure"):
        store.save(second_run_spec, {"summary": {"sharpeRatio": 2.0}})

    assert json.loads(index_path.read_text(encoding="utf-8")) == original_index_payload


def test_run_store_warns_when_index_rebuild_skips_corrupt_run_file(tmp_path: Path, caplog) -> None:
    store = FileRunResultStore(tmp_path)
    run_spec = make_run_spec(
        run_kind="strategy_run",
        strategy_label="alpha",
        fingerprint_seed="alpha",
    )
    store.save(run_spec, {"summary": {"sharpeRatio": 1.0}})
    corrupt_path = tmp_path / "corrupt.json"
    corrupt_path.write_text("{invalid json", encoding="utf-8")
    (tmp_path / RUN_STORE_INDEX_FILENAME).unlink()

    with caplog.at_level(logging.WARNING, logger="app.run_store"):
        records = store.list_records(run_kind="strategy_run")

    assert len(records) == 1
    assert "Skipping corrupt run result file during index rebuild" in caplog.text
    assert str(corrupt_path) in caplog.text


def test_run_store_warns_when_index_rebuild_skips_unsupported_logic_version(
    tmp_path: Path,
    caplog,
) -> None:
    store = FileRunResultStore(tmp_path)
    store.save(
        make_run_spec(run_kind="strategy_run", strategy_label="alpha", fingerprint_seed="alpha"),
        {"summary": {"sharpeRatio": 1.0}},
    )
    stale_path = tmp_path / "stale.json"
    stale_path.write_text(
        json.dumps({"runSpec": {"logicVersion": "old"}, "result": {}}),
        encoding="utf-8",
    )
    (tmp_path / RUN_STORE_INDEX_FILENAME).unlink()

    with caplog.at_level(logging.WARNING, logger="app.run_store"):
        records = store.list_records(run_kind="strategy_run")

    assert len(records) == 1
    assert "unsupported logic version old" in caplog.text
    assert str(stale_path) in caplog.text


def test_predictor_compact_record_reads_evaluation_subject(tmp_path: Path) -> None:
    store = FileRunResultStore(tmp_path)
    run_spec = build_run_spec(
        run_kind="predictor_run",
        strategy_definition={"label": "source strategy", "strategyId": "source"},
        evaluation_subject={
            "kind": "predictor",
            "predictor": {
                "key": "predictor-alpha",
                "label": "Predictor Alpha",
                "timeframe": {"key": "1d"},
                "signalSpec": {
                    "entityIdentifiers": ["AAA", "BBB"],
                    "entityKind": "asset_set",
                    "observationSpec": {
                        "key": "observation-alpha",
                        "label": "Observation Alpha",
                        "assetCount": 2,
                        "fields": ["close"],
                    },
                    "outputSpec": {"outputKind": "score"},
                    "decisionUseSpec": {"useKind": "supplemental"},
                },
                "predictedQuantitySpec": {"quantityKind": "return"},
                "targetSpec": {
                    "key": "target-alpha",
                    "baseline": "zero",
                    "transform": "none",
                    "horizonSpec": {"unit": "bar", "value": 5},
                },
                "featureSpec": {"key": "feature-alpha"},
                "trainingSpec": {"fitMode": "expanding", "minTrainSamples": 10},
                "engineSpec": {
                    "signalSourceSpec": {"signalSourceKind": "ranking_signal", "featureKey": "score"},
                    "learnerSpec": {"learnerKind": "linear_regression"},
                    "combinerSpec": {"combinerKind": "learner_only"},
                },
            },
        },
        market_slice={"period": "10y", "timeframe": {"key": "1d"}},
        evaluation={
            "evaluationSettings": {"splitRatioPct": 70.0},
            "marketDataContexts": [{"timeframe": "1d"}],
            "signalMarketDataContexts": [],
        },
        execution_assumptions={"costModel": {"kind": "pct", "parameters": {"commissionPct": 0.1}}},
        portfolio_state={"kind": "portfolio_state", "positions": []},
        capital_base=100000.0,
    )
    store.save(
        run_spec,
        {
            "overall": {"observationCount": 20, "meanRankIc": 0.2, "meanTopMinusBottomPct": 1.0},
            "test": {"observationCount": 5, "meanRankIc": 0.3, "meanTopMinusBottomPct": 1.5, "hitRatePct": 60.0},
        },
    )

    records = store.list_compact_records(run_kind="predictor_run", view="predictor")

    assert len(records) == 1
    assert records[0]["predictorKey"] == "predictor-alpha"
    assert records[0]["evaluationSubjectFingerprint"] == run_spec["fingerprints"]["evaluationSubject"]
    assert records[0]["strategyDefinitionFingerprint"] == run_spec["fingerprints"]["strategyDefinition"]
    assert records[0]["allocationFallbackCount"] == 0
    assert records[0]["allocationFallbackRate"] == 0.0
    assert records[0]["diagnosticEventCount"] == 0
    assert records[0]["availabilityWarningCount"] == 0
    assert "strategy" not in store.list_records(run_kind="predictor_run")[0]["runSpec"]


def test_run_store_lists_compact_records_from_index_only(tmp_path: Path) -> None:
    store = FileRunResultStore(tmp_path)
    run_spec = make_run_spec(
        run_kind="strategy_run",
        strategy_label="gamma",
        fingerprint_seed="gamma",
    )
    store.save(run_spec, {"summary": {"sharpeRatio": 3.0, "totalReturnPct": 12.0, "maxDrawdownPct": 4.0}})

    def fail_get_record(_run_key: str) -> dict | None:
        raise AssertionError("get_record should not be called")

    store.get_record = fail_get_record  # type: ignore[method-assign]
    compact_records = store.list_compact_records(run_kind="strategy_run")

    assert len(compact_records) == 1
    assert compact_records[0]["strategyLabel"] == "gamma"
    assert compact_records[0]["sharpeRatio"] == 3.0
    assert compact_records[0]["allocationFallbackCount"] == 0
    assert compact_records[0]["allocationFallbackRate"] == 0.0
    assert compact_records[0]["diagnosticEventCount"] == 0
    assert compact_records[0]["availabilityWarningCount"] == 0


def test_run_store_compact_record_includes_diagnostic_metrics(tmp_path: Path) -> None:
    store = FileRunResultStore(tmp_path)
    run_spec = make_run_spec(
        run_kind="strategy_run",
        strategy_label="diagnostic",
        fingerprint_seed="diagnostic",
    )
    run_spec["evaluation"]["availabilityDiagnostics"] = {"warningCount": 3}
    run_spec["evaluation"]["diagnosticEvents"] = [
        {"kind": "requested_asset_unavailable", "category": "availability"},
        {"kind": "mixed_market_calendar", "category": "calendar"},
    ]
    store.save(
        run_spec,
        {
            "summary": {"sharpeRatio": 3.0, "totalReturnPct": 12.0, "maxDrawdownPct": 4.0},
            "executionTrace": [
                {"eventType": "decision", "allocationFallback": {"fallback": "equal_weight"}},
                {"eventType": "rebalance"},
                {"eventType": "decision"},
            ],
        },
    )

    compact_records = store.list_compact_records(run_kind="strategy_run")

    assert len(compact_records) == 1
    assert compact_records[0]["allocationFallbackCount"] == 1
    assert compact_records[0]["allocationFallbackRate"] == 0.5
    assert compact_records[0]["diagnosticEventCount"] == 2
    assert compact_records[0]["availabilityWarningCount"] == 3


def test_run_store_backfills_diagnostic_metrics_for_existing_compact_index(tmp_path: Path) -> None:
    store = FileRunResultStore(tmp_path)
    run_spec = make_run_spec(
        run_kind="strategy_run",
        strategy_label="legacy-index",
        fingerprint_seed="legacy-index",
    )
    store.save(
        run_spec,
        {
            "summary": {"sharpeRatio": 3.0},
            "executionTrace": [
                {"eventType": "decision", "allocationFallback": {"fallback": "equal_weight"}},
                {"eventType": "decision"},
            ],
        },
    )
    index_path = tmp_path / RUN_STORE_INDEX_FILENAME
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    compact_record = index_payload["entries"][0]["genericCompactRecord"]
    for key in (
        "allocationFallbackCount",
        "allocationFallbackRate",
        "diagnosticEventCount",
        "availabilityWarningCount",
    ):
        compact_record.pop(key)
    index_path.write_text(json.dumps(index_payload), encoding="utf-8")

    compact_records = store.list_compact_records(run_kind="strategy_run")

    assert compact_records[0]["allocationFallbackCount"] == 1
    assert compact_records[0]["allocationFallbackRate"] == 0.5
    assert compact_records[0]["diagnosticEventCount"] == 0
    assert compact_records[0]["availabilityWarningCount"] == 0


def test_run_store_rebuild_index_returns_entry_count(tmp_path: Path) -> None:
    store = FileRunResultStore(tmp_path)
    store.save(
        make_run_spec(run_kind="strategy_run", strategy_label="delta", fingerprint_seed="delta"),
        {"summary": {"sharpeRatio": 1.5}},
    )

    payload = store.rebuild_index()

    assert payload == {"entryCount": 1}


def test_run_store_finds_latest_compact_record_by_fingerprint(tmp_path: Path) -> None:
    store = FileRunResultStore(tmp_path)
    first_run_spec = make_run_spec(run_kind="strategy_run", strategy_label="older", fingerprint_seed="shared")
    second_run_spec = make_run_spec(run_kind="strategy_run", strategy_label="newer", fingerprint_seed="shared")

    store.save(first_run_spec, {"summary": {"sharpeRatio": 1.0}})
    store.save(second_run_spec, {"summary": {"sharpeRatio": 2.0}})

    record = store.find_latest_compact_record(
        run_kind="strategy_run",
        strategy_definition_fingerprint=second_run_spec["fingerprints"]["strategyDefinition"],
    )

    assert record is not None
    assert record["strategyLabel"] == "newer"
    assert record["sharpeRatio"] == 2.0
