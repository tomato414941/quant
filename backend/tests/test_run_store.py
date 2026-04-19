from pathlib import Path

from app.run_store import FileRunResultStore, RUN_STORE_INDEX_FILENAME, build_run_spec


def make_run_spec(*, run_kind: str, strategy_label: str, fingerprint_seed: str) -> dict:
    return build_run_spec(
        run_kind=run_kind,
        strategy={"label": strategy_label, "strategyId": strategy_label},
        strategy_definition={"label": strategy_label, "seed": fingerprint_seed},
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
    assert filtered_records[0]["runSpec"]["strategy"]["label"] == "beta"


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
