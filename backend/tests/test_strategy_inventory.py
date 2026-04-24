from __future__ import annotations

import pytest

from app.strategy_catalog import CANONICAL_CANDIDATE_DEFINITIONS
from app.strategy_inventory import (
    EXPLICIT_STRATEGY_INVENTORY_ENTRIES,
    build_strategy_inventory_entries,
    build_strategy_inventory_payload,
)


def test_strategy_inventory_covers_all_canonical_candidate_strategies() -> None:
    canonical_ids = {definition.strategy_id for definition in CANONICAL_CANDIDATE_DEFINITIONS}
    inventory_ids = {entry.strategy_id for entry in build_strategy_inventory_entries()}

    assert inventory_ids == canonical_ids


def test_strategy_inventory_entries_are_unique() -> None:
    entries = build_strategy_inventory_entries()

    assert len({entry.strategy_id for entry in entries}) == len(entries)


def test_explicit_strategy_inventory_entries_reference_known_strategies() -> None:
    canonical_ids = {definition.strategy_id for definition in CANONICAL_CANDIDATE_DEFINITIONS}

    for entry in EXPLICIT_STRATEGY_INVENTORY_ENTRIES:
        assert entry.strategy_id in canonical_ids
        if entry.baseline_strategy_id is not None:
            assert entry.baseline_strategy_id in canonical_ids


@pytest.mark.parametrize(
    "strategy_id",
    [
        "stg-fu-eq",
        "stg-fu-hrp",
        "stg-fu-momo12-top035-hrp",
        "stg-fu-momo2-top035-hrp-month",
        "stg-top3-hrp",
        "stg-dualtop3-hrp",
        "stg-posmom-hrp-month",
        "stg-riskoff-posmom-hrp-month",
        "stg-fu-momolv8515-top025-hrp-month",
        "stg-fu-momo2-top035-pred10mom5050-pw40-hrp-month",
    ],
)
def test_representative_strategy_inventory_entries_are_active(strategy_id: str) -> None:
    entries_by_id = {entry.strategy_id: entry for entry in build_strategy_inventory_entries()}

    assert entries_by_id[strategy_id].status == "active"
    assert entries_by_id[strategy_id].priority in {"high", "medium"}


def test_generated_strategy_inventory_entries_default_to_backlog_low_priority() -> None:
    entries_by_id = {entry.strategy_id: entry for entry in build_strategy_inventory_entries()}

    entry = entries_by_id["stg-fu-rb"]

    assert entry.status == "backlog"
    assert entry.priority == "low"
    assert entry.notes.startswith("Generated backlog entry.")


def test_strategy_inventory_payload_filters_entries() -> None:
    payload = build_strategy_inventory_payload(status="active", priority="high")

    assert payload["kind"] == "strategy_inventory"
    assert payload["schemaVersion"] == "v1"
    assert payload["withLatestRuns"] is False
    assert payload["counts"]["total"] > 0
    assert all(entry["status"] == "active" for entry in payload["entries"])
    assert all(entry["priority"] == "high" for entry in payload["entries"])


def test_strategy_inventory_payload_attaches_latest_run_records() -> None:
    payload = build_strategy_inventory_payload(
        status="active",
        with_latest_runs=True,
        latest_run_records=[
            {
                "runKey": "candidate-newer",
                "savedAtUtc": "2026-04-24T00:00:00Z",
                "strategyId": "stg-fu-momo12-top035-hrp",
                "period": "10y",
                "timeframe": "1d",
                "sharpeRatio": 1.25,
                "totalReturnPct": 42.0,
                "maxDrawdownPct": -12.0,
            },
            {
                "runKey": "baseline",
                "savedAtUtc": "2026-04-24T00:00:00Z",
                "strategyId": "stg-fu-hrp",
                "period": "10y",
                "timeframe": "1d",
                "sharpeRatio": 1.0,
                "totalReturnPct": 35.0,
                "maxDrawdownPct": -10.0,
            },
            {
                "runKey": "candidate-older",
                "savedAtUtc": "2026-04-23T00:00:00Z",
                "strategyId": "stg-fu-momo12-top035-hrp",
                "period": "10y",
                "timeframe": "1d",
                "sharpeRatio": 0.5,
                "totalReturnPct": 20.0,
                "maxDrawdownPct": -20.0,
            },
        ],
    )
    entries_by_id = {entry["strategyId"]: entry for entry in payload["entries"]}

    entry = entries_by_id["stg-fu-momo12-top035-hrp"]

    assert payload["withLatestRuns"] is True
    assert entry["runStatus"] == "available"
    assert entry["latestRun"]["runKey"] == "candidate-newer"
    assert entry["baselineComparison"] == {
        "baselineStrategyId": "stg-fu-hrp",
        "baselineRunKey": "baseline",
        "deltaSharpeRatio": 0.25,
        "deltaTotalReturnPct": 7.0,
        "deltaMaxDrawdownPct": -2.0,
    }


def test_strategy_inventory_payload_marks_missing_latest_runs() -> None:
    payload = build_strategy_inventory_payload(
        status="active",
        with_latest_runs=True,
        latest_run_records=[],
    )

    assert all(entry["runStatus"] == "missing" for entry in payload["entries"])
    assert all(entry["latestRun"] is None for entry in payload["entries"])
    assert all(entry["baselineComparison"] is None for entry in payload["entries"])


def test_strategy_inventory_payload_rejects_unknown_filter() -> None:
    with pytest.raises(ValueError, match="Unknown strategy inventory status"):
        build_strategy_inventory_payload(status="unknown")
