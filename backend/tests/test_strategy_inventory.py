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
    assert payload["counts"]["total"] > 0
    assert all(entry["status"] == "active" for entry in payload["entries"])
    assert all(entry["priority"] == "high" for entry in payload["entries"])


def test_strategy_inventory_payload_rejects_unknown_filter() -> None:
    with pytest.raises(ValueError, match="Unknown strategy inventory status"):
        build_strategy_inventory_payload(status="unknown")
