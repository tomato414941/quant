from __future__ import annotations

from dataclasses import dataclass

from app.strategy_catalog import CANONICAL_CANDIDATE_DEFINITIONS


STRATEGY_INVENTORY_SCHEMA_VERSION = "v1"
STRATEGY_INVENTORY_STATUSES = ("active", "backlog", "rejected")
STRATEGY_INVENTORY_PRIORITIES = ("high", "medium", "low")


@dataclass(frozen=True)
class StrategyInventoryEntry:
    strategy_id: str
    family: str
    role: str
    status: str
    priority: str
    baseline_strategy_id: str | None
    expected_regimes: tuple[str, ...]
    risk_regimes: tuple[str, ...]
    tags: tuple[str, ...]
    notes: str


EXPLICIT_STRATEGY_INVENTORY_ENTRIES = (
    StrategyInventoryEntry(
        strategy_id="stg-fu-eq",
        family="baseline",
        role="baseline",
        status="active",
        priority="high",
        baseline_strategy_id=None,
        expected_regimes=("broad_market",),
        risk_regimes=("benchmark",),
        tags=("full_universe", "equal_weight", "reference_point"),
        notes="Full-universe equal-weight baseline for return and drawdown context.",
    ),
    StrategyInventoryEntry(
        strategy_id="stg-fu-hrp",
        family="baseline",
        role="baseline",
        status="active",
        priority="high",
        baseline_strategy_id="stg-fu-eq",
        expected_regimes=("broad_market", "diversification"),
        risk_regimes=("benchmark",),
        tags=("full_universe", "hrp", "reference_point"),
        notes="Full-universe HRP baseline for portfolio construction effects.",
    ),
    StrategyInventoryEntry(
        strategy_id="stg-fu-momo12-top035-hrp",
        family="momentum_tilt",
        role="candidate",
        status="active",
        priority="high",
        baseline_strategy_id="stg-fu-hrp",
        expected_regimes=("trend", "risk_on"),
        risk_regimes=("momentum_reversal",),
        tags=("full_universe", "momentum", "12m", "hrp"),
        notes="Core annual 12-month momentum tilt candidate.",
    ),
    StrategyInventoryEntry(
        strategy_id="stg-fu-momo2-top035-hrp-month",
        family="short_momentum",
        role="candidate",
        status="active",
        priority="high",
        baseline_strategy_id="stg-fu-hrp",
        expected_regimes=("short_trend", "rotation"),
        risk_regimes=("whipsaw", "high_turnover"),
        tags=("full_universe", "momentum", "2m", "monthly", "hrp"),
        notes="Core monthly 2-month momentum candidate for faster adaptation.",
    ),
    StrategyInventoryEntry(
        strategy_id="stg-top3-hrp",
        family="concentrated_momentum",
        role="candidate",
        status="active",
        priority="medium",
        baseline_strategy_id="stg-fu-hrp",
        expected_regimes=("strong_trend",),
        risk_regimes=("concentration", "momentum_reversal"),
        tags=("top3", "momentum", "hrp"),
        notes="Concentrated top-3 momentum baseline for selection strength.",
    ),
    StrategyInventoryEntry(
        strategy_id="stg-dualtop3-hrp",
        family="defensive_momentum",
        role="candidate",
        status="active",
        priority="medium",
        baseline_strategy_id="stg-top3-hrp",
        expected_regimes=("trend", "risk_off_filtering"),
        risk_regimes=("cash_drag", "missed_recovery"),
        tags=("dual_momentum", "top3", "hrp"),
        notes="Dual momentum variant for defensive selection behavior.",
    ),
    StrategyInventoryEntry(
        strategy_id="stg-posmom-hrp-month",
        family="defensive_momentum",
        role="candidate",
        status="active",
        priority="medium",
        baseline_strategy_id="stg-fu-hrp",
        expected_regimes=("positive_trend",),
        risk_regimes=("cash_drag", "late_entry"),
        tags=("positive_momentum", "monthly", "hrp"),
        notes="Monthly positive momentum universe for defensive participation.",
    ),
    StrategyInventoryEntry(
        strategy_id="stg-riskoff-posmom-hrp-month",
        family="defensive_momentum",
        role="candidate",
        status="active",
        priority="medium",
        baseline_strategy_id="stg-posmom-hrp-month",
        expected_regimes=("risk_off", "positive_trend"),
        risk_regimes=("macro_false_signal", "cash_drag"),
        tags=("risk_regime", "positive_momentum", "monthly", "hrp"),
        notes="Risk-regime gated positive momentum candidate.",
    ),
    StrategyInventoryEntry(
        strategy_id="stg-fu-momolv8515-top025-hrp-month",
        family="low_vol_momentum",
        role="candidate",
        status="active",
        priority="medium",
        baseline_strategy_id="stg-fu-momo12-top035-hrp",
        expected_regimes=("trend", "volatile_market"),
        risk_regimes=("low_vol_crowding",),
        tags=("full_universe", "momentum", "low_vol", "monthly", "hrp"),
        notes="Momentum plus low-vol tilt candidate for robustness checks.",
    ),
    StrategyInventoryEntry(
        strategy_id="stg-fu-momo2-top035-pred10mom5050-pw40-hrp-month",
        family="predictor_augmented_momentum",
        role="candidate",
        status="active",
        priority="high",
        baseline_strategy_id="stg-fu-momo2-top035-hrp-month",
        expected_regimes=("short_trend", "forecastable_rotation"),
        risk_regimes=("prediction_noise", "overfit"),
        tags=("predictor", "momentum", "2m", "monthly", "hrp"),
        notes="Main predictor-augmented candidate against the 2-month momentum baseline.",
    ),
)


def build_strategy_inventory_payload(
    *,
    status: str | None = None,
    priority: str | None = None,
    family: str | None = None,
) -> dict:
    entries = filter_strategy_inventory_entries(
        build_strategy_inventory_entries(),
        status=status,
        priority=priority,
        family=family,
    )
    return {
        "kind": "strategy_inventory",
        "schemaVersion": STRATEGY_INVENTORY_SCHEMA_VERSION,
        "counts": build_strategy_inventory_counts(entries),
        "entries": [serialize_strategy_inventory_entry(entry) for entry in entries],
    }


def build_strategy_inventory_entries() -> list[StrategyInventoryEntry]:
    definitions_by_id = {definition.strategy_id: definition for definition in CANONICAL_CANDIDATE_DEFINITIONS}
    explicit_entries_by_id = {entry.strategy_id: entry for entry in EXPLICIT_STRATEGY_INVENTORY_ENTRIES}
    entries = [
        explicit_entries_by_id.get(strategy_id) or build_default_strategy_inventory_entry(definition)
        for strategy_id, definition in definitions_by_id.items()
    ]
    validate_strategy_inventory_entries(entries, definitions_by_id=definitions_by_id)
    return entries


def build_default_strategy_inventory_entry(definition) -> StrategyInventoryEntry:
    return StrategyInventoryEntry(
        strategy_id=definition.strategy_id,
        family=infer_strategy_family(definition),
        role="candidate",
        status="backlog",
        priority="low",
        baseline_strategy_id=None,
        expected_regimes=(),
        risk_regimes=(),
        tags=build_default_strategy_tags(definition),
        notes="Generated backlog entry. Add explicit research metadata before promoting.",
    )


def infer_strategy_family(definition) -> str:
    strategy_id = definition.strategy_id
    if "pred" in strategy_id:
        return "predictor_augmented_momentum"
    if "momolv" in strategy_id or "lowvol" in strategy_id:
        return "low_vol_momentum"
    if strategy_id.startswith("stg-fu-momo2"):
        return "short_momentum"
    if "dualtop" in strategy_id or "riskoff" in strategy_id or "posmom" in strategy_id:
        return "defensive_momentum"
    if "top3" in strategy_id and not strategy_id.startswith("stg-fu-"):
        return "concentrated_momentum"
    if "momo" in strategy_id:
        return "momentum_tilt"
    if strategy_id.startswith("stg-fu-"):
        return "baseline"
    return "other"


def build_default_strategy_tags(definition) -> tuple[str, ...]:
    tags = [infer_strategy_family(definition), definition.portfolio_model.key]
    if definition.strategy_id.startswith("stg-fu-"):
        tags.append("full_universe")
    if any(signal.predictor_key for signal in definition.signals):
        tags.append("predictor")
    if definition.execution_plan.rebalance_schedule == "month_end":
        tags.append("monthly")
    return tuple(dict.fromkeys(tags))


def filter_strategy_inventory_entries(
    entries: list[StrategyInventoryEntry],
    *,
    status: str | None,
    priority: str | None,
    family: str | None,
) -> list[StrategyInventoryEntry]:
    validate_strategy_inventory_filter("status", status, STRATEGY_INVENTORY_STATUSES)
    validate_strategy_inventory_filter("priority", priority, STRATEGY_INVENTORY_PRIORITIES)
    filtered_entries = entries
    if status is not None:
        filtered_entries = [entry for entry in filtered_entries if entry.status == status]
    if priority is not None:
        filtered_entries = [entry for entry in filtered_entries if entry.priority == priority]
    if family is not None:
        filtered_entries = [entry for entry in filtered_entries if entry.family == family]
    return filtered_entries


def validate_strategy_inventory_filter(name: str, value: str | None, allowed_values: tuple[str, ...]) -> None:
    if value is not None and value not in allowed_values:
        raise ValueError(f"Unknown strategy inventory {name}: {value}")


def build_strategy_inventory_counts(entries: list[StrategyInventoryEntry]) -> dict:
    return {
        "total": len(entries),
        "byStatus": count_strategy_inventory_values(entries, "status"),
        "byPriority": count_strategy_inventory_values(entries, "priority"),
        "byFamily": count_strategy_inventory_values(entries, "family"),
    }


def count_strategy_inventory_values(entries: list[StrategyInventoryEntry], field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        value = str(getattr(entry, field_name))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def serialize_strategy_inventory_entry(entry: StrategyInventoryEntry) -> dict:
    return {
        "strategyId": entry.strategy_id,
        "family": entry.family,
        "role": entry.role,
        "status": entry.status,
        "priority": entry.priority,
        "baselineStrategyId": entry.baseline_strategy_id,
        "expectedRegimes": list(entry.expected_regimes),
        "riskRegimes": list(entry.risk_regimes),
        "tags": list(entry.tags),
        "notes": entry.notes,
    }


def validate_strategy_inventory_entries(
    entries: list[StrategyInventoryEntry],
    *,
    definitions_by_id: dict[str, object],
) -> None:
    entry_ids = [entry.strategy_id for entry in entries]
    duplicate_ids = sorted({strategy_id for strategy_id in entry_ids if entry_ids.count(strategy_id) > 1})
    if duplicate_ids:
        raise ValueError(f"Duplicate strategy inventory entry id(s): {', '.join(duplicate_ids)}")

    missing_ids = sorted(set(definitions_by_id) - set(entry_ids))
    if missing_ids:
        raise ValueError(f"Missing strategy inventory entry id(s): {', '.join(missing_ids)}")

    unknown_ids = sorted(set(entry_ids) - set(definitions_by_id))
    if unknown_ids:
        raise ValueError(f"Unknown strategy inventory entry id(s): {', '.join(unknown_ids)}")

    for entry in entries:
        if entry.status not in STRATEGY_INVENTORY_STATUSES:
            raise ValueError(f"Unknown strategy inventory status for {entry.strategy_id}: {entry.status}")
        if entry.priority not in STRATEGY_INVENTORY_PRIORITIES:
            raise ValueError(f"Unknown strategy inventory priority for {entry.strategy_id}: {entry.priority}")
        if entry.baseline_strategy_id is not None and entry.baseline_strategy_id not in definitions_by_id:
            raise ValueError(
                f"Unknown baseline strategy id for {entry.strategy_id}: {entry.baseline_strategy_id}"
            )


__all__ = [
    "EXPLICIT_STRATEGY_INVENTORY_ENTRIES",
    "STRATEGY_INVENTORY_PRIORITIES",
    "STRATEGY_INVENTORY_SCHEMA_VERSION",
    "STRATEGY_INVENTORY_STATUSES",
    "StrategyInventoryEntry",
    "build_strategy_inventory_entries",
    "build_strategy_inventory_payload",
    "filter_strategy_inventory_entries",
    "serialize_strategy_inventory_entry",
    "validate_strategy_inventory_entries",
]
