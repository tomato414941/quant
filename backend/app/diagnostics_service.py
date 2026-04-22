from __future__ import annotations


INVALIDATING_SEVERITY = "invalidating"
WARNING_SEVERITY = "warning"
INFO_SEVERITY = "info"


def build_evaluation_diagnostic_events(
    *,
    availability_diagnostics: dict[str, object] | None,
    instrument_diagnostics: dict[str, object] | None,
) -> list[dict[str, object]]:
    events = []
    events.extend(build_availability_diagnostic_events(availability_diagnostics or {}))
    events.extend(build_instrument_diagnostic_events(instrument_diagnostics or {}))
    return events


def build_availability_diagnostic_events(
    availability_diagnostics: dict[str, object],
) -> list[dict[str, object]]:
    events = []
    for warning in availability_diagnostics.get("actionableWarnings") or []:
        events.append(normalize_availability_warning(warning, severity=INVALIDATING_SEVERITY))
    for warning in availability_diagnostics.get("calendarBoundaryWarnings") or []:
        events.append(normalize_availability_warning(warning, severity=INFO_SEVERITY))
    for warning in availability_diagnostics.get("assetLifecycleWarnings") or []:
        events.append(
            normalize_availability_warning(
                warning,
                severity=INFO_SEVERITY,
                scope="asset_lifecycle",
            )
        )
    return events


def normalize_availability_warning(
    warning: dict[str, object],
    *,
    severity: str,
    scope: str = "market_data",
) -> dict[str, object]:
    event = {
        "kind": str(warning.get("kind") or "availability_warning"),
        "category": "availability",
        "severity": severity,
        "scope": scope,
        "reason": str(warning.get("message") or warning.get("kind") or "availability warning"),
        "evidence": compact_mapping(warning),
    }
    add_if_present(event, "symbol", warning, ("symbol", "asset", "ticker"))
    add_if_present(event, "timeframe", warning, ("timeframe", "timeframeKey"))
    add_dates(event, warning)
    return event


def build_instrument_diagnostic_events(
    instrument_diagnostics: dict[str, object],
) -> list[dict[str, object]]:
    events = []
    unknown_symbols = [str(symbol) for symbol in instrument_diagnostics.get("unknownSymbols") or []]
    if unknown_symbols:
        events.append(
            {
                "kind": "unknown_symbols",
                "category": "instrument",
                "severity": INVALIDATING_SEVERITY,
                "scope": "instrument_registry",
                "reason": "unknown instrument symbols",
                "symbols": unknown_symbols,
                "evidence": {"unknownSymbols": unknown_symbols},
            }
        )
    if instrument_diagnostics.get("mixedMarketCalendar"):
        events.append(
            {
                "kind": "mixed_market_calendar",
                "category": "calendar",
                "severity": INFO_SEVERITY,
                "scope": "instrument_registry",
                "reason": "mixed market calendars",
                "evidence": {
                    "marketCalendars": instrument_diagnostics.get("marketCalendars") or {},
                    "assetClassCounts": instrument_diagnostics.get("assetClassCounts") or {},
                },
            }
        )
    return events


def summarize_diagnostic_events(events: list[dict[str, object]]) -> dict[str, object]:
    severity_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    flags = []
    for event in events:
        severity = str(event.get("severity") or "unknown")
        category = str(event.get("category") or "unknown")
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        category_counts[category] = category_counts.get(category, 0) + 1
        flag = diagnostic_flag(event)
        if flag is not None:
            flags.append(flag)
    return {
        "eventCount": len(events),
        "severityCounts": dict(sorted(severity_counts.items())),
        "categoryCounts": dict(sorted(category_counts.items())),
        "flags": sorted(set(flags)),
    }


def diagnostic_flag(event: dict[str, object]) -> str | None:
    kind = event.get("kind")
    category = event.get("category")
    severity = event.get("severity")
    if severity == INVALIDATING_SEVERITY and category == "availability":
        return "actionable availability warning"
    if kind == "unknown_symbols":
        return "unknown symbols"
    if kind == "mixed_market_calendar":
        return "mixed calendar"
    return None


def has_invalidating_diagnostic(events: list[dict[str, object]], *, category: str | None = None) -> bool:
    return any(
        event.get("severity") == INVALIDATING_SEVERITY
        and (category is None or event.get("category") == category)
        for event in events
    )


def representative_diagnostic_event(events: list[dict[str, object]]) -> dict[str, object] | None:
    if not events:
        return None
    return sorted(
        events,
        key=lambda event: (
            severity_priority(str(event.get("severity") or "")),
            str(event.get("category") or ""),
            str(event.get("kind") or ""),
        ),
    )[0]


def severity_priority(severity: str) -> int:
    priorities = {
        INVALIDATING_SEVERITY: 0,
        WARNING_SEVERITY: 1,
        INFO_SEVERITY: 2,
    }
    return priorities.get(severity, 9)


def compact_mapping(values: dict[str, object]) -> dict[str, object]:
    return {
        str(key): value
        for key, value in values.items()
        if value is not None
    }


def add_if_present(
    target: dict[str, object],
    target_key: str,
    source: dict[str, object],
    source_keys: tuple[str, ...],
) -> None:
    for source_key in source_keys:
        value = source.get(source_key)
        if value is not None:
            target[target_key] = value
            return


def add_dates(target: dict[str, object], source: dict[str, object]) -> None:
    dates = {
        key: value
        for key, value in source.items()
        if key.endswith("Date") and value is not None
    }
    if dates:
        target["dates"] = dates
