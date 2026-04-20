from __future__ import annotations

import argparse
import copy
from datetime import date
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

from app.comparison_service import (
    build_comparison_payload,
    build_comparison_payload_from_run_spec_payload,
    build_walk_forward_comparison_payload,
    build_comparison_run_spec_payload,
    build_latest_run_payload,
    build_run_catalog_payload,
    build_run_result_store,
)
from app.default_comparison import DEFAULT_COMPARISON_SPEC
from app.market_data import fetch_market_universe_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app")
    subparsers = parser.add_subparsers(dest="command", required=True)

    comparison_parser = subparsers.add_parser(
        "comparison-summary",
        help="Print a terminal summary of the current comparison.",
    )
    comparison_parser.add_argument("--top", type=int, default=5)
    comparison_parser.add_argument(
        "--universe",
        choices=("crypto_included", "btc_only", "no_crypto"),
        default="crypto_included",
        help="Run the comparison against a fixed universe variant.",
    )
    comparison_parser.add_argument("--json", action="store_true", dest="as_json")
    comparison_parser.add_argument("--walk-forward", action="store_true")
    comparison_parser.add_argument("--walk-forward-start-year", type=int, default=2020)
    comparison_parser.add_argument("--walk-forward-end-year", type=int, default=2025)

    comparison_run_spec_parser = subparsers.add_parser(
        "comparison-run-spec",
        help="Print the canonical comparison run spec for reproducible reruns.",
    )
    comparison_run_spec_parser.add_argument("--json", action="store_true", dest="as_json")

    rerun_comparison_spec_parser = subparsers.add_parser(
        "rerun-comparison-spec",
        help="Rerun a saved comparison-run-spec JSON payload.",
    )
    rerun_comparison_spec_parser.add_argument("spec_file")
    rerun_comparison_spec_parser.add_argument("--top", type=int, default=5)
    rerun_comparison_spec_parser.add_argument("--json", action="store_true", dest="as_json")

    run_catalog_parser = subparsers.add_parser(
        "run-catalog",
        help="List saved run records from the local run store.",
    )
    run_catalog_parser.add_argument("--limit", type=int, default=20)
    run_catalog_parser.add_argument("--run-kind", dest="run_kind")
    run_catalog_parser.add_argument("--generation-method", dest="generation_method")
    run_catalog_parser.add_argument("--strategy-definition-fingerprint", dest="strategy_definition_fingerprint")
    run_catalog_parser.add_argument("--market-data-fingerprint", dest="market_data_fingerprint")
    run_catalog_parser.add_argument("--evaluation-fingerprint", dest="evaluation_fingerprint")
    run_catalog_parser.add_argument("--json", action="store_true", dest="as_json")

    latest_run_parser = subparsers.add_parser(
        "latest-run",
        help="Resolve the latest saved run for a fingerprint filter set.",
    )
    latest_run_parser.add_argument("--run-kind", dest="run_kind")
    latest_run_parser.add_argument("--generation-method", dest="generation_method")
    latest_run_parser.add_argument("--strategy-definition-fingerprint", dest="strategy_definition_fingerprint")
    latest_run_parser.add_argument("--market-data-fingerprint", dest="market_data_fingerprint")
    latest_run_parser.add_argument("--evaluation-fingerprint", dest="evaluation_fingerprint")
    latest_run_parser.add_argument("--json", action="store_true", dest="as_json")

    rebuild_index_parser = subparsers.add_parser(
        "rebuild-run-index",
        help="Rebuild the local run store index file.",
    )
    rebuild_index_parser.add_argument("--json", action="store_true", dest="as_json")

    return parser


UNIVERSE_VARIANTS = {
    "crypto_included": {
        "label": "Crypto included",
        "excluded_tickers": set(),
    },
    "btc_only": {
        "label": "BTC only",
        "excluded_tickers": {"ETH-USD"},
    },
    "no_crypto": {
        "label": "No crypto",
        "excluded_tickers": {"BTC-USD", "ETH-USD"},
    },
}


def apply_comparison_universe_variant(comparison, universe_key: str):
    variant = UNIVERSE_VARIANTS[universe_key]
    excluded_tickers = set(variant["excluded_tickers"])
    if not excluded_tickers:
        return comparison

    filtered_comparison = copy.deepcopy(comparison)
    filtered_comparison.comparison_id = f"{comparison.comparison_id}__{universe_key}"
    filtered_comparison.candidate_strategies = [
        filter_strategy_definition_universe(strategy, excluded_tickers, universe_key)
        for strategy in comparison.candidate_strategies
    ]
    filtered_comparison.reference_strategies = [
        filter_strategy_definition_universe(strategy, excluded_tickers, universe_key)
        for strategy in comparison.reference_strategies
    ]
    portfolio_state = comparison.run_spec.portfolio_state
    retained_weights = {
        asset: weight
        for asset, weight in portfolio_state.current_weights.items()
        if asset not in excluded_tickers
    }
    removed_weight = sum(
        weight
        for asset, weight in portfolio_state.current_weights.items()
        if asset in excluded_tickers
    )
    filtered_comparison.run_spec.portfolio_state = replace(
        portfolio_state,
        current_weights=retained_weights,
        cash_weight=portfolio_state.cash_weight + removed_weight,
    )
    return filtered_comparison


def filter_strategy_definition_universe(strategy_definition, excluded_tickers: set[str], universe_key: str):
    tickers = tuple(
        ticker
        for ticker in strategy_definition.investment_universe.tickers
        if ticker not in excluded_tickers
    )
    if len(tickers) < 2:
        raise ValueError("Universe variant must retain at least two assets per strategy.")
    investment_universe = replace(
        strategy_definition.investment_universe,
        key=f"{strategy_definition.investment_universe.key}__{universe_key}",
        label=f"{strategy_definition.investment_universe.label} / {UNIVERSE_VARIANTS[universe_key]['label']}",
        tickers=tickers,
    )
    return replace(
        strategy_definition,
        investment_universe=investment_universe,
        signals=tuple(
            filter_strategy_signal_universe(signal, tickers, universe_key)
            for signal in strategy_definition.signals
        ),
    )


def filter_strategy_signal_universe(signal, tickers: tuple[str, ...], universe_key: str):
    observation_spec = replace(
        signal.observation_spec,
        key=f"{signal.observation_spec.key}__{universe_key}",
        tickers=tickers,
    )
    data_source_spec = signal.data_source_spec
    if data_source_spec is not None:
        data_source_spec = replace(
            data_source_spec,
            observation_spec=replace(
                data_source_spec.observation_spec,
                key=f"{data_source_spec.observation_spec.key}__{universe_key}",
                tickers=tickers,
            ),
        )
    return replace(
        signal,
        observation_spec=observation_spec,
        data_source_spec=data_source_spec,
    )


def format_percent(value: float) -> str:
    return f"{value:+.2f}%"


def parse_iso_date(value: object) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def timeframe_calendar_boundary_days(timeframe_key: object) -> int:
    if timeframe_key in {"1wk", "1w"}:
        return 7
    if timeframe_key == "1mo":
        return 31
    return 1


def is_small_calendar_gap(
    warning: dict,
    availability_policy: dict | None,
    left_date_key: str,
    right_date_key: str,
) -> bool:
    left_date = parse_iso_date(warning.get(left_date_key))
    right_date = parse_iso_date(warning.get(right_date_key))
    if left_date is None or right_date is None:
        return False
    max_stale_bars = int((availability_policy or {}).get("maxStaleBars") or 0)
    boundary_days = timeframe_calendar_boundary_days(warning.get("timeframe"))
    max_calendar_gap_days = max(max_stale_bars, boundary_days)
    return abs((right_date - left_date).days) <= max_calendar_gap_days


def classify_availability_warning(warning: dict, availability_policy: dict | None) -> str:
    kind = warning.get("kind")
    if kind == "requested_asset_unavailable":
        return "actionable"
    if kind == "aligned_start_after_requested_start":
        if is_small_calendar_gap(warning, availability_policy, "requestedStartDate", "alignedStartDate"):
            return "calendar"
        return "actionable"
    if kind == "aligned_end_before_requested_end":
        if is_small_calendar_gap(warning, availability_policy, "requestedEndDate", "alignedEndDate"):
            return "calendar"
        return "actionable"
    if kind == "asset_available_after_aligned_start":
        if is_small_calendar_gap(warning, availability_policy, "alignedStartDate", "firstValidDate"):
            return "calendar"
        return "actionable"
    if kind == "asset_unavailable_before_aligned_end":
        if is_small_calendar_gap(warning, availability_policy, "lastValidDate", "alignedEndDate"):
            return "calendar"
        return "actionable"
    return "actionable"


def split_availability_warnings(warnings: list[dict], availability_policy: dict | None) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {
        "actionable": [],
        "calendar": [],
        "info": [],
    }
    for warning in warnings:
        classification = classify_availability_warning(warning, availability_policy)
        grouped.setdefault(classification, []).append(warning)
    return grouped


def render_availability_warnings(lines: list[str], warnings: list[dict], availability_policy: dict | None) -> None:
    if not warnings:
        return
    grouped_warnings = split_availability_warnings(warnings, availability_policy)
    actionable_warnings = grouped_warnings["actionable"]
    calendar_warnings = grouped_warnings["calendar"]
    if actionable_warnings:
        lines.append("Warnings:")
        for warning in actionable_warnings:
            lines.append(f"- {warning['message']}")
    if calendar_warnings:
        if actionable_warnings:
            lines.append("")
        lines.append(
            "Calendar differences: "
            f"{len(calendar_warnings)} market-calendar boundary warning(s) hidden from Warnings."
        )
    if actionable_warnings or calendar_warnings:
        lines.append("")


def portfolio_segment_summary(run: dict, segment: str) -> dict:
    return run["splitAnalysis"][segment]["portfolio"]


def sort_candidate_runs(candidate_runs: list[dict]) -> list[dict]:
    return sorted(
        candidate_runs,
        key=lambda run: (
            -portfolio_segment_summary(run, "test")["sharpeRatio"],
            -portfolio_segment_summary(run, "test")["totalReturnPct"],
            portfolio_segment_summary(run, "test")["maxDrawdownPct"],
        ),
    )


def render_comparison_summary(payload: dict, *, top: int) -> str:
    comparison = payload["comparison"]
    candidate_runs = sort_candidate_runs(payload["candidateRuns"])
    lines = [
        f"Comparison: {comparison['title']} ({comparison['comparisonId']})",
        f"Question: {comparison['question']}",
        (
            "Selection policy: "
            f"{comparison['selectionPolicy']['primaryMetric']} / "
            f"{comparison['selectionPolicy']['secondaryMetric']} / "
            f"{comparison['selectionPolicy']['tertiaryMetric']}"
        ),
        (
            "Run store: "
            f"cached={payload['runStoreSummary']['cachedRunCount']} "
            f"computed={payload['runStoreSummary']['computedRunCount']}"
        ),
        (
            "Counts: "
            f"candidate={len(payload['candidateRuns'])} "
            f"reference={len(payload['referenceRuns'])} "
            f"predictor={len(payload['predictorRuns'])}"
        ),
        "",
    ]

    evaluation = comparison.get("runSpec", {}).get("evaluation", {})
    render_availability_warnings(lines, evaluation.get("warnings", []), evaluation.get("availabilityPolicy"))

    lines.append(f"Top {min(top, len(candidate_runs))} candidate runs by test performance:")

    for index, run in enumerate(candidate_runs[:top], start=1):
        test_summary = portfolio_segment_summary(run, "test")
        train_summary = portfolio_segment_summary(run, "train")
        overall_summary = run["summary"]
        lines.append(f"{index}. {run['strategy']['label']} [{run['key']}]")
        lines.append(
            "   "
            f"Test Sharpe {test_summary['sharpeRatio']:.3f} | "
            f"Return {format_percent(test_summary['totalReturnPct'])} | "
            f"MDD {format_percent(test_summary['maxDrawdownPct'])} | "
            f"Turnover {format_percent(test_summary['turnoverPct'])}"
        )
        lines.append(
            "   "
            f"Train Sharpe {train_summary['sharpeRatio']:.3f} | "
            f"Summary Sharpe {overall_summary['sharpeRatio']:.3f}"
        )

    return "\n".join(lines)



def format_count_range(min_count: int, max_count: int) -> str:
    if min_count == max_count:
        return str(min_count)
    return f"{min_count}-{max_count}"


def render_walk_forward_summary(payload: dict, *, top: int) -> str:
    comparison = payload["comparison"]
    walk_forward = payload["walkForward"]
    candidate_results = payload["candidateResults"]
    evaluation = comparison.get("runSpec", {}).get("evaluation", {})
    lines = [
        f"Comparison: {comparison['title']} ({comparison['comparisonId']})",
        f"Question: {comparison['question']}",
        (
            "Walk-forward: "
            f"{walk_forward['startYear']}-{walk_forward['endYear']} "
            f"({walk_forward['windowCount']} windows)"
        ),
        (
            "Ranking policy: average test Sharpe / minimum test Sharpe / "
            "average test return / average test max drawdown"
        ),
        (
            "Run store: "
            f"cached={payload['runStoreSummary']['cachedRunCount']} "
            f"computed={payload['runStoreSummary']['computedRunCount']}"
        ),
        "",
    ]

    availability_policy = evaluation.get("availabilityPolicy")
    if availability_policy:
        lines.append(
            "Availability policy: "
            f"minHistoryBars={availability_policy.get('minHistoryBars')} "
            f"maxStaleBars={availability_policy.get('maxStaleBars')} "
            f"delistedAssetPolicy={availability_policy.get('delistedAssetPolicy')}"
        )
    availability_summary = evaluation.get("availabilitySummary")
    if availability_summary:
        min_available = int(availability_summary.get("minAvailableAssetCount", 0))
        max_available = int(availability_summary.get("maxAvailableAssetCount", 0))
        lines.append(
            "Market availability: "
            f"available assets {format_count_range(min_available, max_available)}"
        )
    if availability_policy or availability_summary:
        lines.append("")

    render_availability_warnings(lines, evaluation.get("warnings", []), availability_policy)

    lines.append(f"Top {min(top, len(candidate_results))} candidate strategies by walk-forward test performance:")
    for index, result in enumerate(candidate_results[:top], start=1):
        lines.append(f"{index}. {result['strategyLabel']} [{result['strategyKey']}]")
        lines.append(
            "   "
            f"Avg Sharpe {result['averageSharpeRatio']:.3f} | "
            f"Min Sharpe {result['minimumSharpeRatio']:.3f} | "
            f"Avg Return {format_percent(result['averageTotalReturnPct'])} | "
            f"Avg MDD {format_percent(result['averageMaxDrawdownPct'])} | "
            f"Positive Years {result['positiveReturnWindowCount']}/{result['windowCount']}"
        )
        lines.append(
            "   "
            f"Avg Turnover {format_percent(result['averageTurnoverPct'])}"
        )
        if "minTestEligibleAssetCount" in result:
            lines.append(
                "   "
                "Test eligible assets "
                f"{format_count_range(int(result['minTestEligibleAssetCount']), int(result['maxTestEligibleAssetCount']))} | "
                f"Newly eligible {result['testNewlyEligibleAssetCount']} | "
                f"Removed {result['testRemovedAssetCount']}"
            )

    return "\n".join(lines)


def render_comparison_run_spec(payload: dict) -> str:
    lines = [
        f"Comparison: {payload['title']} ({payload['comparisonId']})",
        (
            "Selection policy: "
            f"{payload['selectionPolicy']['primaryMetric']} / "
            f"{payload['selectionPolicy']['secondaryMetric']} / "
            f"{payload['selectionPolicy']['tertiaryMetric']}"
        ),
        f"Comparison fingerprint: {payload['comparisonFingerprint']}",
        f"Run spec fingerprint: {payload['runSpecFingerprint']}",
        (
            "Strategy counts: "
            f"candidate={payload['candidateStrategyCount']} "
            f"reference={payload['referenceStrategyCount']}"
        ),
        "",
        json.dumps(payload['runSpec'], ensure_ascii=False, indent=2),
    ]
    return "\n".join(lines)


def render_latest_run(payload: dict) -> str:
    record = payload["record"]
    lines = [
        f"Comparison: {payload['comparisonId']}",
        f"Run kind: {payload['runKind'] or '-'}",
    ]
    if record is None:
        lines.append("Record: not found")
        return "\n".join(lines)
    lines.extend(
        [
            f"Run key: {record['runKey']}",
            f"Strategy: {record.get('strategyLabel')}",
            f"Logic version: {record.get('logicVersion')}",
        ]
    )
    return "\n".join(lines)


def render_run_catalog(payload: dict) -> str:
    lines = [
        f"Comparison: {payload['comparisonId']}",
        f"Records: {payload['recordCount']}",
    ]
    for index, record in enumerate(payload["records"], start=1):
        run_spec = record["runSpec"]
        strategy = run_spec["strategy"]
        generation = run_spec.get("generation", {})
        lines.append(
            (
                f"{index}. {record['runKey']} | "
                f"runKind={run_spec['runKind']} | "
                f"strategy={strategy['label']} | "
                f"generation={generation.get('method', '-')}"
            )
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "comparison-summary":
        comparison_spec = apply_comparison_universe_variant(DEFAULT_COMPARISON_SPEC, args.universe)
        if args.walk_forward:
            payload = build_walk_forward_comparison_payload(
                comparison_spec,
                fetch_market_universe_bundle=fetch_market_universe_bundle,
                start_year=args.walk_forward_start_year,
                end_year=args.walk_forward_end_year,
            )
            if args.as_json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                print(render_walk_forward_summary(payload, top=max(args.top, 1)))
            return 0

        payload = build_comparison_payload(
            comparison_spec,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
        )
        if args.as_json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(render_comparison_summary(payload, top=max(args.top, 1)))
        return 0

    if args.command == "comparison-run-spec":
        payload = build_comparison_run_spec_payload(
            DEFAULT_COMPARISON_SPEC,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
        )
        if args.as_json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(render_comparison_run_spec(payload))
        return 0

    if args.command == "rerun-comparison-spec":
        payload = json.loads(Path(args.spec_file).read_text())
        comparison_payload = build_comparison_payload_from_run_spec_payload(
            payload,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
        )
        if args.as_json:
            print(json.dumps(comparison_payload, ensure_ascii=False, indent=2))
        else:
            print(render_comparison_summary(comparison_payload, top=max(args.top, 1)))
        return 0

    if args.command == "run-catalog":
        payload = build_run_catalog_payload(
            DEFAULT_COMPARISON_SPEC,
            limit=max(args.limit, 1),
            run_kind=args.run_kind,
            generation_method=args.generation_method,
            strategy_definition_fingerprint=args.strategy_definition_fingerprint,
            market_data_fingerprint=args.market_data_fingerprint,
            evaluation_fingerprint=args.evaluation_fingerprint,
        )
        if args.as_json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(render_run_catalog(payload))
        return 0

    if args.command == "latest-run":
        payload = build_latest_run_payload(
            DEFAULT_COMPARISON_SPEC,
            run_kind=args.run_kind,
            generation_method=args.generation_method,
            strategy_definition_fingerprint=args.strategy_definition_fingerprint,
            market_data_fingerprint=args.market_data_fingerprint,
            evaluation_fingerprint=args.evaluation_fingerprint,
        )
        if args.as_json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(render_latest_run(payload))
        return 0

    if args.command == "rebuild-run-index":
        payload = build_run_result_store(DEFAULT_COMPARISON_SPEC).rebuild_index()
        if args.as_json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"Rebuilt run store index: {payload['entryCount']} entries")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
