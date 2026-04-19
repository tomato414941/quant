from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from app.comparison_service import (
    build_comparison_payload,
    build_comparison_payload_from_run_spec_payload,
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
    comparison_parser.add_argument("--json", action="store_true", dest="as_json")

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


def format_percent(value: float) -> str:
    return f"{value:+.2f}%"


def sort_candidate_runs(candidate_runs: list[dict]) -> list[dict]:
    return sorted(
        candidate_runs,
        key=lambda run: (
            -run["summary"]["sharpeRatio"],
            -run["summary"]["totalReturnPct"],
            run["summary"]["maxDrawdownPct"],
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
        f"Top {min(top, len(candidate_runs))} candidate runs:",
    ]

    for index, run in enumerate(candidate_runs[:top], start=1):
        summary = run["summary"]
        lines.append(f"{index}. {run['strategy']['label']} [{run['key']}]")
        lines.append(
            "   "
            f"Sharpe {summary['sharpeRatio']:.3f} | "
            f"Return {format_percent(summary['totalReturnPct'])} | "
            f"MDD {format_percent(summary['maxDrawdownPct'])} | "
            f"Turnover {format_percent(summary['turnoverPct'])}"
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
        payload = build_comparison_payload(
            DEFAULT_COMPARISON_SPEC,
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
