from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from app import edge_attribution_service
from app import robustness_service
from app import signal_diagnostics_service
from app.backtest import run_equal_weight_full_period_backtest
from app.comparison_market_context import build_run_result_store
from app.comparison_payloads import (
    build_comparison_payload,
    build_comparison_payload_from_run_spec_payload,
    build_comparison_run_spec_payload,
    build_latest_run_payload,
    build_run_catalog_payload,
)
from app.comparison_walk_forward import build_walk_forward_comparison_payload
from app.default_comparison import DEFAULT_COMPARISON_SPEC
from app.market_data import fetch_market_universe_bundle
from app.market_data import build_market_snapshot_fetcher
from app.market_data import read_market_data_snapshot
from app.instrument_registry import UNIVERSE_VARIANT_KEYS
from app.strategy_inventory import (
    STRATEGY_INVENTORY_PRIORITIES,
    STRATEGY_INVENTORY_STATUSES,
    build_strategy_inventory_payload,
)
from app.cli_renderers import (
    format_count_map,
    format_decimal_distribution,
    format_optional_decimal,
    format_optional_percent,
    format_optional_ratio_percent,
    format_optional_signed_decimal,
    format_percent,
    format_percent_distribution,
    portfolio_segment_summary,
    render_availability_diagnostics,
    render_benchmark_decomposition,
    render_comparison_run_spec,
    render_comparison_summary,
    render_edge_attribution,
    render_instrument_diagnostics,
    render_latest_run,
    render_robustness_progress,
    render_robustness_summary,
    render_run_catalog,
    render_signal_diagnostics,
    render_strategy_inventory,
    render_walk_forward_summary,
    sort_candidate_runs,
    write_json_payload,
)
from app.cli_services import (
    apply_comparison_universe_variant,
    build_benchmark_decomposition_comparison,
    build_benchmark_decomposition_payload,
    build_benchmark_delta,
    build_benchmark_final_weights,
    build_benchmark_reference_strategies,
    build_benchmark_result,
    build_benchmark_worst_window,
    build_equal_weight_hold_benchmark,
    collect_comparison_universe_tickers,
    filter_comparison_strategies,
    filter_strategy_definition_universe,
    filter_strategy_signal_universe,
)


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
        choices=UNIVERSE_VARIANT_KEYS,
        default="crypto_included",
        help="Run the comparison against a fixed universe variant.",
    )
    comparison_parser.add_argument("--json", action="store_true", dest="as_json")
    comparison_parser.add_argument(
        "--refresh-market-data",
        action="store_true",
        help="Fetch live market data for this ad-hoc comparison run.",
    )
    comparison_parser.add_argument("--walk-forward", action="store_true")
    comparison_parser.add_argument("--walk-forward-start-year", type=int, default=2020)
    comparison_parser.add_argument("--walk-forward-end-year", type=int, default=2025)
    comparison_parser.add_argument(
        "--snapshot-spec-file",
        help="Read market data snapshot metadata from a comparison-run-spec JSON payload.",
    )
    comparison_parser.add_argument(
        "--market-snapshot-dir",
        help="Read market data snapshot CSV files from this directory.",
    )
    comparison_parser.add_argument(
        "--strategy-key",
        action="append",
        dest="strategy_keys",
        help="Restrict comparison summary to a strategy key. Can be repeated.",
    )

    robustness_parser = subparsers.add_parser(
        "robustness-summary",
        help="Print a robustness summary across fixed evaluation conditions.",
    )
    robustness_parser.add_argument("--top", type=int, default=10)
    robustness_parser.add_argument(
        "--profile",
        choices=robustness_service.ROBUSTNESS_PROFILE_KEYS,
        default="quick",
        help="Use quick for iteration or standard for the full robustness matrix.",
    )
    robustness_parser.add_argument(
        "--period",
        action="append",
        dest="period_keys",
        choices=[period["key"] for period in robustness_service.ROBUSTNESS_PERIODS],
        help="Restrict robustness scenarios to a period key. Can be repeated.",
    )
    robustness_parser.add_argument(
        "--universe",
        action="append",
        dest="universes",
        choices=robustness_service.ROBUSTNESS_UNIVERSES,
        help="Restrict robustness scenarios to a universe. Can be repeated.",
    )
    robustness_parser.add_argument(
        "--cost-multiplier",
        action="append",
        dest="cost_multipliers",
        type=float,
        choices=robustness_service.ROBUSTNESS_COST_MULTIPLIERS,
        help="Restrict robustness scenarios to a cost multiplier. Can be repeated.",
    )
    robustness_parser.add_argument(
        "--max-weight",
        action="append",
        dest="max_weights",
        type=float,
        choices=robustness_service.ROBUSTNESS_MAX_WEIGHTS,
        help="Restrict robustness scenarios to a max asset weight. Can be repeated.",
    )
    robustness_parser.add_argument(
        "--strategy-key",
        action="append",
        dest="strategy_keys",
        help="Restrict robustness scenarios to a strategy key. Can be repeated.",
    )
    robustness_parser.add_argument("--progress", action="store_true")
    robustness_parser.add_argument("--output", help="Write the full robustness JSON payload to this path.")
    robustness_parser.add_argument("--json", action="store_true", dest="as_json")

    signal_diagnostics_parser = subparsers.add_parser(
        "signal-diagnostics",
        help="Print signal score versus forward return diagnostics.",
    )
    signal_diagnostics_parser.add_argument(
        "--strategy-key",
        action="append",
        dest="strategy_keys",
        help="Restrict diagnostics to a strategy key. Can be repeated.",
    )
    signal_diagnostics_parser.add_argument(
        "--period",
        default=DEFAULT_COMPARISON_SPEC.run_spec.market_slice.period,
        help="Market data period to evaluate.",
    )
    signal_diagnostics_parser.add_argument(
        "--universe",
        choices=UNIVERSE_VARIANT_KEYS,
        default="crypto_included",
        help="Run diagnostics against a fixed universe variant.",
    )
    signal_diagnostics_parser.add_argument(
        "--horizon",
        action="append",
        dest="horizons",
        default=None,
        help="Forward horizon such as 1d, 5d, or 21d. Can be repeated.",
    )
    signal_diagnostics_parser.add_argument(
        "--observation-schedule",
        choices=signal_diagnostics_service.SIGNAL_DIAGNOSTIC_OBSERVATION_SCHEDULES,
        default="strategy",
        help="Observation cadence for score/forward-return diagnostics.",
    )
    signal_diagnostics_parser.add_argument("--json", action="store_true", dest="as_json")

    edge_attribution_parser = subparsers.add_parser(
        "edge-attribution",
        help="Decompose one strategy into cash, universe, selection, and full components.",
    )
    edge_attribution_parser.add_argument(
        "--strategy-key",
        required=True,
        help="Strategy key to decompose.",
    )
    edge_attribution_parser.add_argument(
        "--period",
        default=DEFAULT_COMPARISON_SPEC.run_spec.market_slice.period,
        help="Market data period to evaluate.",
    )
    edge_attribution_parser.add_argument(
        "--universe",
        choices=UNIVERSE_VARIANT_KEYS,
        default="crypto_included",
        help="Run attribution against a fixed universe variant.",
    )
    edge_attribution_parser.add_argument("--walk-forward-start-year", type=int, default=2020)
    edge_attribution_parser.add_argument("--walk-forward-end-year", type=int, default=2025)
    edge_attribution_parser.add_argument("--json", action="store_true", dest="as_json")

    comparison_run_spec_parser = subparsers.add_parser(
        "comparison-run-spec",
        help="Print the canonical comparison run spec for reproducible reruns.",
    )
    comparison_run_spec_parser.add_argument("--json", action="store_true", dest="as_json")
    comparison_run_spec_parser.add_argument("--write-market-snapshots", action="store_true")
    comparison_run_spec_parser.add_argument("--market-snapshot-dir")

    market_snapshot_parser = subparsers.add_parser(
        "market-snapshot-create",
        help="Fetch provider data once, write immutable market snapshots, and print the run spec.",
    )
    market_snapshot_parser.add_argument(
        "--provider",
        default=None,
        help="Market data provider key. Defaults to QUANT_MARKET_DATA_PROVIDER or yfinance.",
    )
    market_snapshot_parser.add_argument(
        "--market-snapshot-dir",
        required=True,
        help="Directory where market snapshot files will be written.",
    )
    market_snapshot_parser.add_argument(
        "--universe",
        choices=UNIVERSE_VARIANT_KEYS,
        default="crypto_included",
        help="Create snapshots for a fixed universe variant.",
    )
    market_snapshot_parser.add_argument(
        "--strategy-key",
        action="append",
        dest="strategy_keys",
        help="Restrict snapshot creation to a strategy key. Can be repeated.",
    )
    market_snapshot_parser.add_argument("--json", action="store_true", dest="as_json")

    backtest_equal_weight_parser = subparsers.add_parser(
        "backtest-equal-weight",
        help="Run a full-period equal-weight backtest from a fixed market snapshot.",
    )
    backtest_equal_weight_parser.add_argument("--snapshot-id", required=True)
    backtest_equal_weight_parser.add_argument("--market-snapshot-dir", required=True)
    backtest_equal_weight_parser.add_argument("--initial-capital", type=float, default=10_000.0)
    backtest_equal_weight_parser.add_argument("--transaction-cost", type=float, default=0.0)
    backtest_equal_weight_parser.add_argument("--bars-per-year", type=float, default=252.0)
    backtest_equal_weight_parser.add_argument("--rebalance-schedule", default="hold")
    backtest_equal_weight_parser.add_argument("--include-series", action="store_true")
    backtest_equal_weight_parser.add_argument("--include-events", action="store_true")

    rerun_comparison_spec_parser = subparsers.add_parser(
        "rerun-comparison-spec",
        help="Rerun a saved comparison-run-spec JSON payload.",
    )
    rerun_comparison_spec_parser.add_argument("spec_file")
    rerun_comparison_spec_parser.add_argument("--top", type=int, default=5)
    rerun_comparison_spec_parser.add_argument("--json", action="store_true", dest="as_json")
    rerun_comparison_spec_parser.add_argument("--market-snapshot-dir")
    rerun_comparison_spec_parser.add_argument(
        "--refresh-market-data",
        action="store_true",
        help="Fetch live market data for this ad-hoc rerun instead of reading snapshots.",
    )

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

    benchmark_parser = subparsers.add_parser(
        "benchmark-decomposition",
        help="Decompose the benchmark strategy across fixed reference variants.",
    )
    benchmark_parser.add_argument("--top", type=int, default=10)
    benchmark_parser.add_argument("--walk-forward-start-year", type=int, default=2020)
    benchmark_parser.add_argument("--walk-forward-end-year", type=int, default=2025)
    benchmark_parser.add_argument("--json", action="store_true", dest="as_json")

    strategy_inventory_parser = subparsers.add_parser(
        "strategy-inventory",
        help="List strategy research inventory metadata.",
    )
    strategy_inventory_parser.add_argument("--status", choices=STRATEGY_INVENTORY_STATUSES)
    strategy_inventory_parser.add_argument("--priority", choices=STRATEGY_INVENTORY_PRIORITIES)
    strategy_inventory_parser.add_argument("--family")
    strategy_inventory_parser.add_argument("--with-latest-runs", action="store_true")
    strategy_inventory_parser.add_argument("--with-evaluation-matrix", action="store_true")
    strategy_inventory_parser.add_argument("--json", action="store_true", dest="as_json")

    return parser










def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "comparison-summary":
        if args.refresh_market_data and (args.snapshot_spec_file or args.market_snapshot_dir):
            parser.error("--refresh-market-data cannot be combined with snapshot options")
        if bool(args.snapshot_spec_file) != bool(args.market_snapshot_dir):
            parser.error("--snapshot-spec-file and --market-snapshot-dir must be provided together")
        if not args.refresh_market_data and not args.snapshot_spec_file:
            parser.error(
                "comparison-summary no longer fetches live market data by default. "
                "Use --snapshot-spec-file with --market-snapshot-dir for reproducible evaluations, "
                "or pass --refresh-market-data for an ad-hoc live-data run."
            )
        comparison_fetch_market_universe_bundle = fetch_market_universe_bundle
        if args.snapshot_spec_file:
            snapshot_payload = json.loads(Path(args.snapshot_spec_file).read_text())
            comparison_fetch_market_universe_bundle = build_market_snapshot_fetcher(
                snapshot_payload,
                args.market_snapshot_dir,
            )
        comparison_spec = apply_comparison_universe_variant(DEFAULT_COMPARISON_SPEC, args.universe)
        comparison_spec = filter_comparison_strategies(
            comparison_spec,
            tuple(args.strategy_keys) if args.strategy_keys else None,
        )
        if args.walk_forward:
            payload = build_walk_forward_comparison_payload(
                comparison_spec,
                fetch_market_universe_bundle=comparison_fetch_market_universe_bundle,
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
            fetch_market_universe_bundle=comparison_fetch_market_universe_bundle,
        )
        if args.as_json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(render_comparison_summary(payload, top=max(args.top, 1)))
        return 0

    if args.command == "edge-attribution":
        comparison_spec = apply_comparison_universe_variant(DEFAULT_COMPARISON_SPEC, args.universe)
        payload = edge_attribution_service.build_edge_attribution_payload(
            comparison_spec,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
            strategy_key=args.strategy_key,
            period=args.period,
            universe=args.universe,
            start_year=args.walk_forward_start_year,
            end_year=args.walk_forward_end_year,
        )
        if args.as_json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(render_edge_attribution(payload))
        return 0

    if args.command == "signal-diagnostics":
        horizons = (
            tuple(signal_diagnostics_service.parse_signal_horizon_label(horizon) for horizon in args.horizons)
            if args.horizons
            else signal_diagnostics_service.DEFAULT_SIGNAL_DIAGNOSTIC_HORIZONS
        )
        comparison_spec = apply_comparison_universe_variant(DEFAULT_COMPARISON_SPEC, args.universe)
        comparison_spec = filter_comparison_strategies(
            comparison_spec,
            tuple(args.strategy_keys) if args.strategy_keys else None,
        )
        payload = signal_diagnostics_service.build_signal_diagnostics_payload(
            comparison_spec,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
            period=args.period,
            universe=args.universe,
            horizons=horizons,
            observation_schedule=args.observation_schedule,
        )
        if args.as_json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(render_signal_diagnostics(payload))
        return 0

    if args.command == "robustness-summary":
        progress_callback = None
        if args.progress:
            progress_callback = lambda event: print(render_robustness_progress(event), file=sys.stderr)
        comparison_spec = filter_comparison_strategies(
            DEFAULT_COMPARISON_SPEC,
            tuple(args.strategy_keys) if args.strategy_keys else None,
        )
        payload = robustness_service.build_robustness_summary_payload(
            comparison_spec,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
            apply_universe_variant=apply_comparison_universe_variant,
            profile=args.profile,
            period_keys=tuple(args.period_keys) if args.period_keys else None,
            universes=tuple(args.universes) if args.universes else None,
            cost_multipliers=tuple(args.cost_multipliers) if args.cost_multipliers else None,
            max_weights=tuple(args.max_weights) if args.max_weights else None,
            progress_callback=progress_callback,
        )
        if args.output:
            write_json_payload(args.output, payload)
        if args.as_json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(render_robustness_summary(payload, top=max(args.top, 1)))
        return 0

    if args.command == "benchmark-decomposition":
        payload = build_benchmark_decomposition_payload(
            DEFAULT_COMPARISON_SPEC,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
            start_year=args.walk_forward_start_year,
            end_year=args.walk_forward_end_year,
        )
        if args.as_json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(render_benchmark_decomposition(payload, top=max(args.top, 1)))
        return 0

    if args.command == "strategy-inventory":
        latest_run_records = None
        if args.with_latest_runs or args.with_evaluation_matrix:
            latest_run_records = build_run_result_store(DEFAULT_COMPARISON_SPEC).list_compact_records(
                run_kind="strategy_run",
                view="generic",
            )
        payload = build_strategy_inventory_payload(
            status=args.status,
            priority=args.priority,
            family=args.family,
            with_latest_runs=args.with_latest_runs,
            latest_run_records=latest_run_records,
            with_evaluation_matrix=args.with_evaluation_matrix,
            evaluation_run_records=latest_run_records,
        )
        if args.as_json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(render_strategy_inventory(payload))
        return 0

    if args.command == "comparison-run-spec":
        if args.write_market_snapshots and not args.market_snapshot_dir:
            parser.error("--write-market-snapshots requires --market-snapshot-dir")
        payload = build_comparison_run_spec_payload(
            DEFAULT_COMPARISON_SPEC,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
            market_snapshot_dir=args.market_snapshot_dir if args.write_market_snapshots else None,
        )
        if args.as_json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(render_comparison_run_spec(payload))
        return 0

    if args.command == "market-snapshot-create":
        comparison_spec = apply_comparison_universe_variant(DEFAULT_COMPARISON_SPEC, args.universe)
        comparison_spec = filter_comparison_strategies(
            comparison_spec,
            tuple(args.strategy_keys) if args.strategy_keys else None,
        )

        def fetch_with_selected_provider(*fetch_args, **fetch_kwargs):
            return fetch_market_universe_bundle(
                *fetch_args,
                provider_key=args.provider,
                **fetch_kwargs,
            )

        payload = build_comparison_run_spec_payload(
            comparison_spec,
            fetch_market_universe_bundle=fetch_with_selected_provider,
            market_snapshot_dir=args.market_snapshot_dir,
        )
        if args.as_json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(render_comparison_run_spec(payload))
        return 0

    if args.command == "backtest-equal-weight":
        bundle, metadata = read_market_data_snapshot(
            args.snapshot_id,
            storage_dir=args.market_snapshot_dir,
        )
        result = run_equal_weight_full_period_backtest(
            closes=bundle["closes"],
            volumes=bundle["volumes"],
            initial_capital=args.initial_capital,
            transaction_cost=args.transaction_cost,
            bars_per_year=args.bars_per_year,
            rebalance_schedule=args.rebalance_schedule,
        )
        payload = {
            "kind": "backtest_equal_weight_result",
            "snapshot": {
                "snapshotId": args.snapshot_id,
                "source": metadata["source"],
                "timeframe": metadata["timeframe"],
                "startDate": metadata["aligned_start_date"],
                "endDate": metadata["aligned_end_date"],
                "rowCount": metadata["row_count"],
                "tickers": metadata["tickers"],
            },
            "result": {
                "kind": result["kind"],
                "strategyKey": result["strategyKey"],
                "summary": result["summary"],
                "firstInvestedDate": result["firstInvestedDate"],
                "finalWeights": result["weights"],
                "seriesCount": len(result["series"]),
                "eventCount": len(result["events"]),
            },
        }
        if args.include_series:
            payload["result"]["series"] = result["series"]
        if args.include_events:
            payload["result"]["events"] = result["events"]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "rerun-comparison-spec":
        if args.refresh_market_data and args.market_snapshot_dir:
            parser.error("--refresh-market-data cannot be combined with --market-snapshot-dir")
        if not args.refresh_market_data and not args.market_snapshot_dir:
            parser.error(
                "rerun-comparison-spec requires --market-snapshot-dir for reproducible evaluations, "
                "or --refresh-market-data for an ad-hoc live-data rerun."
            )
        payload = json.loads(Path(args.spec_file).read_text())
        rerun_fetch_market_universe_bundle = (
            build_market_snapshot_fetcher(payload, args.market_snapshot_dir)
            if args.market_snapshot_dir
            else fetch_market_universe_bundle
        )
        comparison_payload = build_comparison_payload_from_run_spec_payload(
            payload,
            fetch_market_universe_bundle=rerun_fetch_market_universe_bundle,
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
