from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
import sys
from pathlib import Path
from typing import Sequence

from app import edge_attribution_service
from app import robustness_service
from app import signal_diagnostics_service
from app.diagnostics_service import summarize_diagnostic_events
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
from app.portfolio import (
    build_evaluator_strategy_spec,
    build_investment_universe_spec,
    build_risk_controls_spec,
    build_strategy_definition_from_evaluator_strategy_spec,
)
from app.strategy_presets import (
    DEFAULT_INVESTMENT_UNIVERSE,
    EQUAL_WEIGHT,
    ETF_ONLY_INVESTMENT_UNIVERSE,
    FULL_UNIVERSE,
    REFERENCE_HOLD_EXECUTION_POLICY,
)
from app.instrument_registry import (
    UNIVERSE_VARIANT_KEYS,
    get_universe_variant,
    resolve_universe_variant_excluded_tickers,
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
    comparison_parser.add_argument("--walk-forward", action="store_true")
    comparison_parser.add_argument("--walk-forward-start-year", type=int, default=2020)
    comparison_parser.add_argument("--walk-forward-end-year", type=int, default=2025)

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

    benchmark_parser = subparsers.add_parser(
        "benchmark-decomposition",
        help="Decompose the benchmark strategy across fixed reference variants.",
    )
    benchmark_parser.add_argument("--top", type=int, default=10)
    benchmark_parser.add_argument("--walk-forward-start-year", type=int, default=2020)
    benchmark_parser.add_argument("--walk-forward-end-year", type=int, default=2025)
    benchmark_parser.add_argument("--json", action="store_true", dest="as_json")

    return parser


def collect_comparison_universe_tickers(comparison) -> tuple[str, ...]:
    tickers: dict[str, None] = {}
    for asset in comparison.run_spec.portfolio_state.current_weights:
        tickers.setdefault(asset, None)
    for strategy in comparison.candidate_strategies + comparison.reference_strategies:
        for ticker in strategy.investment_universe.tickers:
            tickers.setdefault(ticker, None)
        for signal in strategy.signals:
            for ticker in signal.observation_spec.tickers:
                tickers.setdefault(ticker, None)
    return tuple(tickers)


def filter_comparison_strategies(comparison, strategy_keys: tuple[str, ...] | None):
    if not strategy_keys:
        return comparison

    selected_keys = set(strategy_keys)
    available_keys = {
        strategy.key
        for strategy in comparison.candidate_strategies + comparison.reference_strategies
    }
    missing_keys = sorted(selected_keys - available_keys)
    if missing_keys:
        raise ValueError(f"Unknown strategy key(s): {', '.join(missing_keys)}")

    filtered_comparison = copy.deepcopy(comparison)
    filtered_comparison.comparison_id = f"{comparison.comparison_id}__strategies_{'_'.join(strategy_keys)}"
    filtered_comparison.candidate_strategies = [
        strategy
        for strategy in comparison.candidate_strategies
        if strategy.key in selected_keys
    ]
    filtered_comparison.reference_strategies = [
        strategy
        for strategy in comparison.reference_strategies
        if strategy.key in selected_keys
    ]
    if not filtered_comparison.candidate_strategies and not filtered_comparison.reference_strategies:
        raise ValueError("At least one strategy must be selected.")
    return filtered_comparison


def apply_comparison_universe_variant(comparison, universe_key: str):
    variant = get_universe_variant(universe_key)
    excluded_tickers = resolve_universe_variant_excluded_tickers(
        collect_comparison_universe_tickers(comparison),
        universe_key,
    )
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
    variant = get_universe_variant(universe_key)
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
        label=f"{strategy_definition.investment_universe.label} / {variant.label}",
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


def format_optional_percent(value: object) -> str:
    if value is None:
        return "n/a"
    return format_percent(float(value))


def format_optional_decimal(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def format_optional_ratio_percent(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def format_percent_distribution(distribution: dict | None) -> str:
    if not distribution or int(distribution.get("count", 0)) <= 0:
        return "n/a"
    return (
        f"min {format_optional_percent(distribution.get('minimum'))} / "
        f"med {format_optional_percent(distribution.get('median'))} / "
        f"max {format_optional_percent(distribution.get('maximum'))}"
    )


def format_decimal_distribution(distribution: dict | None) -> str:
    if not distribution or int(distribution.get("count", 0)) <= 0:
        return "n/a"
    return (
        f"min {format_optional_decimal(distribution.get('minimum'))} / "
        f"med {format_optional_decimal(distribution.get('median'))} / "
        f"max {format_optional_decimal(distribution.get('maximum'))}"
    )


def format_count_map(counts: dict, *, limit: int = 3) -> str:
    if not counts:
        return "none"
    pairs = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[:limit]
    return ", ".join(f"{key}={value}" for key, value in pairs)


def render_availability_diagnostics(lines: list[str], diagnostics: dict | None) -> None:
    if not diagnostics:
        return
    actionable_warnings = diagnostics.get("actionableWarnings") or []
    calendar_boundary_warning_count = int(diagnostics.get("calendarBoundaryWarningCount") or 0)
    asset_lifecycle_warning_count = int(diagnostics.get("assetLifecycleWarningCount") or 0)
    if actionable_warnings:
        lines.append("Warnings:")
        for warning in actionable_warnings:
            lines.append(f"- {warning['message']}")
    if calendar_boundary_warning_count:
        if actionable_warnings:
            lines.append("")
        lines.append(
            "Calendar boundary differences: "
            f"{calendar_boundary_warning_count} classified as non-actionable."
        )
    if asset_lifecycle_warning_count:
        if actionable_warnings or calendar_boundary_warning_count:
            lines.append("")
        lines.append(
            "Asset lifecycle differences: "
            f"{asset_lifecycle_warning_count} classified as non-actionable."
        )
    if actionable_warnings or calendar_boundary_warning_count or asset_lifecycle_warning_count:
        lines.append("")


def render_instrument_diagnostics(lines: list[str], diagnostics: dict | None) -> None:
    if not diagnostics or not diagnostics.get("mixedMarketCalendar"):
        return
    calendars = ", ".join((diagnostics.get("marketCalendars") or {}).keys())
    lines.append(f"Calendar diagnostics: mixed market calendars {calendars}.")
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


BENCHMARK_BASELINE_KEY = "ref-fu-eq-cash-15"


def build_benchmark_decomposition_payload(
    comparison,
    *,
    fetch_market_universe_bundle,
    start_year: int = 2020,
    end_year: int = 2025,
) -> dict:
    benchmark_comparison = build_benchmark_decomposition_comparison(comparison)
    walk_forward_payload = build_walk_forward_comparison_payload(
        benchmark_comparison,
        fetch_market_universe_bundle=fetch_market_universe_bundle,
        start_year=start_year,
        end_year=end_year,
    )
    evaluation = walk_forward_payload.get("comparison", {}).get("runSpec", {}).get("evaluation", {})
    diagnostic_events = evaluation.get("diagnosticEvents") or []
    diagnostic_summary = summarize_diagnostic_events(diagnostic_events)
    benchmarks = [
        build_benchmark_result(result, diagnostic_summary=diagnostic_summary)
        for result in walk_forward_payload.get("referenceResults", [])
    ]
    baseline = next(
        (benchmark for benchmark in benchmarks if benchmark["strategyKey"] == BENCHMARK_BASELINE_KEY),
        None,
    )
    if baseline is not None:
        for benchmark in benchmarks:
            benchmark["deltaVsBaseline"] = build_benchmark_delta(benchmark, baseline)

    return {
        "kind": "benchmark_decomposition",
        "schemaVersion": "v1",
        "baselineKey": BENCHMARK_BASELINE_KEY,
        "walkForward": walk_forward_payload["walkForward"],
        "runStoreSummary": walk_forward_payload["runStoreSummary"],
        "diagnosticSummary": diagnostic_summary,
        "benchmarks": benchmarks,
    }


def build_benchmark_decomposition_comparison(comparison):
    benchmark_comparison = copy.deepcopy(comparison)
    benchmark_comparison.comparison_id = f"{comparison.comparison_id}__benchmark_decomposition"
    benchmark_comparison.title = "Benchmark decomposition"
    benchmark_comparison.question = "ref-fu-eq-cash の強さを固定 benchmark variants で分解する"
    benchmark_comparison.candidate_strategies = []
    benchmark_comparison.reference_strategies = build_benchmark_reference_strategies()
    return benchmark_comparison


def build_benchmark_reference_strategies() -> list:
    spy_universe = build_investment_universe_spec(
        key="spy_only_v1",
        label="SPY only",
        tickers=("SPY",),
    )
    return [
        build_equal_weight_hold_benchmark(
            strategy_id="ref-fu-eq-cash-15",
            universe=DEFAULT_INVESTMENT_UNIVERSE,
            cash_weight=0.15,
            label="全20資産等金額 + CASH 15%",
        ),
        build_equal_weight_hold_benchmark(
            strategy_id="ref-fu-eq-cash-0",
            universe=DEFAULT_INVESTMENT_UNIVERSE,
            cash_weight=0.0,
            label="全20資産等金額 + CASH 0%",
        ),
        build_equal_weight_hold_benchmark(
            strategy_id="ref-fu-eq-cash-25",
            universe=DEFAULT_INVESTMENT_UNIVERSE,
            cash_weight=0.25,
            label="全20資産等金額 + CASH 25%",
        ),
        build_equal_weight_hold_benchmark(
            strategy_id="ref-etf-eq-cash-15",
            universe=ETF_ONLY_INVESTMENT_UNIVERSE,
            cash_weight=0.15,
            label="ETF only 等金額 + CASH 15%",
        ),
        build_equal_weight_hold_benchmark(
            strategy_id="ref-spy-hold",
            universe=spy_universe,
            cash_weight=0.0,
            label="SPY 100% buy and hold",
        ),
        build_equal_weight_hold_benchmark(
            strategy_id="ref-spy-cash-15",
            universe=spy_universe,
            cash_weight=0.15,
            label="SPY 85% + CASH 15%",
        ),
    ]


def build_equal_weight_hold_benchmark(*, strategy_id: str, universe, cash_weight: float, label: str):
    max_investment_ratio = round(1.0 - float(cash_weight), 10)
    strategy = build_evaluator_strategy_spec(
        strategy_id=strategy_id,
        investment_universe=universe,
        selection=FULL_UNIVERSE,
        portfolio_model=EQUAL_WEIGHT,
        execution_policy=REFERENCE_HOLD_EXECUTION_POLICY,
        risk_controls=build_risk_controls_spec(
            max_investment_ratio=max_investment_ratio,
            max_weight=None,
        ),
        label=label,
        description=f"{label} benchmark",
    )
    return build_strategy_definition_from_evaluator_strategy_spec(strategy)


def build_benchmark_result(result: dict, *, diagnostic_summary: dict) -> dict:
    worst_window = build_benchmark_worst_window(result.get("windows", []))
    return {
        "strategyKey": result["strategyKey"],
        "label": result["strategyLabel"],
        "summary": {
            "averageSharpeRatio": result["averageSharpeRatio"],
            "minimumSharpeRatio": result["minimumSharpeRatio"],
            "averageTotalReturnPct": result["averageTotalReturnPct"],
            "averageMaxDrawdownPct": result["averageMaxDrawdownPct"],
            "averageTurnoverPct": result["averageTurnoverPct"],
            "positiveReturnWindowCount": result["positiveReturnWindowCount"],
            "windowCount": result["windowCount"],
        },
        "windows": result.get("windows", []),
        "worstWindow": worst_window,
        "finalWeights": build_benchmark_final_weights(result["strategy"]),
        "diagnosticSummary": diagnostic_summary,
    }


def build_benchmark_worst_window(windows: list[dict]) -> dict | None:
    if not windows:
        return None
    return min(
        windows,
        key=lambda window: (
            float(window["test"]["sharpeRatio"]),
            float(window["test"]["totalReturnPct"]),
            -float(window["test"]["maxDrawdownPct"]),
            int(window["year"]),
        ),
    )


def build_benchmark_final_weights(strategy_payload: dict) -> list[dict]:
    core = strategy_payload["components"]["core"]
    optional = strategy_payload["components"]["optional"]
    tickers = core["investmentUniverse"]["tickers"]
    max_investment_ratio = float(optional["riskControls"]["maxInvestmentPct"]) / 100
    asset_weight_pct = round(max_investment_ratio * 100 / len(tickers), 2)
    rows = [
        {"asset": ticker, "weightPct": asset_weight_pct}
        for ticker in tickers
    ]
    cash_weight_pct = round((1.0 - max_investment_ratio) * 100, 2)
    if cash_weight_pct > 0:
        rows.append({"asset": "CASH", "weightPct": cash_weight_pct})
    return rows


def build_benchmark_delta(benchmark: dict, baseline: dict) -> dict:
    benchmark_summary = benchmark["summary"]
    baseline_summary = baseline["summary"]
    return {
        "averageSharpeRatio": round(
            float(benchmark_summary["averageSharpeRatio"]) - float(baseline_summary["averageSharpeRatio"]),
            6,
        ),
        "minimumSharpeRatio": round(
            float(benchmark_summary["minimumSharpeRatio"]) - float(baseline_summary["minimumSharpeRatio"]),
            6,
        ),
        "averageTotalReturnPct": round(
            float(benchmark_summary["averageTotalReturnPct"]) - float(baseline_summary["averageTotalReturnPct"]),
            6,
        ),
        "averageMaxDrawdownPct": round(
            float(benchmark_summary["averageMaxDrawdownPct"]) - float(baseline_summary["averageMaxDrawdownPct"]),
            6,
        ),
        "averageTurnoverPct": round(
            float(benchmark_summary["averageTurnoverPct"]) - float(baseline_summary["averageTurnoverPct"]),
            6,
        ),
    }


def render_benchmark_decomposition(payload: dict, *, top: int) -> str:
    walk_forward = payload["walkForward"]
    benchmarks = payload["benchmarks"]
    lines = [
        "Benchmark decomposition",
        f"Baseline: {payload['baselineKey']}",
        (
            "Walk-forward: "
            f"{walk_forward['startYear']}-{walk_forward['endYear']} "
            f"({walk_forward['windowCount']} windows)"
        ),
        (
            "Run store: "
            f"cached={payload['runStoreSummary']['cachedRunCount']} "
            f"computed={payload['runStoreSummary']['computedRunCount']}"
        ),
        "",
        f"Top {min(top, len(benchmarks))} benchmarks by walk-forward performance:",
    ]
    for index, benchmark in enumerate(benchmarks[:top], start=1):
        summary = benchmark["summary"]
        delta = benchmark.get("deltaVsBaseline", {})
        lines.append(f"{index}. {benchmark['label']} [{benchmark['strategyKey']}]")
        lines.append(
            "   "
            f"Avg Sharpe {summary['averageSharpeRatio']:.3f} | "
            f"Min Sharpe {summary['minimumSharpeRatio']:.3f} | "
            f"Avg Return {format_percent(summary['averageTotalReturnPct'])} | "
            f"Avg MDD {format_percent(summary['averageMaxDrawdownPct'])} | "
            f"Avg Turnover {format_percent(summary['averageTurnoverPct'])}"
        )
        if delta:
            lines.append(
                "   "
                "Delta vs ref-fu-eq-cash-15: "
                f"Sharpe {delta['averageSharpeRatio']:+.3f} | "
                f"Min Sharpe {delta['minimumSharpeRatio']:+.3f} | "
                f"Return {format_percent(delta['averageTotalReturnPct'])} | "
                f"Turnover {format_percent(delta['averageTurnoverPct'])}"
            )
        worst_window = benchmark.get("worstWindow")
        if worst_window:
            test = worst_window["test"]
            lines.append(
                "   "
                f"Worst window {worst_window['year']} | "
                f"Sharpe {test['sharpeRatio']:.3f} | "
                f"Return {format_percent(test['totalReturnPct'])} | "
                f"MDD {format_percent(test['maxDrawdownPct'])}"
            )
    return "\n".join(lines)


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
    render_instrument_diagnostics(lines, evaluation.get("instrumentDiagnostics"))
    render_availability_diagnostics(lines, evaluation.get("availabilityDiagnostics"))

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

    render_instrument_diagnostics(lines, evaluation.get("instrumentDiagnostics"))
    render_availability_diagnostics(lines, evaluation.get("availabilityDiagnostics"))

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


def render_edge_attribution(payload: dict) -> str:
    walk_forward = payload["walkForward"]
    lines = [
        f"Edge attribution: {payload['strategyLabel']} [{payload['strategyKey']}]",
        f"Period: {payload['period']} | universe {payload['universe']}",
        (
            "Walk-forward: "
            f"{walk_forward['startYear']}-{walk_forward['endYear']} "
            f"({walk_forward['windowCount']} windows)"
        ),
        f"Baseline: {payload['baselineKey']}",
        (
            "Run store: "
            f"cached={payload['runStoreSummary']['cachedRunCount']} "
            f"computed={payload['runStoreSummary']['computedRunCount']}"
        ),
        "",
        "Components:",
    ]
    for component in payload.get("components") or []:
        summary = component["summary"]
        delta = component.get("deltaVsBaseline") or {}
        lines.append(f"- {component['label']} [{component['componentKey']}]")
        lines.append(
            "   "
            f"Avg Sharpe {summary['averageSharpeRatio']:.3f} | "
            f"Min Sharpe {summary['minimumSharpeRatio']:.3f} | "
            f"Avg Return {format_percent(summary['averageTotalReturnPct'])} | "
            f"Avg CAGR {format_percent(summary['averageCagrPct'])} | "
            f"Avg MDD {format_percent(summary['averageMaxDrawdownPct'])} | "
            f"Avg Turnover {format_percent(summary['averageTurnoverPct'])}"
        )
        if delta:
            lines.append(
                "   "
                "Delta vs universe: "
                f"Sharpe {delta['averageSharpeRatio']:+.3f} | "
                f"Return {format_percent(delta['averageTotalReturnPct'])} | "
                f"CAGR {format_percent(delta['averageCagrPct'])} | "
                f"Turnover {format_percent(delta['averageTurnoverPct'])}"
            )
    effect = payload.get("effectSummary") or {}
    full_vs_universe = effect.get("fullEffectVsUniverse")
    full_vs_cash = effect.get("fullEffectVsCash")
    if full_vs_universe and full_vs_cash:
        lines.extend([
            "",
            (
                "Full effect vs universe: "
                f"Sharpe {full_vs_universe['averageSharpeRatio']:+.3f} | "
                f"Return {format_percent(full_vs_universe['averageTotalReturnPct'])} | "
                f"CAGR {format_percent(full_vs_universe['averageCagrPct'])}"
            ),
            (
                "Full effect vs cash: "
                f"Sharpe {full_vs_cash['averageSharpeRatio']:+.3f} | "
                f"Return {format_percent(full_vs_cash['averageTotalReturnPct'])} | "
                f"CAGR {format_percent(full_vs_cash['averageCagrPct'])}"
            ),
        ])
    return "\n".join(lines)


def render_signal_diagnostics(payload: dict) -> str:
    lines = [
        (
            f"Signal diagnostics: {payload['strategyCount']} strategies | "
            f"period {payload['period']} | universe {payload['universe']} | "
            f"horizons {', '.join(payload.get('horizons') or [])}"
        )
    ]
    for result in payload.get("strategyResults") or []:
        lines.append(f"{result['strategyLabel']} [{result['strategyKey']}]")
        for horizon_result in result.get("horizonResults") or []:
            lines.append(
                "   "
                f"{horizon_result['horizon']} | "
                f"samples {horizon_result['sampleCount']} | "
                f"rank IC {format_optional_decimal(horizon_result.get('rankIc'))} | "
                f"spread {format_optional_percent(horizon_result.get('topMinusBottomForwardReturnPct'))} | "
                f"top {format_optional_percent(horizon_result.get('topBucketForwardReturnPct'))} | "
                f"bottom {format_optional_percent(horizon_result.get('bottomBucketForwardReturnPct'))} | "
                f"hit {format_optional_ratio_percent(horizon_result.get('hitRate'))}"
            )
    return "\n".join(lines)


def render_robustness_summary(payload: dict, *, top: int) -> str:
    matrix = payload["matrix"]
    strategy_results = payload["strategyResults"]
    periods = ", ".join(period["label"] for period in matrix["periods"])
    universes = ", ".join(matrix["universes"])
    cost_multipliers = ", ".join(f"x{value:.1f}" for value in matrix["costMultipliers"])
    max_weights = ", ".join(f"{value:.0f}%" for value in matrix["maxWeightPcts"])
    decision_counts = payload.get("decisionSummary", {}).get("counts", {})
    decision_summary = ", ".join(
        f"{decision}={decision_counts.get(decision, 0)}"
        for decision in robustness_service.DECISION_PRIORITY
    )
    lines = [
        f"Robustness summary: {payload['scenarioCount']} scenarios ({payload.get('profile', 'unknown')} profile)",
        f"Decisions: {decision_summary}",
        f"Periods: {periods}",
        f"Universes: {universes}",
        f"Costs: {cost_multipliers} | Max weights: {max_weights}",
        (
            "Ranking policy: decision / worst Sharpe / top-5 stability / "
            "average Sharpe"
        ),
        (
            "Run store: "
            f"cached={payload['runStoreSummary']['cachedRunCount']} "
            f"computed={payload['runStoreSummary']['computedRunCount']}"
        ),
        "",
        f"Top {min(top, len(strategy_results))} strategies by robustness:",
    ]
    for index, result in enumerate(strategy_results[:top], start=1):
        risks = ", ".join(result["diagnosticFlags"]) if result["diagnosticFlags"] else "none"
        reasons = ", ".join(result.get("decisionReasons") or []) or "none"
        lines.append(f"{index}. {result['strategyLabel']} [{result['strategyKey']}]")
        lines.append(
            "   "
            f"Decision {result['decision']} | "
            f"Reasons: {reasons} | "
            f"Avg Sharpe {result['averageSharpeRatio']:.3f} | "
            f"Worst Sharpe {result['worstSharpeRatio']:.3f} | "
            f"Top5 {result['top5ScenarioCount']}/{result['scenarioCount']}"
        )
        lines.append(
            "   "
            f"Avg Return {format_percent(result['averageTotalReturnPct'])} | "
            f"Worst MDD {format_percent(result['worstMaxDrawdownPct'])} | "
            f"Avg Turnover {format_percent(result['averageTurnoverPct'])} | "
            f"Max Turnover {format_percent(result['maxTurnoverPct'])}"
        )
        diversification = result.get("diversificationSummary") or {}
        exposure = result.get("exposureSummary") or {}
        if diversification:
            lines.append(
                "   "
                f"Holdings avg {diversification['averageHoldingCount']:.1f} | "
                f"min {diversification['minimumHoldingCount']} | "
                f"Top5 weight {format_percent(diversification['averageTop5WeightPct'])} | "
                f"Max asset {format_percent(diversification['maximumSingleAssetWeightPct'])}"
            )
        if exposure:
            lines.append(
                "   "
                f"Invested {format_percent(exposure['averageInvestedWeightPct'])} | "
                f"Cash {format_percent(exposure['averageCashWeightPct'])}"
            )
        execution_decisions = result.get("executionDecisionSummary") or {}
        if int(execution_decisions.get("decisionCount", 0)) > 0:
            lines.append(
                "   "
                f"Exec decisions {execution_decisions['decisionCount']} | "
                f"rebalance {execution_decisions['rebalanceCount']} | "
                f"no-trade {execution_decisions['noTradeCount']} | "
                f"avg edge {format_optional_percent(execution_decisions.get('averageEstimatedEdgePct'))} | "
                f"avg cost {format_optional_percent(execution_decisions.get('averageEstimatedCostPct'))}"
            )
            lines.append(
                "   "
                f"Exec reasons {format_count_map(execution_decisions.get('reasonCounts') or {})}"
            )
            lines.append(
                "   "
                f"Exec edge sources {format_count_map(execution_decisions.get('edgeSourceCounts') or {})}"
            )
            lines.append(
                "   "
                "Exec edge "
                f"{format_percent_distribution(execution_decisions.get('estimatedEdgePctDistribution'))} | "
                "cost "
                f"{format_percent_distribution(execution_decisions.get('estimatedCostPctDistribution'))} | "
                "edge-cost "
                f"{format_percent_distribution(execution_decisions.get('estimatedEdgeAfterCostPctDistribution'))}"
            )
            lines.append(
                "   "
                "Exec realized edge "
                f"{format_percent_distribution(execution_decisions.get('realizedEdgePctDistribution'))} | "
                "realized edge-cost "
                f"{format_percent_distribution(execution_decisions.get('realizedEdgeAfterCostPctDistribution'))}"
            )
            lines.append(
                "   "
                "Exec edge hit "
                f"{format_optional_ratio_percent(execution_decisions.get('edgeHitRate'))} | "
                f"corr {format_optional_decimal(execution_decisions.get('estimatedVsRealizedEdgeCorrelation'))} | "
                f"samples {execution_decisions.get('edgeHitSampleCount', 0)}"
            )
            lines.append(
                "   "
                "Exec confidence "
                f"{format_decimal_distribution(execution_decisions.get('confidenceDistribution'))}"
            )
        delta = result.get("deltaVsBaseline")
        if delta:
            lines.append(
                "   "
                f"Delta vs {payload['baselineKey']}: "
                f"Sharpe {delta['averageSharpeRatio']:+.3f} | "
                f"Worst Sharpe {delta['worstSharpeRatio']:+.3f} | "
                f"Return {format_percent(delta['averageTotalReturnPct'])} | "
                f"MDD {format_percent(delta['worstMaxDrawdownPct'])} | "
                f"Turnover {format_percent(delta['averageTurnoverPct'])}"
            )
        worst = result["worstScenario"]
        lines.append(
            "   "
            f"Crypto sensitivity {result['cryptoSensitivity']:+.3f} | "
            f"Cost sensitivity {result['costSensitivity']:+.3f} | "
            f"Risks: {risks}"
        )
        diagnostic_summary = result.get("diagnosticSummary") or {}
        if diagnostic_summary.get("eventCount"):
            severity_counts = diagnostic_summary.get("severityCounts") or {}
            lines.append(
                "   "
                f"Diagnostics {diagnostic_summary['eventCount']} events | "
                f"invalidating={severity_counts.get('invalidating', 0)} "
                f"warning={severity_counts.get('warning', 0)} "
                f"info={severity_counts.get('info', 0)}"
            )
        representative = result.get("representativeDiagnostic")
        if representative:
            lines.append(
                "   "
                "Representative diagnostic "
                f"{representative.get('severity')}/"
                f"{representative.get('category')}/"
                f"{representative.get('kind')}: "
                f"{representative.get('reason')}"
            )
        lines.append(
            "   "
            f"Worst scenario {worst['scenarioKey']} | "
            f"Min Sharpe {worst['minimumSharpeRatio']:.3f} | "
            f"Rank {worst['rank']}"
        )
        worst_window = result.get("worstWindow") or {}
        window = worst_window.get("window") or {}
        test_metrics = window.get("test") or {}
        test_availability = window.get("testAvailability") or {}
        if window:
            eligible_min = test_availability.get("minEligibleAssetCount", "?")
            eligible_max = test_availability.get("maxEligibleAssetCount", "?")
            lines.append(
                "   "
                f"Worst window {worst_window['scenarioKey']} / {window['year']} "
                f"{window['testStartDate']}..{window['testEndDate']} | "
                f"Sharpe {test_metrics['sharpeRatio']:.3f} | "
                f"Return {format_percent(test_metrics['totalReturnPct'])} | "
                f"MDD {format_percent(test_metrics['maxDrawdownPct'])} | "
                f"Eligible {eligible_min}-{eligible_max}"
            )
    return "\n".join(lines)


def render_robustness_progress(event: dict) -> str:
    scenario = event["scenario"]
    prefix = f"scenario {event['scenarioIndex']}/{event['scenarioCount']}"
    if event["kind"] == "scenario_started":
        return f"{prefix} started: {scenario['key']}"
    run_store_summary = event["runStoreSummary"]
    return (
        f"{prefix} done: {scenario['key']} | "
        f"cached={run_store_summary['cachedRunCount']} "
        f"computed={run_store_summary['computedRunCount']} "
        f"elapsed={event['elapsedSeconds']:.3f}s"
    )


def write_json_payload(path: str, payload: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


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
