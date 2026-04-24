from __future__ import annotations

import copy
from dataclasses import replace

from app.comparison_service import build_walk_forward_comparison_payload
from app.diagnostics_service import summarize_diagnostic_events
from app.instrument_registry import get_universe_variant, resolve_universe_variant_excluded_tickers
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


BENCHMARK_BASELINE_KEY = "ref-fu-eq-cash-15"


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
