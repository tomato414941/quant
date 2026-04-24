from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from app.comparison_models import ComparisonSpec
from app.comparison_market_context import collect_comparison_tickers, resolve_strategy_market_data_timeframe
from app.instrument_registry import get_normalized_asset_class
from app.portfolio import (
    build_executable_evaluator_strategy_spec_from_definition,
    compute_strategy_score_series,
    get_strategy_signal_execution_contexts,
    prepare_strategy_signal_data,
    should_rebalance,
)

DEFAULT_SIGNAL_DIAGNOSTIC_HORIZONS = (1, 5, 21)
DEFAULT_SIGNAL_DIAGNOSTIC_BUCKET_COUNT = 5
MIN_SIGNAL_DIAGNOSTIC_ASSET_COUNT = 2
MIN_SIGNAL_DIAGNOSTIC_HISTORY_BARS = 3
SIGNAL_DIAGNOSTIC_OBSERVATION_SCHEDULES = (
    "strategy",
    "daily",
    "month_end",
    "quarter_end",
    "year_end",
)
RESOLVED_SIGNAL_DIAGNOSTIC_OBSERVATION_SCHEDULES = (
    "daily",
    "month_end",
    "quarter_end",
    "year_end",
    "hold",
)


def parse_signal_horizon_label(label: str) -> int:
    normalized = label.strip().lower()
    if not normalized.endswith("d"):
        raise ValueError("Signal diagnostic horizons must use day labels such as 1d, 5d, or 21d.")
    value = int(normalized[:-1])
    if value <= 0:
        raise ValueError("Signal diagnostic horizon must be positive.")
    return value


def format_signal_horizon_label(horizon_bars: int) -> str:
    return f"{int(horizon_bars)}d"

def normalize_signal_diagnostic_observation_schedule(observation_schedule: str) -> str:
    normalized = observation_schedule.strip().lower()
    if normalized == "every_bar":
        return "daily"
    if normalized not in SIGNAL_DIAGNOSTIC_OBSERVATION_SCHEDULES:
        raise ValueError("Unsupported signal diagnostic observation schedule.")
    return normalized


def resolve_signal_diagnostic_observation_schedule(*, strategy, observation_schedule: str) -> str:
    normalized = normalize_signal_diagnostic_observation_schedule(observation_schedule)
    if normalized != "strategy":
        return normalized

    strategy_schedule = str(strategy.decision_schedule or strategy.execution_policy.rebalance_schedule)
    if strategy_schedule == "every_bar":
        return "daily"
    if strategy_schedule not in RESOLVED_SIGNAL_DIAGNOSTIC_OBSERVATION_SCHEDULES:
        raise ValueError("Unsupported strategy decision schedule for signal diagnostics.")
    return strategy_schedule


def describe_observation_count_semantics(*, observation_schedule: str, resolved_observation_schedule: str) -> str:
    if resolved_observation_schedule == "daily":
        return "daily overlapping forward-return windows"
    if observation_schedule == "strategy":
        return "strategy decision dates only"
    if resolved_observation_schedule == "hold":
        return "single initial observation after minimum history"
    return f"{resolved_observation_schedule} observations only"


def should_include_signal_observation(
    *,
    index: int,
    index_values: pd.Index,
    resolved_observation_schedule: str,
) -> bool:
    if resolved_observation_schedule == "daily":
        return True
    if resolved_observation_schedule == "hold":
        return index == MIN_SIGNAL_DIAGNOSTIC_HISTORY_BARS
    return should_rebalance(
        index_values[index - 1],
        index_values[index],
        resolved_observation_schedule,
    )



def optional_round(value: float | None, digits: int = 4) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def summarize_signal_horizon_observations(
    observations: list[dict[str, float]],
    *,
    horizon_bars: int,
    bucket_count: int = DEFAULT_SIGNAL_DIAGNOSTIC_BUCKET_COUNT,
) -> dict[str, object]:
    rank_ics = [observation["rank_ic"] for observation in observations if math.isfinite(observation["rank_ic"])]
    top_returns = [observation["top_return"] for observation in observations]
    bottom_returns = [observation["bottom_return"] for observation in observations]
    spreads = [observation["spread"] for observation in observations]
    hit_values = [observation["spread"] > 0 for observation in observations]
    sample_count = sum(int(observation["sample_count"]) for observation in observations)
    return {
        "horizon": format_signal_horizon_label(horizon_bars),
        "horizonBars": int(horizon_bars),
        "bucketCount": int(bucket_count),
        "observationCount": len(observations),
        "sampleCount": sample_count,
        "rankIc": optional_round(float(np.mean(rank_ics)) if rank_ics else None),
        "topMinusBottomForwardReturnPct": optional_round(float(np.mean(spreads)) * 100 if spreads else None),
        "topBucketForwardReturnPct": optional_round(float(np.mean(top_returns)) * 100 if top_returns else None),
        "bottomBucketForwardReturnPct": optional_round(float(np.mean(bottom_returns)) * 100 if bottom_returns else None),
        "hitRate": optional_round(float(np.mean(hit_values)) if hit_values else None),
    }


def build_signal_horizon_observation(
    scores: pd.Series,
    forward_returns: pd.Series,
    *,
    bucket_count: int,
) -> dict[str, float] | None:
    aligned = (
        pd.DataFrame({"score": scores, "forward_return": forward_returns})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(aligned) < MIN_SIGNAL_DIAGNOSTIC_ASSET_COUNT or aligned["score"].nunique() < 2:
        return None
    score_ranks = aligned["score"].rank(method="average", pct=True)
    return_ranks = aligned["forward_return"].rank(method="average", pct=True)
    rank_ic = float(score_ranks.corr(return_ranks))
    if not math.isfinite(rank_ic):
        return None

    effective_bucket_count = max(2, min(int(bucket_count), len(aligned)))
    bottom_cutoff = 1.0 / effective_bucket_count
    top_cutoff = 1.0 - bottom_cutoff
    top_returns = aligned.loc[score_ranks >= top_cutoff, "forward_return"]
    bottom_returns = aligned.loc[score_ranks <= bottom_cutoff, "forward_return"]
    if top_returns.empty or bottom_returns.empty:
        return None
    top_return = float(top_returns.mean())
    bottom_return = float(bottom_returns.mean())
    return {
        "rank_ic": rank_ic,
        "top_return": top_return,
        "bottom_return": bottom_return,
        "spread": top_return - bottom_return,
        "sample_count": float(len(aligned)),
    }


def get_asset_class(ticker: str) -> str:
    return get_normalized_asset_class(str(ticker))


def group_assets_by_class(columns: pd.Index) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for column in columns:
        grouped.setdefault(get_asset_class(str(column)), []).append(str(column))
    return grouped


def summarize_yearly_signal_observations(
    observations_by_year: dict[int, list[dict[str, float]]],
    *,
    horizon_bars: int,
    bucket_count: int,
) -> list[dict[str, object]]:
    results = []
    for year, observations in sorted(observations_by_year.items()):
        summary = summarize_signal_horizon_observations(
            observations,
            horizon_bars=horizon_bars,
            bucket_count=bucket_count,
        )
        results.append({"year": int(year), **summary})
    return results


def summarize_asset_class_signal_observations(
    observations_by_asset_class: dict[str, list[dict[str, float]]],
    *,
    horizon_bars: int,
    bucket_count: int,
    asset_class_counts: dict[str, int],
) -> list[dict[str, object]]:
    results = []
    for asset_class, observations in sorted(observations_by_asset_class.items()):
        summary = summarize_signal_horizon_observations(
            observations,
            horizon_bars=horizon_bars,
            bucket_count=bucket_count,
        )
        results.append({
            "assetClass": asset_class,
            "assetCount": int(asset_class_counts.get(asset_class, 0)),
            **summary,
        })
    return results


def has_positive_signal(result: dict[str, object]) -> bool:
    return (
        result.get("rankIc") is not None
        and result.get("topMinusBottomForwardReturnPct") is not None
        and result.get("hitRate") is not None
        and float(result["rankIc"]) > 0
        and float(result["topMinusBottomForwardReturnPct"]) > 0
        and float(result["hitRate"]) > 0.5
    )


def build_signal_diagnostics_diagnosis(
    *,
    horizon_results: list[dict[str, object]],
    yearly_results: list[dict[str, object]],
    asset_class_results: list[dict[str, object]],
) -> dict[str, object]:
    valid_horizons = [result for result in horizon_results if result.get("rankIc") is not None]
    if not valid_horizons:
        return {
            "primaryFinding": "insufficient_signal_data",
            "flags": ["insufficient_signal_data"],
            "positiveHorizonCount": 0,
            "validHorizonCount": 0,
            "positiveYearCount": 0,
            "validYearCount": 0,
            "positiveAssetClassCount": 0,
            "validAssetClassCount": 0,
        }

    positive_horizons = [result for result in valid_horizons if has_positive_signal(result)]
    valid_years = [result for result in yearly_results if result.get("rankIc") is not None]
    positive_years = [result for result in valid_years if has_positive_signal(result)]
    valid_asset_classes = [result for result in asset_class_results if result.get("rankIc") is not None]
    positive_asset_classes = [result for result in valid_asset_classes if has_positive_signal(result)]
    avg_rank_ic = float(np.mean([float(result["rankIc"]) for result in valid_horizons]))
    avg_spread = float(np.mean([float(result["topMinusBottomForwardReturnPct"]) for result in valid_horizons]))
    avg_hit_rate = float(np.mean([
        float(result["hitRate"]) for result in valid_horizons if result.get("hitRate") is not None
    ]))

    flags = []
    if not positive_horizons or avg_rank_ic < 0.02 or avg_spread <= 0 or avg_hit_rate < 0.52:
        flags.append("weak_signal")
    if valid_years and len(positive_years) / len(valid_years) < 0.6:
        flags.append("unstable_signal")

    crypto_results = [result for result in valid_asset_classes if result.get("assetClass") == "crypto"]
    non_crypto_results = [result for result in valid_asset_classes if result.get("assetClass") != "crypto"]
    crypto_positive = any(has_positive_signal(result) for result in crypto_results)
    non_crypto_positive = any(has_positive_signal(result) for result in non_crypto_results)
    if crypto_positive and not non_crypto_positive:
        flags.append("crypto_dependent_signal")

    if not flags:
        primary_finding = "usable_signal"
    elif "weak_signal" in flags and positive_horizons:
        primary_finding = "usable_but_weak_signal"
    else:
        primary_finding = flags[0]
    return {
        "primaryFinding": primary_finding,
        "flags": flags or ["no_obvious_signal_issue"],
        "averageRankIc": optional_round(avg_rank_ic),
        "averageTopMinusBottomForwardReturnPct": optional_round(avg_spread),
        "averageHitRate": optional_round(avg_hit_rate),
        "positiveHorizonCount": len(positive_horizons),
        "validHorizonCount": len(valid_horizons),
        "positiveYearCount": len(positive_years),
        "validYearCount": len(valid_years),
        "positiveAssetClassCount": len(positive_asset_classes),
        "validAssetClassCount": len(valid_asset_classes),
    }


def build_empty_strategy_signal_diagnostics(
    *,
    strategy_definition,
    horizons: Sequence[int],
    bucket_count: int,
    observation_schedule: str,
    resolved_observation_schedule: str,
    observation_count_semantics: str,
) -> dict[str, object]:
    horizon_results = [
        summarize_signal_horizon_observations([], horizon_bars=int(horizon), bucket_count=bucket_count)
        for horizon in horizons
    ]
    return {
        "strategyKey": strategy_definition.key,
        "strategyLabel": strategy_definition.label,
        "observationSchedule": observation_schedule,
        "resolvedObservationSchedule": resolved_observation_schedule,
        "observationCountSemantics": observation_count_semantics,
        "horizonResults": horizon_results,
        "yearlyResults": [],
        "assetClassResults": [],
        "diagnosis": build_signal_diagnostics_diagnosis(
            horizon_results=horizon_results,
            yearly_results=[],
            asset_class_results=[],
        ),
    }


def build_strategy_signal_diagnostics(
    *,
    strategy_definition,
    closes: pd.DataFrame,
    volumes: pd.DataFrame | None,
    horizons: Sequence[int] = DEFAULT_SIGNAL_DIAGNOSTIC_HORIZONS,
    bucket_count: int = DEFAULT_SIGNAL_DIAGNOSTIC_BUCKET_COUNT,
    observation_schedule: str = "strategy",
) -> dict[str, object]:
    strategy = build_executable_evaluator_strategy_spec_from_definition(strategy_definition)
    resolved_observation_schedule = resolve_signal_diagnostic_observation_schedule(
        strategy=strategy,
        observation_schedule=observation_schedule,
    )
    observation_count_semantics = describe_observation_count_semantics(
        observation_schedule=observation_schedule,
        resolved_observation_schedule=resolved_observation_schedule,
    )
    selection_contexts, predictor_context = get_strategy_signal_execution_contexts(strategy)
    available_assets = [asset for asset in strategy.investment_universe.tickers if asset in closes.columns]
    if len(available_assets) < MIN_SIGNAL_DIAGNOSTIC_ASSET_COUNT:
        return build_empty_strategy_signal_diagnostics(
            strategy_definition=strategy_definition,
            horizons=horizons,
            bucket_count=bucket_count,
            observation_schedule=observation_schedule,
            resolved_observation_schedule=resolved_observation_schedule,
            observation_count_semantics=observation_count_semantics,
        )

    scoped_closes = closes[available_assets].replace([np.inf, -np.inf], np.nan).ffill().dropna(how="all")
    scoped_volumes = None if volumes is None else volumes.reindex(scoped_closes.index)[available_assets]
    market_returns = scoped_closes.pct_change(fill_method=None).dropna(how="all")
    scoped_volumes = None if scoped_volumes is None else scoped_volumes.reindex(market_returns.index)
    signal_returns, signal_volumes, signal_bars_per_year = prepare_strategy_signal_data(
        history_returns=market_returns,
        volume_history=scoped_volumes,
        strategy=strategy,
        selection_contexts=selection_contexts,
    )
    signal_returns = signal_returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if signal_volumes is not None:
        signal_volumes = signal_volumes.reindex(signal_returns.index)

    max_horizon = max(int(horizon) for horizon in horizons)
    observations_by_horizon: dict[int, list[dict[str, float]]] = {int(horizon): [] for horizon in horizons}
    yearly_observations_by_horizon: dict[int, dict[int, list[dict[str, float]]]] = {
        int(horizon): {} for horizon in horizons
    }
    asset_class_observations_by_horizon: dict[int, dict[str, list[dict[str, float]]]] = {
        int(horizon): {} for horizon in horizons
    }
    assets_by_class = group_assets_by_class(signal_returns.columns)
    asset_class_counts = {asset_class: len(assets) for asset_class, assets in assets_by_class.items()}
    if len(signal_returns) <= MIN_SIGNAL_DIAGNOSTIC_HISTORY_BARS + max_horizon:
        horizon_results = [
            summarize_signal_horizon_observations(
                observations_by_horizon[int(horizon)],
                horizon_bars=int(horizon),
                bucket_count=bucket_count,
            )
            for horizon in horizons
        ]
        return {
            "strategyKey": strategy_definition.key,
            "strategyLabel": strategy_definition.label,
            "observationSchedule": observation_schedule,
            "resolvedObservationSchedule": resolved_observation_schedule,
            "observationCountSemantics": observation_count_semantics,
            "horizonResults": horizon_results,
            "yearlyResults": [],
            "assetClassResults": [],
            "diagnosis": build_signal_diagnostics_diagnosis(
                horizon_results=horizon_results,
                yearly_results=[],
                asset_class_results=[],
            ),
        }

    for index in range(MIN_SIGNAL_DIAGNOSTIC_HISTORY_BARS, len(signal_returns) - max_horizon):
        if not should_include_signal_observation(
            index=index,
            index_values=signal_returns.index,
            resolved_observation_schedule=resolved_observation_schedule,
        ):
            continue
        history_returns = signal_returns.iloc[: index + 1]
        history_volumes = None if signal_volumes is None else signal_volumes.iloc[: index + 1]
        current_date = str(signal_returns.index[index])
        current_year = int(pd.Timestamp(signal_returns.index[index]).year)
        try:
            scores = compute_strategy_score_series(
                history_returns,
                history_volumes,
                strategy,
                bars_per_year=signal_bars_per_year,
                current_date=current_date,
                predictor_panel=None,
                selection_contexts=selection_contexts,
                predictor_context=predictor_context,
            )
        except ValueError:
            continue
        if scores is None:
            continue
        scores = scores.reindex(signal_returns.columns)
        for horizon in horizons:
            horizon = int(horizon)
            forward_window = signal_returns.iloc[index + 1 : index + horizon + 1]
            if len(forward_window) < horizon:
                continue
            forward_returns = (1.0 + forward_window).prod(axis=0) - 1.0
            observation = build_signal_horizon_observation(
                scores,
                forward_returns.reindex(signal_returns.columns),
                bucket_count=bucket_count,
            )
            if observation is None:
                continue
            observations_by_horizon[horizon].append(observation)
            yearly_observations_by_horizon[horizon].setdefault(current_year, []).append(observation)
            for asset_class, class_assets in assets_by_class.items():
                if len(class_assets) < MIN_SIGNAL_DIAGNOSTIC_ASSET_COUNT:
                    continue
                class_observation = build_signal_horizon_observation(
                    scores.reindex(class_assets),
                    forward_returns.reindex(class_assets),
                    bucket_count=bucket_count,
                )
                if class_observation is not None:
                    asset_class_observations_by_horizon[horizon].setdefault(
                        asset_class,
                        [],
                    ).append(class_observation)

    horizon_results = [
        summarize_signal_horizon_observations(
            observations_by_horizon[int(horizon)],
            horizon_bars=int(horizon),
            bucket_count=bucket_count,
        )
        for horizon in horizons
    ]
    yearly_results = [
        result
        for horizon in horizons
        for result in summarize_yearly_signal_observations(
            yearly_observations_by_horizon[int(horizon)],
            horizon_bars=int(horizon),
            bucket_count=bucket_count,
        )
    ]
    asset_class_results = [
        result
        for horizon in horizons
        for result in summarize_asset_class_signal_observations(
            asset_class_observations_by_horizon[int(horizon)],
            horizon_bars=int(horizon),
            bucket_count=bucket_count,
            asset_class_counts=asset_class_counts,
        )
    ]
    return {
        "strategyKey": strategy_definition.key,
        "strategyLabel": strategy_definition.label,
        "observationSchedule": observation_schedule,
        "resolvedObservationSchedule": resolved_observation_schedule,
        "observationCountSemantics": observation_count_semantics,
        "horizonResults": horizon_results,
        "yearlyResults": yearly_results,
        "assetClassResults": asset_class_results,
        "diagnosis": build_signal_diagnostics_diagnosis(
            horizon_results=horizon_results,
            yearly_results=yearly_results,
            asset_class_results=asset_class_results,
        ),
    }


def build_signal_diagnostics_payload(
    comparison: ComparisonSpec,
    *,
    fetch_market_universe_bundle: Callable,
    period: str | None = None,
    universe: str = "crypto_included",
    horizons: Sequence[int] = DEFAULT_SIGNAL_DIAGNOSTIC_HORIZONS,
    bucket_count: int = DEFAULT_SIGNAL_DIAGNOSTIC_BUCKET_COUNT,
    observation_schedule: str = "strategy",
) -> dict[str, object]:
    market_period = period or comparison.run_spec.market_slice.period
    strategy_definitions = comparison.candidate_strategies + comparison.reference_strategies
    comparison_tickers = collect_comparison_tickers(
        comparison,
        strategy_definitions=strategy_definitions,
    )
    market_timeframes = {}
    for strategy_definition in strategy_definitions:
        timeframe = resolve_strategy_market_data_timeframe(strategy_definition)
        market_timeframes[timeframe.key] = timeframe
    start_date = comparison.run_spec.market_slice.start_date if market_period == comparison.run_spec.market_slice.period else None
    end_date = comparison.run_spec.market_slice.end_date if market_period == comparison.run_spec.market_slice.period else None
    market_bundles_by_timeframe = {}
    metadata_by_timeframe = {}
    for timeframe_key, timeframe in market_timeframes.items():
        market_bundle, metadata = fetch_market_universe_bundle(
            tickers=comparison_tickers,
            period=market_period,
            timeframe=timeframe.yfinance_interval,
            start_date=start_date,
            end_date=end_date,
        )
        market_bundles_by_timeframe[timeframe_key] = market_bundle
        metadata_by_timeframe[timeframe_key] = metadata

    strategy_results = []
    for strategy_definition in strategy_definitions:
        timeframe = resolve_strategy_market_data_timeframe(strategy_definition)
        market_bundle = market_bundles_by_timeframe[timeframe.key]
        strategy_results.append(
            build_strategy_signal_diagnostics(
                strategy_definition=strategy_definition,
                closes=market_bundle["closes"],
                volumes=market_bundle.get("volumes"),
                horizons=horizons,
                bucket_count=bucket_count,
                observation_schedule=observation_schedule,
            )
        )

    return {
        "kind": "signal_diagnostics",
        "schemaVersion": "v1",
        "period": market_period,
        "universe": universe,
        "horizons": [format_signal_horizon_label(int(horizon)) for horizon in horizons],
        "bucketCount": int(bucket_count),
        "observationSchedule": normalize_signal_diagnostic_observation_schedule(observation_schedule),
        "strategyCount": len(strategy_results),
        "marketData": metadata_by_timeframe,
        "strategyResults": strategy_results,
    }
