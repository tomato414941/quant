from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from app.comparison_models import ComparisonSpec
from app.comparison_service import collect_comparison_tickers, resolve_strategy_market_data_timeframe
from app.portfolio import (
    build_executable_evaluator_strategy_spec_from_definition,
    compute_strategy_score_series,
    get_strategy_signal_execution_contexts,
    prepare_strategy_signal_data,
)

DEFAULT_SIGNAL_DIAGNOSTIC_HORIZONS = (1, 5, 21)
DEFAULT_SIGNAL_DIAGNOSTIC_BUCKET_COUNT = 5
MIN_SIGNAL_DIAGNOSTIC_ASSET_COUNT = 2
MIN_SIGNAL_DIAGNOSTIC_HISTORY_BARS = 3


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


def build_strategy_signal_diagnostics(
    *,
    strategy_definition,
    closes: pd.DataFrame,
    volumes: pd.DataFrame | None,
    horizons: Sequence[int] = DEFAULT_SIGNAL_DIAGNOSTIC_HORIZONS,
    bucket_count: int = DEFAULT_SIGNAL_DIAGNOSTIC_BUCKET_COUNT,
) -> dict[str, object]:
    strategy = build_executable_evaluator_strategy_spec_from_definition(strategy_definition)
    selection_contexts, predictor_context = get_strategy_signal_execution_contexts(strategy)
    available_assets = [asset for asset in strategy.investment_universe.tickers if asset in closes.columns]
    if len(available_assets) < MIN_SIGNAL_DIAGNOSTIC_ASSET_COUNT:
        return {
            "strategyKey": strategy_definition.key,
            "strategyLabel": strategy_definition.label,
            "horizonResults": [
                summarize_signal_horizon_observations([], horizon_bars=horizon, bucket_count=bucket_count)
                for horizon in horizons
            ],
        }

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
    if len(signal_returns) <= MIN_SIGNAL_DIAGNOSTIC_HISTORY_BARS + max_horizon:
        return {
            "strategyKey": strategy_definition.key,
            "strategyLabel": strategy_definition.label,
            "horizonResults": [
                summarize_signal_horizon_observations(
                    observations_by_horizon[int(horizon)],
                    horizon_bars=int(horizon),
                    bucket_count=bucket_count,
                )
                for horizon in horizons
            ],
        }

    for index in range(MIN_SIGNAL_DIAGNOSTIC_HISTORY_BARS, len(signal_returns) - max_horizon):
        history_returns = signal_returns.iloc[: index + 1]
        history_volumes = None if signal_volumes is None else signal_volumes.iloc[: index + 1]
        current_date = str(signal_returns.index[index])
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
        for horizon in horizons:
            horizon = int(horizon)
            forward_window = signal_returns.iloc[index + 1 : index + horizon + 1]
            if len(forward_window) < horizon:
                continue
            forward_returns = (1.0 + forward_window).prod(axis=0) - 1.0
            observation = build_signal_horizon_observation(
                scores.reindex(signal_returns.columns),
                forward_returns.reindex(signal_returns.columns),
                bucket_count=bucket_count,
            )
            if observation is not None:
                observations_by_horizon[horizon].append(observation)

    return {
        "strategyKey": strategy_definition.key,
        "strategyLabel": strategy_definition.label,
        "horizonResults": [
            summarize_signal_horizon_observations(
                observations_by_horizon[int(horizon)],
                horizon_bars=int(horizon),
                bucket_count=bucket_count,
            )
            for horizon in horizons
        ],
    }


def build_signal_diagnostics_payload(
    comparison: ComparisonSpec,
    *,
    fetch_market_universe_bundle: Callable,
    period: str | None = None,
    universe: str = "crypto_included",
    horizons: Sequence[int] = DEFAULT_SIGNAL_DIAGNOSTIC_HORIZONS,
    bucket_count: int = DEFAULT_SIGNAL_DIAGNOSTIC_BUCKET_COUNT,
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
            )
        )

    return {
        "kind": "signal_diagnostics",
        "schemaVersion": "v1",
        "period": market_period,
        "universe": universe,
        "horizons": [format_signal_horizon_label(int(horizon)) for horizon in horizons],
        "bucketCount": int(bucket_count),
        "strategyCount": len(strategy_results),
        "marketData": metadata_by_timeframe,
        "strategyResults": strategy_results,
    }
