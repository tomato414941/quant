from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from app.portfolio_domain import *


def build_default_availability_policy() -> dict[str, object]:
    return {
        "kind": "asset_availability_policy",
        "minHistoryBars": 252,
        "maxStaleBars": 5,
        "delistedAssetPolicy": "liquidate_to_cash",
        "missingReturnPolicy": "reject_if_held",
    }


def resolve_effective_min_history_bars(
    history_returns: pd.DataFrame,
    availability_policy: dict[str, object],
) -> int:
    configured_min = int(availability_policy.get("minHistoryBars", 252))
    return max(1, min(configured_min, len(history_returns)))


def resolve_available_assets(history_returns: pd.DataFrame) -> list[str]:
    if history_returns.empty:
        return []
    latest_row = history_returns.iloc[-1]
    return [str(asset) for asset, value in latest_row.items() if pd.notna(value)]


def resolve_eligible_assets(
    history_returns: pd.DataFrame,
    availability_policy: dict[str, object],
) -> list[str]:
    if history_returns.empty:
        return []
    min_history_bars = resolve_effective_min_history_bars(history_returns, availability_policy)
    latest_row = history_returns.iloc[-1]
    valid_counts = history_returns.notna().sum(axis=0)
    return [
        str(asset)
        for asset in history_returns.columns
        if pd.notna(latest_row[asset]) and int(valid_counts[asset]) >= min_history_bars
    ]


def prepare_history_returns_for_assets(
    history_returns: pd.DataFrame,
    eligible_assets: list[str],
) -> pd.DataFrame:
    if not eligible_assets:
        return history_returns.iloc[0:0, 0:0].copy()
    return history_returns.loc[:, eligible_assets].dropna(how="any")


def prepare_volume_history_for_assets(
    volume_history: pd.DataFrame | None,
    clean_history_returns: pd.DataFrame,
) -> pd.DataFrame | None:
    if volume_history is None:
        return None
    return volume_history.loc[clean_history_returns.index, clean_history_returns.columns]


def zero_weights_outside_assets(
    weights: np.ndarray,
    universe_columns: pd.Index,
    allowed_assets: list[str],
) -> np.ndarray:
    allowed_asset_set = set(allowed_assets)
    scoped_weights = [
        float(weight) if str(asset) in allowed_asset_set else 0.0
        for asset, weight in zip(universe_columns, weights, strict=True)
    ]
    return np.asarray(scoped_weights, dtype="float64")


def compute_dynamic_portfolio_allocation(
    *,
    history_returns: pd.DataFrame,
    volume_history: pd.DataFrame | None,
    strategy: EvaluatorStrategySpec,
    portfolio_model: PortfolioModelSpec,
    bars_per_year: float,
    universe_columns: pd.Index,
    max_investment_ratio: float,
    max_weight: float | None,
    previous_weights: np.ndarray | None,
    transaction_cost: float,
    current_date: str | None,
    predictor_panel: pd.DataFrame | None,
    selection_contexts: list[dict[str, object]] | None,
    predictor_context: dict[str, object] | None,
    availability_policy: dict[str, object],
    allocation_fn: Callable[..., tuple[list[str], np.ndarray]],
) -> tuple[list[str], np.ndarray]:
    eligible_assets = resolve_eligible_assets(history_returns, availability_policy)
    if len(eligible_assets) == 1:
        weights = np.zeros(len(universe_columns), dtype="float64")
        asset = eligible_assets[0]
        asset_index = list(universe_columns).index(asset)
        asset_weight = max_investment_ratio
        if max_weight is not None:
            asset_weight = min(asset_weight, max_weight)
        weights[asset_index] = asset_weight
        return [asset], weights
    if len(eligible_assets) < 2:
        return [], np.zeros(len(universe_columns), dtype="float64")
    clean_history_returns = prepare_history_returns_for_assets(history_returns, eligible_assets)
    if len(clean_history_returns) < 3 or len(clean_history_returns.columns) < 2:
        return [], np.zeros(len(universe_columns), dtype="float64")
    clean_volume_history = prepare_volume_history_for_assets(volume_history, clean_history_returns)
    scoped_previous_weights = None
    if previous_weights is not None:
        scoped_previous_weights = zero_weights_outside_assets(previous_weights, universe_columns, eligible_assets)
    try:
        return allocation_fn(
            history_returns=clean_history_returns,
            volume_history=clean_volume_history,
            strategy=strategy,
            portfolio_model=portfolio_model,
            bars_per_year=bars_per_year,
            universe_columns=universe_columns,
            max_investment_ratio=max_investment_ratio,
            max_weight=max_weight,
            previous_weights=scoped_previous_weights,
            transaction_cost=transaction_cost,
            current_date=current_date,
            predictor_panel=predictor_panel,
            selection_contexts=selection_contexts,
            predictor_context=predictor_context,
        )
    except ValueError:
        return [], np.zeros(len(universe_columns), dtype="float64")
