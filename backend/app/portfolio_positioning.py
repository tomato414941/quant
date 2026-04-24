from __future__ import annotations

import numpy as np
import pandas as pd

from app.portfolio_allocation import PortfolioAllocationInput, expand_weights, fit_portfolio_model
from app.portfolio_domain import *
from app.portfolio_forecast import build_strategy_forecast_snapshot
from app.portfolio_selection import (
    prepare_strategy_predictor_panel,
    prepare_strategy_signal_data,
    resolve_primary_selection_spec,
    select_assets,
)
from app.portfolio_tilt import apply_weight_tilt


def compute_portfolio_allocation(
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
    selection_contexts: list[dict[str, object]] | None = None,
    predictor_context: dict[str, object] | None = None,
) -> tuple[list[str], np.ndarray]:
    signal_returns, signal_volumes, signal_bars_per_year = prepare_strategy_signal_data(
        history_returns=history_returns,
        volume_history=volume_history,
        strategy=strategy,
        selection_contexts=selection_contexts,
    )
    selected_assets = select_assets(
        signal_returns,
        signal_volumes,
        strategy,
        bars_per_year=signal_bars_per_year,
        selection_contexts=selection_contexts,
    )
    if not selected_assets:
        return [], np.zeros(len(universe_columns), dtype="float64")
    strategy_returns = filter_positive_variance_assets(signal_returns[selected_assets])
    selected_assets = list(strategy_returns.columns)
    if len(selected_assets) < 2:
        return [], np.zeros(len(universe_columns), dtype="float64")
    selected_previous_weights = None
    if previous_weights is not None:
        previous_weight_map = {
            str(asset): float(weight)
            for asset, weight in zip(universe_columns, previous_weights, strict=True)
        }
        selected_previous_weights = np.asarray(
            [previous_weight_map.get(asset, 0.0) for asset in selected_assets],
            dtype="float64",
        )
    prepared_predictor_panel = prepare_strategy_predictor_panel(
        predictor_panel=predictor_panel,
        strategy=strategy,
        predictor_context=predictor_context,
    )
    weights = fit_portfolio_model(
        PortfolioAllocationInput(
            returns=strategy_returns,
            portfolio_model=portfolio_model,
            max_investment_ratio=max_investment_ratio,
            max_weight=max_weight,
            previous_weights=selected_previous_weights,
            transaction_cost=transaction_cost,
        )
    ) * max_investment_ratio
    if portfolio_model.model_type != "mean_risk_utility":
        weights = apply_strategy_weight_tilt(
            weights=weights,
            history_returns=strategy_returns,
            strategy=strategy,
            bars_per_year=signal_bars_per_year,
            max_investment_ratio=max_investment_ratio,
            max_weight=max_weight,
            current_date=current_date,
            predictor_panel=prepared_predictor_panel,
            selection_contexts=selection_contexts,
            predictor_context=predictor_context,
        )
    expanded_weights = expand_weights(
        universe_columns=universe_columns,
        selected_columns=strategy_returns.columns,
        selected_weights=weights,
    )
    return selected_assets, expanded_weights


def apply_strategy_weight_tilt(
    *,
    weights: np.ndarray,
    history_returns: pd.DataFrame,
    strategy: EvaluatorStrategySpec,
    bars_per_year: float,
    max_investment_ratio: float,
    max_weight: float | None,
    current_date: str | None,
    predictor_panel: pd.DataFrame | None,
    selection_contexts: list[dict[str, object]] | None = None,
    predictor_context: dict[str, object] | None = None,
) -> np.ndarray:
    primary_selection = resolve_primary_selection_spec(
        strategy,
        selection_contexts=selection_contexts,
    )
    score_parameters = dict(primary_selection.ranking_signal.score_parameters)
    if "tilt_strength" not in score_parameters:
        return weights

    forecast = build_strategy_forecast_snapshot(
        history_returns,
        None,
        strategy,
        bars_per_year=bars_per_year,
        current_date=current_date,
        predictor_panel=predictor_panel,
        selection_contexts=selection_contexts,
        predictor_context=predictor_context,
    )
    if forecast is None:
        return weights
    percentile_ranks = forecast.percentile_rank.reindex(history_returns.columns)
    tilt_strength = float(score_parameters.get("tilt_strength", 0.5))
    tilt_shape = float(score_parameters.get("tilt_shape", 0.0))
    rank_values = percentile_ranks.to_numpy(dtype="float64")
    return apply_weight_tilt(
        weights=weights,
        rank_values=rank_values,
        tilt_strength=tilt_strength,
        tilt_shape=tilt_shape,
        max_weight=max_weight,
    )


def filter_positive_variance_assets(returns: pd.DataFrame) -> pd.DataFrame:
    positive_variance = returns.var(axis=0) > 1e-12
    filtered = returns.loc[:, positive_variance]
    if filtered.empty:
        raise ValueError("At least one asset with positive variance is required.")
    return filtered
