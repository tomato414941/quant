from __future__ import annotations

import numpy as np
import pandas as pd

from app.portfolio_allocation import (
    PortfolioAllocationInput,
    build_equal_weight_fallback,
    compute_portfolio_weights,
    expand_weights,
)
from app.portfolio_domain import *
from app.portfolio_tilt import apply_weight_tilt
from app.portfolio_market_data import (
    prepare_signal_component_data,
    prepare_strategy_market_data,
    resample_market_frame_to_timeframe,
    resample_returns_frame_to_timeframe,
    resolve_timeframe_bars_per_year,
)
from app.portfolio_metrics import (
    compute_split_index,
    empty_number_distribution,
    serialize_weights,
    should_rebalance,
    summarize_availability_series,
    summarize_metrics_from_bar_returns,
    summarize_optional_number_distribution,
    summarize_portfolio_decision_events,
    summarize_portfolio_metrics,
    summarize_segment_from_returns,
)
from app.portfolio_availability import (
    build_default_availability_policy,
    compute_dynamic_portfolio_allocation as _compute_dynamic_portfolio_allocation,
    prepare_history_returns_for_assets,
    prepare_volume_history_for_assets,
    resolve_available_assets,
    resolve_effective_min_history_bars,
    resolve_eligible_assets,
    zero_weights_outside_assets,
)
from app.portfolio_execution import (
    COST_AWARE_NO_TRADE_DECISION_POLICY,
    DECISION_POLICY_EXTENSION_KEY,
    DEFAULT_CONFIDENCE_FLOOR,
    DEFAULT_NO_TRADE_BAND,
    DIRECT_SCORE_DECISION_POLICY,
    SUPPORTED_DECISION_POLICIES,
    PortfolioDecision,
    build_decision_forecast_snapshot,
    build_portfolio_decision,
    compute_edge_hit,
    compute_forecast_edge,
    compute_realized_decision_edge_pct,
    compute_weighted_forecast_confidence,
    resolve_decision_policy_kind,
    resolve_selected_assets_from_weights,
    serialize_execution_trace_event,
    serialize_portfolio_decision_event,
)
from app.portfolio_forecast import (
    SIGNAL_RETURN_PROXY_EDGE_SOURCE,
    ForecastSnapshot,
    build_asset_ranking_specs,
    build_asset_ranking_specs_from_strategy_definitions,
    build_predictor_specs,
    build_strategy_forecast_snapshot,
    compute_expected_return_proxy,
    compute_forecast_confidence_series,
    compute_prediction_feature_frame,
    compute_prediction_supplemented_score_series,
    compute_predictor_panel,
    compute_signal_return_proxy_series,
    compute_strategy_score_series,
    deserialize_predictor_panel,
    evaluate_asset_ranking_spec,
    evaluate_predictor_spec,
    extract_ranking_score_parameters,
    freeze_parameter_value,
    serialize_predictor_panel,
    serialize_predictor_spec,
    summarize_prediction_observations,
    summarize_ranking_observations,
)
from app.portfolio_selection import (
    build_default_strategy_predictor_context,
    build_default_strategy_selection_context,
    build_runtime_signal_execution_contexts,
    build_selection_spec_from_signal_payload,
    compute_macro_proxy_rank,
    compute_strategy_score_series_base,
    compute_strategy_selection_score_series,
    compute_strategy_selection_trailing_total_returns,
    compute_trailing_total_returns,
    compute_volume_strength,
    compute_window_total_returns,
    convert_horizon_spec_to_bars,
    convert_window_spec_to_bars,
    extract_predictor_signal_payload,
    get_ranking_window_bars,
    get_score_parameter_window_bars,
    get_strategy_definition_signal_execution_contexts,
    get_strategy_selection_components,
    get_strategy_signal_execution_contexts,
    parse_score_parameter_strings,
    prepare_strategy_predictor_panel,
    prepare_strategy_signal_data,
    resolve_decision_schedule,
    resolve_predictor_use_spec,
    resolve_predictor_snapshot,
    resolve_primary_selection_spec,
    resolve_runtime_signal_execution_contexts,
    resolve_strategy_market_data_timeframe_key,
    select_assets,
    select_positive_trend_short_reversal_assets,
    select_risk_regime_positive_momentum_assets,
    standardize_prediction_series,
)
from app.portfolio_costs import (
    build_flat_cost_model,
    compute_trade_cost,
    cost_rate_from_parameters,
    impact_rate_from_parameters,
    resolve_cost_model_inputs,
)
from app.portfolio_positioning import (
    apply_strategy_weight_tilt,
    compute_portfolio_allocation,
    filter_positive_variance_assets,
)
from app.portfolio_runs import (
    compare_portfolio_models as _compare_portfolio_models,
    compare_portfolio_runs as _compare_portfolio_runs,
    evaluate_strategy_definition_run as _evaluate_strategy_definition_run,
    evaluate_strategy_run as _evaluate_strategy_run,
    run_portfolio_backtest as _run_portfolio_backtest,
)
from app.portfolio_state import (
    build_portfolio_state,
    resolve_initial_weights,
    serialize_portfolio_state,
)


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
) -> tuple[list[str], np.ndarray]:
    return _compute_dynamic_portfolio_allocation(
        history_returns=history_returns,
        volume_history=volume_history,
        strategy=strategy,
        portfolio_model=portfolio_model,
        bars_per_year=bars_per_year,
        universe_columns=universe_columns,
        max_investment_ratio=max_investment_ratio,
        max_weight=max_weight,
        previous_weights=previous_weights,
        transaction_cost=transaction_cost,
        current_date=current_date,
        predictor_panel=predictor_panel,
        selection_contexts=selection_contexts,
        predictor_context=predictor_context,
        availability_policy=availability_policy,
        allocation_fn=compute_portfolio_allocation,
    )


def compare_portfolio_runs(
    closes: pd.DataFrame,
    volumes: pd.DataFrame | None,
    strategies: list[EvaluatorStrategySpec],
    initial_capital: float,
    split_ratio: float,
    bars_per_year: float = 252.0,
    execution_assumptions: dict | None = None,
    cost_model: dict | None = None,
    transaction_cost: float | None = None,
    portfolio_state: PortfolioState | None = None,
    predictor_panels_by_strategy: dict[str, pd.DataFrame] | None = None,
    strategy_signal_execution_contexts_by_key: dict[str, tuple[list[dict[str, object]], dict[str, object] | None]] | None = None,
    availability_policy: dict[str, object] | None = None,
) -> list[dict]:
    return _compare_portfolio_runs(
        closes=closes,
        volumes=volumes,
        strategies=strategies,
        initial_capital=initial_capital,
        split_ratio=split_ratio,
        bars_per_year=bars_per_year,
        execution_assumptions=execution_assumptions,
        cost_model=cost_model,
        transaction_cost=transaction_cost,
        portfolio_state=portfolio_state,
        predictor_panels_by_strategy=predictor_panels_by_strategy,
        strategy_signal_execution_contexts_by_key=strategy_signal_execution_contexts_by_key,
        availability_policy=availability_policy,
        dynamic_allocation_fn=compute_dynamic_portfolio_allocation,
    )


def evaluate_strategy_run(
    closes: pd.DataFrame,
    volumes: pd.DataFrame | None,
    strategy: EvaluatorStrategySpec,
    initial_capital: float,
    split_ratio: float,
    bars_per_year: float = 252.0,
    execution_assumptions: dict | None = None,
    cost_model: dict | None = None,
    transaction_cost: float | None = None,
    portfolio_state: PortfolioState | None = None,
    predictor_panel: pd.DataFrame | None = None,
    signal_execution_contexts: tuple[list[dict[str, object]], dict[str, object] | None] | None = None,
    availability_policy: dict[str, object] | None = None,
) -> dict:
    return _evaluate_strategy_run(
        closes=closes,
        volumes=volumes,
        strategy=strategy,
        initial_capital=initial_capital,
        split_ratio=split_ratio,
        bars_per_year=bars_per_year,
        execution_assumptions=execution_assumptions,
        cost_model=cost_model,
        transaction_cost=transaction_cost,
        portfolio_state=portfolio_state,
        predictor_panel=predictor_panel,
        signal_execution_contexts=signal_execution_contexts,
        availability_policy=availability_policy,
        dynamic_allocation_fn=compute_dynamic_portfolio_allocation,
    )


def evaluate_strategy_definition_run(
    closes: pd.DataFrame,
    volumes: pd.DataFrame | None,
    strategy_definition: StrategyDefinition,
    initial_capital: float,
    split_ratio: float,
    bars_per_year: float = 252.0,
    execution_assumptions: dict | None = None,
    cost_model: dict | None = None,
    transaction_cost: float | None = None,
    portfolio_state: PortfolioState | None = None,
    predictor_panel: pd.DataFrame | None = None,
    availability_policy: dict[str, object] | None = None,
) -> dict:
    return _evaluate_strategy_definition_run(
        closes=closes,
        volumes=volumes,
        strategy_definition=strategy_definition,
        initial_capital=initial_capital,
        split_ratio=split_ratio,
        bars_per_year=bars_per_year,
        execution_assumptions=execution_assumptions,
        cost_model=cost_model,
        transaction_cost=transaction_cost,
        portfolio_state=portfolio_state,
        predictor_panel=predictor_panel,
        availability_policy=availability_policy,
        dynamic_allocation_fn=compute_dynamic_portfolio_allocation,
    )


def compare_portfolio_models(
    closes: pd.DataFrame,
    volumes: pd.DataFrame | None,
    portfolio_models: list[PortfolioModelSpec],
    initial_capital: float,
    split_ratio: float,
    transaction_cost: float,
    bars_per_year: float = 252.0,
    portfolio_state: PortfolioState | None = None,
) -> list[dict]:
    return _compare_portfolio_models(
        closes=closes,
        volumes=volumes,
        portfolio_models=portfolio_models,
        initial_capital=initial_capital,
        split_ratio=split_ratio,
        transaction_cost=transaction_cost,
        bars_per_year=bars_per_year,
        portfolio_state=portfolio_state,
        dynamic_allocation_fn=compute_dynamic_portfolio_allocation,
    )


def run_portfolio_backtest(*args, **kwargs) -> dict:
    kwargs.setdefault("dynamic_allocation_fn", compute_dynamic_portfolio_allocation)
    return _run_portfolio_backtest(*args, **kwargs)
