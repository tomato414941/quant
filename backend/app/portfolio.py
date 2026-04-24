from __future__ import annotations

import math
import numpy as np
import pandas as pd
from app.instrument_registry import get_instrument
from app.portfolio_allocation import (
    PortfolioAllocationInput,
    build_equal_weight_fallback,
    expand_weights,
    fit_portfolio_model,
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
    compute_expected_return_proxy_series,
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


def build_portfolio_state(
    *,
    current_weights: dict[str, float],
    cash_weight: float = 0.0,
) -> PortfolioState:
    normalized_weights = {
        asset.strip().upper(): float(weight)
        for asset, weight in current_weights.items()
        if asset.strip()
    }
    if cash_weight < 0 or cash_weight > 1:
        raise ValueError("Cash weight must be between 0 and 1.")
    if any(weight < 0 for weight in normalized_weights.values()):
        raise ValueError("Current weights must be non-negative.")
    total_weight = sum(normalized_weights.values()) + cash_weight
    if total_weight > 1.000001:
        raise ValueError("Portfolio state weights must sum to 1 or less.")
    return PortfolioState(current_weights=normalized_weights, cash_weight=float(cash_weight))


def serialize_portfolio_state(portfolio_state: PortfolioState) -> dict:
    rows = [
        {"asset": asset, "weightPct": round(weight * 100, 2)}
        for asset, weight in portfolio_state.current_weights.items()
        if weight > 0
    ]
    if portfolio_state.cash_weight > 0:
        rows.append({"asset": "CASH", "weightPct": round(portfolio_state.cash_weight * 100, 2)})
    rows.sort(key=lambda item: item["weightPct"], reverse=True)
    return {"weights": rows}


def build_flat_cost_model(*, commission_pct: float, slippage_pct: float = 0.0) -> dict:
    return {
        "kind": "flat_cost",
        "parameters": {
            "commissionPct": float(commission_pct),
            "slippagePct": float(slippage_pct),
        },
        "perAssetOverrides": {},
    }


def cost_rate_from_parameters(parameters: dict[str, float]) -> float:
    commission_pct = float(parameters.get("commissionPct", 0.0))
    slippage_pct = float(parameters.get("slippagePct", 0.0))
    return (commission_pct + slippage_pct) / 100


def impact_rate_from_parameters(parameters: dict[str, float]) -> float:
    return float(parameters.get("impactCoefficientPct", 0.0)) / 100


def resolve_cost_model_inputs(
    *,
    universe_columns: pd.Index,
    cost_model: dict,
) -> dict[str, np.ndarray | float | int]:
    kind = cost_model.get("kind", "flat_cost")
    parameters = cost_model.get("parameters", {})
    default_rate = cost_rate_from_parameters(parameters)
    default_impact_rate = impact_rate_from_parameters(parameters)
    linear_rates = np.repeat(default_rate, len(universe_columns)).astype("float64")
    impact_rates = np.repeat(default_impact_rate, len(universe_columns)).astype("float64")
    adv_window_bars = max(1, int(float(parameters.get("advWindowBars", 20))))
    min_adv_notional = float(parameters.get("minAdvNotional", 1_000_000.0))

    if kind == "flat_cost":
        return {
            "defaultLinearRate": default_rate,
            "linearRates": linear_rates,
            "impactRates": impact_rates,
            "advWindowBars": adv_window_bars,
            "minAdvNotional": min_adv_notional,
        }
    if kind not in {"asset_specific_linear_cost", "asset_specific_adv_cost"}:
        raise ValueError("Unsupported cost model.")

    overrides = cost_model.get("perAssetOverrides", {})
    for index, asset in enumerate(universe_columns):
        override = overrides.get(str(asset))
        if not override:
            continue
        linear_rates[index] = cost_rate_from_parameters(override)
        impact_rates[index] = impact_rate_from_parameters({**parameters, **override})

    return {
        "defaultLinearRate": default_rate,
        "linearRates": linear_rates,
        "impactRates": impact_rates,
        "advWindowBars": adv_window_bars,
        "minAdvNotional": min_adv_notional,
    }


def compute_trade_cost(
    *,
    weight_delta: np.ndarray,
    linear_cost_rates: np.ndarray,
    impact_cost_rates: np.ndarray,
    portfolio_equity: float,
    price_snapshot: pd.Series | None,
    volume_history: pd.DataFrame | None,
    adv_window_bars: int,
    min_adv_notional: float,
) -> float:
    linear_cost = float(np.dot(weight_delta, linear_cost_rates))
    if (
        portfolio_equity <= 0
        or price_snapshot is None
        or volume_history is None
        or len(volume_history) == 0
        or np.allclose(impact_cost_rates, 0.0)
    ):
        return linear_cost

    recent_volumes = volume_history.tail(adv_window_bars)
    if recent_volumes.empty:
        return linear_cost

    aligned_prices = price_snapshot.reindex(recent_volumes.columns).astype("float64").fillna(0.0)
    adv_shares = recent_volumes.mean(axis=0).astype("float64").fillna(0.0)
    adv_notional = np.maximum((adv_shares * aligned_prices).to_numpy(dtype="float64"), min_adv_notional)
    trade_notional = weight_delta * float(portfolio_equity)
    participation = np.clip(trade_notional / adv_notional, 0.0, None)
    impact_rates = impact_cost_rates * np.sqrt(participation)
    impact_cost = float(np.dot(weight_delta, impact_rates))
    return linear_cost + impact_cost


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
    if not strategies:
        raise ValueError("At least one strategy is required.")
    if initial_capital <= 0:
        raise ValueError("Initial capital must be positive.")
    if execution_assumptions is None:
        execution_assumptions = {
            "kind": "close_execution_assumptions",
            "label": "終値約定",
            "parameters": {
                "fillPrice": "close",
            },
            "costModel": None,
        }
    if cost_model is None:
        cost_model = execution_assumptions.get("costModel")
    if cost_model is None:
        if transaction_cost is None:
            raise ValueError("Either cost_model or transaction_cost must be provided.")
        if transaction_cost < 0 or transaction_cost >= 1:
            raise ValueError("Transaction cost must be between 0 and 1.")
        cost_model = build_flat_cost_model(commission_pct=transaction_cost * 100, slippage_pct=0.0)

    if strategy_signal_execution_contexts_by_key is None:
        strategy_signal_execution_contexts_by_key = {}
    if availability_policy is None:
        availability_policy = build_default_availability_policy()

    runs: list[dict] = []
    for strategy in strategies:
        signal_execution_contexts = resolve_runtime_signal_execution_contexts(
            strategy,
            strategy_signal_execution_contexts_by_key.get(strategy.key),
        )
        selection_contexts, predictor_context = signal_execution_contexts
        rebalance_schedule = strategy.execution_policy.rebalance_schedule
        decision_schedule = resolve_decision_schedule(strategy)
        strategy_universe = [
            asset
            for asset in strategy.investment_universe.tickers
            if asset in closes.columns
        ]
        if len(strategy_universe) < 1:
            raise ValueError("Strategy investment universe must contain at least one available asset.")

        strategy_closes = closes[strategy_universe]
        strategy_volumes = volumes[strategy_universe] if volumes is not None else None
        returns = strategy_closes.pct_change(fill_method=None).iloc[1:]
        returns = returns.dropna(how="all")
        if len(returns) < 6:
            raise ValueError("At least 6 return rows are required for portfolio comparison.")
        strategy_volumes = strategy_volumes.loc[returns.index] if strategy_volumes is not None else None
        cost_inputs = resolve_cost_model_inputs(
            universe_columns=returns.columns,
            cost_model=cost_model,
        )
        default_transaction_cost = float(cost_inputs["defaultLinearRate"])
        asset_transaction_costs = np.asarray(cost_inputs["linearRates"], dtype="float64")
        asset_impact_costs = np.asarray(cost_inputs["impactRates"], dtype="float64")
        adv_window_bars = int(cost_inputs["advWindowBars"])
        min_adv_notional = float(cost_inputs["minAdvNotional"])

        split_index = compute_split_index(len(returns), split_ratio)
        train_returns = returns.iloc[:split_index]
        initial_portfolio_weights = resolve_initial_weights(
            universe_columns=returns.columns,
            portfolio_state=portfolio_state,
        )

        portfolio_model = strategy.portfolio_model
        risk_controls = strategy.risk_controls
        initial_selected_assets, initial_weights = compute_dynamic_portfolio_allocation(
            history_returns=train_returns,
            volume_history=strategy_volumes.loc[train_returns.index] if strategy_volumes is not None else None,
            strategy=strategy,
            portfolio_model=portfolio_model,
            bars_per_year=bars_per_year,
            universe_columns=returns.columns,
            max_investment_ratio=risk_controls.max_investment_ratio,
            max_weight=risk_controls.max_weight,
            previous_weights=initial_portfolio_weights,
            transaction_cost=default_transaction_cost,
            current_date=str(returns.index[split_index]) if split_index < len(returns.index) else None,
            predictor_panel=None if predictor_panels_by_strategy is None else predictor_panels_by_strategy.get(strategy.key),
            selection_contexts=selection_contexts,
            predictor_context=predictor_context,
            availability_policy=availability_policy,
        )
        backtest = run_portfolio_backtest(
            closes=strategy_closes,
            returns=returns,
            volumes=strategy_volumes,
            split_index=split_index,
            split_ratio=split_ratio,
            strategy=strategy,
            portfolio_model=portfolio_model,
            bars_per_year=bars_per_year,
            warmup_weights=initial_portfolio_weights,
            initial_weights=initial_weights,
            initial_selected_assets=initial_selected_assets,
            max_investment_ratio=risk_controls.max_investment_ratio,
            initial_capital=initial_capital,
            transaction_cost=default_transaction_cost,
            asset_transaction_costs=asset_transaction_costs,
            asset_impact_costs=asset_impact_costs,
            adv_window_bars=adv_window_bars,
            min_adv_notional=min_adv_notional,
            max_weight=risk_controls.max_weight,
            decision_schedule=decision_schedule,
            rebalance_schedule=rebalance_schedule,
            predictor_panel=None if predictor_panels_by_strategy is None else predictor_panels_by_strategy.get(strategy.key),
            selection_contexts=selection_contexts,
            predictor_context=predictor_context,
            availability_policy=availability_policy,
        )

        runs.append(
            {
                "kind": "run_result",
                "schemaVersion": "v1",
                "key": strategy.key,
                "strategy": serialize_evaluator_strategy_spec(strategy),
                "weights": serialize_weights(
                    returns.columns,
                    backtest["latestWeights"],
                    risk_controls.max_investment_ratio,
                ),
                "selectedAssets": backtest["latestSelectedAssets"],
                "summary": backtest["summary"],
                "splitAnalysis": backtest["splitAnalysis"],
                "series": backtest["series"],
                "availabilitySummary": backtest["availabilitySummary"],
                "decisionSummary": backtest["decisionSummary"],
                "decisionEvents": backtest["decisionEvents"],
                "executionTrace": backtest["executionTrace"],
                "availabilityPolicy": availability_policy,
            }
        )

    return runs


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
    return compare_portfolio_runs(
        closes=closes,
        volumes=volumes,
        strategies=[strategy],
        bars_per_year=bars_per_year,
        initial_capital=initial_capital,
        split_ratio=split_ratio,
        execution_assumptions=execution_assumptions,
        cost_model=cost_model,
        transaction_cost=transaction_cost,
        portfolio_state=portfolio_state,
        predictor_panels_by_strategy=None if predictor_panel is None else {strategy.key: predictor_panel},
        strategy_signal_execution_contexts_by_key=(
            None
            if signal_execution_contexts is None
            else {strategy.key: signal_execution_contexts}
        ),
        availability_policy=availability_policy,
    )[0]


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
    try:
        strategy = build_executable_evaluator_strategy_spec_from_definition(strategy_definition)
    except ValueError as exc:
        raise ValueError(
            f"Strategy definition {strategy_definition.strategy_id} is not executable: {exc}"
        ) from exc
    signal_execution_contexts = get_strategy_definition_signal_execution_contexts(strategy_definition)
    run = evaluate_strategy_run(
        closes=closes,
        volumes=volumes,
        strategy=strategy,
        bars_per_year=bars_per_year,
        initial_capital=initial_capital,
        split_ratio=split_ratio,
        execution_assumptions=execution_assumptions,
        cost_model=cost_model,
        transaction_cost=transaction_cost,
        portfolio_state=portfolio_state,
        predictor_panel=predictor_panel,
        signal_execution_contexts=signal_execution_contexts,
        availability_policy=availability_policy,
    )
    run["key"] = strategy_definition.key
    run["strategy"] = serialize_strategy_definition(strategy_definition)
    return run


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
    return compare_portfolio_runs(
        closes=closes,
        volumes=volumes,
        strategies=[
            build_evaluator_strategy_spec(
                investment_universe=build_investment_universe_spec(
                    tickers=list(closes.columns),
                    key="ad_hoc_universe",
                    label="Ad hoc universe",
                ),
                selection=build_selection_spec("full_universe"),
                portfolio_model=portfolio_model,
                execution_policy=build_execution_policy_spec(
                    key="hold",
                    label="保有",
                    entry="hold",
                    rebalance_schedule="hold",
                ),
                risk_controls=build_risk_controls_spec(
                    max_investment_ratio=1.0,
                    max_weight=None,
                ),
            )
            for portfolio_model in portfolio_models
        ],
        bars_per_year=bars_per_year,
        initial_capital=initial_capital,
        split_ratio=split_ratio,
        execution_assumptions={
            "kind": "close_execution_assumptions",
            "label": "終値約定",
            "parameters": {
                "fillPrice": "close",
            },
            "costModel": build_flat_cost_model(
                commission_pct=transaction_cost * 100,
                slippage_pct=0.0,
            ),
        },
        transaction_cost=transaction_cost,
        portfolio_state=portfolio_state,
    )


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


def run_portfolio_backtest(
    closes: pd.DataFrame,
    returns: pd.DataFrame,
    volumes: pd.DataFrame | None,
    bars_per_year: float,
    split_index: int,
    split_ratio: float,
    strategy: EvaluatorStrategySpec,
    portfolio_model: PortfolioModelSpec,
    warmup_weights: np.ndarray,
    initial_weights: np.ndarray,
    initial_selected_assets: list[str],
    max_investment_ratio: float,
    initial_capital: float,
    transaction_cost: float,
    asset_transaction_costs: np.ndarray,
    asset_impact_costs: np.ndarray,
    adv_window_bars: int,
    min_adv_notional: float,
    max_weight: float | None,
    decision_schedule: str,
    rebalance_schedule: str,
    predictor_panel: pd.DataFrame | None,
    selection_contexts: list[dict[str, object]] | None = None,
    predictor_context: dict[str, object] | None = None,
    availability_policy: dict[str, object] | None = None,
) -> dict:
    if availability_policy is None:
        availability_policy = build_default_availability_policy()

    portfolio_equity = initial_capital
    portfolio_returns: list[float] = []
    series: list[dict] = []
    train_dates: list[str] = []
    test_dates: list[str] = []
    train_portfolio_returns: list[float] = []
    test_portfolio_returns: list[float] = []
    train_turnover = 0.0
    test_turnover = 0.0
    current_weights = warmup_weights.copy()
    current_selected_assets: list[str] = []
    latest_weights = current_weights.copy()
    latest_selected_assets = list(current_selected_assets)
    pending_decision_weights = initial_weights.copy()
    pending_decision_selected_assets = list(initial_selected_assets)
    next_rebalance_weights: np.ndarray | None = None
    next_rebalance_selected_assets: list[str] | None = None
    previous_eligible_assets: set[str] = set()
    decision_events: list[dict] = []
    execution_trace: list[dict] = []

    for index, (date, row) in enumerate(returns.iterrows()):
        trade_turnover = 0.0
        trade_cost = 0.0
        phase = "train" if index < split_index else "test"
        history_through_current = returns.iloc[: index + 1]
        available_assets = resolve_available_assets(history_through_current)
        eligible_assets = resolve_eligible_assets(history_through_current, availability_policy)
        eligible_asset_set = set(eligible_assets)
        newly_eligible_assets = sorted(eligible_asset_set - previous_eligible_assets)
        removed_assets = sorted(previous_eligible_assets - eligible_asset_set)

        tradable_weights = zero_weights_outside_assets(current_weights, returns.columns, eligible_assets)
        forced_weight_delta = np.abs(tradable_weights - current_weights)
        if forced_weight_delta.sum() > 0:
            forced_turnover = float(forced_weight_delta.sum())
            trade_turnover += forced_turnover
            forced_trade_cost = compute_trade_cost(
                weight_delta=forced_weight_delta,
                linear_cost_rates=asset_transaction_costs,
                impact_cost_rates=asset_impact_costs,
                portfolio_equity=portfolio_equity,
                price_snapshot=closes.iloc[max(index - 1, 0)],
                volume_history=volumes.iloc[:index] if volumes is not None else None,
                adv_window_bars=adv_window_bars,
                min_adv_notional=min_adv_notional,
            )
            trade_cost += forced_trade_cost
            current_weights = tradable_weights
            current_selected_assets = [asset for asset in current_selected_assets if asset in eligible_asset_set]
            execution_trace.append(
                serialize_execution_trace_event(
                    date=str(date),
                    event_type="forced_universe_change",
                    phase=phase,
                    universe_columns=returns.columns,
                    max_investment_ratio=max_investment_ratio,
                    available_asset_count=len(available_assets),
                    eligible_asset_count=len(eligible_assets),
                    selected_assets=current_selected_assets,
                    target_weights=tradable_weights,
                    executed_weights=current_weights,
                    decision_action="forced_rebalance",
                    decision_reason="asset_unavailable",
                    turnover_pct=forced_turnover * 100,
                    estimated_cost_pct=round(forced_trade_cost * 100, 4),
                    estimated_edge_pct=None,
                    edge_source=None,
                    average_confidence=None,
                )
            )

        if index == split_index:
            split_history_returns = returns.iloc[:index]
            split_history_volumes = volumes.iloc[:index] if volumes is not None else None
            split_decision = build_portfolio_decision(
                history_returns=split_history_returns,
                volume_history=split_history_volumes,
                strategy=strategy,
                bars_per_year=bars_per_year,
                universe_columns=returns.columns,
                current_weights=current_weights,
                current_selected_assets=current_selected_assets,
                target_weights=initial_weights,
                target_selected_assets=initial_selected_assets,
                transaction_cost=transaction_cost,
                current_date=str(date),
                predictor_panel=predictor_panel,
                selection_contexts=selection_contexts,
                predictor_context=predictor_context,
                availability_policy=availability_policy,
            )
            decision_events.append(
                serialize_portfolio_decision_event(
                    str(date),
                    split_decision,
                    universe_columns=returns.columns,
                    current_weights=current_weights,
                    target_weights=initial_weights,
                    realized_returns=row,
                )
            )
            execution_trace.append(
                serialize_execution_trace_event(
                    date=str(date),
                    event_type="decision",
                    phase=phase,
                    universe_columns=returns.columns,
                    max_investment_ratio=max_investment_ratio,
                    available_asset_count=len(available_assets),
                    eligible_asset_count=len(eligible_assets),
                    selected_assets=split_decision.selected_assets,
                    target_weights=initial_weights,
                    executed_weights=current_weights,
                    decision_action=split_decision.action,
                    decision_reason=split_decision.reason,
                    turnover_pct=split_decision.turnover * 100,
                    estimated_cost_pct=split_decision.estimated_cost_pct,
                    estimated_edge_pct=split_decision.estimated_edge_pct,
                    edge_source=split_decision.edge_source,
                    average_confidence=split_decision.average_confidence,
                )
            )
            next_rebalance_weights = split_decision.weights.copy()
            next_rebalance_selected_assets = list(split_decision.selected_assets)

        if index >= split_index and next_rebalance_weights is not None:
            rebalanced_weights = zero_weights_outside_assets(
                next_rebalance_weights,
                returns.columns,
                eligible_assets,
            )
            selected_assets = [asset for asset in (next_rebalance_selected_assets or []) if asset in eligible_asset_set]
            weight_delta = np.abs(rebalanced_weights - current_weights)
            rebalance_turnover = float(weight_delta.sum())
            trade_turnover += rebalance_turnover
            rebalance_trade_cost = compute_trade_cost(
                weight_delta=weight_delta,
                linear_cost_rates=asset_transaction_costs,
                impact_cost_rates=asset_impact_costs,
                portfolio_equity=portfolio_equity,
                price_snapshot=closes.iloc[max(index - 1, 0)],
                volume_history=volumes.iloc[:index] if volumes is not None else None,
                adv_window_bars=adv_window_bars,
                min_adv_notional=min_adv_notional,
            )
            trade_cost += rebalance_trade_cost
            current_weights = rebalanced_weights
            current_selected_assets = list(selected_assets)
            latest_weights = rebalanced_weights.copy()
            latest_selected_assets = list(current_selected_assets)
            execution_trace.append(
                serialize_execution_trace_event(
                    date=str(date),
                    event_type="rebalance",
                    phase=phase,
                    universe_columns=returns.columns,
                    max_investment_ratio=max_investment_ratio,
                    available_asset_count=len(available_assets),
                    eligible_asset_count=len(eligible_assets),
                    selected_assets=current_selected_assets,
                    target_weights=next_rebalance_weights,
                    executed_weights=current_weights,
                    decision_action="rebalance",
                    decision_reason="scheduled_rebalance",
                    turnover_pct=rebalance_turnover * 100,
                    estimated_cost_pct=round(rebalance_trade_cost * 100, 4),
                    estimated_edge_pct=None,
                    edge_source=None,
                    average_confidence=None,
                )
            )
            next_rebalance_weights = None
            next_rebalance_selected_assets = None

        row_returns = row.reindex(returns.columns).fillna(0.0).to_numpy(dtype="float64")
        portfolio_return = float(np.dot(row_returns, current_weights))
        portfolio_return -= trade_cost

        portfolio_equity *= 1 + portfolio_return
        portfolio_returns.append(portfolio_return)
        segment_dates = train_dates if index < split_index else test_dates
        segment_portfolio_returns = train_portfolio_returns if index < split_index else test_portfolio_returns
        segment_dates.append(str(date))
        segment_portfolio_returns.append(portfolio_return)
        if index < split_index:
            train_turnover += trade_turnover
        else:
            test_turnover += trade_turnover
        series.append(
            {
                "date": str(date),
                "portfolioEquity": round(portfolio_equity, 2),
                "portfolioReturnPct": round(portfolio_return * 100, 2),
                "availableAssetCount": len(available_assets),
                "eligibleAssetCount": len(eligible_assets),
                "newlyEligibleAssets": newly_eligible_assets,
                "removedAssets": removed_assets,
            }
        )
        previous_eligible_assets = eligible_asset_set

        if index <= split_index or index >= len(returns) - 1:
            continue

        previous_date = returns.index[index - 1]
        if should_rebalance(
            previous_date=previous_date,
            current_date=date,
            rebalance_schedule=decision_schedule,
        ):
            target_selected_assets, target_weights = compute_dynamic_portfolio_allocation(
                history_returns=returns.iloc[: index + 1],
                volume_history=volumes.iloc[: index + 1] if volumes is not None else None,
                strategy=strategy,
                portfolio_model=portfolio_model,
                bars_per_year=bars_per_year,
                universe_columns=returns.columns,
                max_investment_ratio=max_investment_ratio,
                max_weight=max_weight,
                previous_weights=current_weights,
                transaction_cost=transaction_cost,
                current_date=str(date),
                predictor_panel=predictor_panel,
                selection_contexts=selection_contexts,
                predictor_context=predictor_context,
                availability_policy=availability_policy,
            )
            portfolio_decision = build_portfolio_decision(
                history_returns=returns.iloc[: index + 1],
                volume_history=volumes.iloc[: index + 1] if volumes is not None else None,
                strategy=strategy,
                bars_per_year=bars_per_year,
                universe_columns=returns.columns,
                current_weights=current_weights,
                current_selected_assets=current_selected_assets,
                target_weights=target_weights,
                target_selected_assets=target_selected_assets,
                transaction_cost=transaction_cost,
                current_date=str(date),
                predictor_panel=predictor_panel,
                selection_contexts=selection_contexts,
                predictor_context=predictor_context,
                availability_policy=availability_policy,
            )
            decision_events.append(
                serialize_portfolio_decision_event(
                    str(date),
                    portfolio_decision,
                    universe_columns=returns.columns,
                    current_weights=current_weights,
                    target_weights=target_weights,
                    realized_returns=returns.iloc[index + 1] if index + 1 < len(returns) else None,
                )
            )
            execution_trace.append(
                serialize_execution_trace_event(
                    date=str(date),
                    event_type="decision",
                    phase=phase,
                    universe_columns=returns.columns,
                    max_investment_ratio=max_investment_ratio,
                    available_asset_count=len(available_assets),
                    eligible_asset_count=len(eligible_assets),
                    selected_assets=portfolio_decision.selected_assets,
                    target_weights=target_weights,
                    executed_weights=current_weights,
                    decision_action=portfolio_decision.action,
                    decision_reason=portfolio_decision.reason,
                    turnover_pct=portfolio_decision.turnover * 100,
                    estimated_cost_pct=portfolio_decision.estimated_cost_pct,
                    estimated_edge_pct=portfolio_decision.estimated_edge_pct,
                    edge_source=portfolio_decision.edge_source,
                    average_confidence=portfolio_decision.average_confidence,
                )
            )
            pending_decision_selected_assets = list(portfolio_decision.selected_assets)
            pending_decision_weights = portfolio_decision.weights.copy()
        if should_rebalance(
            previous_date=previous_date,
            current_date=date,
            rebalance_schedule=rebalance_schedule,
        ):
            next_rebalance_weights = pending_decision_weights.copy()
            next_rebalance_selected_assets = list(pending_decision_selected_assets)

    train_summary = summarize_segment_from_returns(
        dates=train_dates,
        portfolio_returns=train_portfolio_returns,
        bars_per_year=bars_per_year,
        turnover=train_turnover,
    )
    test_summary = summarize_segment_from_returns(
        dates=test_dates,
        portfolio_returns=test_portfolio_returns,
        bars_per_year=bars_per_year,
        turnover=test_turnover,
    )
    return {
        "summary": test_summary["portfolio"],
        "series": series,
        "latestWeights": latest_weights,
        "latestSelectedAssets": latest_selected_assets,
        "availabilitySummary": summarize_availability_series(series),
        "decisionSummary": summarize_portfolio_decision_events(decision_events),
        "decisionEvents": decision_events,
        "executionTrace": execution_trace,
        "splitAnalysis": {
            "config": {"splitRatioPct": round(split_ratio * 100, 1)},
            "train": train_summary,
            "test": test_summary,
        },
    }


def resolve_initial_weights(
    *,
    universe_columns: pd.Index,
    portfolio_state: PortfolioState | None,
) -> np.ndarray:
    if portfolio_state is None:
        return np.zeros(len(universe_columns), dtype="float64")
    return np.asarray(
        [portfolio_state.current_weights.get(str(asset), 0.0) for asset in universe_columns],
        dtype="float64",
    )
