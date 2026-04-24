from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from app.portfolio_availability import (
    prepare_history_returns_for_assets,
    prepare_volume_history_for_assets,
    resolve_eligible_assets,
)
from app.portfolio_domain import *
from app.portfolio_forecast import ForecastSnapshot, build_strategy_forecast_snapshot
from app.portfolio_metrics import serialize_weights

DECISION_POLICY_EXTENSION_KEY = "decision_policy"
DIRECT_SCORE_DECISION_POLICY = "direct_score_to_weight"
COST_AWARE_NO_TRADE_DECISION_POLICY = "cost_aware_no_trade"
SUPPORTED_DECISION_POLICIES = {
    DIRECT_SCORE_DECISION_POLICY,
    COST_AWARE_NO_TRADE_DECISION_POLICY,
}
DEFAULT_NO_TRADE_BAND = 0.02
DEFAULT_CONFIDENCE_FLOOR = 0.25


@dataclass(frozen=True)
class PortfolioDecision:
    selected_assets: list[str]
    weights: np.ndarray
    policy: str
    action: str
    reason: str
    turnover: float
    estimated_cost_pct: float
    estimated_edge_pct: float | None
    edge_source: str | None
    average_confidence: float | None


def resolve_decision_policy_kind(strategy: EvaluatorStrategySpec) -> str:
    policy = dict(strategy.extensions).get(DECISION_POLICY_EXTENSION_KEY, COST_AWARE_NO_TRADE_DECISION_POLICY)
    if policy not in SUPPORTED_DECISION_POLICIES:
        raise ValueError(f"Unsupported decision policy: {policy}")
    return policy


def resolve_selected_assets_from_weights(universe_columns: pd.Index, weights: np.ndarray) -> list[str]:
    return [
        str(asset)
        for asset, weight in zip(universe_columns, weights, strict=True)
        if abs(float(weight)) > 1e-9
    ]


def build_decision_forecast_snapshot(
    *,
    history_returns: pd.DataFrame,
    volume_history: pd.DataFrame | None,
    strategy: EvaluatorStrategySpec,
    bars_per_year: float,
    current_date: str | None,
    predictor_panel: pd.DataFrame | None,
    selection_contexts: list[dict[str, object]] | None,
    predictor_context: dict[str, object] | None,
    availability_policy: dict[str, object],
) -> ForecastSnapshot | None:
    eligible_assets = resolve_eligible_assets(history_returns, availability_policy)
    if len(eligible_assets) < 2:
        return None
    clean_history_returns = prepare_history_returns_for_assets(history_returns, eligible_assets)
    if len(clean_history_returns) < 3 or len(clean_history_returns.columns) < 2:
        return None
    clean_volume_history = prepare_volume_history_for_assets(volume_history, clean_history_returns)
    try:
        return build_strategy_forecast_snapshot(
            clean_history_returns,
            clean_volume_history,
            strategy,
            bars_per_year=bars_per_year,
            current_date=current_date,
            predictor_panel=predictor_panel,
            selection_contexts=selection_contexts,
            predictor_context=predictor_context,
        )
    except ValueError:
        return None


def compute_weighted_forecast_confidence(
    forecast: ForecastSnapshot | None,
    universe_columns: pd.Index,
    target_weights: np.ndarray,
) -> float | None:
    if forecast is None:
        return None
    confidence = forecast.confidence.reindex(universe_columns).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    weight_values = np.abs(np.asarray(target_weights, dtype="float64"))
    weight_sum = float(weight_values.sum())
    if weight_sum <= 0:
        finite_confidence = confidence.replace([np.inf, -np.inf], np.nan).dropna()
        return None if finite_confidence.empty else float(finite_confidence.mean())
    return float(np.dot(confidence.to_numpy(dtype="float64"), weight_values) / weight_sum)


def compute_forecast_edge(
    forecast: ForecastSnapshot | None,
    universe_columns: pd.Index,
    current_weights: np.ndarray,
    target_weights: np.ndarray,
) -> float | None:
    if forecast is None or forecast.expected_return_proxy is None:
        return None
    proxy = forecast.expected_return_proxy.reindex(universe_columns).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    weight_delta = np.asarray(target_weights, dtype="float64") - np.asarray(current_weights, dtype="float64")
    return float(np.dot(weight_delta, proxy.to_numpy(dtype="float64")))


def build_portfolio_decision(
    *,
    history_returns: pd.DataFrame,
    volume_history: pd.DataFrame | None,
    strategy: EvaluatorStrategySpec,
    bars_per_year: float,
    universe_columns: pd.Index,
    current_weights: np.ndarray,
    current_selected_assets: list[str],
    target_weights: np.ndarray,
    target_selected_assets: list[str],
    transaction_cost: float,
    current_date: str | None,
    predictor_panel: pd.DataFrame | None,
    selection_contexts: list[dict[str, object]] | None,
    predictor_context: dict[str, object] | None,
    availability_policy: dict[str, object],
) -> PortfolioDecision:
    policy = resolve_decision_policy_kind(strategy)
    turnover = float(np.abs(target_weights - current_weights).sum())
    estimated_cost = turnover * float(transaction_cost)
    if policy == DIRECT_SCORE_DECISION_POLICY:
        return PortfolioDecision(
            selected_assets=list(target_selected_assets),
            weights=target_weights.copy(),
            policy=policy,
            action="rebalance",
            reason="direct_policy",
            turnover=turnover,
            estimated_cost_pct=round(estimated_cost * 100, 4),
            estimated_edge_pct=None,
            edge_source=None,
            average_confidence=None,
        )

    if float(np.sum(target_weights)) <= 0:
        return PortfolioDecision(
            selected_assets=[],
            weights=target_weights.copy(),
            policy=policy,
            action="rebalance",
            reason="target_cash",
            turnover=turnover,
            estimated_cost_pct=round(estimated_cost * 100, 4),
            estimated_edge_pct=None,
            edge_source=None,
            average_confidence=None,
        )

    forecast = build_decision_forecast_snapshot(
        history_returns=history_returns,
        volume_history=volume_history,
        strategy=strategy,
        bars_per_year=bars_per_year,
        current_date=current_date,
        predictor_panel=predictor_panel,
        selection_contexts=selection_contexts,
        predictor_context=predictor_context,
        availability_policy=availability_policy,
    )
    average_confidence = compute_weighted_forecast_confidence(forecast, universe_columns, target_weights)
    estimated_edge = compute_forecast_edge(forecast, universe_columns, current_weights, target_weights)
    edge_source = None if estimated_edge is None or forecast is None else forecast.edge_source
    no_trade_reason = None
    if turnover <= DEFAULT_NO_TRADE_BAND:
        no_trade_reason = "turnover_below_band"
    elif average_confidence is not None and average_confidence < DEFAULT_CONFIDENCE_FLOOR:
        no_trade_reason = "confidence_below_floor"
    elif estimated_edge is not None and estimated_edge <= estimated_cost:
        no_trade_reason = "edge_below_cost"

    if no_trade_reason is not None:
        held_assets = current_selected_assets or resolve_selected_assets_from_weights(universe_columns, current_weights)
        return PortfolioDecision(
            selected_assets=list(held_assets),
            weights=current_weights.copy(),
            policy=policy,
            action="no_trade",
            reason=no_trade_reason,
            turnover=turnover,
            estimated_cost_pct=round(estimated_cost * 100, 4),
            estimated_edge_pct=None if estimated_edge is None else round(estimated_edge * 100, 4),
            edge_source=edge_source,
            average_confidence=None if average_confidence is None else round(average_confidence, 4),
        )

    return PortfolioDecision(
        selected_assets=list(target_selected_assets),
        weights=target_weights.copy(),
        policy=policy,
        action="rebalance",
        reason="edge_after_cost",
        turnover=turnover,
        estimated_cost_pct=round(estimated_cost * 100, 4),
        estimated_edge_pct=None if estimated_edge is None else round(estimated_edge * 100, 4),
        edge_source=edge_source,
        average_confidence=None if average_confidence is None else round(average_confidence, 4),
    )


def compute_realized_decision_edge_pct(
    *,
    universe_columns: pd.Index,
    current_weights: np.ndarray,
    target_weights: np.ndarray,
    realized_returns: pd.Series | None,
) -> float | None:
    if realized_returns is None:
        return None
    clean_returns = realized_returns.reindex(universe_columns).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    weight_delta = np.asarray(target_weights, dtype="float64") - np.asarray(current_weights, dtype="float64")
    return round(float(np.dot(weight_delta, clean_returns.to_numpy(dtype="float64"))) * 100, 4)


def compute_edge_hit(estimated_edge_pct: float | None, realized_edge_pct: float | None) -> bool | None:
    if estimated_edge_pct is None or realized_edge_pct is None:
        return None
    if not math.isfinite(float(estimated_edge_pct)) or not math.isfinite(float(realized_edge_pct)):
        return None
    return float(np.sign(float(estimated_edge_pct))) == float(np.sign(float(realized_edge_pct)))


def serialize_portfolio_decision_event(
    date: str,
    decision: PortfolioDecision,
    *,
    universe_columns: pd.Index | None = None,
    current_weights: np.ndarray | None = None,
    target_weights: np.ndarray | None = None,
    realized_returns: pd.Series | None = None,
) -> dict[str, object]:
    realized_edge_pct = None
    if universe_columns is not None and current_weights is not None and target_weights is not None:
        realized_edge_pct = compute_realized_decision_edge_pct(
            universe_columns=universe_columns,
            current_weights=current_weights,
            target_weights=target_weights,
            realized_returns=realized_returns,
        )
    realized_edge_after_cost_pct = (
        None if realized_edge_pct is None else round(realized_edge_pct - decision.estimated_cost_pct, 4)
    )
    return {
        "date": date,
        "policy": decision.policy,
        "action": decision.action,
        "reason": decision.reason,
        "turnoverPct": round(decision.turnover * 100, 2),
        "estimatedCostPct": decision.estimated_cost_pct,
        "estimatedEdgePct": decision.estimated_edge_pct,
        "edgeSource": decision.edge_source,
        "realizedEdgePct": realized_edge_pct,
        "realizedEdgeAfterCostPct": realized_edge_after_cost_pct,
        "edgeHit": compute_edge_hit(decision.estimated_edge_pct, realized_edge_pct),
        "averageConfidence": decision.average_confidence,
        "selectedAssetCount": len(decision.selected_assets),
    }


def serialize_execution_trace_event(
    *,
    date: str,
    event_type: str,
    phase: str,
    universe_columns: pd.Index,
    max_investment_ratio: float,
    available_asset_count: int,
    eligible_asset_count: int,
    selected_assets: list[str],
    target_weights: np.ndarray | None,
    executed_weights: np.ndarray | None,
    decision_action: str | None,
    decision_reason: str | None,
    turnover_pct: float | None,
    estimated_cost_pct: float | None,
    estimated_edge_pct: float | None,
    edge_source: str | None,
    average_confidence: float | None,
) -> dict[str, object]:
    return {
        "date": date,
        "eventType": event_type,
        "phase": phase,
        "availableAssetCount": available_asset_count,
        "eligibleAssetCount": eligible_asset_count,
        "selectedAssets": list(selected_assets),
        "targetWeights": serialize_weights(universe_columns, target_weights, max_investment_ratio)
        if target_weights is not None
        else [],
        "executedWeights": serialize_weights(universe_columns, executed_weights, max_investment_ratio)
        if executed_weights is not None
        else [],
        "decisionAction": decision_action,
        "decisionReason": decision_reason,
        "turnoverPct": None if turnover_pct is None else round(float(turnover_pct), 2),
        "estimatedCostPct": estimated_cost_pct,
        "estimatedEdgePct": estimated_edge_pct,
        "edgeSource": edge_source,
        "averageConfidence": average_confidence,
    }
