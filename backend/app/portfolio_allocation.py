from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from skfolio.optimization import HierarchicalRiskParity, MeanRisk, RiskBudgeting

from app.portfolio_domain import PortfolioModelSpec


@dataclass(frozen=True)
class PortfolioAllocationInput:
    returns: pd.DataFrame
    portfolio_model: PortfolioModelSpec
    max_investment_ratio: float
    max_weight: float | None
    previous_weights: np.ndarray | None
    transaction_cost: float


@dataclass(frozen=True)
class ForecastAllocationInput:
    expected_returns: np.ndarray
    confidence: np.ndarray | None
    risk_proxy: np.ndarray | None
    transaction_cost: float


@dataclass(frozen=True)
class PortfolioAllocationResult:
    weights: np.ndarray
    fallback_metadata: dict[str, object] | None = None


def expand_weights(
    universe_columns: pd.Index,
    selected_columns: pd.Index,
    selected_weights: np.ndarray,
) -> np.ndarray:
    weight_map = {
        str(asset): float(weight)
        for asset, weight in zip(selected_columns, selected_weights, strict=True)
    }
    return np.asarray([weight_map.get(str(asset), 0.0) for asset in universe_columns], dtype="float64")


def fit_portfolio_model(allocation_input: PortfolioAllocationInput) -> np.ndarray:
    return fit_portfolio_model_result(allocation_input).weights


def fit_portfolio_model_result(allocation_input: PortfolioAllocationInput) -> PortfolioAllocationResult:
    return fit_risk_structure_portfolio_model_result(allocation_input)


def fit_risk_structure_portfolio_model(allocation_input: PortfolioAllocationInput) -> np.ndarray:
    return fit_risk_structure_portfolio_model_result(allocation_input).weights


def fit_risk_structure_portfolio_model_result(allocation_input: PortfolioAllocationInput) -> PortfolioAllocationResult:
    returns = allocation_input.returns
    portfolio_model = allocation_input.portfolio_model
    asset_count = len(returns.columns)
    if asset_count == 0:
        raise ValueError("At least one asset is required.")

    raw_max_weight = None
    if allocation_input.max_weight is not None:
        raw_max_weight = min(1.0, allocation_input.max_weight / allocation_input.max_investment_ratio)
    has_insufficient_optimizer_history = len(returns.index) < max(2, asset_count)

    if portfolio_model.model_type == "equal_weight":
        weights = build_equal_weight_fallback(asset_count, raw_max_weight)
        fallback_metadata = None
    elif portfolio_model.model_type == "risk_budgeting":
        if has_insufficient_optimizer_history:
            weights = build_equal_weight_fallback(asset_count, raw_max_weight)
            fallback_metadata = build_allocation_fallback_metadata(
                portfolio_model.model_type,
                "insufficient_history",
            )
        else:
            try:
                estimator = RiskBudgeting(
                    max_weights=raw_max_weight if raw_max_weight is not None else 1.0,
                    transaction_costs=allocation_input.transaction_cost,
                    previous_weights=allocation_input.previous_weights,
                )
                estimator.fit(returns)
                weights = estimator.weights_
                fallback_metadata = None
            except Exception as exc:
                weights = build_equal_weight_fallback(asset_count, raw_max_weight)
                fallback_metadata = build_allocation_fallback_metadata(
                    portfolio_model.model_type,
                    "optimizer_exception",
                    exc,
                )
    elif portfolio_model.model_type == "minimum_variance":
        if has_insufficient_optimizer_history:
            weights = build_equal_weight_fallback(asset_count, raw_max_weight)
            fallback_metadata = build_allocation_fallback_metadata(
                portfolio_model.model_type,
                "insufficient_history",
            )
        else:
            try:
                estimator = MeanRisk(
                    max_weights=raw_max_weight if raw_max_weight is not None else 1.0,
                    transaction_costs=allocation_input.transaction_cost,
                    previous_weights=allocation_input.previous_weights,
                )
                estimator.fit(returns)
                weights = estimator.weights_
                fallback_metadata = None
            except Exception as exc:
                weights = build_equal_weight_fallback(asset_count, raw_max_weight)
                fallback_metadata = build_allocation_fallback_metadata(
                    portfolio_model.model_type,
                    "optimizer_exception",
                    exc,
                )
    elif portfolio_model.model_type == "hierarchical_risk_parity":
        if has_insufficient_optimizer_history:
            weights = build_equal_weight_fallback(asset_count, raw_max_weight)
            fallback_metadata = build_allocation_fallback_metadata(
                portfolio_model.model_type,
                "insufficient_history",
            )
        elif asset_count <= 2:
            try:
                estimator = RiskBudgeting(
                    max_weights=raw_max_weight if raw_max_weight is not None else 1.0,
                    transaction_costs=allocation_input.transaction_cost,
                    previous_weights=allocation_input.previous_weights,
                )
                estimator.fit(returns)
                weights = estimator.weights_
                fallback_metadata = None
            except Exception as exc:
                weights = build_equal_weight_fallback(asset_count, raw_max_weight)
                fallback_metadata = build_allocation_fallback_metadata(
                    portfolio_model.model_type,
                    "optimizer_exception",
                    exc,
                )
        else:
            try:
                estimator = HierarchicalRiskParity(
                    max_weights=raw_max_weight if raw_max_weight is not None else 1.0,
                    transaction_costs=allocation_input.transaction_cost,
                    previous_weights=allocation_input.previous_weights,
                )
                estimator.fit(returns)
                weights = estimator.weights_
                fallback_metadata = None
            except Exception as exc:
                weights = build_equal_weight_fallback(asset_count, raw_max_weight)
                fallback_metadata = build_allocation_fallback_metadata(
                    portfolio_model.model_type,
                    "optimizer_exception",
                    exc,
                )
    elif portfolio_model.model_type in {"mean_risk_utility", "mean_risk_utility_conservative"}:
        weights = build_uncalibrated_forecast_allocator_fallback(asset_count, raw_max_weight)
        fallback_metadata = build_allocation_fallback_metadata(
            portfolio_model.model_type,
            "uncalibrated_forecast",
        )
    else:
        raise ValueError("Unsupported portfolio model.")

    weights = np.asarray(weights, dtype="float64")
    weight_sum = weights.sum()
    if weight_sum <= 0:
        raise ValueError("Portfolio model returned invalid weights.")
    if raw_max_weight is not None and weight_sum < 0.999999:
        return PortfolioAllocationResult(weights=weights, fallback_metadata=fallback_metadata)
    return PortfolioAllocationResult(weights=weights / weight_sum, fallback_metadata=fallback_metadata)


def build_equal_weight_fallback(asset_count: int, raw_max_weight: float | None) -> np.ndarray:
    weights = np.repeat(1 / asset_count, asset_count)
    if raw_max_weight is not None:
        weights = np.minimum(weights, raw_max_weight)
    return weights


def build_uncalibrated_forecast_allocator_fallback(asset_count: int, raw_max_weight: float | None) -> np.ndarray:
    return build_equal_weight_fallback(asset_count, raw_max_weight)


def build_allocation_fallback_metadata(
    model_type: str,
    reason: str,
    exception: Exception | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "modelType": model_type,
        "reason": reason,
        "fallback": "equal_weight",
    }
    if exception is not None:
        metadata["exceptionType"] = type(exception).__name__
        metadata["message"] = str(exception)
    return metadata
