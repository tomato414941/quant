from __future__ import annotations

import numpy as np
import pandas as pd


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
