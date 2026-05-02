from __future__ import annotations

import numpy as np
import pandas as pd

from app.portfolio_accounting import MISSING_RETURN_POLICY_REJECT_IF_HELD, resolve_accounting_returns
from app.portfolio_allocation import PortfolioAllocationInput, compute_portfolio_allocation_result
from app.portfolio_costs import build_flat_cost_model, compute_trade_cost, resolve_cost_model_inputs
from app.portfolio_domain import PortfolioModelSpec
from app.portfolio_metrics import serialize_weights, should_rebalance, summarize_segment_from_returns


def run_equal_weight_full_period_backtest(
    *,
    closes: pd.DataFrame,
    volumes: pd.DataFrame | None = None,
    initial_capital: float = 10_000.0,
    transaction_cost: float = 0.0,
    bars_per_year: float = 252.0,
    rebalance_schedule: str = "hold",
    missing_return_policy: str = MISSING_RETURN_POLICY_REJECT_IF_HELD,
) -> dict[str, object]:
    if closes.empty:
        raise ValueError("closes must contain at least one row.")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive.")
    if transaction_cost < 0 or transaction_cost >= 1:
        raise ValueError("transaction_cost must be between 0 and 1.")

    returns = closes.pct_change(fill_method=None).iloc[1:].dropna(how="all")
    if returns.empty:
        raise ValueError("At least one return row is required.")
    if volumes is not None:
        volumes = volumes.reindex(closes.index)

    universe_columns = returns.columns
    target_weights = np.repeat(1.0 / len(universe_columns), len(universe_columns)).astype("float64")
    current_weights = np.zeros(len(universe_columns), dtype="float64")
    cost_model = build_flat_cost_model(commission_pct=transaction_cost * 100, slippage_pct=0.0)
    cost_inputs = resolve_cost_model_inputs(universe_columns=universe_columns, cost_model=cost_model)

    portfolio_equity = float(initial_capital)
    portfolio_returns: list[float] = []
    dates: list[str] = []
    series: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    total_turnover = 0.0

    for index, (date, row) in enumerate(returns.iterrows()):
        should_trade = index == 0
        if index > 0:
            should_trade = should_rebalance(returns.index[index - 1], date, rebalance_schedule)

        trade_cost = 0.0
        if should_trade:
            weight_delta = np.abs(target_weights - current_weights)
            turnover = float(weight_delta.sum())
            trade_cost = compute_trade_cost(
                weight_delta=weight_delta,
                linear_cost_rates=np.asarray(cost_inputs["linearRates"], dtype="float64"),
                impact_cost_rates=np.asarray(cost_inputs["impactRates"], dtype="float64"),
                portfolio_equity=portfolio_equity,
                price_snapshot=closes.iloc[index],
                volume_history=volumes.iloc[: index + 1] if volumes is not None else None,
                adv_window_bars=int(cost_inputs["advWindowBars"]),
                min_adv_notional=float(cost_inputs["minAdvNotional"]),
            )
            current_weights = target_weights.copy()
            total_turnover += turnover
            events.append(
                {
                    "date": str(date),
                    "eventType": "rebalance",
                    "turnoverPct": round(turnover * 100, 2),
                    "estimatedCostPct": round(trade_cost * 100, 4),
                    "targetWeights": serialize_weights(universe_columns, target_weights, 1.0),
                    "executedWeights": serialize_weights(universe_columns, current_weights, 1.0),
                }
            )

        accounting = resolve_accounting_returns(
            row=row,
            universe_columns=universe_columns,
            current_weights=current_weights,
            date=date,
            phase="full_period",
            missing_return_policy=missing_return_policy,
        )
        events.extend(accounting.events)
        portfolio_return = accounting.portfolio_return - trade_cost
        portfolio_equity *= 1 + portfolio_return
        portfolio_returns.append(portfolio_return)
        dates.append(str(date))
        series.append(
            {
                "date": str(date),
                "portfolioEquity": round(portfolio_equity, 2),
                "portfolioReturnPct": round(portfolio_return * 100, 2),
                "weights": serialize_weights(universe_columns, current_weights, 1.0),
            }
        )

    summary = summarize_segment_from_returns(
        dates=dates,
        portfolio_returns=portfolio_returns,
        bars_per_year=bars_per_year,
        turnover=total_turnover,
    )["portfolio"]
    return {
        "kind": "full_period_backtest",
        "strategyKey": "equal_weight",
        "summary": summary,
        "series": series,
        "events": events,
        "weights": serialize_weights(universe_columns, current_weights, 1.0),
        "firstInvestedDate": events[0]["date"] if events else None,
    }


def run_portfolio_model_full_period_backtest(
    *,
    closes: pd.DataFrame,
    portfolio_model: PortfolioModelSpec,
    strategy_key: str,
    volumes: pd.DataFrame | None = None,
    initial_capital: float = 10_000.0,
    transaction_cost: float = 0.0,
    bars_per_year: float = 252.0,
    rebalance_schedule: str = "hold",
    missing_return_policy: str = MISSING_RETURN_POLICY_REJECT_IF_HELD,
) -> dict[str, object]:
    if closes.empty:
        raise ValueError("closes must contain at least one row.")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive.")
    if transaction_cost < 0 or transaction_cost >= 1:
        raise ValueError("transaction_cost must be between 0 and 1.")

    returns = closes.pct_change(fill_method=None).iloc[1:].dropna(how="all")
    if returns.empty:
        raise ValueError("At least one return row is required.")
    if volumes is not None:
        volumes = volumes.reindex(closes.index)

    universe_columns = returns.columns
    current_weights = np.zeros(len(universe_columns), dtype="float64")
    cost_model = build_flat_cost_model(commission_pct=transaction_cost * 100, slippage_pct=0.0)
    cost_inputs = resolve_cost_model_inputs(universe_columns=universe_columns, cost_model=cost_model)

    portfolio_equity = float(initial_capital)
    portfolio_returns: list[float] = []
    dates: list[str] = []
    series: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    total_turnover = 0.0

    for index, (date, row) in enumerate(returns.iterrows()):
        should_trade = index == 0
        if index > 0:
            should_trade = should_rebalance(returns.index[index - 1], date, rebalance_schedule)

        trade_cost = 0.0
        if should_trade:
            allocation_result = compute_portfolio_allocation_result(
                PortfolioAllocationInput(
                    returns=returns.iloc[: index + 1],
                    portfolio_model=portfolio_model,
                    max_investment_ratio=1.0,
                    max_weight=None,
                    previous_weights=current_weights,
                    transaction_cost=transaction_cost,
                )
            )
            target_weights = allocation_result.weights
            weight_delta = np.abs(target_weights - current_weights)
            turnover = float(weight_delta.sum())
            trade_cost = compute_trade_cost(
                weight_delta=weight_delta,
                linear_cost_rates=np.asarray(cost_inputs["linearRates"], dtype="float64"),
                impact_cost_rates=np.asarray(cost_inputs["impactRates"], dtype="float64"),
                portfolio_equity=portfolio_equity,
                price_snapshot=closes.iloc[index],
                volume_history=volumes.iloc[: index + 1] if volumes is not None else None,
                adv_window_bars=int(cost_inputs["advWindowBars"]),
                min_adv_notional=float(cost_inputs["minAdvNotional"]),
            )
            current_weights = target_weights.copy()
            total_turnover += turnover
            event = {
                "date": str(date),
                "eventType": "rebalance",
                "turnoverPct": round(turnover * 100, 2),
                "estimatedCostPct": round(trade_cost * 100, 4),
                "targetWeights": serialize_weights(universe_columns, target_weights, 1.0),
                "executedWeights": serialize_weights(universe_columns, current_weights, 1.0),
            }
            if allocation_result.fallback_metadata is not None:
                event["allocationFallback"] = allocation_result.fallback_metadata
            events.append(event)

        accounting = resolve_accounting_returns(
            row=row,
            universe_columns=universe_columns,
            current_weights=current_weights,
            date=date,
            phase="full_period",
            missing_return_policy=missing_return_policy,
        )
        events.extend(accounting.events)
        portfolio_return = accounting.portfolio_return - trade_cost
        portfolio_equity *= 1 + portfolio_return
        portfolio_returns.append(portfolio_return)
        dates.append(str(date))
        series.append(
            {
                "date": str(date),
                "portfolioEquity": round(portfolio_equity, 2),
                "portfolioReturnPct": round(portfolio_return * 100, 2),
                "weights": serialize_weights(universe_columns, current_weights, 1.0),
            }
        )

    summary = summarize_segment_from_returns(
        dates=dates,
        portfolio_returns=portfolio_returns,
        bars_per_year=bars_per_year,
        turnover=total_turnover,
    )["portfolio"]
    return {
        "kind": "full_period_backtest",
        "strategyKey": strategy_key,
        "summary": summary,
        "series": series,
        "events": events,
        "weights": serialize_weights(universe_columns, current_weights, 1.0),
        "firstInvestedDate": events[0]["date"] if events else None,
    }
