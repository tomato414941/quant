from __future__ import annotations

import math
import statistics
from dataclasses import dataclass


TRADING_DAYS_PER_YEAR = 252


@dataclass
class PricePoint:
    date: str
    close: float


def run_backtest(
    prices: list[PricePoint],
    threshold: float,
    initial_capital: float,
    holding_days: int = 1,
    transaction_cost: float = 0.0,
) -> dict:
    if len(prices) < 3:
        raise ValueError("At least 3 prices are required.")
    if threshold <= 0 or threshold >= 1:
        raise ValueError("Threshold must be between 0 and 1.")
    if initial_capital <= 0:
        raise ValueError("Initial capital must be positive.")
    if holding_days <= 0 or holding_days > 30:
        raise ValueError("Holding days must be between 1 and 30.")
    if transaction_cost < 0 or transaction_cost >= 1:
        raise ValueError("Transaction cost must be between 0 and 1.")

    returns = [0.0]
    for index in range(1, len(prices)):
        previous_close = prices[index - 1].close
        current_close = prices[index].close
        returns.append((current_close / previous_close) - 1)

    strategy_equity = initial_capital
    benchmark_equity = initial_capital
    strategy_returns: list[float] = []
    benchmark_returns: list[float] = []
    trade_returns: list[float] = []
    series: list[dict] = []
    trade_count = 0
    holding_remaining = 0
    current_trade_growth = 1.0

    for index, point in enumerate(prices):
        previous_day_return = returns[index - 1] if index >= 1 else 0.0
        current_day_return = returns[index]
        signal = previous_day_return <= -threshold and holding_remaining == 0
        if signal:
            holding_remaining = holding_days
            current_trade_growth = 1.0
            trade_count += 1

        position = 1 if holding_remaining > 0 else 0
        strategy_return = current_day_return * position
        if signal:
            strategy_return -= transaction_cost

        if position == 1:
            current_trade_growth *= 1 + strategy_return

        strategy_equity *= 1 + strategy_return
        benchmark_equity *= 1 + current_day_return
        strategy_returns.append(strategy_return)
        benchmark_returns.append(current_day_return)

        series.append(
            {
                "date": point.date,
                "close": round(point.close, 2),
                "previousDayReturnPct": round(previous_day_return * 100, 2),
                "position": position,
                "signal": signal,
                "strategyEquity": round(strategy_equity, 2),
                "benchmarkEquity": round(benchmark_equity, 2),
                "strategyReturnPct": round(strategy_return * 100, 2),
            }
        )

        if holding_remaining > 0:
            holding_remaining -= 1
            if holding_remaining == 0:
                trade_returns.append(current_trade_growth - 1)

    summary = {
        "strategy": {
            "totalReturnPct": round(percent_return(strategy_equity, initial_capital), 2),
            "cagrPct": round(cagr(strategy_equity, initial_capital, len(prices)), 2),
            "sharpeRatio": round(sharpe_ratio(strategy_returns), 2),
            "maxDrawdownPct": round(max_drawdown(series, "strategyEquity"), 2),
            "tradeCount": trade_count,
            "winRatePct": round(win_rate(trade_returns), 2),
        },
        "benchmark": {
            "totalReturnPct": round(percent_return(benchmark_equity, initial_capital), 2),
            "cagrPct": round(cagr(benchmark_equity, initial_capital, len(prices)), 2),
            "sharpeRatio": round(sharpe_ratio(benchmark_returns), 2),
            "maxDrawdownPct": round(max_drawdown(series, "benchmarkEquity"), 2),
        },
        "config": {
            "thresholdPct": round(threshold * 100, 2),
            "initialCapital": round(initial_capital, 2),
            "holdingDays": holding_days,
            "transactionCostPct": round(transaction_cost * 100, 3),
            "holdingRule": f"Buy for {holding_days} day(s) after a drop larger than threshold.",
        },
    }

    return {"summary": summary, "series": series}


def run_grid_search(
    prices: list[PricePoint],
    thresholds: list[float],
    holding_days_options: list[int],
    initial_capital: float,
    transaction_cost: float = 0.0,
) -> dict:
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    if not holding_days_options:
        raise ValueError("At least one holding-day value is required.")

    results: list[dict] = []
    benchmark_summary: dict | None = None

    for threshold in thresholds:
        for holding_days in holding_days_options:
            backtest = run_backtest(
                prices=prices,
                threshold=threshold,
                initial_capital=initial_capital,
                holding_days=holding_days,
                transaction_cost=transaction_cost,
            )
            summary = backtest["summary"]
            benchmark_summary = summary["benchmark"]
            strategy_summary = summary["strategy"]
            results.append(
                {
                    "thresholdPct": summary["config"]["thresholdPct"],
                    "holdingDays": summary["config"]["holdingDays"],
                    "totalReturnPct": strategy_summary["totalReturnPct"],
                    "cagrPct": strategy_summary["cagrPct"],
                    "sharpeRatio": strategy_summary["sharpeRatio"],
                    "maxDrawdownPct": strategy_summary["maxDrawdownPct"],
                    "tradeCount": strategy_summary["tradeCount"],
                    "winRatePct": strategy_summary["winRatePct"],
                }
            )

    results.sort(
        key=lambda item: (
            item["sharpeRatio"],
            item["totalReturnPct"],
            -item["maxDrawdownPct"],
        ),
        reverse=True,
    )

    for index, result in enumerate(results, start=1):
        result["rank"] = index

    return {
        "config": {
            "thresholdValuesPct": [round(value * 100, 2) for value in thresholds],
            "holdingDaysValues": holding_days_options,
            "initialCapital": round(initial_capital, 2),
            "transactionCostPct": round(transaction_cost * 100, 3),
        },
        "results": results,
    }


def compare_tickers(
    datasets: list[dict],
    threshold: float,
    holding_days: int,
    initial_capital: float,
    transaction_cost: float,
) -> dict:
    comparisons: list[dict] = []
    for dataset in datasets:
        backtest = run_backtest(
            prices=dataset["prices"],
            threshold=threshold,
            initial_capital=initial_capital,
            holding_days=holding_days,
            transaction_cost=transaction_cost,
        )
        summary = backtest["summary"]
        comparisons.append(
            {
                "ticker": dataset["ticker"],
                "period": dataset["period"],
                "strategy": summary["strategy"],
                "benchmark": summary["benchmark"],
            }
        )

    comparisons.sort(
        key=lambda item: (
            item["strategy"]["sharpeRatio"],
            item["strategy"]["totalReturnPct"],
        ),
        reverse=True,
    )

    return {
        "config": {
            "thresholdPct": round(threshold * 100, 2),
            "holdingDays": holding_days,
            "initialCapital": round(initial_capital, 2),
            "transactionCostPct": round(transaction_cost * 100, 3),
        },
        "results": comparisons,
    }


def percent_return(final_value: float, initial_value: float) -> float:
    return ((final_value / initial_value) - 1) * 100


def cagr(final_value: float, initial_value: float, periods: int) -> float:
    years = max((periods - 1) / TRADING_DAYS_PER_YEAR, 1 / TRADING_DAYS_PER_YEAR)
    return (((final_value / initial_value) ** (1 / years)) - 1) * 100


def sharpe_ratio(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0

    mean_return = statistics.fmean(returns)
    volatility = statistics.pstdev(returns)
    if volatility == 0:
        return 0.0
    return math.sqrt(TRADING_DAYS_PER_YEAR) * (mean_return / volatility)


def max_drawdown(series: list[dict], equity_key: str) -> float:
    peak = series[0][equity_key]
    max_dd = 0.0
    for point in series:
        equity = point[equity_key]
        peak = max(peak, equity)
        drawdown = (equity / peak) - 1
        max_dd = min(max_dd, drawdown)
    return abs(max_dd) * 100


def win_rate(trade_returns: list[float]) -> float:
    if not trade_returns:
        return 0.0
    wins = sum(1 for value in trade_returns if value > 0)
    return (wins / len(trade_returns)) * 100
