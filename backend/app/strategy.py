from __future__ import annotations

import math
import statistics
from dataclasses import dataclass


TRADING_DAYS_PER_YEAR = 252
SUPPORTED_STRATEGIES = {"mean_reversion", "momentum"}
STRATEGY_LABELS = {
    "mean_reversion": "逆張り",
    "momentum": "上昇継続",
}
DEFAULT_HYPOTHESES = {
    "mean_reversion": "大きく動いた翌日に短期反発が出る",
    "momentum": "大きく動いた翌日に短期継続が出る",
}


@dataclass
class PricePoint:
    date: str
    close: float


@dataclass(frozen=True)
class StrategyDefinition:
    key: str
    engine: str
    label: str
    hypothesis: str
    threshold: float
    holding_days: int


def build_strategy_definition(
    engine: str,
    threshold: float,
    holding_days: int,
    *,
    key: str | None = None,
    label: str | None = None,
    hypothesis: str | None = None,
) -> StrategyDefinition:
    validate_strategy_inputs(engine=engine, threshold=threshold, holding_days=holding_days)
    threshold_pct = round(threshold * 100, 2)
    return StrategyDefinition(
        key=key or f"{engine}_{str(threshold_pct).replace('.', '_')}_{holding_days}d",
        engine=engine,
        label=label or f"{STRATEGY_LABELS[engine]} {threshold_pct:.1f}% / {holding_days}日",
        hypothesis=hypothesis or DEFAULT_HYPOTHESES[engine],
        threshold=threshold,
        holding_days=holding_days,
    )


def serialize_strategy_definition(strategy_definition: StrategyDefinition) -> dict:
    return {
        "key": strategy_definition.key,
        "engine": strategy_definition.engine,
        "label": strategy_definition.label,
        "hypothesis": strategy_definition.hypothesis,
        "thresholdPct": round(strategy_definition.threshold * 100, 2),
        "holdingDays": strategy_definition.holding_days,
    }


def run_backtest(
    prices: list[PricePoint],
    strategy_definition: StrategyDefinition,
    initial_capital: float,
    transaction_cost: float = 0.0,
) -> dict:
    validate_backtest_inputs(
        prices=prices,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
        strategy_definition=strategy_definition,
    )

    returns = build_returns(prices)
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
        signal = should_enter_trade(
            strategy_definition=strategy_definition,
            previous_day_return=previous_day_return,
            holding_remaining=holding_remaining,
        )
        if signal:
            holding_remaining = strategy_definition.holding_days
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
        "strategy": summarize_metrics(strategy_equity, initial_capital, len(prices), strategy_returns, series, "strategyEquity"),
        "benchmark": summarize_metrics(benchmark_equity, initial_capital, len(prices), benchmark_returns, series, "benchmarkEquity"),
        "config": {
            "strategyDefinition": serialize_strategy_definition(strategy_definition),
            "strategyId": strategy_definition.engine,
            "strategyLabel": strategy_definition.label,
            "thresholdPct": round(strategy_definition.threshold * 100, 2),
            "initialCapital": round(initial_capital, 2),
            "holdingDays": strategy_definition.holding_days,
            "transactionCostPct": round(transaction_cost * 100, 3),
        },
    }
    summary["strategy"]["tradeCount"] = trade_count
    summary["strategy"]["winRatePct"] = round(win_rate(trade_returns), 2)

    return {"summary": summary, "series": series}


def run_split_backtest(
    prices: list[PricePoint],
    strategy_definition: StrategyDefinition,
    initial_capital: float,
    transaction_cost: float = 0.0,
    split_ratio: float = 0.7,
) -> dict:
    validate_split_ratio(split_ratio)
    split_index = int(len(prices) * split_ratio)
    split_index = min(max(split_index, 3), len(prices) - 3)

    training_prices = prices[:split_index]
    testing_prices = prices[split_index:]
    if len(training_prices) < 3 or len(testing_prices) < 3:
        raise ValueError("Split ratio leaves too little data for train/test analysis.")

    training_result = run_backtest(
        prices=training_prices,
        strategy_definition=strategy_definition,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
    )
    testing_result = run_backtest(
        prices=testing_prices,
        strategy_definition=strategy_definition,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
    )

    return {
        "config": {
            "splitRatioPct": round(split_ratio * 100, 1),
        },
        "train": summarize_segment(training_prices, training_result["summary"]),
        "test": summarize_segment(testing_prices, testing_result["summary"]),
    }


def run_grid_search(
    prices: list[PricePoint],
    thresholds: list[float],
    holding_days_options: list[int],
    initial_capital: float,
    transaction_cost: float = 0.0,
    strategy: str = "mean_reversion",
) -> dict:
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    if not holding_days_options:
        raise ValueError("At least one holding-day value is required.")
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError("Unsupported strategy.")

    results: list[dict] = []
    for threshold in thresholds:
        for holding_days in holding_days_options:
            backtest = run_backtest(
                prices=prices,
                strategy_definition=build_strategy_definition(
                    engine=strategy,
                    threshold=threshold,
                    holding_days=holding_days,
                ),
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
            )
            summary = backtest["summary"]
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
            "strategyId": strategy,
            "strategyLabel": STRATEGY_LABELS[strategy],
            "thresholdValuesPct": [round(value * 100, 2) for value in thresholds],
            "holdingDaysValues": holding_days_options,
            "initialCapital": round(initial_capital, 2),
            "transactionCostPct": round(transaction_cost * 100, 3),
        },
        "results": results,
    }


def compare_tickers(
    datasets: list[dict],
    strategy_definition: StrategyDefinition,
    initial_capital: float,
    transaction_cost: float,
) -> dict:
    comparisons: list[dict] = []
    for dataset in datasets:
        backtest = run_backtest(
            prices=dataset["prices"],
            strategy_definition=strategy_definition,
            initial_capital=initial_capital,
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
            "strategyDefinition": serialize_strategy_definition(strategy_definition),
            "strategyId": strategy_definition.engine,
            "strategyLabel": strategy_definition.label,
            "thresholdPct": round(strategy_definition.threshold * 100, 2),
            "holdingDays": strategy_definition.holding_days,
            "initialCapital": round(initial_capital, 2),
            "transactionCostPct": round(transaction_cost * 100, 3),
        },
        "results": comparisons,
    }


def compare_periods(
    datasets: list[dict],
    strategy_definition: StrategyDefinition,
    initial_capital: float,
    transaction_cost: float,
) -> dict:
    comparisons: list[dict] = []
    for dataset in datasets:
        backtest = run_backtest(
            prices=dataset["prices"],
            strategy_definition=strategy_definition,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
        )
        summary = backtest["summary"]
        comparisons.append(
            {
                "period": dataset["period"],
                "strategy": summary["strategy"],
                "benchmark": summary["benchmark"],
            }
        )

    return {
        "config": {
            "strategyDefinition": serialize_strategy_definition(strategy_definition),
            "strategyId": strategy_definition.engine,
            "strategyLabel": strategy_definition.label,
            "thresholdPct": round(strategy_definition.threshold * 100, 2),
            "holdingDays": strategy_definition.holding_days,
            "initialCapital": round(initial_capital, 2),
            "transactionCostPct": round(transaction_cost * 100, 3),
        },
        "results": comparisons,
    }


def compare_strategies(
    prices: list[PricePoint],
    strategy_definitions: list[StrategyDefinition],
    initial_capital: float,
    transaction_cost: float,
    split_ratio: float,
) -> list[dict]:
    if not strategy_definitions:
        raise ValueError("At least one strategy definition is required.")

    comparisons: list[dict] = []
    for strategy_definition in strategy_definitions:
        backtest = run_backtest(
            prices=prices,
            strategy_definition=strategy_definition,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
        )
        comparisons.append(
            {
                "definition": serialize_strategy_definition(strategy_definition),
                "summary": backtest["summary"]["strategy"],
                "benchmark": backtest["summary"]["benchmark"],
                "splitAnalysis": run_split_backtest(
                    prices=prices,
                    strategy_definition=strategy_definition,
                    initial_capital=initial_capital,
                    transaction_cost=transaction_cost,
                    split_ratio=split_ratio,
                ),
                "series": backtest["series"],
            }
        )

    return comparisons


def validate_backtest_inputs(
    prices: list[PricePoint],
    initial_capital: float,
    transaction_cost: float,
    strategy_definition: StrategyDefinition,
) -> None:
    if len(prices) < 3:
        raise ValueError("At least 3 prices are required.")
    if initial_capital <= 0:
        raise ValueError("Initial capital must be positive.")
    if transaction_cost < 0 or transaction_cost >= 1:
        raise ValueError("Transaction cost must be between 0 and 1.")
    validate_strategy_inputs(
        engine=strategy_definition.engine,
        threshold=strategy_definition.threshold,
        holding_days=strategy_definition.holding_days,
    )


def validate_strategy_inputs(engine: str, threshold: float, holding_days: int) -> None:
    if threshold <= 0 or threshold >= 1:
        raise ValueError("Threshold must be between 0 and 1.")
    if holding_days <= 0 or holding_days > 30:
        raise ValueError("Holding days must be between 1 and 30.")
    if engine not in SUPPORTED_STRATEGIES:
        raise ValueError("Unsupported strategy.")


def validate_split_ratio(split_ratio: float) -> None:
    if split_ratio <= 0.5 or split_ratio >= 0.95:
        raise ValueError("Split ratio must be between 0.5 and 0.95.")


def build_returns(prices: list[PricePoint]) -> list[float]:
    returns = [0.0]
    for index in range(1, len(prices)):
        previous_close = prices[index - 1].close
        current_close = prices[index].close
        returns.append((current_close / previous_close) - 1)
    return returns


def should_enter_trade(
    strategy_definition: StrategyDefinition,
    previous_day_return: float,
    holding_remaining: int,
) -> bool:
    if holding_remaining != 0:
        return False
    if strategy_definition.engine == "mean_reversion":
        return previous_day_return <= -strategy_definition.threshold
    if strategy_definition.engine == "momentum":
        return previous_day_return >= strategy_definition.threshold
    raise ValueError("Unsupported strategy.")


def summarize_segment(prices: list[PricePoint], summary: dict) -> dict:
    return {
        "startDate": prices[0].date,
        "endDate": prices[-1].date,
        "dayCount": len(prices),
        "strategy": summary["strategy"],
        "benchmark": summary["benchmark"],
    }


def summarize_metrics(
    final_value: float,
    initial_value: float,
    periods: int,
    returns: list[float],
    series: list[dict],
    equity_key: str,
) -> dict:
    return {
        "totalReturnPct": round(percent_return(final_value, initial_value), 2),
        "cagrPct": round(cagr(final_value, initial_value, periods), 2),
        "sharpeRatio": round(sharpe_ratio(returns), 2),
        "maxDrawdownPct": round(max_drawdown(series, equity_key), 2),
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
    return wins / len(trade_returns) * 100
