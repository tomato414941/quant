from __future__ import annotations

import csv
import io
import math
import random
import statistics
from dataclasses import dataclass
from datetime import date, timedelta


TRADING_DAYS_PER_YEAR = 252


@dataclass
class PricePoint:
    date: str
    close: float


def generate_demo_prices(days: int = 260, seed: int = 7) -> list[PricePoint]:
    rng = random.Random(seed)
    current_price = 100.0
    current_date = date(2025, 1, 2)
    prices: list[PricePoint] = []

    for index in range(days):
        if index > 0:
            drift = 0.0005
            shock = rng.gauss(0, 0.015)
            if index % 37 == 0:
                shock -= 0.045
            current_price *= 1 + drift + shock

        prices.append(
            PricePoint(
                date=current_date.isoformat(),
                close=round(max(current_price, 1.0), 2),
            )
        )
        current_date += timedelta(days=1)

    return prices


def parse_uploaded_prices(file_bytes: bytes) -> list[PricePoint]:
    text = file_bytes.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    field_map = {normalize_column_name(name): name for name in reader.fieldnames or []}

    date_field = first_matching_field(field_map, ["date", "datetime", "timestamp"])
    close_field = first_matching_field(field_map, ["close", "adjclose", "adjustedclose"])
    if not date_field or not close_field:
        raise ValueError("CSV must contain date and close columns.")

    prices: list[PricePoint] = []
    for row in reader:
        raw_date = (row.get(date_field) or "").strip()
        raw_close = (row.get(close_field) or "").strip()
        if not raw_date or not raw_close:
            continue

        try:
            prices.append(PricePoint(date=raw_date, close=float(raw_close)))
        except ValueError as exc:
            raise ValueError("CSV contains invalid numeric values.") from exc

    if len(prices) < 3:
        raise ValueError("At least 3 rows are required for a backtest.")

    prices.sort(key=lambda item: item.date)
    return prices


def normalize_column_name(name: str) -> str:
    return "".join(char for char in name.lower() if char.isalnum())


def first_matching_field(field_map: dict[str, str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in field_map:
            return field_map[candidate]
    return None


def run_backtest(
    prices: list[PricePoint],
    threshold: float,
    initial_capital: float,
) -> dict:
    if len(prices) < 3:
        raise ValueError("At least 3 prices are required.")
    if threshold <= 0 or threshold >= 1:
        raise ValueError("Threshold must be between 0 and 1.")
    if initial_capital <= 0:
        raise ValueError("Initial capital must be positive.")

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
    previous_position = 0

    for index, point in enumerate(prices):
        previous_day_return = returns[index - 1] if index >= 1 else 0.0
        current_day_return = returns[index]
        signal = previous_day_return <= -threshold
        position = 1 if signal else 0
        strategy_return = current_day_return * position

        if position == 1 and previous_position == 0:
            trade_count += 1
        if position == 1:
            trade_returns.append(strategy_return)

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
        previous_position = position

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
            "holdingRule": "Buy for one day after a drop larger than threshold.",
        },
    }

    return {"summary": summary, "series": series}


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
