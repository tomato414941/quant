from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.dashboard_config import DEFAULT_DASHBOARD_CONFIG
from app.dashboard_service import build_dashboard_payload
from app.market_data import SUPPORTED_PERIODS, fetch_market_prices, fetch_market_universe
from app.strategy import (
    SUPPORTED_STRATEGIES,
    build_strategy_definition,
    compare_periods,
    compare_tickers,
    run_backtest,
    run_grid_search,
    run_split_backtest,
)


app = FastAPI(title="Quant API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|[\w\.-]+)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/dashboard")
def dashboard() -> dict:
    try:
        return build_dashboard_payload(
            DEFAULT_DASHBOARD_CONFIG,
            fetch_market_universe=fetch_market_universe,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/backtest")
def backtest_market(
    ticker: str = "SPY",
    period: str = "2y",
    threshold: float = 0.03,
    initial_capital: float = 10_000,
    holding_days: int = 1,
    transaction_cost: float = 0.0,
    strategy: str = "mean_reversion",
    split_ratio: float = 0.7,
) -> dict:
    try:
        strategy_definition = build_strategy_definition(
            engine=parse_strategy(strategy),
            threshold=threshold,
            holding_days=holding_days,
        )
        prices, metadata = fetch_market_prices(ticker=ticker, period=period)
        payload = run_backtest(
            prices=prices,
            strategy_definition=strategy_definition,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
        )
        payload["splitAnalysis"] = run_split_backtest(
            prices=prices,
            strategy_definition=strategy_definition,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
            split_ratio=split_ratio,
        )
        payload["dataset"] = metadata
        return payload
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/grid-search")
def grid_search_market(
    ticker: str = "SPY",
    period: str = "2y",
    threshold_values: str = Query("1.5,2,3,4,5"),
    holding_days_values: str = Query("1,2,3"),
    initial_capital: float = 10_000,
    transaction_cost: float = 0.0,
    strategy: str = "mean_reversion",
) -> dict:
    try:
        prices, metadata = fetch_market_prices(ticker=ticker, period=period)
        payload = run_grid_search(
            prices=prices,
            thresholds=parse_percentage_values(threshold_values),
            holding_days_options=parse_integer_values(holding_days_values),
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
            strategy=parse_strategy(strategy),
        )
        payload["dataset"] = metadata
        return payload
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/ticker-compare")
def ticker_compare_market(
    tickers: str = Query("SPY,QQQ,IWM,TLT,GLD,BTC-USD"),
    period: str = "2y",
    threshold: float = 0.03,
    initial_capital: float = 10_000,
    holding_days: int = 1,
    transaction_cost: float = 0.0,
    strategy: str = "mean_reversion",
) -> dict:
    try:
        strategy_definition = build_strategy_definition(
            engine=parse_strategy(strategy),
            threshold=threshold,
            holding_days=holding_days,
        )
        ticker_values = parse_ticker_values(tickers)
        datasets = []
        for ticker in ticker_values:
            prices, metadata = fetch_market_prices(ticker=ticker, period=period)
            datasets.append(
                {
                    "ticker": metadata["ticker"],
                    "period": metadata["period"],
                    "prices": prices,
                }
            )

        return compare_tickers(
            datasets=datasets,
            strategy_definition=strategy_definition,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/period-compare")
def period_compare_market(
    ticker: str = "SPY",
    periods: str = Query("6mo,1y,2y,3y,5y"),
    threshold: float = 0.03,
    initial_capital: float = 10_000,
    holding_days: int = 1,
    transaction_cost: float = 0.0,
    strategy: str = "mean_reversion",
) -> dict:
    try:
        strategy_definition = build_strategy_definition(
            engine=parse_strategy(strategy),
            threshold=threshold,
            holding_days=holding_days,
        )
        datasets = []
        normalized_ticker = ticker.strip().upper()
        for period in parse_period_values(periods):
            prices, metadata = fetch_market_prices(ticker=normalized_ticker, period=period)
            datasets.append(
                {
                    "ticker": metadata["ticker"],
                    "period": metadata["period"],
                    "prices": prices,
                }
            )

        payload = compare_periods(
            datasets=datasets,
            strategy_definition=strategy_definition,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
        )
        payload["dataset"] = {"ticker": normalized_ticker, "source": "Yahoo Finance via yfinance"}
        return payload
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def parse_strategy(raw_value: str) -> str:
    strategy = raw_value.strip().lower()
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError("Unsupported strategy.")
    return strategy


def parse_percentage_values(raw_values: str) -> list[float]:
    values = parse_csv_numbers(raw_values)
    percentages = sorted({value / 100 for value in values})
    if any(value <= 0 or value >= 1 for value in percentages):
        raise ValueError("Threshold values must be between 0 and 100.")
    return percentages


def parse_integer_values(raw_values: str) -> list[int]:
    values = [int(value) for value in parse_csv_numbers(raw_values)]
    unique_values = sorted(set(values))
    if any(value <= 0 or value > 30 for value in unique_values):
        raise ValueError("Holding-day values must be between 1 and 30.")
    return unique_values


def parse_csv_numbers(raw_values: str) -> list[float]:
    try:
        values = [float(value.strip()) for value in raw_values.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError("Parameter lists must be comma-separated numbers.") from exc

    if not values:
        raise ValueError("At least one parameter value is required.")
    return values


def parse_ticker_values(raw_values: str) -> list[str]:
    values = [value.strip().upper() for value in raw_values.split(",") if value.strip()]
    unique_values = list(dict.fromkeys(values))
    if not unique_values:
        raise ValueError("At least one ticker is required.")
    return unique_values


def parse_period_values(raw_values: str) -> list[str]:
    values = [value.strip() for value in raw_values.split(",") if value.strip()]
    unique_values = list(dict.fromkeys(values))
    if not unique_values:
        raise ValueError("At least one period is required.")
    if any(value not in SUPPORTED_PERIODS for value in unique_values):
        raise ValueError("Unsupported period list.")
    return unique_values
