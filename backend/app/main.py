from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.market_data import fetch_market_prices
from app.strategy import run_backtest, run_grid_search


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


@app.get("/api/backtest")
def backtest_market(
    ticker: str = "SPY",
    period: str = "2y",
    threshold: float = 0.03,
    initial_capital: float = 10_000,
    holding_days: int = 1,
) -> dict:
    try:
        prices, metadata = fetch_market_prices(ticker=ticker, period=period)
        payload = run_backtest(
            prices=prices,
            threshold=threshold,
            initial_capital=initial_capital,
            holding_days=holding_days,
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
) -> dict:
    try:
        prices, metadata = fetch_market_prices(ticker=ticker, period=period)
        payload = run_grid_search(
            prices=prices,
            thresholds=parse_percentage_values(threshold_values),
            holding_days_options=parse_integer_values(holding_days_values),
            initial_capital=initial_capital,
        )
        payload["dataset"] = metadata
        return payload
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
