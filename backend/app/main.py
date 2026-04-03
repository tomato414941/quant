from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.strategy import generate_demo_prices, parse_uploaded_prices, run_backtest


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


@app.get("/api/backtest/demo")
def backtest_demo(
    threshold: float = 0.03,
    initial_capital: float = 10_000,
) -> dict:
    try:
        return run_backtest(
            prices=generate_demo_prices(),
            threshold=threshold,
            initial_capital=initial_capital,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/backtest/upload")
async def backtest_upload(
    file: UploadFile = File(...),
    threshold: float = Form(0.03),
    initial_capital: float = Form(10_000),
) -> dict:
    try:
        contents = await file.read()
        prices = parse_uploaded_prices(contents)
        return run_backtest(
            prices=prices,
            threshold=threshold,
            initial_capital=initial_capital,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
