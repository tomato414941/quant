from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeframeSpec:
    key: str
    label: str
    yfinance_interval: str
    bar_seconds: int
    bars_per_year: float


def build_timeframe_spec(
    *,
    key: str,
    label: str,
    yfinance_interval: str,
    bar_seconds: int,
    bars_per_year: float,
) -> TimeframeSpec:
    return TimeframeSpec(
        key=key,
        label=label,
        yfinance_interval=yfinance_interval,
        bar_seconds=bar_seconds,
        bars_per_year=float(bars_per_year),
    )


DEFAULT_DAILY_TIMEFRAME = build_timeframe_spec(
    key="1d",
    label="日次",
    yfinance_interval="1d",
    bar_seconds=86_400,
    bars_per_year=252,
)

DEFAULT_WEEKLY_TIMEFRAME = build_timeframe_spec(
    key="1w",
    label="週次",
    yfinance_interval="1wk",
    bar_seconds=604_800,
    bars_per_year=52,
)

DEFAULT_MONTHLY_TIMEFRAME = build_timeframe_spec(
    key="1mo",
    label="月次",
    yfinance_interval="1mo",
    bar_seconds=2_592_000,
    bars_per_year=12,
)
