from __future__ import annotations


DEFAULT_DASHBOARD_CONFIG = {
    "ticker": "SPY",
    "period": "2y",
    "period_compare_periods": ["6mo", "1y", "2y", "3y", "5y"],
    "comparison_tickers": ["SPY", "QQQ", "IWM", "TLT", "GLD", "BTC-USD"],
    "strategy": "mean_reversion",
    "threshold": 0.03,
    "holding_days": 1,
    "split_ratio": 0.7,
    "initial_capital": 10_000,
    "transaction_cost": 0.001,
    "threshold_grid": [0.015, 0.02, 0.03, 0.04, 0.05],
    "holding_days_grid": [1, 2, 3],
}
