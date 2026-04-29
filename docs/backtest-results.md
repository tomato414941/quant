# Full-Period Backtest Results

この文書は、固定market snapshotで再現できるfull-period backtestの手動記録である。

Leaderboardではない。Strategy間の順位付けではなく、「このsnapshotと条件ではこの結果になる」ことを確認するための再現メモとして扱う。

## stg-fu-eq / ETF 2015-2025

Last updated: 2026-04-29

Purpose: `stg-fu-eq` の full-period backtest が固定snapshotから再現できることを記録する。

Command:

```bash
cd backend
uv run python -m app backtest-strategy \
  --strategy-key stg-fu-eq \
  --snapshot-id 580d7527df6c22a918306b55a29b7c756bd05480094aec732dc3d67d2d48ce6a \
  --market-snapshot-dir data/market_snapshots_yfinance_only_etf \
  --rebalance-schedule year_end \
  --transaction-cost 0.001
```

Input:

| Field | Value |
| --- | --- |
| Strategy | `stg-fu-eq` |
| Backtest kind | `full_period_backtest` |
| Snapshot ID | `580d7527df6c22a918306b55a29b7c756bd05480094aec732dc3d67d2d48ce6a` |
| Source | Yahoo Finance via yfinance |
| Timeframe | `1d` |
| Start date | `2015-01-02` |
| End date | `2025-12-31` |
| Row count | `2766` |
| Ticker count | `18` |
| Rebalance schedule | `year_end` |
| Transaction cost | `0.001` |

Tickers:

`SPY`, `QQQ`, `IWM`, `EFA`, `EEM`, `EWJ`, `EWZ`, `VNQ`, `TLT`, `IEF`, `LQD`, `HYG`, `TIP`, `GLD`, `SLV`, `DBC`, `USO`, `UUP`

Output summary:

| Metric | Value |
| --- | ---: |
| First invested date | `2015-01-05` |
| Total return | `113.63%` |
| CAGR | `7.16%` |
| Sharpe | `0.69` |
| Max drawdown | `24.99%` |
| Turnover | `100.0%` |
| Series count | `2765` |
| Event count | `11` |

Notes:

- This is not the same as `comparison-summary`.
- `comparison-summary` uses the existing comparison/run workflow.
- This result is the full-period `backtest-strategy` output for one supported strategy.
