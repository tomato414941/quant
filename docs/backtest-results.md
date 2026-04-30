# Full-Period Backtest Results

この文書は、固定market snapshotで再現できるfull-period backtestの手動記録である。固定snapshot、command、input、output summaryを記録する再現性ログとして扱う。

Leaderboardではない。Strategy間の順位付けではなく、「このsnapshotと条件ではこの結果になる」ことを確認するための再現メモとして扱う。

Scope:

- 代表的な結果、回帰確認に使う結果、判断の根拠になる結果だけを残す。
- すべての一括実行結果を貼り付けない。16 strategy matrix などの全量出力は
  `backend/data/` 配下の ignored local output として扱う。
- 記録する場合は、snapshot ID、snapshot dir、strategy key、cost、再現コマンド、
  参照したJSON fieldを併記する。
- 採用候補や順位付けの読み取りは `docs/leaderboard.md` に置く。

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

Source fields in the `backtest-strategy` JSON payload:

| Metric | JSON field |
| --- | --- |
| First invested date | `result.firstInvestedDate` |
| Total return | `result.summary.totalReturnPct` |
| CAGR | `result.summary.cagrPct` |
| Sharpe | `result.summary.sharpeRatio` |
| Max drawdown | `result.summary.maxDrawdownPct` |
| Turnover | `result.summary.turnoverPct` |
| Series count | `result.seriesCount` |
| Event count | `result.eventCount` |

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

## Risk Model Variants / Golden Snapshot

Last updated: 2026-04-30

Purpose: risk model variants の full-period backtest が committed golden snapshot から再現できることを記録する。

Command:

```bash
cd backend
uv run quant backtest-strategy \
  --strategy-key <strategy-key> \
  --snapshot-id 0d1aaa50ac7a9f4c7e82ee584c7a1bb0a6ce27f9434bf1884bcaee9a4432cd88 \
  --market-snapshot-dir tests/fixtures/golden_market/market_snapshots \
  --rebalance-schedule hold \
  --transaction-cost 0
```

Input:

| Field | Value |
| --- | --- |
| Strategies | `stg-fu-rb`, `stg-fu-minvar`, `stg-fu-hrp` |
| Backtest kind | `full_period_backtest` |
| Snapshot ID | `0d1aaa50ac7a9f4c7e82ee584c7a1bb0a6ce27f9434bf1884bcaee9a4432cd88` |
| Source | `test` |
| Timeframe | `1d` |
| Start date | `2025-01-01` |
| End date | `2025-01-07` |
| Row count | `7` |
| Ticker count | `18` |
| Rebalance schedule | `hold` |
| Transaction cost | `0` |

Output summary:

| Strategy | First invested date | Return | CAGR | Sharpe | Max DD | Turnover | Events |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `stg-fu-rb` | `2025-01-02` | `4.66%` | `578.07%` | `36.8` | `0.0%` | `100.0%` | `1` |
| `stg-fu-minvar` | `2025-01-02` | `4.66%` | `578.07%` | `36.8` | `0.0%` | `100.0%` | `1` |
| `stg-fu-hrp` | `2025-01-02` | `4.66%` | `578.07%` | `36.8` | `0.0%` | `100.0%` | `1` |

Notes:

- The golden snapshot is intentionally tiny and has only one return row at the
  first rebalance.
- These variants therefore record
  `allocationFallback.reason=insufficient_history` and use equal-weight fallback
  for this fixture.
