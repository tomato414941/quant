# ETF Full-Universe No-Signal Allocation

この文書は、評価済みrunを人間が比較するための手動スナップショットである。

CLI/API機能の仕様ではない。今の段階ではコードを増やさず、比較の読み方だけをdocsで試す。

公開リポジトリ上では、ここを「最新runの自動集計」ではなく「選んだrunを手で記録した比較表」として読む。
再現や追加検証に進む前に、対象のEvaluation Contextと更新日を確認する。

Raw matrix outputs do not belong here. Keep generated files under ignored paths
such as `backend/data/`, then copy the comparison table and source metadata that
are useful to read by hand.

## Questions This Answers

- 評価済みrunをどう並べて見るか？
- 比較時にどの指標を同時に見るべきか？

## Questions This Does Not Answer

- 最新runを自動集計する方法は何か？
- 最終採用Strategyはどれか？
- 異なるEvaluation Contextを混ぜた総合順位は何か？

## Reading Rules

- Evaluation Contextを混ぜて順位付けしない。
- Sharpe単独で判断しない。
- CAGR、max drawdown、turnover、baseline差分も見る。
- 手動スナップショットなので更新日を書く。
- 自動更新ではないため、表が最新runを網羅しているとは限らない。
- この文書は比較メモであり、最終採用判断ではない。
- 生成された全量出力をそのまま貼らず、人間が読む表として整える。

## ETF 2015-2025

Last updated: 2026-04-30

Source: local generated matrix
`backend/data/backtest-matrix-yfinance-only-etf.md`. The generated file is an
ignored local artifact; the table below is the committed manual snapshot.

Snapshot:

| Field | Value |
| --- | --- |
| Snapshot ID | `580d7527df6c22a918306b55a29b7c756bd05480094aec732dc3d67d2d48ce6a` |
| Source | Yahoo Finance via yfinance |
| Period | `2015-01-02` to `2025-12-31` |
| Timeframe | `1d` |
| Rows | `2766` |
| Transaction cost | `0.001` |

Reproduction command:

```bash
cd backend
uv run python ../scripts/backtest_matrix.py \
  --snapshot-id 580d7527df6c22a918306b55a29b7c756bd05480094aec732dc3d67d2d48ce6a \
  --market-snapshot-dir data/market_snapshots_yfinance_only_etf \
  --transaction-cost 0.001 \
  --output data/backtest-matrix-yfinance-only-etf.md
```

Ranked by Sharpe:

| Rank | Strategy ID | Return | CAGR | Sharpe | Max DD | Turnover | Events |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `stg-fu-rb-day` | 62.25% | 4.51% | 0.91 | 11.83% | 636.78% | 2765 |
| 2 | `stg-fu-rb-week` | 61.46% | 4.46% | 0.89 | 11.98% | 465.69% | 574 |
| 3 | `stg-fu-minvar-day` | 36.41% | 2.87% | 0.88 | 6.72% | 1620.85% | 2765 |
| 4 | `stg-fu-rb-month` | 61.17% | 4.45% | 0.87 | 12.29% | 383.75% | 132 |
| 5 | `stg-fu-minvar-week` | 35.84% | 2.83% | 0.85 | 7.52% | 1049.83% | 574 |
| 6 | `stg-fu-minvar-month` | 35.65% | 2.82% | 0.82 | 7.22% | 768.62% | 132 |
| 7 | `stg-fu-hrp-week` | 46.54% | 3.54% | 0.79 | 11.50% | 1295.91% | 574 |
| 8 | `stg-fu-hrp-day` | 46.49% | 3.54% | 0.79 | 11.19% | 2174.52% | 2765 |
| 9 | `stg-fu-rb` | 57.99% | 4.26% | 0.75 | 14.32% | 226.10% | 11 |
| 10 | `stg-fu-hrp-month` | 45.49% | 3.48% | 0.75 | 12.31% | 840.75% | 132 |
| 11 | `stg-fu-eq` | 113.63% | 7.16% | 0.69 | 24.99% | 100.00% | 11 |
| 12 | `stg-fu-eq-month` | 113.63% | 7.16% | 0.69 | 24.99% | 100.00% | 132 |
| 13 | `stg-fu-eq-week` | 113.63% | 7.16% | 0.69 | 24.99% | 100.00% | 574 |
| 14 | `stg-fu-eq-day` | 113.63% | 7.16% | 0.69 | 24.99% | 100.00% | 2765 |
| 15 | `stg-fu-hrp` | 47.46% | 3.60% | 0.67 | 13.51% | 375.35% | 11 |
| 16 | `stg-fu-minvar` | 34.65% | 2.75% | 0.64 | 12.81% | 407.18% | 11 |

Interpretation:

- `stg-fu-rb-day` has the highest Sharpe in this snapshot, but daily rebalance
  drives high turnover and event count.
- `stg-fu-rb-week` and `stg-fu-rb-month` are close on Sharpe/CAGR/drawdown, with
  weekly taking more turnover than monthly.
- Equal-weight variants have the highest CAGR, but materially larger drawdown.
- Minimum-variance variants reduce drawdown, but the return tradeoff is large in
  this context.
- HRP variants sit in the middle on drawdown, but turnover is high for weekly
  and daily schedules.

### Saved Comparison-Summary Runs

Last updated: 2026-04-27

Source: latest matching local saved `strategy_run` records for `period=2015_2025`, non-walk-forward, `maxWeightPct=45.0`, saved at 2026-04-21 00:07:51-00:08:04 UTC.

Re-run command attempted:

```bash
cd backend
uv run python -m app comparison-summary --universe only_etf --strategy-key stg-fu-eq --strategy-key stg-fu-rb --strategy-key stg-fu-minvar --strategy-key stg-fu-hrp --top 4 --json
```

Live re-run blocker: yfinance returned empty frames for every requested ETF in this environment, ending with `ValueError: At least two tickers with valid market data are required`. The table below uses already-saved local run results, not newly downloaded live data.

Dataset snapshot id: unavailable in these saved run records. Market data fingerprint for the source batch: `b1c8f1c100dacba02ffe8b1f641f72c3c5e737e45bd7a029a994aa8d5651c2b3`.

Fallback rate: unavailable in the saved compact metrics for this batch.

| Rank | Strategy ID | Sharpe | CAGR | Max DD | Turnover | Fallback Rate | Dataset Snapshot ID | Note |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | `stg-fu-minvar` | 1.55 | 5.30% | 3.09% | 147.52% | N/A | N/A | Run file `72dc5d5b069cd7f08e05f6ef55a98e0ee28436043211a6112803fcd8b9d6a7b5.json`; market data fingerprint above. |
| 2 | `stg-fu-rb` | 1.48 | 7.24% | 4.47% | 82.06% | N/A | N/A | Run file `b57a18b4cfc3569f32fa26d28641d9b4d93ebe73794247299ecbe3d5f8e26477.json`; market data fingerprint above. |
| 3 | `stg-fu-hrp` | 1.39 | 6.04% | 4.47% | 144.06% | N/A | N/A | Run file `4e83759d5a366c7193ddb45a9d4377219911d31c48595481c8fabe3c3ed851ad.json`; market data fingerprint above. |
| 4 | `stg-fu-eq` | 1.19 | 12.26% | 9.39% | 23.50% | N/A | N/A | Run file `22b4e8f5b7e5b4284caaf3f2f42521400339a3d0f7969fa31c842347d06e1e5b.json`; market data fingerprint above. |

## ETF 2015-2025 Cost Sensitivity

Last updated: 2026-04-30

Transaction cost is modeled as a flat proportional cost per traded notional. It
is a simplified proxy for commissions, bid-ask spread, and slippage. It does not
model ETF-specific liquidity, order size, taxes, or expense ratios.

Cost points:

| Transaction Cost | Label | Intended Reading |
| ---: | --- | --- |
| `0.00025` | Low | Liquid ETF / low-slippage assumption. |
| `0.0005` | Base | Practical baseline for liquid US ETF trading. |
| `0.001` | Conservative | Current fixed-snapshot matrix assumption. |
| `0.002` | Stress | High-cost stress case, not a normal baseline. |

| Transaction Cost | Risk Budgeting Weekly | Sharpe | CAGR | Max DD | Turnover | Monthly Comparison |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| `0.00025` | `stg-fu-rb-week` | 0.89 | 4.50% | 11.97% | 465.69% | `stg-fu-rb-month`: Sharpe 0.88, CAGR 4.47%, MDD 12.30%, turnover 383.75%. |
| `0.0005` | `stg-fu-rb-week` | 0.89 | 4.49% | 11.97% | 465.69% | `stg-fu-rb-month`: Sharpe 0.88, CAGR 4.46%, MDD 12.29%, turnover 383.75%. |
| `0.001` | `stg-fu-rb-week` | 0.89 | 4.46% | 11.98% | 465.69% | `stg-fu-rb-month`: Sharpe 0.87, CAGR 4.45%, MDD 12.29%, turnover 383.75%. |
| `0.002` | `stg-fu-rb-week` | 0.88 | 4.42% | 12.00% | 465.69% | `stg-fu-rb-month`: Sharpe 0.86, CAGR 4.41%, MDD 12.30%, turnover 383.75%. |

Interpretation:

- `stg-fu-rb-week` stays slightly ahead of `stg-fu-rb-month` across the tested
  cost range.
- The advantage is small, while weekly turnover is consistently higher.
- This is a cost sensitivity read, not a final conclusion.
