# ETF Full-Universe No-Signal Allocation

この文書は、評価済みrunを人間が比較するために手動で記録した比較表である。

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
- 手動で記録した比較表なので更新日を書く。
- 自動更新ではないため、表が最新runを網羅しているとは限らない。
- この文書は比較メモであり、最終採用判断ではない。
- 生成された全量出力をそのまま貼らず、人間が読む表として整える。

## ETF 2015-2025

Last updated: 2026-04-30

Source: local generated matrix
`backend/data/backtest-matrix-yfinance-only-etf.md`. The generated file is an
ignored local artifact; the table below is the committed comparison record.

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

`Result ID` is a deterministic SHA-256 hash of the strategy result object plus
the market data snapshot ID and run assumptions.

| Rank | Result ID | Strategy ID | Return | CAGR | Sharpe | Max DD | Turnover | Events |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `0a31e8838a7c429c62afc386f8c341104a8da766d5fd9f91d46fdd76f8c9cacf` | `stg-fu-rb-day` | 62.25% | 4.51% | 0.91 | 11.83% | 636.78% | 2765 |
| 2 | `2129ef8d242600c8e6e9c6e1ae4274d73d36d7bd9a815737b8a1e0fa46943821` | `stg-fu-rb-week` | 61.46% | 4.46% | 0.89 | 11.98% | 465.69% | 574 |
| 3 | `ec49b091053c79e1862a27a6325e94f4bb841a22aab20b895ce85b7bf0b8d4e5` | `stg-fu-minvar-day` | 36.41% | 2.87% | 0.88 | 6.72% | 1620.85% | 2765 |
| 4 | `e98757a547ff67b27951ffa75efa528c76e36eda2917010a02e9fbc222f3428e` | `stg-fu-rb-month` | 61.17% | 4.45% | 0.87 | 12.29% | 383.75% | 132 |
| 5 | `10d72360d9fa7da38cedaa0b2c8c9ce23946ff1fa08905f4b28a8658216f31eb` | `stg-fu-minvar-week` | 35.84% | 2.83% | 0.85 | 7.52% | 1049.83% | 574 |
| 6 | `150daddfd9de9e8f82e5749e8aefff64a835be918fcfd6e5c255a08cd1e97111` | `stg-fu-minvar-month` | 35.65% | 2.82% | 0.82 | 7.22% | 768.62% | 132 |
| 7 | `37b60556375741e06414429bcd3d4d5e87a41acbafb1a6357c1be4f296168a35` | `stg-fu-hrp-week` | 46.54% | 3.54% | 0.79 | 11.50% | 1295.91% | 574 |
| 8 | `8053cb03577945c9892ff440919400d6046cf0363e440ec61a691232fbe567a0` | `stg-fu-hrp-day` | 46.49% | 3.54% | 0.79 | 11.19% | 2174.52% | 2765 |
| 9 | `8401632025546570da76448a3aecee1e529e1239168a2b8e523ba6b988833069` | `stg-fu-rb` | 57.99% | 4.26% | 0.75 | 14.32% | 226.10% | 11 |
| 10 | `0bcc01bb7e0b6bbc0113906ed9ad5cd9f793f77c4333f43891531e0de05c8123` | `stg-fu-hrp-month` | 45.49% | 3.48% | 0.75 | 12.31% | 840.75% | 132 |
| 11 | `118b6f2d0aac7bcad6bf912c35fc00872464379e01bfb5f79b391a252ea56c76` | `stg-fu-eq` | 113.63% | 7.16% | 0.69 | 24.99% | 100.00% | 11 |
| 12 | `ebc29481c7f3e1d051fe068f3a63f486f10832535ac5fae5ba7cfa6ca41bb3ae` | `stg-fu-eq-month` | 113.63% | 7.16% | 0.69 | 24.99% | 100.00% | 132 |
| 13 | `023fc2e4b596b36e4b72284499728002d2c89e735eb91e2d1b24d3d71774d3a6` | `stg-fu-eq-week` | 113.63% | 7.16% | 0.69 | 24.99% | 100.00% | 574 |
| 14 | `63d9bc6890561107cd7cd4fbd602d033b430b431c00dba6c40ac379050e74fbf` | `stg-fu-eq-day` | 113.63% | 7.16% | 0.69 | 24.99% | 100.00% | 2765 |
| 15 | `ddce5e04c7fbf057e5e64d75f440f317d081afe81a234a85a99f676413962404` | `stg-fu-hrp` | 47.46% | 3.60% | 0.67 | 13.51% | 375.35% | 11 |
| 16 | `cad3a8fc254e99d954ec2b90b549c2e28fbaf568decacbb0dee3d9c9f7908537` | `stg-fu-minvar` | 34.65% | 2.75% | 0.64 | 12.81% | 407.18% | 11 |

Interpretation:

- `stg-fu-rb-day` has the highest Sharpe in this comparison, but daily rebalance
  drives high turnover and event count.
- `stg-fu-rb-week` and `stg-fu-rb-month` are close on Sharpe/CAGR/drawdown, with
  weekly taking more turnover than monthly.
- Equal-weight variants have the highest CAGR, but materially larger drawdown.
- Minimum-variance variants reduce drawdown, but the return tradeoff is large in
  this context.
- HRP variants sit in the middle on drawdown, but turnover is high for weekly
  and daily schedules.

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
