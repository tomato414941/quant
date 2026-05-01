# Leaderboard

この文書は、評価済みrunを人間が比較するための手動スナップショットである。

CLI/API機能の仕様ではない。今の段階ではコードを増やさず、比較の読み方だけをdocsで試す。

公開リポジトリ上では、ここを「最新runの自動集計」ではなく「選んだrunを手で記録した比較表」として読む。
再現や追加検証に進む前に、対象のEvaluation Contextと更新日を確認する。

Raw matrix outputs do not belong here. Keep full generated tables under ignored
paths such as `backend/data/`, then copy only the decision-relevant rows,
ranking, and interpretation into this document.

## Questions This Answers

- 評価済みrunをどう並べて見るか？
- 次に深掘りするStrategy候補はどれか？
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
- この文書は議論の入口であり、最終採用判断ではない。
- 生成された全量表をそのまま貼らず、比較判断に必要な要約だけを載せる。
- 次に深掘りする採用候補はここに置く。最終採用判断は将来の
  decision note で別に記録する。

## ETF 2015-2025

### Fixed-Snapshot Backtest-Strategy Matrix

Last updated: 2026-04-30

Source: selected rows from local generated matrix
`backend/data/backtest-matrix-yfinance-only-etf.md`. The full matrix is an
ignored local artifact, not a committed documentation source of truth.

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

Decision summary:

| Role | Strategy ID | Sharpe | CAGR | Max DD | Turnover | Note |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Baseline | `stg-fu-eq` | 0.69 | 7.16% | 24.99% | 100.00% | Highest CAGR in this matrix, but largest drawdown among the selected rows. |
| Main candidate | `stg-fu-rb-month` | 0.87 | 4.45% | 12.29% | 383.75% | Better Sharpe/drawdown balance than equal weight with moderate rebalance frequency. |
| Main candidate | `stg-fu-rb-week` | 0.89 | 4.46% | 11.98% | 465.69% | Slightly better Sharpe/MDD than monthly, with higher turnover. |
| Low-drawdown reference | `stg-fu-minvar-month` | 0.82 | 2.82% | 7.22% | 768.62% | Strong drawdown control, but low CAGR and high turnover. |
| Middle reference | `stg-fu-hrp-week` | 0.79 | 3.54% | 11.50% | 1295.91% | Middle risk profile, but turnover is high. |

Interpretation:

- Primary candidate: `stg-fu-rb-week`.
- Conservative candidate: `stg-fu-rb-month`.
- `stg-fu-eq` remains the baseline because it has the highest CAGR, but its
  drawdown is materially larger.
- Minimum-variance variants reduce drawdown, but the return tradeoff is large in
  this context.
- Daily variants are not current main candidates because turnover is high:
  `stg-fu-rb-day` 636.78%, `stg-fu-minvar-day` 1620.85%, and `stg-fu-hrp-day`
  2174.52%.

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

Candidate cost points:

| Transaction Cost | Label | Intended Reading |
| ---: | --- | --- |
| `0.00025` | Low | Liquid ETF / low-slippage assumption. |
| `0.0005` | Base | Practical baseline candidate for liquid US ETF trading. |
| `0.001` | Conservative | Current fixed-snapshot matrix assumption. |
| `0.002` | Stress | High-cost stress case, not a normal baseline. |

| Transaction Cost | Main Candidate | Sharpe | CAGR | Max DD | Turnover | Note |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| `0.00025` | `stg-fu-rb-week` | 0.89 | 4.50% | 11.97% | 465.69% | `stg-fu-rb-month`: Sharpe 0.88, CAGR 4.47%, MDD 12.30%, turnover 383.75%. |
| `0.0005` | `stg-fu-rb-week` | 0.89 | 4.49% | 11.97% | 465.69% | `stg-fu-rb-month`: Sharpe 0.88, CAGR 4.46%, MDD 12.29%, turnover 383.75%. |
| `0.001` | `stg-fu-rb-week` | 0.89 | 4.46% | 11.98% | 465.69% | `stg-fu-rb-month`: Sharpe 0.87, CAGR 4.45%, MDD 12.29%, turnover 383.75%. |
| `0.002` | `stg-fu-rb-week` | 0.88 | 4.42% | 12.00% | 465.69% | `stg-fu-rb-month`: Sharpe 0.86, CAGR 4.41%, MDD 12.30%, turnover 383.75%. |

Interpretation:

- `stg-fu-rb-week` stays slightly ahead of `stg-fu-rb-month` across the tested
  cost range.
- The advantage is small, while weekly turnover is consistently higher.
- Keep `stg-fu-rb-week` as the primary candidate and `stg-fu-rb-month` as the
  conservative candidate.
- Run walk-forward stability only if another validation gate is needed before a
  stronger adoption decision.

## ETF Walk-Forward 2020-2025

Last updated: TBD

| Rank | Strategy ID | Average Sharpe | Average CAGR | Worst Max DD | Stability | Note |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| TBD | TBD | TBD | TBD | TBD | TBD | TBD |
