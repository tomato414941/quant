# Leaderboard

この文書は、評価済みrunを人間が比較するための手動スナップショットである。

CLI/API機能の仕様ではない。今の段階ではコードを増やさず、比較の読み方だけをdocsで試す。

公開リポジトリ上では、ここを「最新runの自動集計」ではなく「選んだrunを手で記録した比較表」として読む。
再現や追加検証に進む前に、対象のEvaluation Contextと更新日を確認する。

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

## ETF 2015-2025

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

## ETF 2015-2025 Cost x2

Last updated: TBD

| Rank | Strategy ID | Sharpe | CAGR | Max DD | Turnover | Baseline Delta | Note |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## ETF Walk-Forward 2020-2025

Last updated: TBD

| Rank | Strategy ID | Average Sharpe | Average CAGR | Worst Max DD | Stability | Note |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| TBD | TBD | TBD | TBD | TBD | TBD | TBD |
