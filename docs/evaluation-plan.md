# Evaluation Plan

この文書は、今回どの `Strategy` をどの `Evaluation Context` で評価するかを管理する。

## Questions This Answers

- 今回の評価対象Strategy setはどれか？
- 今回追跡するEvaluation Contextはどれか？
- 次にどの評価を埋めるか？

## Questions This Does Not Answer

- このプロジェクトに存在する全Strategyは何か？
- Evaluation Context候補は他に何があるか？
- どのStrategyが最終的に最良か？

詳細は次の文書を参照する。

- 全Strategy: [Strategy Catalog](./strategy-catalog.md)
- 評価条件: [Evaluation Contexts](./evaluation-contexts.md)
- 評価済みrunの比較: [Leaderboard](./leaderboard.md)

## Current Plan

現在の計画は、`active 10 strategies x tracked 3 evaluation contexts` を埋めることである。

Strategy set:
- Source: [Strategy Catalog](./strategy-catalog.md)
- Filter: `status = active`
- Count: 10

Evaluation contexts:
- Source: [Evaluation Contexts](./evaluation-contexts.md)
- Count: 3

## Evaluation Matrix

`missing` は「その評価条件に一致する保存済みrunが見つからない」という意味であり、戦略が失敗したという意味ではない。

| Strategy ID | `etf_2015_2025` | `etf_cost_2x_2015_2025` | `etf_walk_forward_2020_2025` |
| --- | --- | --- | --- |
| `stg-fu-eq` | missing | missing | missing |
| `stg-fu-hrp` | missing | missing | missing |
| `stg-fu-momo12-top035-hrp` | missing | missing | missing |
| `stg-fu-momolv8515-top025-hrp-month` | missing | missing | missing |
| `stg-top3-hrp` | missing | missing | missing |
| `stg-dualtop3-hrp` | missing | missing | missing |
| `stg-posmom-hrp-month` | missing | missing | missing |
| `stg-riskoff-posmom-hrp-month` | missing | missing | missing |
| `stg-fu-momo2-top035-hrp-month` | missing | missing | missing |
| `stg-fu-momo2-top035-pred10mom5050-pw40-hrp-month` | missing | missing | missing |

## Run Status Meaning

- `missing`: 評価未実行、または保存済みrunが評価条件に一致していない
- `available`: 評価条件に一致する保存済みrunが存在する
- `accepted`: 評価結果を確認し、次の比較対象として採用する
- `rejected`: 評価結果を確認し、現時点では採用しない
- `deferred`: 評価結果だけでは判断せず、追加検証へ回す

## Next Evaluation Order

評価は次の順序で埋める。

1. `etf_2015_2025` をactive 10戦略で埋める。
2. `etf_cost_2x_2015_2025` をactive 10戦略で埋め、コスト感応度を見る。
3. `etf_walk_forward_2020_2025` をactive 10戦略で埋め、時間安定性を見る。

評価結果の良し悪しは、少なくとも基本評価とコスト感応度を並べてから判断する。1つの条件だけで構造的に不可能とは結論づけない。

## Code Helper Relationship

`strategy-inventory --with-evaluation-matrix` は、この文書の評価対象と保存済みrunの突き合わせを補助するための確認コマンドである。

評価条件の追加、削除、優先順位変更は、まずdocsを更新してからコード側へ反映する。
