# Evaluation Plan

この文書は、今回評価するStrategy setとEvaluation Contextを定義する。
初めて読む場合は、この文書で「何を評価するか」を確認し、結果の読み方は
[ETF Full-Universe No-Signal Allocation](./leaderboard/etf-full-universe-no-signal-allocation.md) の手動で記録した比較表を見る。

## Questions This Answers

- 今回どのStrategy setを評価するか？
- どのEvaluation Contextで評価するか？
- 次にどの結果をLeaderboardへ記録するか？

## Questions This Does Not Answer

- このプロジェクトに存在する全Strategyは何か？
- Evaluation Context候補は何か？
- 評価結果の順位はどうか？
- Strategy x Evaluation Contextの全ケース一覧は何か？

詳細は次の文書を参照する。

- 全Strategy: [Strategy Catalog](./strategy-catalog.md)
- 評価Context: [Evaluation Contexts](./evaluation-contexts.md)
- 評価済みrunの比較: [ETF Full-Universe No-Signal Allocation](./leaderboard/etf-full-universe-no-signal-allocation.md)

## Current Evaluation Set

| Field | Value |
| --- | --- |
| Evaluation Set ID | `no_signal_allocation` |
| Purpose | signalなしのETF配分モデルを比較する |
| Strategies | `stg-fu-eq`, `stg-fu-rb`, `stg-fu-minvar`, `stg-fu-hrp` |
| Evaluation Contexts | `2015_2025` |
| Status | evaluated |

Generated outputs are not tracked in this document. Local matrix files and raw
helper outputs belong under ignored paths such as `backend/data/`; selected
reproducibility records go to [Full-Period Backtest Results](./backtest-results.md),
and comparison summaries go to [ETF Full-Universe No-Signal Allocation](./leaderboard/etf-full-universe-no-signal-allocation.md).

## Next Evaluation Order

1. 追加評価が必要になった場合だけ、対象のStrategy setとEvaluation Contextを明示する。
2. 実行結果は再現性ログか手動で記録した比較表として記録する。

Leaderboardへの記録は手動で行う。評価runの保存やhelper出力が存在しても、
この文書で選んだEvaluation SetとContextだけを混ぜずに転記する。

## Code Helper Relationship

コード側のevaluation matrix helperは、保存済みrunの突き合わせを補助するための互換ビューである。

評価対象Strategy setや評価Contextの追加、削除、優先順位変更は、まずdocsを更新してからコード側へ反映する。
