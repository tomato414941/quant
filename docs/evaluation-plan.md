# Evaluation Plan

この文書は、今回評価するStrategy setとEvaluation Contextを定義する。
初めて読む場合は、この文書で「何を評価するか」を確認し、結果の読み方は
[Leaderboard](./leaderboard.md) の手動スナップショットを見る。

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
- 評価済みrunの比較: [Leaderboard](./leaderboard.md)

## Current Evaluation Set

| Field | Value |
| --- | --- |
| Evaluation Set ID | `no_signal_allocation_v1` |
| Purpose | signalなしのETF配分モデルを比較する |
| Strategies | `stg-fu-eq`, `stg-fu-rb`, `stg-fu-minvar`, `stg-fu-hrp` |
| Evaluation Contexts | `2015_2025` |
| Status | planned |

## Next Evaluation Order

1. `no_signal_allocation_v1` を `2015_2025` で評価する。
2. 結果を [Leaderboard](./leaderboard.md) に手動スナップショットとして記録する。
3. 必要になったらcost sensitivityやwalk-forwardをEvaluation Context候補から正式化する。

Leaderboardへの記録は手動で行う。評価runの保存やhelper出力が存在しても、
この文書で選んだEvaluation SetとContextだけを混ぜずに転記する。

## Code Helper Relationship

コード側のevaluation matrix helperは、保存済みrunの突き合わせを補助するための互換ビューである。

評価対象Strategy setや評価Contextの追加、削除、優先順位変更は、まずdocsを更新してからコード側へ反映する。
