# Quant Refactor Plan

## Goal

このプロジェクトの最終形は、`1つの戦略 = 1つの時間軸` ではなく、
`1つの戦略 = 複数 signal / feature / predictor / execution rule の組み合わせ`
として表現・実行・比較できる状態にすること。

年次・月次・週次・日次を特別扱いせず、同じ構造の中でフラットに扱う。
中心となる概念は次の4つ。

- `data_timeframe`
- `signal_timeframe`
- `decision_schedule`
- `rebalance_schedule`

これにより、

- 日足価格を使う
- 週次 signal を計算する
- 月次 macro を補助 signal に使う
- 判断は毎日
- 売買は月末

のような構成を自然に表現できるようにする。

## Current Snapshot

- 正本の戦略定義は `StrategyDefinition`
- canonical candidate catalog の正本は strategy definitions
- `comparison_service` は `StrategyDefinition` を直接実行入口として扱う
- `StrategySpec` は公開正本ではなく、低レベル evaluator 用の内部 DTO として残る
- `comparison_service` は strategy / condition / parameter / predictor / ranking の runSpec を `StrategyDefinition` 正本で生成する
- `executionSupport` は `strategySpecAdapter*` と `directExecution*` で実行互換性を表す
- runtime の `extensions` fallback は通常実行経路から削除済み
- `evaluate_strategy_definition_run` が multi-timeframe / predictor / multi-selection を通す
- run store は `v59` で `strategyDefinition` と `evaluationSubject` の fingerprint を正本化した
- strategy / condition / parameter / predictor / ranking run は新規 runSpec で legacy `strategy` payload を生成しない
- predictor / ranking run は由来を `strategyDefinition`、評価対象を `evaluationSubject` として分離する
- 外部入口は `comparison` 系に統一済み
- 現在の全体テスト: `107 passed`

## 21 Steps

### Done

1. `StrategyDefinition` を導入する
2. `StrategyExecutionPlanSpec` を導入する
3. `StrategySignalSpec` を導入する
4. `legacy StrategySpec -> definition` 変換を実装する
5. `definition -> legacy StrategySpec` 変換を実装する
6. legacy adapter の互換判定を独立させる
7. `ComparisonSpec` が definition を受けられるようにする
8. `comparison_service` が definition を正規化して実行できるようにする
9. candidate 定義を `definition first` に移行する
10. baseline / canonical catalog を definition 正本に寄せる
11. `selection × portfolio_model × execution` の product builder を作る
12. predictor を signal として builder に取り込む
13. `DEFAULT_COMPARISON_SPEC` を definition 起点にする
14. API payload で definition と互換情報を見えるようにする
15. `FeatureDefinition` と `DataSourceDefinition` を分ける
16. 異粒度データを結合する `alignment_policy` を導入する
21. `StrategyBlueprintSpec` を `StrategyDefinition` にリネームする

17. signal ごとに `data_timeframe` と `signal_timeframe` を自然に実行できるようにする
18. legacy adapter を通さず comparison layer が definition を直接評価する経路を作る
19. run store を `strategyDefinition + evaluationSubject + marketData + evaluation + logicVersion` fingerprint ベースへ寄せる
20. 評価条件と CLI を正本化し、同条件で再検証できる状態にする

### Done Notes

Step 20 の完了内容:
- `comparison-run-spec` を CLI / API から取得できる
- canonical payload に strategy definitions / condition variants / comparison fingerprint を含める
- `rerun-comparison-spec` で保存済み JSON から同条件再実行できる
- rerun 時に `runSpecFingerprint` / `comparisonFingerprint` の整合性を検証できる
- strategy / condition / parameter / predictor / ranking run の run spec が strategy definition を持つ
- predictor / ranking run は `evaluationSubject` で実評価対象を再現可能にした
- rerun は definition-only payload を前提にする
- `StrategySpec.extensions` は runtime context の復元元ではない
- `comparison_service` から `StrategySpec` 変換の直接 import を削除済み
- parameter sweep は `StrategySpec` を経由せず `StrategyDefinition` を直接生成する
- ranking 用の evaluator DTO 変換は `portfolio.py` の helper に閉じた
- `legacyAdapter*` 表示は削除し、`strategySpecAdapter*` 表示へ一本化した
- 個別 subject fingerprint filter は必要になった段階で追加する

### In Progress

なし。次は step 22 相当として、低レベル evaluator 内部の `StrategySpec` DTO bridge をさらに薄くする。

## Remaining Duplication

いま残っている主な二重構造は次の通り。

- low-level evaluator 内部に残る `StrategySpec` DTO bridge

外部入口の二重化はほぼ解消済み。

- API は `/api/comparison`
- CLI は `comparison-summary`
- candidate catalog の production 正本は strategy definitions

## Near-Term Next Steps

1. step 22 相当として、low-level evaluator 内部の `StrategySpec` DTO bridge をさらに薄くする
2. 必要になった段階で `evaluationSubjectFingerprint` filter を API / CLI に追加する
3. 新しい構造で戦略探索・比較の改善に戻る
