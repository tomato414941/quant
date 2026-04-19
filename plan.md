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
- runtime の legacy fallback は `portfolio.py` 先頭の helper にかなり集約済み
- `evaluate_strategy_definition_run` が multi-timeframe / predictor / multi-selection を通す
- run store は `v58` で strategy / condition / parameter sweep の `strategyDefinition` を正本化した
- 外部入口は `comparison` 系に統一済み
- 現在の全体テスト: `105 passed`

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

### In Progress

19. run store を `strategy definition fingerprint + data fingerprint + logicVersion` ベースへ寄せる
状況:
- runSpec に fingerprint を保存済み
- compact/index payload でも fingerprint を露出済み
- API / CLI で fingerprint filter を利用可能
- `_index.json` による index 化と index-only 一覧は実装済み
- `rebuild-run-index` で復旧できる
- fingerprint 条件から最新 run を直接引く lookup を API / CLI で利用できる
- condition sweep / parameter sweep でも run spec に strategy definition を保存する
- `RUN_STORE_LOGIC_VERSION` は `v58`
- ただし predictor / ranking run は strategyDefinition 以外の strategy payload をまだ使う

20. 評価条件と CLI を正本化し、同条件で再検証できる状態にする
状況:
- `comparison-run-spec` を CLI / API から取得できる
- canonical payload に strategy definitions / condition variants / comparison fingerprint を含める
- `rerun-comparison-spec` で保存済み JSON から同条件再実行できる
- rerun 時に `runSpecFingerprint` / `comparisonFingerprint` の整合性を検証できる
- strategy / condition / parameter sweep の run spec が strategy definition を持つ
- rerun は definition-only payload を前提にする
- ただし predictor / ranking run の payload 正本化はまだ残る

## Remaining Duplication

いま残っている主な二重構造は次の通り。

- low-level evaluator 内部に残る `StrategySpec` DTO bridge
- runtime 境界に残る最終的な `extensions` fallback
- predictor / ranking run の run spec がまだ strategyDefinition 正本ではない点

外部入口の二重化はほぼ解消済み。

- API は `/api/comparison`
- CLI は `comparison-summary`
- candidate catalog の production 正本は strategy definitions

## Near-Term Next Steps

1. predictor / ranking run の run spec も definition-aware に整理する
2. low-level evaluator の `StrategySpec` DTO bridge をさらに薄くする
3. 残っている `extensions` fallback を legacy helper の外から見えない形まで縮める
4. run store を fingerprint-first な検索・再利用モデルへさらに寄せ切る
5. `plan.md` の完了条件を step ごとにより厳密に固定する
