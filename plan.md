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
- `comparison_service` は strategy definitions / normalized execution context を直接参照する
- `StrategySpec` は explicit execution context と `executionMode` を持ち、payload でも structured に見える
- runtime の legacy fallback は `portfolio.py` 先頭の helper にかなり集約済み
- direct execution は multi-timeframe / predictor / multi-selection のかなりの範囲を通せる
- run store は latest fingerprint lookup / rerun payload validation / sweep run spec の strategy definition 保存まで入った
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

### In Progress

17. signal ごとに `data_timeframe` と `signal_timeframe` を自然に実行できるようにする
状況:
- signal 単位の timeframe / alignment metadata は表現済み
- raw daily data と weekly/monthly signal の分離は一部実行済み
- predictor overlay も signal timeframe に揃えて評価できる
- multi-selection も direct execution でかなり通る
- ただし runtime はまだ完全な definition-native execution model ではない

18. legacy adapter を通さず definition を直接評価する経路を作る
状況:
- direct execution path は存在する
- `StrategySpec` の explicit execution context と `executionMode` は導入済み
- `StrategySpec <-> definition` の往復でも execution metadata をかなり保持できる
- ただしまだ完全な definition-native evaluator ではない
- runtime 境界には `extensions` fallback が残っているが、かなり legacy helper に閉じた

19. run store を `strategy spec fingerprint + data fingerprint + logicVersion` ベースへ寄せる
状況:
- runSpec に fingerprint を保存済み
- compact/index payload でも fingerprint を露出済み
- API / CLI で fingerprint filter を利用可能
- `_index.json` による index 化と index-only 一覧は実装済み
- `rebuild-run-index` で復旧できる
- fingerprint 条件から最新 run を直接引く lookup を API / CLI で利用できる
- condition sweep / parameter sweep でも run spec に strategy definition を保存する
- ただし run store 全体はまだ完全な fingerprint-first 運用にはなっていない

20. 評価条件と CLI を正本化し、同条件で再検証できる状態にする
状況:
- `comparison-run-spec` を CLI / API から取得できる
- canonical payload に strategy definitions / condition variants / comparison fingerprint を含める
- `rerun-comparison-spec` で保存済み JSON から同条件再実行できる
- rerun 時に `runSpecFingerprint` / `comparisonFingerprint` の整合性を検証できる
- strategy / condition / parameter sweep の run spec が strategy definition を持つため、再検証材料はかなり揃った
- ただし definition-native evaluator と完全に一体化した再検証運用にはまだ届いていない

## Remaining Duplication

いま残っている主な二重構造は次の通り。

- direct execution bridge と definition-native execution の二重経路
- runtime 境界に残る最終的な `extensions` fallback
- run store が definition fingerprint ではなく legacy payload 依存な点

外部入口の二重化はほぼ解消済み。

- API は `/api/comparison`
- CLI は `comparison-summary`
- candidate catalog の production 正本は strategy definitions

## Near-Term Next Steps

1. definition-native evaluator を進めて、direct execution bridge をさらに薄くする
2. 残っている `extensions` fallback を helper の外から見えない形まで縮める
3. run store を fingerprint-first な検索・再利用モデルへさらに寄せ切る
4. `comparison-run-spec` ベースの再検証運用を definition-native evaluator 側へさらに寄せる
5. `plan.md` の完了条件を step ごとにより厳密に固定する
