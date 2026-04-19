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

これにより、次のような構成を自然に表現できるようにする。

- 日足価格を使う
- 週次 signal を計算する
- 月次 macro を補助 signal に使う
- 判断は毎日
- 売買は月末

## Current State

基盤整理は一段落している。公開正本は `StrategyDefinition`、低レベル evaluator 内部 DTO は `EvaluatorStrategySpec` として分離済み。

現在の状態:
- canonical candidate catalog の正本は strategy definitions
- `comparison_service` は `StrategyDefinition` を直接実行入口として扱う
- `comparison_service` は strategy / condition / parameter / predictor / ranking の runSpec を `StrategyDefinition` 正本で生成する
- `executionSupport` は `evaluatorAdapter*` と `directExecution*` で実行互換性を表す
- runtime の `extensions` fallback は通常実行経路から削除済み
- `evaluate_strategy_definition_run` が multi-timeframe / predictor / multi-selection を通す
- run store は `v59` で `strategyDefinition` と `evaluationSubject` の fingerprint を正本化した
- strategy / condition / parameter / predictor / ranking run は新規 runSpec で legacy `strategy` payload を生成しない
- predictor / ranking run は由来を `strategyDefinition`、評価対象を `evaluationSubject` として分離する
- 外部入口は `comparison` 系に統一済み
- 現在の全体テスト: `107 passed`

## Completed Foundation

完了した基盤:
- `StrategyDefinition` / `StrategyExecutionPlanSpec` / `StrategySignalSpec` を導入した
- `StrategyBlueprintSpec` を `StrategyDefinition` にリネームした
- legacy evaluator DTO との相互変換と互換判定を導入した
- `ComparisonSpec` と `comparison_service` を definition-first に移行した
- candidate catalog / baseline / canonical catalog を strategy definitions 正本に寄せた
- `selection × portfolio_model × execution` の product builder を作った
- predictor を signal として builder に取り込んだ
- `FeatureDefinition` と `DataSourceDefinition` を分けた
- 異粒度データを結合する `alignment_policy` を導入した
- signal ごとの `data_timeframe` / `signal_timeframe` を評価経路に通した
- comparison layer が legacy adapter を直接使わず definition を評価できる経路を作った
- run store を `strategyDefinition + evaluationSubject + marketData + evaluation + logicVersion` fingerprint ベースへ寄せた
- `comparison-run-spec` / `rerun-comparison-spec` で同条件再実行できる状態にした
- API payload の互換表示を `evaluatorAdapter*` / `directExecution*` に揃えた
- production で不要だった evaluator DTO list helper を削除した

重要な完了条件:
- rerun 時に `runSpecFingerprint` / `comparisonFingerprint` の整合性を検証できる
- rerun は definition-only payload を前提にする
- `EvaluatorStrategySpec.extensions` は runtime context の復元元ではない
- `comparison_service` から `EvaluatorStrategySpec` 変換の直接 import を削除済み
- parameter sweep は `EvaluatorStrategySpec` を経由せず `StrategyDefinition` を直接生成する
- candidate catalog tests は evaluator DTO 変換比較ではなく `StrategyDefinition` を直接検証する

## Active Decision

次の大きな分岐は次のどちらか。

- evaluator 本体をさらに `StrategyDefinition` 直接評価へ寄せる
- 日次中心の戦略探索・比較へ戻る

現時点では、戦略探索に戻るほうを優先する。理由は、外部正本と再現性の基盤は整っており、evaluator 内部 DTO を完全撤去するよりも、実際に勝てる候補を探すほうが目的に近いから。

## Next Work

次にやること:
- 現在もっとも良い既存結果を確認する
- その結果を reference として固定し、同条件で再実行する
- daily-first の candidate set を作る
- 年次トップに対して、日次 / 週次 signal、月次 rebalance、predictor overlay の候補を比較する
- 良い候補が出たら、その strategy family を広げる

成功条件:
- 比較条件が runSpec として保存され、rerun できる
- reference の年次トップと daily-first 候補を同じ market data / cost / split 条件で比較できる
- 勝敗だけでなく、年率リターン、最大DD、Sharpe、取引頻度を確認できる

## Parking Lot

後で必要になったらやること:
- `evaluationSubjectFingerprint` filter を API / CLI に追加する
- evaluator 本体から `EvaluatorStrategySpec` DTO 依存をさらに減らす
- `portfolio.py` の低レベル evaluator を `StrategyDefinition` 直接評価へ段階移行する
- strategy search の結果保存・ランキング・比較 UI 相当の CLI 出力を改善する
