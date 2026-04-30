# Evaluation Contexts

この文書は、Strategyを評価するときの測定条件を定義する。

## Questions This Answers

- 現在正式に追跡している評価Contextは何か？
- それぞれの評価Contextは何を確認するためのものか？
- 今後増やす候補には何があるか？

## Questions This Does Not Answer

- どのStrategyを今回評価するか？
- 評価結果の順位はどうか？
- どのStrategyを採用すべきか？
- Strategy x Evaluation Contextの全ケース一覧は何か？

## Summary

| Metric | Count |
| --- | ---: |
| Tracked evaluation contexts | 1 |

現在正式に追跡しているEvaluation Contextは1個である。これは候補が1個しかないという意味ではない。まずStrategy定義を上書きしない最小Contextから始め、比較の土台を作る。

## How To Read

- `Evaluation Context ID`: 評価条件の安定参照ID。
- `Display Name`: 人間が読むための評価条件名。
- `Period`: 評価に使うデータ期間。
- `Evaluation Method`: 評価方法。例: `single_run`, `walk_forward`。
- `Cost Assumption`: コードやrunと対応する取引コスト前提の参照名。
- `Execution Assumption`: 約定や執行に関する前提。
- `Purpose`: この評価条件で何を確認するか。
- `Human Reviewed`: 人間がこの行の記載内容を確認したか。

この文書は評価Contextの定義であり、Strategy x Evaluation Contextの全ケース一覧は管理しない。ケース一覧は数が膨らむため、必要に応じてコードやrun管理で扱う。

Evaluation ContextはStrategy定義を上書きしない。Universe、rebalance frequency、position limitsなどはStrategy側に従う。

## Evaluation Contexts

| Evaluation Context ID | Display Name | Period | Evaluation Method | Cost Assumption | Execution Assumption | Purpose | Human Reviewed | Reviewed At | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `2015_2025` | 2015-2025 Full-Period Evaluation | 2015-2025 | single_run | `retail_multi_asset_default` | `close_execution` | 期間全体でリターン、ドローダウン、turnoverを見る | no |  |  |

## Cost Assumptions

| Cost Assumption | Summary | Source |
| --- | --- | --- |
| `retail_multi_asset_default` | 個人投資家が複数資産を売買する前提の既存コストプロファイル。詳細な計算はコード定義に従う。 | code |

## Execution Assumptions

| Execution Assumption | Summary | Source |
| --- | --- | --- |
| `close_execution` | 終値約定を前提にする。 | code |

## Candidate Future Contexts

今後増やす候補は次の通り。

- `crypto_only`: cryptoだけで評価する
- `etf_plus_crypto`: ETFとcryptoを混ぜて評価する
- shorter period: より短い期間で評価する
- longer period: より長い期間で評価する
- cost sensitivity contexts: 取引コスト前提を変えて評価する
- walk-forward contexts: 年次などの窓で時間安定性を評価する

これらはすぐに正式追跡対象へ追加しない。まず現在の1Contextで比較の読み方を固めてから、必要なものを追加する。

## Relationship To Code

コード上の `EvaluationProfileRegistry` は、保存済みrunを機械的に突き合わせるための補助ビューである。

評価条件の意思決定正本はこの文書に置く。コードへ反映するのは、docs上で評価条件が安定してからにする。

## Data Snapshot Contract

Evaluation run specs include `marketDataContexts[].datasetSnapshot` as the fixed market data contract for a run. Provider APIs are used to create local snapshots; reproducible evaluations read those snapshots and validate their metadata/content fingerprints instead of fetching provider data.

Two code paths consume these snapshots today:

- `comparison-summary`: existing strategy comparison/run plumbing. It is useful
  for comparing strategy definitions under the current research workflow, but
  should not be described as a plain full-period backtest.
- `backtest-strategy`: full-period backtest over one fixed market snapshot. This
  is the simpler entrypoint for asking how a supported strategy behaved across
  the whole snapshot period. It currently supports 16 full-universe ETF variants:
  four portfolio models across annual, monthly, weekly, and daily rebalance
  schedules.

The v1 contract records:

- `source`: data provider label.
- `createdAtUtc`: snapshot metadata creation time.
- `timeframe`: data timeframe such as `1d`, `1wk`, or `1mo`.
- `period`, `start`, `end`: requested evaluation range.
- `requestedTickers`: requested universe symbols.
- `availableTickers`: symbols with usable rows after provider filtering.
- `rowCount`: aligned market data row count.
- `adjustmentPolicy`: price adjustment policy used by the provider.
- `contentFingerprint`: stable identifier derived from `closes.csv` and `volumes.csv`.
- `fingerprint` and `snapshotId`: stable identifiers derived from the snapshot metadata excluding `createdAtUtc`.

Provider-backed snapshot creation writes `manifest.json`, `closes.csv`, and `volumes.csv` under the local snapshot directory. Evaluation and rerun commands resolve market data by `snapshotId` and reject mismatched metadata or CSV content. Live provider fetches are for creating/updating snapshots or explicit ad-hoc runs, not the intended reproducible evaluation path.

Committed fixtures may include small generated/golden snapshots for tests. Full provider downloads remain local generated data and must not be committed.
