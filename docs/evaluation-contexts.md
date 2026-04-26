# Evaluation Context Catalog

この文書は、Strategyを評価する条件の台帳である。

## Questions This Answers

- 現在正式に追跡している評価条件は何個あるか？
- それぞれの評価条件は何を確認するためのものか？
- 今後増やす候補には何があるか？

## Questions This Does Not Answer

- どのStrategyを今回評価するか？
- 評価結果の順位はどうか？
- どのStrategyを採用すべきか？
- Strategy x Evaluation Contextの全ケース一覧は何か？

## Summary

| Metric | Count |
| --- | ---: |
| Tracked evaluation contexts | 3 |

現在正式に追跡しているEvaluation Contextは3個である。これは候補が3個しかないという意味ではない。まず管理可能な最小セットを埋め、比較の土台を作るために3個へ絞っている。

## How To Read

- `Evaluation Context ID`: 評価条件の安定参照ID。
- `Display Name`: 人間が読むための評価条件名。
- `Universe`: 評価時に使う投資対象集合。Strategy側のUniverseと混同しない。
- `Period`: 評価に使うデータ期間。
- `Mode`: 評価方法。例: `single_run`, `walk_forward`。
- `Cost Model`: 取引コスト前提。
- `Max Weight`: 1資産あたりの最大weight制約。
- `Rebalance Assumption`: Strategy定義のリバランス頻度を使うか、評価側で固定するか。
- `Purpose`: この評価条件で何を確認するか。
- `Human Reviewed`: 人間がこの行の記載内容を確認したか。

この文書は評価条件のカタログであり、Strategy x Evaluation Contextの全ケース一覧は管理しない。ケース一覧は数が膨らむため、必要に応じてコードやrun管理で扱う。

## Evaluation Contexts

| Evaluation Context ID | Display Name | Universe | Period | Mode | Cost Model | Max Weight | Rebalance Assumption | Purpose | Human Reviewed | Reviewed At | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `etf_2015_2025` | ETF 2015-2025 | ETF-only | 2015-2025 | single_run | normal | 45% | use_strategy_definition | 基本評価 | no |  | ETF-onlyで基本的なリターン、ドローダウン、turnoverを見る。 |
| `etf_cost_2x_2015_2025` | ETF 2015-2025 Cost x2 | ETF-only | 2015-2025 | single_run | 2x | 45% | use_strategy_definition | コスト感応度 | no |  | 基本評価に対して取引コストを強めたときの劣化を見る。 |
| `etf_walk_forward_2020_2025` | ETF Walk-Forward 2020-2025 | ETF-only | 2015-2025 train / 2020-2025 test windows | walk_forward | normal | 45% | use_strategy_definition | 時間安定性 | no |  | 年次walk-forwardで、特定期間だけに依存していないかを見る。 |

## Candidate Future Contexts

今後増やす候補は次の通り。

- `crypto_only`: cryptoだけで評価する
- `etf_plus_crypto`: ETFとcryptoを混ぜて評価する
- shorter period: より短い期間で評価する
- longer period: より長い期間で評価する
- cost variants: cost 0.5x、3xなどで評価する
- max weight variants: max weight 25%、30%、45%などで評価する
- rebalance frequency variants: daily、weekly、monthlyなどで評価する

これらはすぐに正式追跡対象へ追加しない。現在の3条件が埋まってから、必要なものを追加する。

## Relationship To Code

コード上の `EvaluationProfileRegistry` は、保存済みrunを機械的に突き合わせるための補助ビューである。

評価条件の意思決定正本はこの文書に置く。コードへ反映するのは、docs上で評価条件が安定してからにする。
