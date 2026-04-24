# Evaluation Contexts

この文書は、Strategyを評価する条件の台帳である。

## Questions This Answers

- 現在正式に追跡している評価条件は何個あるか？
- それぞれの評価条件は何を確認するためのものか？
- 今後増やす候補には何があるか？

## Questions This Does Not Answer

- どのStrategyを今回評価するか？
- 評価結果の順位はどうか？
- どのStrategyを採用すべきか？

## Current Tracked Contexts

現在正式に追跡しているEvaluation Contextは3個である。

これは候補が3個しかないという意味ではない。まず管理可能な最小セットを埋め、比較の土台を作るために3個へ絞っている。

| Evaluation Context ID | Universe | Period | Cost | Max Weight | Kind | Purpose |
| --- | --- | --- | --- | --- | --- | --- |
| `etf_2015_2025` | ETF | 2015-2025 | normal | 45% | single run | 基本評価 |
| `etf_cost_2x_2015_2025` | ETF | 2015-2025 | 2x | 45% | single run | コスト感応度 |
| `etf_walk_forward_2020_2025` | ETF | 2015-2025 | normal | 45% | walk-forward | 時間安定性 |

## Candidate Future Contexts

今後増やす候補は次の通り。

- `crypto_only`: cryptoだけで評価する
- `etf_plus_crypto`: ETFとcryptoを混ぜて評価する
- shorter period: より短い期間で評価する
- longer period: より長い期間で評価する
- cost variants: cost 0.5x、3xなどで評価する
- max weight variants: max weight 25%、30%、45%などで評価する
- rebalance frequency variants: daily、weekly、monthlyなどで評価する

これらはすぐに正式追跡対象へ追加しない。active 10戦略の3条件が埋まってから、必要なものを追加する。

## Relationship To Code

コード上の `EvaluationProfileRegistry` は、保存済みrunを機械的に突き合わせるための補助ビューである。

評価条件の意思決定正本はこの文書に置く。コードへ反映するのは、docs上で評価条件が安定してからにする。
