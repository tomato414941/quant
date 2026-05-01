# Strategy Catalog

この文書は、公開リポジトリで共有するStrategy台帳である。

公開版では、Strategyの管理方針と人間レビュー済みの代表Strategyだけを記載する。未レビューの候補、細かい探索パラメータ、研究途中のアイデアは公開台帳には載せない。

## Questions This Answers

- このプロジェクトではStrategyをどの粒度で管理するか？
- 人間レビュー済みの代表Strategyは何か？
- Strategy ID、Slug、Display Nameをどう使い分けるか？

## Questions This Does Not Answer

- 未レビュー候補を含む全探索リストは何か？
- どのStrategyが最終的に最良か？
- 各runの評価結果はどうか？
- 今回どのStrategyを評価対象にするか？

## Public Scope

この公開版カタログは、研究途中の全候補を網羅しない。

公開する情報:

- 人間レビュー済みのStrategy
- 比較の読み方に必要な構成要素
- 評価結果ではなく、Strategy定義の概要

公開しない情報:

- 未レビューの探索候補一覧
- 細かいパラメータ探索の全履歴
- 公開前に確認していない研究メモ

## Summary

| Metric | Count |
| --- | ---: |
| Public reviewed strategies | 4 |

## How To Read

- `Universe`: 投資対象集合。現状は主に `ETF`。
- `Selection`: 候補資産をどう選ぶか。例: `full_universe`。
- `Signal`: rankingやtiltに使う情報。`none` はシグナルなしを表す。
- `Portfolio Model`: weightを決めるモデル。例: `equal_weight`, `equal_risk_contribution`, `minimum_variance`, `hrp`。
- `Rebalance Frequency`: 配分を見直す頻度。
- `Overlay`: 追加的なゲートや予測補助。
- `Role`: 比較上の役割。
- `Tags`: Strategyの性質を補助的に表す。評価対象かどうか、採用可否、優先順位は表さない。

この文書は評価結果を書く場所ではない。評価済みrunの比較は [ETF Full-Universe No-Signal Allocation](./leaderboard/etf-full-universe-no-signal-allocation.md) に置く。

## Naming Convention

StrategyのIDと名前は分けて扱う。

- `Strategy ID`: コードや評価結果と対応させるための安定参照ID。将来的には意味を持たない不透明IDへ移行する。
- `Slug`: 人間が短く参照するための短縮名。Strategyの内容を軽く示してよいが、全パラメータを詰め込まない。
- `Display Name`: 議論や表示で使う自然言語名。`Universe` という語は使わず、対象資産集合は `ETF` のように短く表す。
- `Key Parameters`: 比較に効く主要パラメータ。IDへ全て詰め込まない。
- 現在の `stg-fu-...` 形式のIDは互換用の既存IDであり、新しい命名としては推奨しない。
- 将来の `Strategy ID` は `stg-` + ランダムな短い英数字のような不透明IDを想定する。IDからStrategyの内容を推測してはいけない。

## Reviewed Strategies

The table below lists the four public reviewed annual strategy definitions.
Monthly, weekly, and daily `backtest-strategy` variants are runner-supported
rebalance-frequency variants of these definitions; they are not yet separate
reviewed catalog entries.

| Strategy ID | Slug | Display Name | Universe | Selection | Signal | Portfolio Model | Rebalance Frequency | Overlay | Role | Tags | Key Parameters | Human Reviewed | Reviewed At | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `stg-fu-eq` | `etf-eq` | ETF Equal Weight Allocation | ETF | full_universe | none | equal_weight | annual | none | reference | `etf_universe, full_universe, equal_weight, reference_point` | `universe=ETF, selection=full_universe, signal=none, portfolio=equal_weight, rebalance_frequency=annual, overlay=none` | yes | 2026-04-25 | ETF equal-weight reference point for return and drawdown context. Strategy ID and slug are expected to change when naming is cleaned up. Proposed opaque Strategy ID: `stg-k7m4qa`. |
| `stg-fu-rb` | `etf-erc` | ETF Equal Risk Contribution Allocation | ETF | full_universe | none | equal_risk_contribution | annual | none | reference_candidate | `etf_universe, full_universe, equal_risk_contribution` | `universe=ETF, selection=full_universe, signal=none, portfolio=equal_risk_contribution, rebalance_frequency=annual, overlay=none, code_portfolio_model=risk_budgeting` | yes | 2026-04-25 | ETF full-universe reference candidate. Current code model is named `risk_budgeting` but behaves like equal-risk-contribution allocation. Strategy ID and slug are expected to change when naming is cleaned up. Proposed opaque Strategy ID: `stg-p9x2vt`. |
| `stg-fu-minvar` | `etf-minvar` | ETF Minimum Variance Allocation | ETF | full_universe | none | minimum_variance | annual | none | reference_candidate | `etf_universe, full_universe, minimum_variance` | `universe=ETF, selection=full_universe, signal=none, portfolio=minimum_variance, rebalance_frequency=annual, overlay=none` | yes | 2026-04-25 | ETF full-universe minimum variance allocation reference candidate. Strategy ID and slug are expected to change when naming is cleaned up. Proposed opaque Strategy ID: `stg-r4n8cw`. |
| `stg-fu-hrp` | `etf-hrp` | ETF Hierarchical Risk Parity Allocation | ETF | full_universe | none | hrp | annual | none | reference | `etf_universe, full_universe, hrp, reference_point` | `universe=ETF, selection=full_universe, signal=none, portfolio=hrp, rebalance_frequency=annual, overlay=none` | yes | 2026-04-25 | ETF HRP reference point for portfolio construction effects. Proposed opaque Strategy ID: `stg-h6q3dz`. |

## Relationship To Evaluation Plan

Strategyを評価対象へ載せるかどうかは、このカタログでは決めない。評価対象セットは [Evaluation Plan](./evaluation-plan.md) で別途定義する。

## Private Research Notes

未レビュー候補を含む詳細な探索リストは、公開対象ではないローカル資料として管理する。公開前にレビュー済みとなったStrategyだけを、この文書へ追加する。
