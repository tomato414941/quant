# Strategy Catalog

この文書は、プロジェクトに存在する全Strategyの台帳である。

## Questions This Answers

- このプロジェクトにはどのStrategyが存在するか？
- それぞれのStrategyは今 `active` / `backlog` のどちらか？
- 評価計画へ載せる候補はどれか？

## Questions This Does Not Answer

- どのStrategyが最終的に最良か？
- 各runの評価結果はどうか？
- 今回どのEvaluation Contextで評価するか？

## Summary

| Metric | Count |
| --- | ---: |
| Total strategies | 43 |
| Active strategies | 10 |
| Backlog strategies | 33 |

| Family | Count |
| --- | ---: |
| `baseline` | 4 |
| `concentrated_momentum` | 4 |
| `defensive_momentum` | 3 |
| `low_vol_momentum` | 5 |
| `momentum_tilt` | 21 |
| `other` | 2 |
| `predictor_augmented_momentum` | 3 |
| `short_momentum` | 1 |

## How To Read

- `active`: 現在の評価計画に載せる対象
- `backlog`: 存在するが、明示的な研究metadataを追加するまで評価計画へ載せない対象
- `priority`: 評価計画上の優先度
- `baselineStrategyId`: 比較時に主な参照先とするStrategy

この文書は評価結果を書く場所ではない。評価済みrunの比較は [Leaderboard](./leaderboard.md) に置く。

## Naming Convention

StrategyのIDと名前は分けて扱う。

- `Strategy ID`: コード互換のための安定参照ID。既存の `stg-fu-...` を維持する。
- `Slug`: 人間が短く参照するための短縮名。IDより読みやすく、全パラメータを詰め込みすぎない。
- `Display Name`: 議論や表示で使う自然言語名。Strategy IDより変更しやすい。
- `Key Parameters`: 比較に効く主要パラメータ。IDへ全て詰め込まない。

### ID Token Glossary

| Token | Meaning |
| --- | --- |
| `stg` | strategy |
| `fu` | full ETF universe |
| `eq` | equal weight |
| `hrp` | hierarchical risk parity |
| `momo` | momentum |
| `momolv` | momentum + low volatility |
| `top035` | top-weight emphasis 0.35 |
| `pred` | predictor augmented |
| `month` | monthly variant |

## Active Strategies

| Strategy ID | Slug | Display Name | Family | Role | Priority | Baseline | Key Parameters | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `stg-fu-eq` | `fu-eq` | ETF Universe Equal Weight | baseline | baseline | high | none | `universe=ETF, portfolio=equal_weight` | ETF equal-weight baseline for return and drawdown context. |
| `stg-fu-hrp` | `fu-hrp` | ETF Universe HRP | baseline | baseline | high | `stg-fu-eq` | `universe=ETF, portfolio=hrp` | ETF HRP baseline for portfolio construction effects. |
| `stg-fu-momo12-top035-hrp` | `fu-momo12-hrp` | ETF 12M Momentum Tilt, HRP | momentum_tilt | candidate | high | `stg-fu-hrp` | `universe=ETF, momentum=12m, top_weight=0.35, portfolio=hrp` | Core annual 12-month momentum tilt candidate. |
| `stg-fu-momolv8515-top025-hrp-month` | `fu-momolv-month` | ETF Momentum + Low Vol Tilt, HRP, Monthly | low_vol_momentum | candidate | medium | `stg-fu-momo12-top035-hrp` | `universe=ETF, momentum_weight=0.85, low_vol_weight=0.15, top_weight=0.25, portfolio=hrp, frequency=monthly` | Momentum plus low-vol tilt candidate for robustness checks. |
| `stg-top3-hrp` | `top3-hrp` | ETF Top 3 Momentum, HRP | concentrated_momentum | candidate | medium | `stg-fu-hrp` | `universe=ETF, selection=top3, portfolio=hrp` | Concentrated top-3 momentum baseline for selection strength. |
| `stg-dualtop3-hrp` | `dualtop3-hrp` | ETF Dual Momentum Top 3, HRP | defensive_momentum | candidate | medium | `stg-top3-hrp` | `universe=ETF, selection=dual_momentum_top3, portfolio=hrp` | Dual momentum variant for defensive selection behavior. |
| `stg-posmom-hrp-month` | `posmom-month` | ETF Positive Momentum, HRP, Monthly | defensive_momentum | candidate | medium | `stg-fu-hrp` | `universe=ETF, filter=positive_momentum, portfolio=hrp, frequency=monthly` | Monthly positive momentum universe for defensive participation. |
| `stg-riskoff-posmom-hrp-month` | `riskoff-posmom-month` | ETF Risk-Off Gated Positive Momentum, HRP, Monthly | defensive_momentum | candidate | medium | `stg-posmom-hrp-month` | `universe=ETF, filter=risk_off_positive_momentum, portfolio=hrp, frequency=monthly` | Risk-regime gated positive momentum candidate. |
| `stg-fu-momo2-top035-hrp-month` | `fu-momo2-month` | ETF 2M Momentum Tilt, HRP, Monthly | short_momentum | candidate | high | `stg-fu-hrp` | `universe=ETF, momentum=2m, top_weight=0.35, portfolio=hrp, frequency=monthly` | Core monthly 2-month momentum candidate for faster adaptation. |
| `stg-fu-momo2-top035-pred10mom5050-pw40-hrp-month` | `fu-momo2-pred-month` | ETF 2M Momentum + Predictor Overlay, HRP, Monthly | predictor_augmented_momentum | candidate | high | `stg-fu-momo2-top035-hrp-month` | `universe=ETF, momentum=2m, top_weight=0.35, predictor=10bar_momentum_50_50, predictor_weight=0.40, portfolio=hrp, frequency=monthly` | Main predictor-augmented candidate against the 2-month momentum baseline. |

## Backlog Strategies

### Baseline

| Strategy ID | Role | Priority | Notes |
| --- | --- | --- | --- |
| `stg-fu-rb` | candidate | low | Risk budgeting baseline backlog entry. |
| `stg-fu-minvar` | candidate | low | Minimum variance baseline backlog entry. |

### Momentum Tilt

| Strategy ID | Role | Priority | Notes |
| --- | --- | --- | --- |
| `stg-fu-momo12-lin025-hrp` | candidate | low | Momentum tilt backlog entry. |
| `stg-fu-momo6-top035-hrp` | candidate | low | Momentum tilt backlog entry. |
| `stg-fu-momo9-top035-hrp` | candidate | low | Momentum tilt backlog entry. |
| `stg-fu-momo8-top035-hrp` | candidate | low | Momentum tilt backlog entry. |
| `stg-fu-momo10-top035-hrp` | candidate | low | Momentum tilt backlog entry. |
| `stg-fu-momo11-top035-hrp` | candidate | low | Momentum tilt backlog entry. |
| `stg-fu-momo3-top035-hrp` | candidate | low | Momentum tilt backlog entry. |
| `stg-fu-momo15-top035-hrp` | candidate | low | Momentum tilt backlog entry. |
| `stg-fu-momomac8515-top025-hrp` | candidate | low | Momentum macro tilt backlog entry. |
| `stg-fu-momo12-soft025-hrp` | candidate | low | Soft momentum tilt backlog entry. |
| `stg-fu-momo12-lin050-hrp` | candidate | low | Momentum tilt backlog entry. |
| `stg-fu-momo12-lin100-hrp` | candidate | low | Momentum tilt backlog entry. |
| `stg-fu-momomac8515-top025-hrp-month` | candidate | low | Monthly momentum macro tilt backlog entry. |
| `stg-fu-momo12-soft025-hrp-month` | candidate | low | Monthly soft momentum tilt backlog entry. |
| `stg-fu-momo12-lin050-hrp-month` | candidate | low | Monthly momentum tilt backlog entry. |
| `stg-fu-momo12-top035-mru` | candidate | low | Mean risk utility momentum backlog entry. |
| `stg-fu-momo12-top035-mruc` | candidate | low | Conservative mean risk utility momentum backlog entry. |
| `stg-etf-momo12-top035-hrp` | candidate | low | ETF momentum tilt backlog entry. |
| `stg-fu-momo9-top035-hrp-1w` | candidate | low | Weekly 9-month momentum backlog entry. |
| `stg-fu-momo9-top035-hrp-1mo` | candidate | low | Monthly 9-month momentum backlog entry. |

### Low Vol Momentum

| Strategy ID | Role | Priority | Notes |
| --- | --- | --- | --- |
| `stg-fu-momolv7030-top025-hrp` | candidate | low | Low-vol momentum backlog entry. |
| `stg-fu-momolv8515-top025-hrp` | candidate | low | Low-vol momentum backlog entry. |
| `stg-poslowvol-hrp` | candidate | low | Positive low-vol backlog entry. |
| `stg-trailmomlowvol-hrp` | candidate | low | Trailing momentum low-vol backlog entry. |

### Concentrated Momentum

| Strategy ID | Role | Priority | Notes |
| --- | --- | --- | --- |
| `stg-top3-eq` | candidate | low | Top-3 equal-weight backlog entry. |
| `stg-top3-rb` | candidate | low | Top-3 risk-budgeting backlog entry. |
| `stg-top3-minvar` | candidate | low | Top-3 minimum-variance backlog entry. |

### Other

| Strategy ID | Role | Priority | Notes |
| --- | --- | --- | --- |
| `stg-posvol-hrp` | candidate | low | Positive volume backlog entry. |
| `stg-posrev5-trend60-hrp-month` | candidate | low | Positive reversal/trend backlog entry. |

### Predictor Augmented Momentum

| Strategy ID | Role | Priority | Notes |
| --- | --- | --- | --- |
| `stg-fu-momo2-top035-pred10mom9010-hrp-month` | candidate | low | Predictor-augmented monthly momentum backlog entry. |
| `stg-fu-momo2-top035-pred10mom5050-pw40-hrp-daily` | candidate | low | Predictor-augmented daily momentum backlog entry. |

## Relationship To Evaluation Plan

Strategyを評価対象へ昇格させる場合は、まずこのカタログでstatus、priority、notesを整理する。その後 [Evaluation Plan](./evaluation-plan.md) の対象Strategy setへ反映する。
