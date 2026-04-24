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

## Active Strategies

| Strategy ID | Family | Role | Priority | Baseline | Notes |
| --- | --- | --- | --- | --- | --- |
| `stg-fu-eq` | baseline | baseline | high | none | ETF equal-weight baseline for return and drawdown context. |
| `stg-fu-hrp` | baseline | baseline | high | `stg-fu-eq` | ETF HRP baseline for portfolio construction effects. |
| `stg-fu-momo12-top035-hrp` | momentum_tilt | candidate | high | `stg-fu-hrp` | Core annual 12-month momentum tilt candidate. |
| `stg-fu-momolv8515-top025-hrp-month` | low_vol_momentum | candidate | medium | `stg-fu-momo12-top035-hrp` | Momentum plus low-vol tilt candidate for robustness checks. |
| `stg-top3-hrp` | concentrated_momentum | candidate | medium | `stg-fu-hrp` | Concentrated top-3 momentum baseline for selection strength. |
| `stg-dualtop3-hrp` | defensive_momentum | candidate | medium | `stg-top3-hrp` | Dual momentum variant for defensive selection behavior. |
| `stg-posmom-hrp-month` | defensive_momentum | candidate | medium | `stg-fu-hrp` | Monthly positive momentum universe for defensive participation. |
| `stg-riskoff-posmom-hrp-month` | defensive_momentum | candidate | medium | `stg-posmom-hrp-month` | Risk-regime gated positive momentum candidate. |
| `stg-fu-momo2-top035-hrp-month` | short_momentum | candidate | high | `stg-fu-hrp` | Core monthly 2-month momentum candidate for faster adaptation. |
| `stg-fu-momo2-top035-pred10mom5050-pw40-hrp-month` | predictor_augmented_momentum | candidate | high | `stg-fu-momo2-top035-hrp-month` | Main predictor-augmented candidate against the 2-month momentum baseline. |

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
