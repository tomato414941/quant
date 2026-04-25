# Strategy Catalog

この文書は、プロジェクトに存在する全Strategyのフラットな台帳である。

## Questions This Answers

- このプロジェクトにはどのStrategyが存在するか？
- それぞれのStrategyはどのfamily、role、priority、tagを持つか？
- Strategyを人間が読むときの短い名前は何か？

## Questions This Does Not Answer

- どのStrategyが最終的に最良か？
- 各runの評価結果はどうか？
- 今回どのStrategyを評価対象にするか？

## Summary

| Metric | Count |
| --- | ---: |
| Total strategies | 43 |

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

- `Family`: Strategyの大まかな分類。
- `Role`: baselineやcandidateなど、比較上の役割。
- `Priority`: 現在のコードinventoryにある優先度。docs上では評価対象の確定を意味しない。
- `Baseline`: 比較時に主な参照先になりうるStrategy。
- `Tags`: Strategyの性質を表す。評価対象かどうか、採用可否、優先順位は表さない。

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

## Strategies

| Strategy ID | Slug | Display Name | Family | Role | Priority | Baseline | Tags | Key Parameters | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `stg-fu-eq` | `fu-eq` | ETF Universe Equal Weight | baseline | baseline | high | none | `baseline, etf_universe, equal_weight, reference_point` | `universe=ETF, portfolio=equal_weight` | ETF equal-weight baseline for return and drawdown context. |
| `stg-fu-rb` | `fu-rb` | TBD | baseline | candidate | low | none | `candidate, baseline, risk_budgeting, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-minvar` | `fu-minvar` | TBD | baseline | candidate | low | none | `candidate, baseline, minimum_variance, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-hrp` | `fu-hrp` | ETF Universe HRP | baseline | baseline | high | `stg-fu-eq` | `baseline, etf_universe, hrp, reference_point` | `universe=ETF, portfolio=hrp` | ETF HRP baseline for portfolio construction effects. |
| `stg-fu-momo12-lin025-hrp` | `fu-momo12-lin025-hrp` | TBD | momentum_tilt | candidate | low | none | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo12-top035-hrp` | `fu-momo12-hrp` | ETF 12M Momentum Tilt, HRP | momentum_tilt | candidate | high | `stg-fu-hrp` | `candidate, momentum_tilt, etf_universe, momentum, 12m, hrp` | `universe=ETF, momentum=12m, top_weight=0.35, portfolio=hrp` | Core annual 12-month momentum tilt candidate. |
| `stg-fu-momo6-top035-hrp` | `fu-momo6-top035-hrp` | TBD | momentum_tilt | candidate | low | none | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo9-top035-hrp` | `fu-momo9-top035-hrp` | TBD | momentum_tilt | candidate | low | none | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo8-top035-hrp` | `fu-momo8-top035-hrp` | TBD | momentum_tilt | candidate | low | none | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo10-top035-hrp` | `fu-momo10-top035-hrp` | TBD | momentum_tilt | candidate | low | none | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo11-top035-hrp` | `fu-momo11-top035-hrp` | TBD | momentum_tilt | candidate | low | none | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo3-top035-hrp` | `fu-momo3-top035-hrp` | TBD | momentum_tilt | candidate | low | none | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo15-top035-hrp` | `fu-momo15-top035-hrp` | TBD | momentum_tilt | candidate | low | none | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momolv7030-top025-hrp` | `fu-momolv7030-top025-hrp` | TBD | low_vol_momentum | candidate | low | none | `candidate, low_vol_momentum, hierarchical_risk_parity, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momolv8515-top025-hrp` | `fu-momolv8515-top025-hrp` | TBD | low_vol_momentum | candidate | low | none | `candidate, low_vol_momentum, hierarchical_risk_parity, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momomac8515-top025-hrp` | `fu-momomac8515-top025-hrp` | TBD | momentum_tilt | candidate | low | none | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo12-soft025-hrp` | `fu-momo12-soft025-hrp` | TBD | momentum_tilt | candidate | low | none | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo12-lin050-hrp` | `fu-momo12-lin050-hrp` | TBD | momentum_tilt | candidate | low | none | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo12-lin100-hrp` | `fu-momo12-lin100-hrp` | TBD | momentum_tilt | candidate | low | none | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momolv8515-top025-hrp-month` | `fu-momolv-month` | ETF Momentum + Low Vol Tilt, HRP, Monthly | low_vol_momentum | candidate | medium | `stg-fu-momo12-top035-hrp` | `candidate, low_vol_momentum, etf_universe, momentum, low_vol, monthly, hrp` | `universe=ETF, momentum_weight=0.85, low_vol_weight=0.15, top_weight=0.25, portfolio=hrp, frequency=monthly` | Momentum plus low-vol tilt candidate for robustness checks. |
| `stg-fu-momomac8515-top025-hrp-month` | `fu-momomac8515-top025-hrp-month` | TBD | momentum_tilt | candidate | low | none | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe, monthly` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo12-soft025-hrp-month` | `fu-momo12-soft025-hrp-month` | TBD | momentum_tilt | candidate | low | none | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe, monthly` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo12-lin050-hrp-month` | `fu-momo12-lin050-hrp-month` | TBD | momentum_tilt | candidate | low | none | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe, monthly` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo12-top035-mru` | `fu-momo12-top035-mru` | TBD | momentum_tilt | candidate | low | none | `candidate, momentum_tilt, mean_risk_utility, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo12-top035-mruc` | `fu-momo12-top035-mruc` | TBD | momentum_tilt | candidate | low | none | `candidate, momentum_tilt, mean_risk_utility_conservative, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-top3-eq` | `top3-eq` | TBD | concentrated_momentum | candidate | low | none | `candidate, concentrated_momentum, equal_weight` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-top3-rb` | `top3-rb` | TBD | concentrated_momentum | candidate | low | none | `candidate, concentrated_momentum, risk_budgeting` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-top3-minvar` | `top3-minvar` | TBD | concentrated_momentum | candidate | low | none | `candidate, concentrated_momentum, minimum_variance` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-top3-hrp` | `top3-hrp` | ETF Top 3 Momentum, HRP | concentrated_momentum | candidate | medium | `stg-fu-hrp` | `candidate, concentrated_momentum, top3, momentum, hrp` | `universe=ETF, selection=top3, portfolio=hrp` | Concentrated top-3 momentum baseline for selection strength. |
| `stg-dualtop3-hrp` | `dualtop3-hrp` | ETF Dual Momentum Top 3, HRP | defensive_momentum | candidate | medium | `stg-top3-hrp` | `candidate, defensive_momentum, dual_momentum, top3, hrp` | `universe=ETF, selection=dual_momentum_top3, portfolio=hrp` | Dual momentum variant for defensive selection behavior. |
| `stg-poslowvol-hrp` | `poslowvol-hrp` | TBD | low_vol_momentum | candidate | low | none | `candidate, low_vol_momentum, hierarchical_risk_parity` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-trailmomlowvol-hrp` | `trailmomlowvol-hrp` | TBD | low_vol_momentum | candidate | low | none | `candidate, low_vol_momentum, hierarchical_risk_parity` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-posvol-hrp` | `posvol-hrp` | TBD | other | candidate | low | none | `candidate, other, hierarchical_risk_parity` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-posmom-hrp-month` | `posmom-month` | ETF Positive Momentum, HRP, Monthly | defensive_momentum | candidate | medium | `stg-fu-hrp` | `candidate, defensive_momentum, positive_momentum, monthly, hrp` | `universe=ETF, filter=positive_momentum, portfolio=hrp, frequency=monthly` | Monthly positive momentum universe for defensive participation. |
| `stg-posrev5-trend60-hrp-month` | `posrev5-trend60-hrp-month` | TBD | other | candidate | low | none | `candidate, other, hierarchical_risk_parity, monthly` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-riskoff-posmom-hrp-month` | `riskoff-posmom-month` | ETF Risk-Off Gated Positive Momentum, HRP, Monthly | defensive_momentum | candidate | medium | `stg-posmom-hrp-month` | `candidate, defensive_momentum, risk_regime, positive_momentum, monthly, hrp` | `universe=ETF, filter=risk_off_positive_momentum, portfolio=hrp, frequency=monthly` | Risk-regime gated positive momentum candidate. |
| `stg-etf-momo12-top035-hrp` | `etf-momo12-top035-hrp` | TBD | momentum_tilt | candidate | low | none | `candidate, momentum_tilt, hierarchical_risk_parity` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo9-top035-hrp-1w` | `fu-momo9-top035-hrp-1w` | TBD | momentum_tilt | candidate | low | none | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo9-top035-hrp-1mo` | `fu-momo9-top035-hrp-1mo` | TBD | momentum_tilt | candidate | low | none | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo2-top035-hrp-month` | `fu-momo2-month` | ETF 2M Momentum Tilt, HRP, Monthly | short_momentum | candidate | high | `stg-fu-hrp` | `candidate, short_momentum, etf_universe, momentum, 2m, monthly, hrp` | `universe=ETF, momentum=2m, top_weight=0.35, portfolio=hrp, frequency=monthly` | Core monthly 2-month momentum candidate for faster adaptation. |
| `stg-fu-momo2-top035-pred10mom9010-hrp-month` | `fu-momo2-top035-pred10mom9010-hrp-month` | TBD | predictor_augmented_momentum | candidate | low | none | `candidate, predictor_augmented_momentum, hierarchical_risk_parity, etf_universe, predictor, monthly` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo2-top035-pred10mom5050-pw40-hrp-month` | `fu-momo2-pred-month` | ETF 2M Momentum + Predictor Overlay, HRP, Monthly | predictor_augmented_momentum | candidate | high | `stg-fu-momo2-top035-hrp-month` | `candidate, predictor_augmented_momentum, predictor, momentum, 2m, monthly, hrp` | `universe=ETF, momentum=2m, top_weight=0.35, predictor=10bar_momentum_50_50, predictor_weight=0.40, portfolio=hrp, frequency=monthly` | Main predictor-augmented candidate against the 2-month momentum baseline. |
| `stg-fu-momo2-top035-pred10mom5050-pw40-hrp-daily` | `fu-momo2-top035-pred10mom5050-pw40-hrp-daily` | TBD | predictor_augmented_momentum | candidate | low | none | `candidate, predictor_augmented_momentum, hierarchical_risk_parity, etf_universe, predictor` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |

## Relationship To Evaluation Plan

Strategyを評価対象へ載せるかどうかは、このカタログでは決めない。評価対象セットは [Evaluation Plan](./evaluation-plan.md) で別途定義する。
