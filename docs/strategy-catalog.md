# Strategy Catalog

この文書は、プロジェクトに存在する全Strategyのフラットな台帳である。

## Questions This Answers

- このプロジェクトにはどのStrategyが存在するか？
- それぞれのStrategyはどの構成要素を持つか？
- Strategyを人間が読むときの短い名前は何か？

## Questions This Does Not Answer

- どのStrategyが最終的に最良か？
- 各runの評価結果はどうか？
- 今回どのStrategyを評価対象にするか？

## Summary

| Metric | Count |
| --- | ---: |
| Total strategies | 43 |

## How To Read

- `Universe`: 投資対象集合。現状は主に `ETF`。
- `Selection`: 候補資産をどう選ぶか。例: `full_universe`, `top3`, `positive_momentum`。
- `Signal`: rankingやtiltに使う情報。例: `none`, `momentum_12m`, `momentum_low_vol`。
- `Portfolio Model`: weightを決めるモデル。例: `equal_weight`, `risk_budgeting`, `minimum_variance`, `hrp`。
- `Schedule`: 判断・リバランス頻度。例: `annual`, `monthly`, `weekly`, `daily`。
- `Overlay`: 追加的なゲートや予測補助。例: `none`, `risk_off`, `predictor`。
- `Role`: 比較上の役割。例: `reference`, `reference_candidate`, `candidate`。
- `Tags`: Strategyの性質を補助的に表す。評価対象かどうか、採用可否、優先順位は表さない。

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

| Strategy ID | Slug | Display Name | Universe | Selection | Signal | Portfolio Model | Schedule | Overlay | Role | Baseline | Tags | Key Parameters | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `stg-fu-eq` | `fu-eq` | ETF Universe Equal Weight | ETF | full_universe | none | equal_weight | annual | none | reference | none | `etf_universe, full_universe, equal_weight, reference_point` | `universe=ETF, selection=full_universe, signal=none, portfolio=equal_weight, schedule=annual, overlay=none` | ETF equal-weight baseline for return and drawdown context. |
| `stg-fu-rb` | `fu-rb` | ETF Universe Risk Budgeting | ETF | full_universe | none | risk_budgeting | annual | none | reference_candidate | none | `etf_universe, full_universe, risk_budgeting` | `universe=ETF, selection=full_universe, signal=none, portfolio=risk_budgeting, schedule=annual, overlay=none` | ETF full-universe risk budgeting allocation reference candidate. |
| `stg-fu-minvar` | `fu-minvar` | ETF Universe Minimum Variance | ETF | full_universe | none | minimum_variance | annual | none | reference_candidate | none | `etf_universe, full_universe, minimum_variance` | `universe=ETF, selection=full_universe, signal=none, portfolio=minimum_variance, schedule=annual, overlay=none` | ETF full-universe minimum variance allocation reference candidate. |
| `stg-fu-hrp` | `fu-hrp` | ETF Universe HRP | ETF | full_universe | none | hrp | annual | none | reference | `stg-fu-eq` | `etf_universe, full_universe, hrp, reference_point` | `universe=ETF, selection=full_universe, signal=none, portfolio=hrp, schedule=annual, overlay=none` | ETF HRP baseline for portfolio construction effects. |
| `stg-fu-momo12-lin025-hrp` | `fu-momo12-lin025-hrp` | TBD | ETF | full_universe | momentum_12m | hrp | annual | none | candidate | none | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo12-top035-hrp` | `fu-momo12-hrp` | ETF 12M Momentum Tilt, HRP | ETF | full_universe | momentum_12m_top_weighted | hrp | annual | none | candidate | `stg-fu-hrp` | `candidate, etf_universe, momentum, 12m, hrp` | `universe=ETF, selection=full_universe, signal=momentum_12m, top_weight=0.35, portfolio=hrp, schedule=annual, overlay=none` | Core annual 12-month momentum tilt candidate. |
| `stg-fu-momo6-top035-hrp` | `fu-momo6-top035-hrp` | TBD | ETF | full_universe | momentum_6m | hrp | annual | none | candidate | none | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo9-top035-hrp` | `fu-momo9-top035-hrp` | TBD | ETF | full_universe | momentum_9m | hrp | annual | none | candidate | none | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo8-top035-hrp` | `fu-momo8-top035-hrp` | TBD | ETF | full_universe | momentum_8m | hrp | annual | none | candidate | none | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo10-top035-hrp` | `fu-momo10-top035-hrp` | TBD | ETF | full_universe | momentum_10m | hrp | annual | none | candidate | none | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo11-top035-hrp` | `fu-momo11-top035-hrp` | TBD | ETF | full_universe | momentum_11m | hrp | annual | none | candidate | none | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo3-top035-hrp` | `fu-momo3-top035-hrp` | TBD | ETF | full_universe | momentum_3m | hrp | annual | none | candidate | none | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo15-top035-hrp` | `fu-momo15-top035-hrp` | TBD | ETF | full_universe | momentum_15m | hrp | annual | none | candidate | none | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momolv7030-top025-hrp` | `fu-momolv7030-top025-hrp` | TBD | ETF | full_universe | momentum_low_vol | hrp | annual | none | candidate | none | `candidate, low_vol_momentum, hierarchical_risk_parity, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momolv8515-top025-hrp` | `fu-momolv8515-top025-hrp` | TBD | ETF | full_universe | momentum_low_vol | hrp | annual | none | candidate | none | `candidate, low_vol_momentum, hierarchical_risk_parity, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momomac8515-top025-hrp` | `fu-momomac8515-top025-hrp` | TBD | ETF | full_universe | momentum_12m | hrp | annual | none | candidate | none | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo12-soft025-hrp` | `fu-momo12-soft025-hrp` | TBD | ETF | full_universe | momentum_12m | hrp | annual | none | candidate | none | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo12-lin050-hrp` | `fu-momo12-lin050-hrp` | TBD | ETF | full_universe | momentum_12m | hrp | annual | none | candidate | none | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo12-lin100-hrp` | `fu-momo12-lin100-hrp` | TBD | ETF | full_universe | momentum_12m | hrp | annual | none | candidate | none | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momolv8515-top025-hrp-month` | `fu-momolv-month` | ETF Momentum + Low Vol Tilt, HRP, Monthly | ETF | full_universe | momentum_low_vol | hrp | monthly | none | candidate | `stg-fu-momo12-top035-hrp` | `candidate, etf_universe, momentum, low_vol, monthly, hrp` | `universe=ETF, selection=full_universe, signal=momentum_low_vol, momentum_weight=0.85, low_vol_weight=0.15, top_weight=0.25, portfolio=hrp, schedule=monthly, overlay=none` | Momentum plus low-vol tilt candidate for robustness checks. |
| `stg-fu-momomac8515-top025-hrp-month` | `fu-momomac8515-top025-hrp-month` | TBD | ETF | full_universe | momentum_12m | hrp | monthly | none | candidate | none | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe, monthly` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo12-soft025-hrp-month` | `fu-momo12-soft025-hrp-month` | TBD | ETF | full_universe | momentum_12m | hrp | monthly | none | candidate | none | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe, monthly` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo12-lin050-hrp-month` | `fu-momo12-lin050-hrp-month` | TBD | ETF | full_universe | momentum_12m | hrp | monthly | none | candidate | none | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe, monthly` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo12-top035-mru` | `fu-momo12-top035-mru` | TBD | ETF | full_universe | momentum_12m | mean_risk_utility | annual | none | candidate | none | `candidate, momentum_tilt, mean_risk_utility, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo12-top035-mruc` | `fu-momo12-top035-mruc` | TBD | ETF | full_universe | momentum_12m | mean_risk_utility_conservative | annual | none | candidate | none | `candidate, momentum_tilt, mean_risk_utility_conservative, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-top3-eq` | `top3-eq` | TBD | TBD | top3 | momentum | equal_weight | annual | none | candidate | none | `candidate, concentrated_momentum, equal_weight` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-top3-rb` | `top3-rb` | TBD | TBD | top3 | momentum | risk_budgeting | annual | none | candidate | none | `candidate, concentrated_momentum, risk_budgeting` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-top3-minvar` | `top3-minvar` | TBD | TBD | top3 | momentum | minimum_variance | annual | none | candidate | none | `candidate, concentrated_momentum, minimum_variance` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-top3-hrp` | `top3-hrp` | ETF Top 3 Momentum, HRP | ETF | top3 | momentum | hrp | annual | none | candidate | `stg-fu-hrp` | `candidate, top3, momentum, hrp` | `universe=ETF, selection=top3, signal=momentum, portfolio=hrp, schedule=annual, overlay=none` | Concentrated top-3 momentum baseline for selection strength. |
| `stg-dualtop3-hrp` | `dualtop3-hrp` | ETF Dual Momentum Top 3, HRP | ETF | dual_momentum_top3 | momentum | hrp | annual | none | candidate | `stg-top3-hrp` | `candidate, dual_momentum, top3, hrp` | `universe=ETF, selection=dual_momentum_top3, signal=momentum, portfolio=hrp, schedule=annual, overlay=none` | Dual momentum variant for defensive selection behavior. |
| `stg-poslowvol-hrp` | `poslowvol-hrp` | TBD | TBD | positive_low_vol | momentum | hrp | annual | none | candidate | none | `candidate, low_vol_momentum, hierarchical_risk_parity` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-trailmomlowvol-hrp` | `trailmomlowvol-hrp` | TBD | TBD | trailing_momentum_low_vol | momentum | hrp | annual | none | candidate | none | `candidate, low_vol_momentum, hierarchical_risk_parity` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-posvol-hrp` | `posvol-hrp` | TBD | TBD | positive_volume | none | hrp | annual | none | candidate | none | `candidate, other, hierarchical_risk_parity` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-posmom-hrp-month` | `posmom-month` | ETF Positive Momentum, HRP, Monthly | ETF | positive_momentum | momentum | hrp | monthly | none | candidate | `stg-fu-hrp` | `candidate, positive_momentum, monthly, hrp` | `universe=ETF, selection=positive_momentum, signal=momentum, portfolio=hrp, schedule=monthly, overlay=none` | Monthly positive momentum universe for defensive participation. |
| `stg-posrev5-trend60-hrp-month` | `posrev5-trend60-hrp-month` | TBD | TBD | positive_reversal_trend | none | hrp | monthly | none | candidate | none | `candidate, other, hierarchical_risk_parity, monthly` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-riskoff-posmom-hrp-month` | `riskoff-posmom-month` | ETF Risk-Off Gated Positive Momentum, HRP, Monthly | ETF | positive_momentum | momentum | hrp | monthly | risk_off | candidate | `stg-posmom-hrp-month` | `candidate, risk_regime, positive_momentum, monthly, hrp` | `universe=ETF, selection=positive_momentum, signal=momentum, portfolio=hrp, schedule=monthly, overlay=risk_off` | Risk-regime gated positive momentum candidate. |
| `stg-etf-momo12-top035-hrp` | `etf-momo12-top035-hrp` | TBD | ETF | full_universe | momentum_12m | hrp | annual | none | candidate | none | `candidate, momentum_tilt, hierarchical_risk_parity` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo9-top035-hrp-1w` | `fu-momo9-top035-hrp-1w` | TBD | ETF | full_universe | momentum_9m | hrp | weekly | none | candidate | none | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo9-top035-hrp-1mo` | `fu-momo9-top035-hrp-1mo` | TBD | ETF | full_universe | momentum_9m | hrp | monthly | none | candidate | none | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo2-top035-hrp-month` | `fu-momo2-month` | ETF 2M Momentum Tilt, HRP, Monthly | ETF | full_universe | momentum_2m_top_weighted | hrp | monthly | none | candidate | `stg-fu-hrp` | `candidate, etf_universe, momentum, 2m, monthly, hrp` | `universe=ETF, selection=full_universe, signal=momentum_2m, top_weight=0.35, portfolio=hrp, schedule=monthly, overlay=none` | Core monthly 2-month momentum candidate for faster adaptation. |
| `stg-fu-momo2-top035-pred10mom9010-hrp-month` | `fu-momo2-top035-pred10mom9010-hrp-month` | TBD | ETF | full_universe | momentum_2m | hrp | monthly | predictor | candidate | none | `candidate, predictor_augmented_momentum, hierarchical_risk_parity, etf_universe, predictor, monthly` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo2-top035-pred10mom5050-pw40-hrp-month` | `fu-momo2-pred-month` | ETF 2M Momentum + Predictor Overlay, HRP, Monthly | ETF | full_universe | momentum_2m_top_weighted | hrp | monthly | predictor | candidate | `stg-fu-momo2-top035-hrp-month` | `candidate, predictor, momentum, 2m, monthly, hrp` | `universe=ETF, selection=full_universe, signal=momentum_2m, top_weight=0.35, portfolio=hrp, schedule=monthly, overlay=predictor, predictor=10bar_momentum_50_50, predictor_weight=0.40` | Main predictor-augmented candidate against the 2-month momentum baseline. |
| `stg-fu-momo2-top035-pred10mom5050-pw40-hrp-daily` | `fu-momo2-top035-pred10mom5050-pw40-hrp-daily` | TBD | ETF | full_universe | momentum_2m | hrp | daily | predictor | candidate | none | `candidate, predictor_augmented_momentum, hierarchical_risk_parity, etf_universe, predictor` | TBD | Generated backlog entry. Add explicit research metadata before promoting. |

## Relationship To Evaluation Plan

Strategyを評価対象へ載せるかどうかは、このカタログでは決めない。評価対象セットは [Evaluation Plan](./evaluation-plan.md) で別途定義する。
