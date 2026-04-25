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
- `Rebalance Frequency`: 配分を見直す頻度。例: `annual`, `monthly`, `weekly`, `daily`。
- `Overlay`: 追加的なゲートや予測補助。例: `none`, `risk_off`, `predictor`。
- `Role`: 比較上の役割。例: `reference`, `reference_candidate`, `candidate`。
- `Tags`: Strategyの性質を補助的に表す。評価対象かどうか、採用可否、優先順位は表さない。

この文書は評価結果を書く場所ではない。評価済みrunの比較は [Leaderboard](./leaderboard.md) に置く。

## Naming Convention

StrategyのIDと名前は分けて扱う。

- `Strategy ID`: コードや評価結果と対応させるための安定参照ID。将来的には意味を持たない不透明IDへ移行する。
- `Slug`: 人間が短く参照するための短縮名。Strategyの内容を軽く示してよいが、全パラメータを詰め込まない。
- `Display Name`: 議論や表示で使う自然言語名。`Universe` という語は使わず、対象資産集合は `ETF` のように短く表す。
- `Key Parameters`: 比較に効く主要パラメータ。IDへ全て詰め込まない。
- 現在の `stg-fu-...` 形式のIDは互換用の既存IDであり、新しい命名としては推奨しない。
- 将来の `Strategy ID` は `stg-` + ランダムな短い英数字のような不透明IDを想定する。IDからStrategyの内容を推測してはいけない。

## Strategies

| Strategy ID | Slug | Display Name | Universe | Selection | Signal | Portfolio Model | Rebalance Frequency | Overlay | Role | Tags | Key Parameters | Human Reviewed | Reviewed At | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `stg-fu-eq` | `etf-eq` | ETF Equal Weight Allocation | ETF | full_universe | none | equal_weight | annual | none | reference | `etf_universe, full_universe, equal_weight, reference_point` | `universe=ETF, selection=full_universe, signal=none, portfolio=equal_weight, rebalance_frequency=annual, overlay=none` | yes | 2026-04-25 | ETF equal-weight baseline for return and drawdown context. Strategy ID and slug are expected to change when naming is cleaned up. Proposed opaque Strategy ID: `stg-k7m4qa`. |
| `stg-fu-rb` | `etf-erc` | ETF Equal Risk Contribution Allocation | ETF | full_universe | none | equal_risk_contribution | annual | none | reference_candidate | `etf_universe, full_universe, equal_risk_contribution` | `universe=ETF, selection=full_universe, signal=none, portfolio=equal_risk_contribution, rebalance_frequency=annual, overlay=none, code_portfolio_model=risk_budgeting` | yes | 2026-04-25 | ETF full-universe reference candidate. Current code model is named `risk_budgeting` but behaves like equal-risk-contribution allocation. Strategy ID and slug are expected to change when naming is cleaned up. Proposed opaque Strategy ID: `stg-p9x2vt`. |
| `stg-fu-minvar` | `etf-minvar` | ETF Minimum Variance Allocation | ETF | full_universe | none | minimum_variance | annual | none | reference_candidate | `etf_universe, full_universe, minimum_variance` | `universe=ETF, selection=full_universe, signal=none, portfolio=minimum_variance, rebalance_frequency=annual, overlay=none` | yes | 2026-04-25 | ETF full-universe minimum variance allocation reference candidate. Strategy ID and slug are expected to change when naming is cleaned up. Proposed opaque Strategy ID: `stg-r4n8cw`. |
| `stg-fu-hrp` | `etf-hrp` | ETF Hierarchical Risk Parity Allocation | ETF | full_universe | none | hrp | annual | none | reference | `etf_universe, full_universe, hrp, reference_point` | `universe=ETF, selection=full_universe, signal=none, portfolio=hrp, rebalance_frequency=annual, overlay=none` | yes | 2026-04-25 | ETF HRP baseline for portfolio construction effects. Proposed opaque Strategy ID: `stg-h6q3dz`. |
| `stg-fu-momo12-lin025-hrp` | `fu-momo12-lin025-hrp` | TBD | ETF | full_universe | momentum_12m | hrp | annual | none | candidate | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | no |  | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo12-top035-hrp` | `fu-momo12-hrp` | ETF 12M Momentum Tilt, HRP | ETF | full_universe | momentum_12m_top_weighted | hrp | annual | none | candidate | `candidate, etf_universe, momentum, 12m, hrp` | `universe=ETF, selection=full_universe, signal=momentum_12m, top_weight=0.35, portfolio=hrp, rebalance_frequency=annual, overlay=none` | no |  | Core annual 12-month momentum tilt candidate. |
| `stg-fu-momo6-top035-hrp` | `fu-momo6-top035-hrp` | TBD | ETF | full_universe | momentum_6m | hrp | annual | none | candidate | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | no |  | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo9-top035-hrp` | `fu-momo9-top035-hrp` | TBD | ETF | full_universe | momentum_9m | hrp | annual | none | candidate | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | no |  | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo8-top035-hrp` | `fu-momo8-top035-hrp` | TBD | ETF | full_universe | momentum_8m | hrp | annual | none | candidate | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | no |  | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo10-top035-hrp` | `fu-momo10-top035-hrp` | TBD | ETF | full_universe | momentum_10m | hrp | annual | none | candidate | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | no |  | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo11-top035-hrp` | `fu-momo11-top035-hrp` | TBD | ETF | full_universe | momentum_11m | hrp | annual | none | candidate | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | no |  | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo3-top035-hrp` | `fu-momo3-top035-hrp` | TBD | ETF | full_universe | momentum_3m | hrp | annual | none | candidate | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | no |  | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo15-top035-hrp` | `fu-momo15-top035-hrp` | TBD | ETF | full_universe | momentum_15m | hrp | annual | none | candidate | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | no |  | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momolv7030-top025-hrp` | `fu-momolv7030-top025-hrp` | TBD | ETF | full_universe | momentum_low_vol | hrp | annual | none | candidate | `candidate, low_vol_momentum, hierarchical_risk_parity, etf_universe` | TBD | no |  | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momolv8515-top025-hrp` | `fu-momolv8515-top025-hrp` | TBD | ETF | full_universe | momentum_low_vol | hrp | annual | none | candidate | `candidate, low_vol_momentum, hierarchical_risk_parity, etf_universe` | TBD | no |  | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momomac8515-top025-hrp` | `fu-momomac8515-top025-hrp` | TBD | ETF | full_universe | momentum_12m | hrp | annual | none | candidate | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | no |  | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo12-soft025-hrp` | `fu-momo12-soft025-hrp` | TBD | ETF | full_universe | momentum_12m | hrp | annual | none | candidate | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | no |  | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo12-lin050-hrp` | `fu-momo12-lin050-hrp` | TBD | ETF | full_universe | momentum_12m | hrp | annual | none | candidate | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | no |  | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo12-lin100-hrp` | `fu-momo12-lin100-hrp` | TBD | ETF | full_universe | momentum_12m | hrp | annual | none | candidate | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | no |  | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momolv8515-top025-hrp-month` | `fu-momolv-month` | ETF Momentum + Low Vol Tilt, HRP, Monthly | ETF | full_universe | momentum_low_vol | hrp | monthly | none | candidate | `candidate, etf_universe, momentum, low_vol, monthly, hrp` | `universe=ETF, selection=full_universe, signal=momentum_low_vol, momentum_weight=0.85, low_vol_weight=0.15, top_weight=0.25, portfolio=hrp, rebalance_frequency=monthly, overlay=none` | no |  | Momentum plus low-vol tilt candidate for robustness checks. |
| `stg-fu-momomac8515-top025-hrp-month` | `fu-momomac8515-top025-hrp-month` | TBD | ETF | full_universe | momentum_12m | hrp | monthly | none | candidate | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe, monthly` | TBD | no |  | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo12-soft025-hrp-month` | `fu-momo12-soft025-hrp-month` | TBD | ETF | full_universe | momentum_12m | hrp | monthly | none | candidate | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe, monthly` | TBD | no |  | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo12-lin050-hrp-month` | `fu-momo12-lin050-hrp-month` | TBD | ETF | full_universe | momentum_12m | hrp | monthly | none | candidate | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe, monthly` | TBD | no |  | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo12-top035-mru` | `fu-momo12-top035-mru` | TBD | ETF | full_universe | momentum_12m | mean_risk_utility | annual | none | candidate | `candidate, momentum_tilt, mean_risk_utility, etf_universe` | TBD | no |  | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo12-top035-mruc` | `fu-momo12-top035-mruc` | TBD | ETF | full_universe | momentum_12m | mean_risk_utility_conservative | annual | none | candidate | `candidate, momentum_tilt, mean_risk_utility_conservative, etf_universe` | TBD | no |  | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-top3-eq` | `top3-eq` | TBD | TBD | top3 | momentum | equal_weight | annual | none | candidate | `candidate, concentrated_momentum, equal_weight` | TBD | no |  | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-top3-rb` | `top3-rb` | TBD | TBD | top3 | momentum | risk_budgeting | annual | none | candidate | `candidate, concentrated_momentum, risk_budgeting` | TBD | no |  | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-top3-minvar` | `top3-minvar` | TBD | TBD | top3 | momentum | minimum_variance | annual | none | candidate | `candidate, concentrated_momentum, minimum_variance` | TBD | no |  | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-top3-hrp` | `top3-hrp` | ETF Top 3 Momentum, HRP | ETF | top3 | momentum | hrp | annual | none | candidate | `candidate, top3, momentum, hrp` | `universe=ETF, selection=top3, signal=momentum, portfolio=hrp, rebalance_frequency=annual, overlay=none` | no |  | Concentrated top-3 momentum baseline for selection strength. |
| `stg-dualtop3-hrp` | `dualtop3-hrp` | ETF Dual Momentum Top 3, HRP | ETF | dual_momentum_top3 | momentum | hrp | annual | none | candidate | `candidate, dual_momentum, top3, hrp` | `universe=ETF, selection=dual_momentum_top3, signal=momentum, portfolio=hrp, rebalance_frequency=annual, overlay=none` | no |  | Dual momentum variant for defensive selection behavior. |
| `stg-poslowvol-hrp` | `poslowvol-hrp` | TBD | TBD | positive_low_vol | momentum | hrp | annual | none | candidate | `candidate, low_vol_momentum, hierarchical_risk_parity` | TBD | no |  | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-trailmomlowvol-hrp` | `trailmomlowvol-hrp` | TBD | TBD | trailing_momentum_low_vol | momentum | hrp | annual | none | candidate | `candidate, low_vol_momentum, hierarchical_risk_parity` | TBD | no |  | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-posvol-hrp` | `posvol-hrp` | TBD | TBD | positive_volume | none | hrp | annual | none | candidate | `candidate, other, hierarchical_risk_parity` | TBD | no |  | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-posmom-hrp-month` | `posmom-month` | ETF Positive Momentum, HRP, Monthly | ETF | positive_momentum | momentum | hrp | monthly | none | candidate | `candidate, positive_momentum, monthly, hrp` | `universe=ETF, selection=positive_momentum, signal=momentum, portfolio=hrp, rebalance_frequency=monthly, overlay=none` | no |  | Monthly positive momentum universe for defensive participation. |
| `stg-posrev5-trend60-hrp-month` | `posrev5-trend60-hrp-month` | TBD | TBD | positive_reversal_trend | none | hrp | monthly | none | candidate | `candidate, other, hierarchical_risk_parity, monthly` | TBD | no |  | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-riskoff-posmom-hrp-month` | `riskoff-posmom-month` | ETF Risk-Off Gated Positive Momentum, HRP, Monthly | ETF | positive_momentum | momentum | hrp | monthly | risk_off | candidate | `candidate, risk_regime, positive_momentum, monthly, hrp` | `universe=ETF, selection=positive_momentum, signal=momentum, portfolio=hrp, rebalance_frequency=monthly, overlay=risk_off` | no |  | Risk-regime gated positive momentum candidate. |
| `stg-etf-momo12-top035-hrp` | `etf-momo12-top035-hrp` | TBD | ETF | full_universe | momentum_12m | hrp | annual | none | candidate | `candidate, momentum_tilt, hierarchical_risk_parity` | TBD | no |  | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo9-top035-hrp-1w` | `fu-momo9-top035-hrp-1w` | TBD | ETF | full_universe | momentum_9m | hrp | weekly | none | candidate | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | no |  | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo9-top035-hrp-1mo` | `fu-momo9-top035-hrp-1mo` | TBD | ETF | full_universe | momentum_9m | hrp | monthly | none | candidate | `candidate, momentum_tilt, hierarchical_risk_parity, etf_universe` | TBD | no |  | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo2-top035-hrp-month` | `fu-momo2-month` | ETF 2M Momentum Tilt, HRP, Monthly | ETF | full_universe | momentum_2m_top_weighted | hrp | monthly | none | candidate | `candidate, etf_universe, momentum, 2m, monthly, hrp` | `universe=ETF, selection=full_universe, signal=momentum_2m, top_weight=0.35, portfolio=hrp, rebalance_frequency=monthly, overlay=none` | no |  | Core monthly 2-month momentum candidate for faster adaptation. |
| `stg-fu-momo2-top035-pred10mom9010-hrp-month` | `fu-momo2-top035-pred10mom9010-hrp-month` | TBD | ETF | full_universe | momentum_2m | hrp | monthly | predictor | candidate | `candidate, predictor_augmented_momentum, hierarchical_risk_parity, etf_universe, predictor, monthly` | TBD | no |  | Generated backlog entry. Add explicit research metadata before promoting. |
| `stg-fu-momo2-top035-pred10mom5050-pw40-hrp-month` | `fu-momo2-pred-month` | ETF 2M Momentum + Predictor Overlay, HRP, Monthly | ETF | full_universe | momentum_2m_top_weighted | hrp | monthly | predictor | candidate | `candidate, predictor, momentum, 2m, monthly, hrp` | `universe=ETF, selection=full_universe, signal=momentum_2m, top_weight=0.35, portfolio=hrp, rebalance_frequency=monthly, overlay=predictor, predictor=10bar_momentum_50_50, predictor_weight=0.40` | no |  | Main predictor-augmented candidate against the 2-month momentum baseline. |
| `stg-fu-momo2-top035-pred10mom5050-pw40-hrp-daily` | `fu-momo2-top035-pred10mom5050-pw40-hrp-daily` | TBD | ETF | full_universe | momentum_2m | hrp | daily | predictor | candidate | `candidate, predictor_augmented_momentum, hierarchical_risk_parity, etf_universe, predictor` | TBD | no |  | Generated backlog entry. Add explicit research metadata before promoting. |

## Relationship To Evaluation Plan

Strategyを評価対象へ載せるかどうかは、このカタログでは決めない。評価対象セットは [Evaluation Plan](./evaluation-plan.md) で別途定義する。
