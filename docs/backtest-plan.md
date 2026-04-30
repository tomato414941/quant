# Full-Period Backtest Plan

この文書は、`backtest-strategy` をどう広げるかの短期計画である。

目的は、full-period backtest の入口を増やさず、1つのstrategy-level CLIで少しずつ対応Strategyを増やすこと。汎用エンジン化や43Strategy一括対応はまだ行わない。

## Current State

`backtest-strategy` currently supports 16 full-universe ETF variants:

| Portfolio model | Annual | Monthly | Weekly | Daily |
| --- | --- | --- | --- | --- |
| Equal weight | `stg-fu-eq` | `stg-fu-eq-month` | `stg-fu-eq-week` | `stg-fu-eq-day` |
| Risk budgeting | `stg-fu-rb` | `stg-fu-rb-month` | `stg-fu-rb-week` | `stg-fu-rb-day` |
| Minimum variance | `stg-fu-minvar` | `stg-fu-minvar-month` | `stg-fu-minvar-week` | `stg-fu-minvar-day` |
| HRP | `stg-fu-hrp` | `stg-fu-hrp-month` | `stg-fu-hrp-week` | `stg-fu-hrp-day` |

Portfolio-model strategies fall back to equal weight when rebalance history is
insufficient.

`backtest-equal-weight` remains as a lower-level compatibility/debugging entrypoint. Prefer the shared `backtest-strategy` entrypoint for future strategy-level runs.

## Last Completed Candidate

The latest completed step is adding monthly, weekly, and daily rebalance
variants for the four full-universe ETF portfolio models.

Reason:

- It keeps the universe, selection policy, and overlay fixed.
- It isolates rebalance-frequency sensitivity across the same four portfolio
  construction models.
- It does not require adding momentum, predictor, or risk-regime selection logic
  to the full-period backtest runner.

It was added after the four annual full-universe portfolio models remained
stable under the fixed snapshot tests.

## Dispatch Decision

The CLI now uses a small static dispatch map because portfolio-model strategies use a
different runner and portfolio model setup from `stg-fu-eq`.

Acceptable dispatch shape:

```python
BACKTEST_STRATEGY_RUNNERS = {
    "stg-fu-eq": run_stg_fu_eq_backtest,
    "stg-fu-rb": run_stg_fu_rb_backtest,
    "stg-fu-minvar": run_stg_fu_minvar_backtest,
    "stg-fu-hrp": run_stg_fu_hrp_backtest,
}
```

Still avoid:

- automatic strategy discovery
- plugin registration
- generic execution of all existing 43 strategies
- new top-level CLI commands per strategy
- coupling `backtest-strategy` to the existing holdout comparison workflow

## Verification Gate Used For Portfolio Model Strategies

`stg-fu-rb`, `stg-fu-minvar`, and `stg-fu-hrp` added the following test
contract:

- committed golden snapshot CLI test
- exact `result.summary` values for the golden fixture
- `firstInvestedDate`
- `seriesCount`
- `eventCount`
- unsupported strategy still fails clearly

Keep this gate for the next strategy added to `backtest-strategy`.
