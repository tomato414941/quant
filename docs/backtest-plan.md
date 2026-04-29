# Full-Period Backtest Plan

この文書は、`backtest-strategy` をどう広げるかの短期計画である。

目的は、full-period backtest の入口を増やさず、1つのstrategy-level CLIで少しずつ対応Strategyを増やすこと。汎用エンジン化や43Strategy一括対応はまだ行わない。

## Current State

`backtest-strategy` currently supports:

| Strategy | Status | Notes |
| --- | --- | --- |
| `stg-fu-eq` | implemented | Full-period equal-weight backtest from a fixed market snapshot. |

`backtest-equal-weight` remains as a lower-level compatibility/debugging entrypoint. Prefer the shared `backtest-strategy` entrypoint for future strategy-level runs.

## Next Candidate

The next candidate is `stg-fu-rb`.

Reason:

- It is already a public reviewed Strategy in [Strategy Catalog](./strategy-catalog.md).
- It uses the same ETF full universe, no selection signal, annual rebalance, and no overlay as `stg-fu-eq`.
- The main difference from `stg-fu-eq` is the portfolio model: equal-risk-contribution / risk budgeting.
- It is simpler to reason about than minimum variance, HRP, predictor-augmented, defensive momentum, or low-vol momentum variants.

Implement it only after `stg-fu-eq` remains stable under the fixed snapshot tests.

## Dispatch Decision

Keep the current direct guard while only `stg-fu-eq` is supported.

When adding `stg-fu-rb`, decide between these two options at implementation time:

1. Keep a direct branch if the second implementation is still a tiny wrapper.
2. Introduce a small static dispatch map if the second implementation would otherwise duplicate CLI branching or payload construction.

Use a dispatch map when the new strategy needs a different runner, input preparation, validation, or payload metadata. Avoid adding it merely to replace a single `if` branch.

Acceptable dispatch shape:

```python
BACKTEST_STRATEGY_RUNNERS = {
    "stg-fu-eq": run_stg_fu_eq_backtest,
    "stg-fu-rb": run_stg_fu_rb_backtest,
}
```

Avoid for the next step:

- automatic strategy discovery
- plugin registration
- generic execution of all existing 43 strategies
- new top-level CLI commands per strategy
- coupling `backtest-strategy` to the existing holdout comparison workflow

## Verification Gate For Adding stg-fu-rb

Before adding `stg-fu-rb`, define the test contract first:

- committed golden snapshot CLI test
- exact `result.summary` values for the golden fixture
- `firstInvestedDate`
- `seriesCount`
- `eventCount`
- unsupported strategy still fails clearly

Only after that should the runner implementation be added.
