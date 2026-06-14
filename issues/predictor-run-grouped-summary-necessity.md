# Predictor Run Grouped Summary Necessity

## Issue

`/api/predictor-runs` returns `groupedSummaries` in addition to the sorted
predictor run records.

The grouped summaries select the best predictor run within groups such as signal
source, signal source plus horizon, signal source plus period, and signal source
plus learner/combiner.

## Current State

Current flow:

```text
predictor compact records
  -> sort_predictor_run_records()
  -> records
  -> summarize_predictor_record_groups()
  -> groupedSummaries
```

The best record in each group is selected primarily by `testRankIc`, then
`testTopMinusBottomPct`, then `overallRankIc`.

This is part of the current API response shape and is covered by API tests.

## Concern

It is not clear that grouped summaries are necessary in the product sense.

They are useful for quickly scanning predictor research results, but they also
make the API payload larger and more opinionated. The best-run criterion is
fixed around `testRankIc`, while actual research decisions may use different
metrics or may be done in notebooks or client-side tables.

The plain `records` list may be enough for the current project stage.

## Direction

Do not remove `groupedSummaries` now. It is part of the current API contract.

Before changing it, check whether any UI, notebook, or workflow actually uses the
grouped summaries. If not, consider either removing them from the default payload
or making them opt-in through a query option.
