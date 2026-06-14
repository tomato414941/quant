# Predictor Run Sort Contract

## Issue

`/api/predictor-runs` exposes `sort_by` and defaults to `test_rank_ic`.

Supported sort keys are currently:

- `test_rank_ic`
- `overall_rank_ic`
- `test_top_minus_bottom`
- `overall_top_minus_bottom`
- `saved_at`

## Current State

Current flow:

```text
predictor compact records
  -> sort_predictor_run_records(sort_by)
  -> top records
```

The default API response ranks predictor runs by `testRankIc`, with tie-breakers
on `testTopMinusBottomPct`, `overallRankIc`, and `runKey`.

## Concern

Metric sorting is useful for predictor / forecast research, but the default sort
metric can imply an adoption preference.

`testRankIc` may be a reasonable research metric, but it is not necessarily the
only or final decision criterion. A simpler default may be `saved_at`, with
metric sorting used only when explicitly requested.

The current API contract also adds more surface area to maintain before there is
clear evidence that all sort keys are used.

## Direction

Do not change the default now. It is part of the current API behavior and tests.

Before changing it, check actual API/UI/notebook usage. If metric sorting is
mostly exploratory, consider using `saved_at` as the default and keeping metric
sorts as explicit query options.
