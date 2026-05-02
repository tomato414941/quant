# Portfolio Execution Forecast Coupling

## Issue

`portfolio_execution` がforecast constructionを直接呼んでいる。

現状は `build_portfolio_decision()` の中で `build_decision_forecast_snapshot()` を呼び、
そこから `build_strategy_forecast_snapshot()` を呼んでいる。そのため、portfolio executionが
rebalance/no-trade判断だけでなく、strategy score、ranking signal、predictor panel、
signal-derived edge proxyの作り方まで知っている。

## Current State

Current dependency flow:

```text
build_portfolio_decision()
  -> build_decision_forecast_snapshot()
    -> build_strategy_forecast_snapshot()
      -> compute_signal_return_proxy_series()
```

Execution decision then uses the forecast through:

```text
compute_forecast_edge()
```

## Concern

Portfolio execution should decide how to use edge, confidence, turnover, and cost.
It should not need to know how forecast or signal-derived edge proxy is built.

This coupling makes forecast look like a normal execution prerequisite, even when
it should be optional and policy-specific.

It also makes decision tests heavier than necessary because testing no-trade logic
can require strategy score and predictor setup.

## Direction

Do not change behavior immediately.

First, separate the boundary conceptually:

- Forecast/signal layer builds optional edge and confidence inputs.
- Portfolio allocation layer builds target weights.
- Portfolio execution layer consumes current weights, target weights, cost, edge,
  confidence, and policy to decide rebalance or no-trade.

A small future change could add an optional decision context to
`build_portfolio_decision()`, then move forecast construction to the caller. After
that, `portfolio_execution.py` should no longer import `build_strategy_forecast_snapshot`.

