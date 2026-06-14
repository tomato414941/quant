# Cost-Aware No-Trade Default

## Issue

`cost_aware_no_trade` がdefault decision policyになっているため、通常のstrategy executionでも
signal-derived edge proxyが必要な経路に見える。

`compute_signal_return_proxy_series()` は本物のexpected return modelではなく、ranking
scoreを売買コスト比較用の簡易edge proxyへ変換する補助処理である。これがdefault経路に入ると、
signal score、forecast、execution decisionの責務が混ざって見える。

## Current State

- default decision policyは `cost_aware_no_trade`。
- `build_portfolio_decision()` はdefaultでforecast snapshotを作り、edgeとcostを比較する。
- `build_strategy_forecast_snapshot()` はstrategy scoreから `expected_return_proxy` を作る。
- `compute_signal_return_proxy_series()` はそのproxyを作るが、学習済みforecastではない。

## Concern

通常のallocation/backtestに、cost-aware no-trade用のproxy forecastが必須であるかのように見える。

本来は、cost-aware no-tradeを明示的に有効にした場合だけ、
signal scoreからedge proxyを作り、transaction costと比較するのが自然である。

## Direction

すぐにdefaultを変更しない。既存runやテストの挙動が変わるため。

まず次を確認する。

- default decision policyを `cost_aware_no_trade` にしている理由がまだ有効か。
- `direct_score_to_weight` など、proxy forecastを必要としないpolicyをdefaultにできるか。
- `expected_return_proxy` という名前が実態より強すぎないか。
- `compute_signal_return_proxy_series()` をcost-aware no-trade専用の補助として隔離できるか。

必要になったら、default変更は小さなPRで行い、既存評価結果への影響を明示する。

