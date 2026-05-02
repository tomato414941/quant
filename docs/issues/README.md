# Issues

このディレクトリは、今すぐ実装しないが忘れると困る課題を小さく記録する場所である。

実装計画や採用判断ではなく、現在の制約、影響、必要になったときの最小対応だけを書く。

## Current Issues

| Issue | Summary |
| --- | --- |
| [Cost-Aware No-Trade Default](./cost-aware-no-trade-default.md) | `cost_aware_no_trade` がdefaultのため、signal-derived edge proxyが通常経路に見える。 |
| [Portfolio Execution Forecast Coupling](./portfolio-execution-forecast-coupling.md) | portfolio executionがforecast constructionを直接呼んでいる。 |
| [Predictor Feature Recipe Ownership](./predictor-feature-recipe-ownership.md) | `FeatureSpec` がranking-derived feature construction recipeまで持っている。 |
| [Result Lookup](./result-lookup.md) | `resultId` からCLI/APIで結果を引けない。 |
