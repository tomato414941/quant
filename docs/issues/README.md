# Issues

このディレクトリは、今すぐ実装しないが忘れると困る課題を小さく記録する場所である。

実装計画や採用判断ではなく、現在の制約、影響、必要になったときの最小対応だけを書く。

## Current Issues

| Issue | Summary |
| --- | --- |
| [Comparison Module Responsibility Boundary](./comparison-module-responsibility-boundary.md) | comparison系モジュールのserialization/payload/diagnostics境界が曖昧。 |
| [Cost-Aware No-Trade Default](./cost-aware-no-trade-default.md) | `cost_aware_no_trade` がdefaultのため、signal-derived edge proxyが通常経路に見える。 |
| [Portfolio Execution Forecast Coupling](./portfolio-execution-forecast-coupling.md) | portfolio executionがforecast constructionを直接呼んでいる。 |
| [Predictor Feature Recipe Ownership](./predictor-feature-recipe-ownership.md) | `FeatureSpec` がranking-derived feature construction recipeまで持っている。 |
| [Predictor Run Grouped Summary Necessity](./predictor-run-grouped-summary-necessity.md) | `/api/predictor-runs` の `groupedSummaries` が本当に必要か未確認。 |
| [Result Lookup](./result-lookup.md) | `resultId` からCLI/APIで結果を引けない。 |
| [Run Index Compact Record Necessity](./run-index-compact-record-necessity.md) | run indexに`genericCompactRecord`を永続化する必要性が未確認。 |
