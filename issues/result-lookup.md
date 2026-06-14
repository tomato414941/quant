# Result Lookup

## Issue

`ETF Full-Universe No-Signal Allocation` には16本の各結果に `resultId` があるが、現状はそのIDからCLI/APIで結果を引けない。

## Current State

- `resultId` は `scripts/backtest_matrix.py` が生成する。
- leaderboardには `resultId` を手動で記録している。
- run storeやAPI indexには、この `resultId` は保存されていない。

## Direction

必要になったら、保存済みmatrix JSONや軽いlookup helperから始める。run storeへの統合は、必要性が明確になるまで急がない。
