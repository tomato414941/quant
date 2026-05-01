# Quant

Research code and documentation for portfolio strategy evaluation.

This repository is a work-in-progress quant research sandbox. It focuses on defining strategy candidates, evaluation contexts, backtest/run plumbing, and human-readable documentation for comparing portfolio construction ideas.

## Status

This project is experimental and under active design. The documentation is currently the source of truth for evaluation scope and naming decisions, while some implementation helpers still reflect earlier compatibility paths.

## Important Disclaimer

This repository is for research and software development only.

Nothing in this repository is investment advice, financial advice, trading advice, or a recommendation to buy, sell, or hold any asset. Backtests and simulations are not guarantees of future performance. Use at your own risk.

## Repository Layout

- `backend/`: Python backend, strategy definitions, evaluation helpers, run storage, API and CLI entrypoints.
- `docs/`: Human-readable strategy catalog, evaluation context definitions, evaluation plan, leaderboard notes, and design documents.
- `scripts/`: Utility scripts.

## Development

The backend uses `uv` as the primary development entrypoint.

```bash
cd backend
uv sync --dev
uv run pytest -q tests/test_api.py::test_healthcheck tests/test_api.py::test_strategy_inventory_api tests/test_dependencies.py tests/test_instrument_registry.py tests/test_evaluation_profiles.py
```

This is the smoke test: it is intended as a quick sanity check for setup,
inventory, dependencies, and evaluation profile wiring. The full test suite is
heavier and is described in [docs/development.md](docs/development.md). Use the
expanded smoke command in that document when you need the exact current smoke
gate, including CORS parsing checks.

Run the committed golden snapshot evaluation without live market data:

```bash
cd backend
uv run quant comparison-summary \
  --snapshot-spec-file tests/fixtures/golden_market/comparison-run-spec.json \
  --market-snapshot-dir tests/fixtures/golden_market/market_snapshots \
  --universe only_etf \
  --strategy-key stg-fu-eq \
  --top 1
```

There are two evaluation entrypoints with different meanings:

- `comparison-summary`: strategy comparison on the existing comparison/run plumbing. Use this for the current holdout-style research comparison workflow.
- `backtest-strategy`: full-period backtest from a fixed market snapshot. Use this when you want to ask "what happened over the whole snapshot period?" for a supported strategy.

Use `backtest-strategy` for strategy-level full-period backtests. The lower-level
`backtest-equal-weight` command is kept as a compatibility/debugging entrypoint
for the equal-weight engine. Prefer the shared `backtest-strategy` entrypoint
instead of adding one top-level CLI per strategy.

The currently supported `backtest-strategy` keys are listed in
[Full-Period Backtest Plan](docs/backtest-plan.md).

```bash
cd backend
uv run quant backtest-strategy \
  --strategy-key stg-fu-eq \
  --snapshot-id 0d1aaa50ac7a9f4c7e82ee584c7a1bb0a6ce27f9434bf1884bcaee9a4432cd88 \
  --market-snapshot-dir tests/fixtures/golden_market/market_snapshots
```

## What To Read First

Start here if you are viewing the repository for the first time:

1. [Development](docs/development.md): local setup, smoke test, full test, CLI, API commands, and ad-hoc result regeneration.
2. [Strategy Catalog](docs/strategy-catalog.md): strategy definitions and review status.
3. [Evaluation Contexts](docs/evaluation-contexts.md): measurement conditions used to evaluate strategies.
4. [Evaluation Plan](docs/evaluation-plan.md): current strategy set and context selected for evaluation.
5. [ETF Full-Universe No-Signal Allocation](docs/leaderboard/etf-full-universe-no-signal-allocation.md): current manual comparison snapshot for the 16 ETF allocation/rebalance variants.
6. [Full-Period Backtest Results](docs/backtest-results.md): selected reproducible fixed-snapshot backtest records.
7. [Full-Period Backtest Plan](docs/backtest-plan.md): short-term plan for extending `backtest-strategy`.

The leaderboard is not an automatically generated ranking. It is a concise,
human-maintained snapshot of selected evaluated runs.

Documentation responsibility:

- `docs/leaderboard/etf-full-universe-no-signal-allocation.md` is the manual comparison summary. Keep selected rows,
  ranks, and interpretation there.
- `docs/backtest-results.md` is the reproducibility log. Record selected results
  with snapshot IDs, commands, and source fields, not every ad-hoc run.
- `backend/data/` is for ignored local outputs and provider-downloaded data. Files
  such as generated backtest matrices can be regenerated and should not be treated
  as documentation source of truth.

## Data And Secrets

Local data, run results, environment files, credentials, and private keys must not be committed. Generated run data belongs under ignored data directories such as `backend/data/`.

Market data snapshots are metadata-only. Commit snapshot contracts, fingerprints, and provenance fields, but do not commit raw prices, volumes, provider downloads, or generated market data panels.
Tiny synthetic CSV fixtures under `backend/tests/fixtures/` are the exception; they are not provider data and exist only to keep the CLI evaluation path reproducible in tests.

Use ignored directories such as `private/` for local research notes or evaluation notes that should not be published.

Before making the repository public, run a secret scan and review the strategy documentation for research ideas you do not want to publish.

## License

MIT License. See [LICENSE](LICENSE).
