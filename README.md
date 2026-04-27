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
heavier and is described in [docs/development.md](docs/development.md).

## What To Read First

Start here if you are viewing the repository for the first time:

1. [Development](docs/development.md): local setup, smoke test, full test, CLI, and API commands.
2. [Strategy Catalog](docs/strategy-catalog.md): strategy definitions and review status.
3. [Evaluation Contexts](docs/evaluation-contexts.md): measurement conditions used to evaluate strategies.
4. [Evaluation Plan](docs/evaluation-plan.md): current strategy set and context selected for evaluation.
5. [Leaderboard](docs/leaderboard.md): manual snapshots for comparing evaluated runs.

The leaderboard is not an automatically generated ranking. It is a concise,
human-maintained snapshot of selected evaluated runs.

## Data And Secrets

Local data, run results, environment files, credentials, and private keys must not be committed. Generated run data belongs under ignored data directories such as `backend/data/`.

Use ignored directories such as `private/` for local research notes or evaluation notes that should not be published.

Before making the repository public, run a secret scan and review the strategy documentation for research ideas you do not want to publish.

## License

MIT License. See [LICENSE](LICENSE).
