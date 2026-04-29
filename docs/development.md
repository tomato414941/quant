# Development

このプロジェクトの backend は `uv` を正本の開発入口にする。

## Setup

```bash
cd backend
uv sync --dev
```

既存の `requirements.txt` と `requirements-dev.txt` は移行期間の互換ファイルとして残す。
新しい依存は `pyproject.toml` を先に更新する。

## Smoke Test

Use this for a quick public-repo sanity check after setup or small documentation
and wiring changes. It checks the API health endpoint, strategy inventory,
dependency imports, instrument registry, and evaluation profile definitions.

```bash
cd backend
uv run pytest -q \
  tests/test_api.py::test_healthcheck \
  tests/test_api.py::test_parse_cors_allow_origins_uses_safe_defaults_for_empty_values \
  tests/test_api.py::test_parse_cors_allow_origins_trims_comma_separated_values \
  tests/test_api.py::test_cors_allows_default_localhost_origin \
  tests/test_api.py::test_cors_rejects_unlisted_origin \
  tests/test_api.py::test_strategy_inventory_api \
  tests/test_dependencies.py \
  tests/test_instrument_registry.py \
  tests/test_evaluation_profiles.py
```

## Full Test

Use this before larger code changes or release-oriented cleanup. It runs the
entire backend test suite and is expected to be slower than the smoke test.

```bash
cd backend
uv run ruff check .
uv run pytest -q
```

## CLI

`comparison-summary` and `backtest-strategy` are intentionally different
entrypoints:

- `comparison-summary` runs the existing strategy comparison workflow. Treat it
  as comparison/run plumbing, not as a plain full-period backtest.
- `backtest-strategy` runs a full-period backtest from one fixed market
  snapshot. It currently supports `stg-fu-eq` only.

Use `backtest-strategy` as the strategy-level full-period backtest entrypoint.
`backtest-equal-weight` is a lower-level compatibility/debugging command for the
equal-weight engine. Prefer the shared entrypoint over one top-level CLI per
strategy.

```bash
cd backend
uv run quant comparison-summary \
  --snapshot-spec-file tests/fixtures/golden_market/comparison-run-spec.json \
  --market-snapshot-dir tests/fixtures/golden_market/market_snapshots \
  --universe only_etf \
  --strategy-key stg-fu-eq \
  --top 1

uv run quant backtest-strategy \
  --strategy-key stg-fu-eq \
  --snapshot-id 0d1aaa50ac7a9f4c7e82ee584c7a1bb0a6ce27f9434bf1884bcaee9a4432cd88 \
  --market-snapshot-dir tests/fixtures/golden_market/market_snapshots

uv run quant signal-diagnostics --help
```

Live market data runs are ad-hoc only and must opt in explicitly. The default
provider is yfinance. Normal evaluation should first create market snapshots,
then read those snapshots for comparison runs or full-period backtests.

```bash
STOOQ_API_KEY=... uv run quant market-snapshot-create \
  --provider stooq \
  --market-snapshot-dir data/market_snapshots \
  --json > data/comparison-run-spec.json

uv run quant comparison-summary \
  --snapshot-spec-file data/comparison-run-spec.json \
  --market-snapshot-dir data/market_snapshots \
  --top 5

uv run quant backtest-strategy \
  --strategy-key stg-fu-eq \
  --snapshot-id <snapshot-id-from-run-spec> \
  --market-snapshot-dir data/market_snapshots
```

## API

```bash
cd backend
uv run uvicorn app.main:app --reload
```

The API uses an explicit CORS allowlist. By default it allows common local
frontend origins only: `http://localhost:5173`, `http://127.0.0.1:5173`,
`http://localhost:3000`, and `http://127.0.0.1:3000`.

Heavy research endpoints that trigger market data fetches, backtests, run
generation, or reruns are local-client only by default. CORS is not treated as
authentication; expose these endpoints only behind an explicit trusted access
boundary.

For remote previews, set exact origins explicitly.

```bash
QUANT_CORS_ALLOW_ORIGINS=http://100.x.y.z:5173,https://preview.example.com uv run uvicorn app.main:app --reload
```

Web UI やリモート確認が必要な場合は、利用環境ごとの手順に従う。CORS は wildcard ではなく、必要な origin だけを明示する。

## Common Failures

- `ModuleNotFoundError: No module named 'app'`: backend 直下で実行するか、`uv run` を使う。
- `skfolio` の covariance warning: テストでは既知の fallback warning として `pytest.ini` で抑制している。
- 依存が古い: `uv sync --dev` を再実行する。

## Module Boundaries

- 新しい comparison 関連処理は用途別の `comparison_*` module に追加する。
- `app.comparison_service` は移行期間の互換 facade として扱い、新規実装先にしない。
- 新しい portfolio selection / forecast / execution / availability 処理は用途別の `portfolio_*` module に追加する。
- `app.portfolio` は移行期間の互換 facade として扱い、新規実装先にしない。

Portfolio module responsibilities:

- `portfolio_availability`: asset availability, eligibility, and universe scoping.
- `portfolio_allocation`: portfolio model allocation.
- `portfolio_costs`: linear, asset-specific, and impact cost calculations.
- `portfolio_execution`: decision policy, no-trade decisions, and decision event serialization.
- `portfolio_forecast`: forecast snapshots used by decision policy.
- `portfolio_metrics`: return, CAGR, drawdown, turnover, and diagnostics summaries.
- `portfolio_runs`: orchestration of backtest execution; avoid adding domain primitives here.
- `portfolio_state`: initial/current portfolio state normalization.
- `portfolio_tilt` / `portfolio_positioning`: tilt and positioning helpers.

既存互換 import は段階移行中として許容するが、新規 app code は上記の focused module から import する。
