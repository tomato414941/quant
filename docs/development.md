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
uv run pytest -q tests/test_api.py::test_healthcheck tests/test_api.py::test_strategy_inventory_api tests/test_dependencies.py tests/test_instrument_registry.py tests/test_evaluation_profiles.py
```

## Full Test

Use this before larger code changes or release-oriented cleanup. It runs the
entire backend test suite and is expected to be slower than the smoke test.

```bash
cd backend
uv run pytest -q
```

## CLI

```bash
cd backend
uv run quant comparison-summary --top 1
uv run quant signal-diagnostics --help
```

互換入口として次も使える。

```bash
uv run python -m app comparison-summary --top 1
```

## API

```bash
cd backend
uv run uvicorn app.main:app --reload
```

Web UI やリモート確認が必要な場合は、利用環境ごとの手順に従う。

## Common Failures

- `ModuleNotFoundError: No module named 'app'`: backend 直下で実行するか、`uv run` を使う。
- `skfolio` の covariance warning: テストでは既知の fallback warning として `pytest.ini` で抑制している。
- 依存が古い: `uv sync --dev` を再実行する。

## Module Boundaries

- 新しい comparison 関連処理は用途別の `comparison_*` module に追加する。
- `app.comparison_service` は移行期間の互換 facade として扱い、新規実装先にしない。
- 新しい portfolio selection / forecast / execution / availability 処理は用途別の `portfolio_*` module に追加する。
- `app.portfolio` は移行期間の互換 facade として扱い、新規実装先にしない。
