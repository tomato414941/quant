from __future__ import annotations

import ast
from pathlib import Path

import app.comparison_service as comparison_service
import app.portfolio as portfolio


def _module_tree(module: object) -> ast.Module:
    module_path = Path(module.__file__ or "")
    return ast.parse(module_path.read_text())


def test_comparison_service_remains_import_only_facade() -> None:
    tree = _module_tree(comparison_service)

    assert not [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    assert not [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

    imported_modules = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module != "__future__"
        for alias in node.names
    }
    assert imported_modules == {"*"}
    assert {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    } == {
        "app.comparison_market_context",
        "app.comparison_payloads",
        "app.comparison_run_builders",
        "app.comparison_serialization",
        "app.comparison_walk_forward",
        "__future__",
    }


def test_portfolio_facade_only_defines_compatibility_wrappers() -> None:
    tree = _module_tree(portfolio)

    assert not [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

    wrapper_names = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert wrapper_names == {
        "compare_portfolio_models",
        "compare_portfolio_runs",
        "compute_dynamic_portfolio_allocation",
        "evaluate_strategy_definition_run",
        "evaluate_strategy_run",
        "run_portfolio_backtest",
    }
