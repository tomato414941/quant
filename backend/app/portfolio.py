from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from skfolio.optimization import MeanRisk, RiskBudgeting

from app.strategy import (
    cagr,
    max_drawdown,
    percent_return,
    sharpe_ratio,
    validate_split_ratio,
)


SUPPORTED_PORTFOLIO_MODELS = {"equal_weight", "risk_budgeting", "minimum_variance"}
PORTFOLIO_MODEL_LABELS = {
    "equal_weight": "等金額配分",
    "risk_budgeting": "リスク予算配分",
    "minimum_variance": "最小分散",
}
SUPPORTED_PORTFOLIO_STRATEGIES = {"full_universe", "momentum_top3"}
PORTFOLIO_STRATEGY_LABELS = {
    "full_universe": "全資産",
    "momentum_top3": "モメンタム上位3",
}


@dataclass(frozen=True)
class PortfolioStrategyDefinition:
    key: str
    strategy_type: str
    label: str
    description: str


@dataclass(frozen=True)
class PortfolioModelDefinition:
    key: str
    model_type: str
    label: str
    description: str


def build_portfolio_strategy_definition(
    strategy_type: str,
    *,
    key: str | None = None,
    label: str | None = None,
    description: str | None = None,
) -> PortfolioStrategyDefinition:
    if strategy_type not in SUPPORTED_PORTFOLIO_STRATEGIES:
        raise ValueError("Unsupported portfolio strategy.")

    default_descriptions = {
        "full_universe": "全ETFを候補にする",
        "momentum_top3": "学習期間のモメンタム上位3ETFを候補にする",
    }

    return PortfolioStrategyDefinition(
        key=key or strategy_type,
        strategy_type=strategy_type,
        label=label or PORTFOLIO_STRATEGY_LABELS[strategy_type],
        description=description or default_descriptions[strategy_type],
    )


def build_portfolio_model_definition(
    model_type: str,
    *,
    key: str | None = None,
    label: str | None = None,
    description: str | None = None,
) -> PortfolioModelDefinition:
    if model_type not in SUPPORTED_PORTFOLIO_MODELS:
        raise ValueError("Unsupported portfolio model.")

    default_descriptions = {
        "equal_weight": "全資産を同じ比率で持つ",
        "risk_budgeting": "各資産のリスク寄与が近づくように配分する",
        "minimum_variance": "分散が最小になるように配分する",
    }

    return PortfolioModelDefinition(
        key=key or model_type,
        model_type=model_type,
        label=label or PORTFOLIO_MODEL_LABELS[model_type],
        description=description or default_descriptions[model_type],
    )


def serialize_portfolio_model_definition(model_definition: PortfolioModelDefinition) -> dict:
    return {
        "key": model_definition.key,
        "modelType": model_definition.model_type,
        "label": model_definition.label,
        "description": model_definition.description,
    }


def serialize_portfolio_strategy_definition(strategy_definition: PortfolioStrategyDefinition) -> dict:
    return {
        "key": strategy_definition.key,
        "strategyType": strategy_definition.strategy_type,
        "label": strategy_definition.label,
        "description": strategy_definition.description,
    }


def compare_portfolio_runs(
    closes: pd.DataFrame,
    strategy_definitions: list[PortfolioStrategyDefinition],
    model_definitions: list[PortfolioModelDefinition],
    initial_capital: float,
    split_ratio: float,
    transaction_cost: float,
) -> list[dict]:
    if not strategy_definitions:
        raise ValueError("At least one portfolio strategy is required.")
    if not model_definitions:
        raise ValueError("At least one portfolio model is required.")
    if initial_capital <= 0:
        raise ValueError("Initial capital must be positive.")
    if transaction_cost < 0 or transaction_cost >= 1:
        raise ValueError("Transaction cost must be between 0 and 1.")

    returns = closes.pct_change().dropna()
    if len(returns) < 6:
        raise ValueError("At least 6 return rows are required for portfolio comparison.")

    split_index = compute_split_index(len(returns), split_ratio)
    train_returns = returns.iloc[:split_index]
    test_returns = returns.iloc[split_index:]
    benchmark_weights = np.repeat(1 / len(returns.columns), len(returns.columns))
    benchmark_returns = pd.Series(
        returns.to_numpy(dtype="float64") @ benchmark_weights,
        index=returns.index,
        dtype="float64",
    )

    runs: list[dict] = []
    for strategy_definition in strategy_definitions:
        selected_assets = select_assets(train_returns, strategy_definition)
        strategy_train_returns = train_returns[selected_assets]
        for model_definition in model_definitions:
            weights = fit_portfolio_model(strategy_train_returns, model_definition)
            expanded_weights = expand_weights(
                universe_columns=returns.columns,
                selected_columns=strategy_train_returns.columns,
                selected_weights=weights,
            )
            full_backtest = run_portfolio_backtest(
                returns=returns,
                weights=expanded_weights,
                initial_capital=initial_capital,
                benchmark_returns=benchmark_returns,
                transaction_cost=transaction_cost,
            )
            train_backtest = run_portfolio_backtest(
                returns=train_returns,
                weights=expanded_weights,
                initial_capital=initial_capital,
                benchmark_returns=benchmark_returns.loc[train_returns.index],
                transaction_cost=transaction_cost,
            )
            test_backtest = run_portfolio_backtest(
                returns=test_returns,
                weights=expanded_weights,
                initial_capital=initial_capital,
                benchmark_returns=benchmark_returns.loc[test_returns.index],
                transaction_cost=transaction_cost,
            )

            runs.append(
                {
                    "key": f"{strategy_definition.key}__{model_definition.key}",
                    "strategy": serialize_portfolio_strategy_definition(strategy_definition),
                    "portfolioModel": serialize_portfolio_model_definition(model_definition),
                    "weights": serialize_weights(returns.columns, expanded_weights),
                    "selectedAssets": selected_assets,
                    "summary": full_backtest["summary"]["portfolio"],
                    "benchmark": full_backtest["summary"]["benchmark"],
                    "splitAnalysis": {
                        "config": {"splitRatioPct": round(split_ratio * 100, 1)},
                        "train": summarize_segment_from_returns(train_returns, train_backtest["summary"]),
                        "test": summarize_segment_from_returns(test_returns, test_backtest["summary"]),
                    },
                    "series": full_backtest["series"],
                }
            )

    return runs


def compare_portfolio_models(
    closes: pd.DataFrame,
    model_definitions: list[PortfolioModelDefinition],
    initial_capital: float,
    split_ratio: float,
    transaction_cost: float,
) -> list[dict]:
    return compare_portfolio_runs(
        closes=closes,
        strategy_definitions=[build_portfolio_strategy_definition("full_universe")],
        model_definitions=model_definitions,
        initial_capital=initial_capital,
        split_ratio=split_ratio,
        transaction_cost=transaction_cost,
    )


def select_assets(
    returns: pd.DataFrame,
    strategy_definition: PortfolioStrategyDefinition,
) -> list[str]:
    if strategy_definition.strategy_type == "full_universe":
        return list(returns.columns)
    if strategy_definition.strategy_type == "momentum_top3":
        total_returns = (1 + returns).prod() - 1
        selected = total_returns.sort_values(ascending=False).head(min(3, len(total_returns)))
        return list(selected.index)
    raise ValueError("Unsupported portfolio strategy.")


def expand_weights(
    universe_columns: pd.Index,
    selected_columns: pd.Index,
    selected_weights: np.ndarray,
) -> np.ndarray:
    weight_map = {str(asset): float(weight) for asset, weight in zip(selected_columns, selected_weights, strict=True)}
    return np.asarray([weight_map.get(str(asset), 0.0) for asset in universe_columns], dtype="float64")


def fit_portfolio_model(
    returns: pd.DataFrame,
    model_definition: PortfolioModelDefinition,
) -> np.ndarray:
    asset_count = len(returns.columns)
    if asset_count == 0:
        raise ValueError("At least one asset is required.")

    if model_definition.model_type == "equal_weight":
        weights = np.repeat(1 / asset_count, asset_count)
    elif model_definition.model_type == "risk_budgeting":
        estimator = RiskBudgeting()
        estimator.fit(returns)
        weights = estimator.weights_
    elif model_definition.model_type == "minimum_variance":
        estimator = MeanRisk()
        estimator.fit(returns)
        weights = estimator.weights_
    else:
        raise ValueError("Unsupported portfolio model.")

    weights = np.asarray(weights, dtype="float64")
    weight_sum = weights.sum()
    if weight_sum <= 0:
        raise ValueError("Portfolio model returned invalid weights.")
    return weights / weight_sum


def run_portfolio_backtest(
    returns: pd.DataFrame,
    weights: np.ndarray,
    initial_capital: float,
    benchmark_returns: pd.Series,
    transaction_cost: float,
) -> dict:
    portfolio_equity = initial_capital
    benchmark_equity = initial_capital
    portfolio_returns: list[float] = []
    benchmark_return_values: list[float] = []
    series: list[dict] = []

    for index, (date, row) in enumerate(returns.iterrows()):
        portfolio_return = float(np.dot(row.to_numpy(dtype="float64"), weights))
        if index == 0:
            portfolio_return -= transaction_cost * float(np.abs(weights).sum())
        benchmark_return = float(benchmark_returns.loc[date])

        portfolio_equity *= 1 + portfolio_return
        benchmark_equity *= 1 + benchmark_return
        portfolio_returns.append(portfolio_return)
        benchmark_return_values.append(benchmark_return)
        series.append(
            {
                "date": str(date),
                "portfolioEquity": round(portfolio_equity, 2),
                "benchmarkEquity": round(benchmark_equity, 2),
                "portfolioReturnPct": round(portfolio_return * 100, 2),
            }
        )

    summary = {
        "portfolio": summarize_portfolio_metrics(
            final_value=portfolio_equity,
            initial_value=initial_capital,
            periods=len(returns),
            returns=portfolio_returns,
            series=series,
            equity_key="portfolioEquity",
        ),
        "benchmark": summarize_portfolio_metrics(
            final_value=benchmark_equity,
            initial_value=initial_capital,
            periods=len(returns),
            returns=benchmark_return_values,
            series=series,
            equity_key="benchmarkEquity",
        ),
    }
    return {"summary": summary, "series": series}


def summarize_portfolio_metrics(
    final_value: float,
    initial_value: float,
    periods: int,
    returns: list[float],
    series: list[dict],
    equity_key: str,
) -> dict:
    return {
        "totalReturnPct": round(percent_return(final_value, initial_value), 2),
        "cagrPct": round(cagr(final_value, initial_value, periods), 2),
        "sharpeRatio": round(sharpe_ratio(returns), 2),
        "maxDrawdownPct": round(max_drawdown(series, equity_key), 2),
    }


def summarize_segment_from_returns(returns: pd.DataFrame, summary: dict) -> dict:
    prices = [
        {"date": str(index)}
        for index in returns.index
    ]
    return {
        "startDate": prices[0]["date"],
        "endDate": prices[-1]["date"],
        "dayCount": len(prices),
        "portfolio": summary["portfolio"],
        "benchmark": summary["benchmark"],
    }


def serialize_weights(columns: pd.Index, weights: np.ndarray) -> list[dict]:
    weight_map = [
        {"asset": str(asset), "weightPct": round(float(weight) * 100, 2)}
        for asset, weight in zip(columns, weights, strict=True)
    ]
    weight_map.sort(key=lambda item: item["weightPct"], reverse=True)
    return weight_map


def compute_split_index(length: int, split_ratio: float) -> int:
    validate_split_ratio(split_ratio)
    split_index = int(length * split_ratio)
    split_index = min(max(split_index, 3), length - 3)
    return split_index
