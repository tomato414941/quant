from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from skfolio.optimization import HierarchicalRiskParity, MeanRisk, RiskBudgeting

from app.strategy import (
    cagr,
    max_drawdown,
    percent_return,
    sharpe_ratio,
    validate_split_ratio,
)


SUPPORTED_PORTFOLIO_MODELS = {
    "equal_weight",
    "risk_budgeting",
    "minimum_variance",
    "hierarchical_risk_parity",
}
PORTFOLIO_MODEL_LABELS = {
    "equal_weight": "等金額配分",
    "risk_budgeting": "リスク予算配分",
    "minimum_variance": "最小分散",
    "hierarchical_risk_parity": "HRP",
}
SUPPORTED_REBALANCE_FREQUENCIES = {"hold", "monthly", "quarterly", "annual"}
SUPPORTED_PORTFOLIO_STRATEGIES = {
    "full_universe",
    "full_universe_momentum_tilt",
    "momentum_top3",
    "dual_momentum_top3",
    "trailing_momentum_low_vol_universe",
    "positive_momentum_universe",
    "positive_momentum_low_vol_universe",
    "positive_momentum_high_volume_universe",
}
PORTFOLIO_STRATEGY_LABELS = {
    "full_universe": "全資産",
    "full_universe_momentum_tilt": "全資産モメンタム傾斜",
    "momentum_top3": "モメンタム上位3",
    "dual_momentum_top3": "デュアルモメンタム上位3",
    "trailing_momentum_low_vol_universe": "12ヶ月モメンタム低ボラ資産",
    "positive_momentum_universe": "上昇資産",
    "positive_momentum_low_vol_universe": "上昇低ボラ資産",
    "positive_momentum_high_volume_universe": "上昇出来高資産",
}

UNIVERSE_POLICY_LABELS = {
    "all_assets": "全資産",
    "positive_assets_only": "上昇資産のみ",
}
SCORE_MODEL_LABELS = {
    "none": "シグナルなし",
    "momentum": "モメンタム",
    "trailing_momentum_12m": "12ヶ月モメンタム",
    "volatility": "ボラティリティ",
    "volume_strength": "出来高強度",
}
FILTER_RULE_LABELS = {
    "positive_return": "上昇資産のみ",
    "top_3": "上位3",
    "low_volatility_half": "低ボラ半分",
    "high_volume_half": "出来高上位半分",
}
FALLBACK_RULE_LABELS = {
    "none": "フォールバックなし",
    "cash_on_empty": "候補ゼロならCASH",
}


@dataclass(frozen=True)
class UniversePolicyDefinition:
    key: str
    label: str


@dataclass(frozen=True)
class ScoreModelDefinition:
    key: str
    label: str


@dataclass(frozen=True)
class FilterRuleDefinition:
    key: str
    label: str


@dataclass(frozen=True)
class FallbackRuleDefinition:
    key: str
    label: str


@dataclass(frozen=True)
class PortfolioStrategyDefinition:
    key: str
    strategy_type: str
    label: str
    description: str
    feature_inputs: tuple[str, ...]
    universe_policy: UniversePolicyDefinition
    score_model: ScoreModelDefinition
    score_parameters: tuple[tuple[str, float], ...]
    filter_rules: tuple[FilterRuleDefinition, ...]
    fallback_rule: FallbackRuleDefinition


@dataclass(frozen=True)
class PortfolioModelDefinition:
    key: str
    model_type: str
    label: str
    description: str


@dataclass(frozen=True)
class PortfolioState:
    current_weights: dict[str, float]
    cash_weight: float = 0.0


@dataclass(frozen=True)
class PortfolioCandidateDefinition:
    key: str
    strategy_definition: PortfolioStrategyDefinition
    model_definition: PortfolioModelDefinition


def build_portfolio_strategy_definition(
    strategy_type: str,
    *,
    key: str | None = None,
    label: str | None = None,
    description: str | None = None,
    score_parameters: dict[str, float] | None = None,
) -> PortfolioStrategyDefinition:
    if strategy_type not in SUPPORTED_PORTFOLIO_STRATEGIES:
        raise ValueError("Unsupported portfolio strategy.")

    default_descriptions = {
        "full_universe": "全ETFを候補にする",
        "full_universe_momentum_tilt": "全ETFを候補にし、12ヶ月モメンタムで重みを傾ける",
        "momentum_top3": "学習期間のモメンタム上位3ETFを候補にする",
        "dual_momentum_top3": "上昇しているETFだけからモメンタム上位3を候補にする",
        "trailing_momentum_low_vol_universe": "12ヶ月モメンタムが正のETFを候補にし、低ボラ群だけへ配分する",
        "positive_momentum_universe": "上昇しているETFだけを候補にする",
        "positive_momentum_low_vol_universe": "上昇しているETFのうち低ボラ群だけを候補にする",
        "positive_momentum_high_volume_universe": "上昇しているETFのうち出来高が強い群だけを候補にする",
    }
    feature_inputs = {
        "full_universe": ("close",),
        "full_universe_momentum_tilt": ("close",),
        "momentum_top3": ("close",),
        "dual_momentum_top3": ("close",),
        "trailing_momentum_low_vol_universe": ("close",),
        "positive_momentum_universe": ("close",),
        "positive_momentum_low_vol_universe": ("close",),
        "positive_momentum_high_volume_universe": ("close", "volume"),
    }
    universe_policies = {
        "full_universe": UniversePolicyDefinition("all_assets", UNIVERSE_POLICY_LABELS["all_assets"]),
        "full_universe_momentum_tilt": UniversePolicyDefinition(
            "all_assets",
            UNIVERSE_POLICY_LABELS["all_assets"],
        ),
        "momentum_top3": UniversePolicyDefinition("all_assets", UNIVERSE_POLICY_LABELS["all_assets"]),
        "dual_momentum_top3": UniversePolicyDefinition(
            "positive_assets_only",
            UNIVERSE_POLICY_LABELS["positive_assets_only"],
        ),
        "trailing_momentum_low_vol_universe": UniversePolicyDefinition(
            "all_assets",
            UNIVERSE_POLICY_LABELS["all_assets"],
        ),
        "positive_momentum_universe": UniversePolicyDefinition(
            "positive_assets_only",
            UNIVERSE_POLICY_LABELS["positive_assets_only"],
        ),
        "positive_momentum_low_vol_universe": UniversePolicyDefinition(
            "positive_assets_only",
            UNIVERSE_POLICY_LABELS["positive_assets_only"],
        ),
        "positive_momentum_high_volume_universe": UniversePolicyDefinition(
            "positive_assets_only",
            UNIVERSE_POLICY_LABELS["positive_assets_only"],
        ),
    }
    score_models = {
        "full_universe": ScoreModelDefinition("none", SCORE_MODEL_LABELS["none"]),
        "full_universe_momentum_tilt": ScoreModelDefinition(
            "trailing_momentum_12m",
            SCORE_MODEL_LABELS["trailing_momentum_12m"],
        ),
        "momentum_top3": ScoreModelDefinition("momentum", SCORE_MODEL_LABELS["momentum"]),
        "dual_momentum_top3": ScoreModelDefinition("momentum", SCORE_MODEL_LABELS["momentum"]),
        "trailing_momentum_low_vol_universe": ScoreModelDefinition(
            "trailing_momentum_12m",
            SCORE_MODEL_LABELS["trailing_momentum_12m"],
        ),
        "positive_momentum_universe": ScoreModelDefinition("momentum", SCORE_MODEL_LABELS["momentum"]),
        "positive_momentum_low_vol_universe": ScoreModelDefinition("momentum", SCORE_MODEL_LABELS["momentum"]),
        "positive_momentum_high_volume_universe": ScoreModelDefinition(
            "volume_strength",
            SCORE_MODEL_LABELS["volume_strength"],
        ),
    }
    default_score_parameters = {
        "full_universe": {},
        "full_universe_momentum_tilt": {"tilt_strength": 0.5},
        "momentum_top3": {},
        "dual_momentum_top3": {},
        "trailing_momentum_low_vol_universe": {},
        "positive_momentum_universe": {},
        "positive_momentum_low_vol_universe": {},
        "positive_momentum_high_volume_universe": {},
    }
    filter_rules = {
        "full_universe": (),
        "full_universe_momentum_tilt": (),
        "momentum_top3": (
            FilterRuleDefinition("top_3", FILTER_RULE_LABELS["top_3"]),
        ),
        "dual_momentum_top3": (
            FilterRuleDefinition("positive_return", FILTER_RULE_LABELS["positive_return"]),
            FilterRuleDefinition("top_3", FILTER_RULE_LABELS["top_3"]),
        ),
        "trailing_momentum_low_vol_universe": (
            FilterRuleDefinition("positive_return", FILTER_RULE_LABELS["positive_return"]),
            FilterRuleDefinition("low_volatility_half", FILTER_RULE_LABELS["low_volatility_half"]),
        ),
        "positive_momentum_universe": (
            FilterRuleDefinition("positive_return", FILTER_RULE_LABELS["positive_return"]),
        ),
        "positive_momentum_low_vol_universe": (
            FilterRuleDefinition("positive_return", FILTER_RULE_LABELS["positive_return"]),
            FilterRuleDefinition("low_volatility_half", FILTER_RULE_LABELS["low_volatility_half"]),
        ),
        "positive_momentum_high_volume_universe": (
            FilterRuleDefinition("positive_return", FILTER_RULE_LABELS["positive_return"]),
            FilterRuleDefinition("high_volume_half", FILTER_RULE_LABELS["high_volume_half"]),
        ),
    }
    fallback_rules = {
        "full_universe": FallbackRuleDefinition("none", FALLBACK_RULE_LABELS["none"]),
        "full_universe_momentum_tilt": FallbackRuleDefinition("none", FALLBACK_RULE_LABELS["none"]),
        "momentum_top3": FallbackRuleDefinition("none", FALLBACK_RULE_LABELS["none"]),
        "dual_momentum_top3": FallbackRuleDefinition("cash_on_empty", FALLBACK_RULE_LABELS["cash_on_empty"]),
        "trailing_momentum_low_vol_universe": FallbackRuleDefinition(
            "cash_on_empty",
            FALLBACK_RULE_LABELS["cash_on_empty"],
        ),
        "positive_momentum_universe": FallbackRuleDefinition("cash_on_empty", FALLBACK_RULE_LABELS["cash_on_empty"]),
        "positive_momentum_low_vol_universe": FallbackRuleDefinition("cash_on_empty", FALLBACK_RULE_LABELS["cash_on_empty"]),
        "positive_momentum_high_volume_universe": FallbackRuleDefinition(
            "cash_on_empty",
            FALLBACK_RULE_LABELS["cash_on_empty"],
        ),
    }

    return PortfolioStrategyDefinition(
        key=key or strategy_type,
        strategy_type=strategy_type,
        label=label or PORTFOLIO_STRATEGY_LABELS[strategy_type],
        description=description or default_descriptions[strategy_type],
        feature_inputs=feature_inputs[strategy_type],
        universe_policy=universe_policies[strategy_type],
        score_model=score_models[strategy_type],
        score_parameters=tuple(
            sorted((score_parameters or default_score_parameters[strategy_type]).items())
        ),
        filter_rules=filter_rules[strategy_type],
        fallback_rule=fallback_rules[strategy_type],
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
        "hierarchical_risk_parity": "相関クラスタを使って階層的にリスクを分散する",
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
        "featureInputs": list(strategy_definition.feature_inputs),
        "universePolicy": {
            "key": strategy_definition.universe_policy.key,
            "label": strategy_definition.universe_policy.label,
        },
        "scoreModel": {
            "key": strategy_definition.score_model.key,
            "label": strategy_definition.score_model.label,
        },
        "scoreParameters": {
            key: value for key, value in strategy_definition.score_parameters
        },
        "filterRules": [
            {
                "key": filter_rule.key,
                "label": filter_rule.label,
            }
            for filter_rule in strategy_definition.filter_rules
        ],
        "fallbackRule": {
            "key": strategy_definition.fallback_rule.key,
            "label": strategy_definition.fallback_rule.label,
        },
    }


def serialize_portfolio_candidate_definition(
    candidate_definition: PortfolioCandidateDefinition,
) -> dict:
    return {
        "key": candidate_definition.key,
        "strategy": serialize_portfolio_strategy_definition(candidate_definition.strategy_definition),
        "portfolioModel": serialize_portfolio_model_definition(candidate_definition.model_definition),
    }


def build_portfolio_candidate_definition(
    strategy_definition: PortfolioStrategyDefinition,
    model_definition: PortfolioModelDefinition,
    *,
    key: str | None = None,
) -> PortfolioCandidateDefinition:
    return PortfolioCandidateDefinition(
        key=key or f"{strategy_definition.key}__{model_definition.key}",
        strategy_definition=strategy_definition,
        model_definition=model_definition,
    )


def build_portfolio_state(
    *,
    current_weights: dict[str, float],
    cash_weight: float = 0.0,
) -> PortfolioState:
    normalized_weights = {
        asset.strip().upper(): float(weight)
        for asset, weight in current_weights.items()
        if asset.strip()
    }
    if cash_weight < 0 or cash_weight > 1:
        raise ValueError("Cash weight must be between 0 and 1.")
    if any(weight < 0 for weight in normalized_weights.values()):
        raise ValueError("Current weights must be non-negative.")
    total_weight = sum(normalized_weights.values()) + cash_weight
    if total_weight > 1.000001:
        raise ValueError("Portfolio state weights must sum to 1 or less.")
    return PortfolioState(current_weights=normalized_weights, cash_weight=float(cash_weight))


def serialize_portfolio_state(portfolio_state: PortfolioState) -> dict:
    rows = [
        {"asset": asset, "weightPct": round(weight * 100, 2)}
        for asset, weight in portfolio_state.current_weights.items()
        if weight > 0
    ]
    if portfolio_state.cash_weight > 0:
        rows.append({"asset": "CASH", "weightPct": round(portfolio_state.cash_weight * 100, 2)})
    rows.sort(key=lambda item: item["weightPct"], reverse=True)
    return {"weights": rows}


def compare_portfolio_runs(
    closes: pd.DataFrame,
    volumes: pd.DataFrame | None,
    candidate_definitions: list[PortfolioCandidateDefinition],
    initial_capital: float,
    split_ratio: float,
    transaction_cost: float,
    max_investment_ratio: float = 1.0,
    max_weight: float | None = None,
    rebalance_frequency: str = "hold",
    portfolio_state: PortfolioState | None = None,
) -> list[dict]:
    if not candidate_definitions:
        raise ValueError("At least one portfolio candidate is required.")
    if initial_capital <= 0:
        raise ValueError("Initial capital must be positive.")
    if transaction_cost < 0 or transaction_cost >= 1:
        raise ValueError("Transaction cost must be between 0 and 1.")
    if max_investment_ratio <= 0 or max_investment_ratio > 1:
        raise ValueError("Max investment ratio must be between 0 and 1.")
    if max_weight is not None and (max_weight <= 0 or max_weight > 1):
        raise ValueError("Max weight must be between 0 and 1.")
    if rebalance_frequency not in SUPPORTED_REBALANCE_FREQUENCIES:
        raise ValueError("Unsupported rebalance frequency.")

    returns = closes.pct_change().dropna()
    if len(returns) < 6:
        raise ValueError("At least 6 return rows are required for portfolio comparison.")

    split_index = compute_split_index(len(returns), split_ratio)
    train_returns = returns.iloc[:split_index]
    test_returns = returns.iloc[split_index:]
    benchmark_weights = np.repeat(max_investment_ratio / len(returns.columns), len(returns.columns))
    benchmark_returns = pd.Series(
        returns.to_numpy(dtype="float64") @ benchmark_weights,
        index=returns.index,
        dtype="float64",
    )
    initial_portfolio_weights = resolve_initial_weights(
        universe_columns=returns.columns,
        portfolio_state=portfolio_state,
    )

    runs: list[dict] = []
    for candidate_definition in candidate_definitions:
        strategy_definition = candidate_definition.strategy_definition
        model_definition = candidate_definition.model_definition
        initial_selected_assets, initial_weights = compute_portfolio_allocation(
            history_returns=train_returns,
            volume_history=volumes.loc[train_returns.index] if volumes is not None else None,
            strategy_definition=strategy_definition,
            model_definition=model_definition,
            universe_columns=returns.columns,
            max_investment_ratio=max_investment_ratio,
            max_weight=max_weight,
            previous_weights=initial_portfolio_weights,
            transaction_cost=transaction_cost,
        )
        backtest = run_portfolio_backtest(
            returns=returns,
            volumes=volumes,
            split_index=split_index,
            split_ratio=split_ratio,
            strategy_definition=strategy_definition,
            model_definition=model_definition,
            initial_weights=initial_weights,
            initial_selected_assets=initial_selected_assets,
            max_investment_ratio=max_investment_ratio,
            initial_capital=initial_capital,
            benchmark_returns=benchmark_returns,
            transaction_cost=transaction_cost,
            max_weight=max_weight,
            rebalance_frequency=rebalance_frequency,
            portfolio_state=portfolio_state,
        )

        runs.append(
            {
                "key": candidate_definition.key,
                "strategy": serialize_portfolio_strategy_definition(strategy_definition),
                "portfolioModel": serialize_portfolio_model_definition(model_definition),
                "weights": serialize_weights(returns.columns, backtest["latestWeights"], max_investment_ratio),
                "selectedAssets": backtest["latestSelectedAssets"],
                "summary": backtest["summary"]["portfolio"],
                "benchmark": backtest["summary"]["benchmark"],
                "splitAnalysis": backtest["splitAnalysis"],
                "series": backtest["series"],
            }
        )

    return runs


def compare_portfolio_candidate(
    closes: pd.DataFrame,
    volumes: pd.DataFrame | None,
    candidate_definition: PortfolioCandidateDefinition,
    initial_capital: float,
    split_ratio: float,
    transaction_cost: float,
    max_investment_ratio: float = 1.0,
    max_weight: float | None = None,
    rebalance_frequency: str = "hold",
    portfolio_state: PortfolioState | None = None,
) -> dict:
    return compare_portfolio_runs(
        closes=closes,
        volumes=volumes,
        candidate_definitions=[candidate_definition],
        initial_capital=initial_capital,
        split_ratio=split_ratio,
        transaction_cost=transaction_cost,
        max_investment_ratio=max_investment_ratio,
        max_weight=max_weight,
        rebalance_frequency=rebalance_frequency,
        portfolio_state=portfolio_state,
    )[0]


def compare_portfolio_models(
    closes: pd.DataFrame,
    volumes: pd.DataFrame | None,
    model_definitions: list[PortfolioModelDefinition],
    initial_capital: float,
    split_ratio: float,
    transaction_cost: float,
    max_investment_ratio: float = 1.0,
    max_weight: float | None = None,
    rebalance_frequency: str = "hold",
    portfolio_state: PortfolioState | None = None,
) -> list[dict]:
    return compare_portfolio_runs(
        closes=closes,
        volumes=volumes,
        candidate_definitions=[
            build_portfolio_candidate_definition(
                build_portfolio_strategy_definition("full_universe"),
                model_definition,
            )
            for model_definition in model_definitions
        ],
        initial_capital=initial_capital,
        split_ratio=split_ratio,
        transaction_cost=transaction_cost,
        max_investment_ratio=max_investment_ratio,
        max_weight=max_weight,
        rebalance_frequency=rebalance_frequency,
        portfolio_state=portfolio_state,
    )


def select_assets(
    returns: pd.DataFrame,
    volume_history: pd.DataFrame | None,
    strategy_definition: PortfolioStrategyDefinition,
) -> list[str]:
    trailing_total_returns = compute_trailing_total_returns(returns)

    if strategy_definition.strategy_type == "full_universe":
        return list(returns.columns)
    if strategy_definition.strategy_type == "full_universe_momentum_tilt":
        return list(returns.columns)
    if strategy_definition.strategy_type == "momentum_top3":
        selected = trailing_total_returns.sort_values(ascending=False).head(min(3, len(trailing_total_returns)))
        return list(selected.index)
    if strategy_definition.strategy_type == "dual_momentum_top3":
        positive_returns = trailing_total_returns[trailing_total_returns > 0]
        if positive_returns.empty:
            return []
        selected = positive_returns.sort_values(ascending=False).head(min(3, len(positive_returns)))
        return list(selected.index)
    if strategy_definition.strategy_type == "trailing_momentum_low_vol_universe":
        positive_assets = trailing_total_returns[trailing_total_returns > 0].sort_values(ascending=False)
        if positive_assets.empty:
            return []
        volatilities = returns[positive_assets.index].std()
        low_vol_threshold = float(volatilities.quantile(0.6))
        selected = volatilities[volatilities <= low_vol_threshold].sort_values()
        if selected.empty:
            return [str(volatilities.idxmin())]
        return list(selected.index)
    if strategy_definition.strategy_type == "positive_momentum_universe":
        positive_returns = trailing_total_returns[trailing_total_returns > 0]
        if positive_returns.empty:
            return []
        return list(positive_returns.sort_values(ascending=False).index)
    if strategy_definition.strategy_type == "positive_momentum_low_vol_universe":
        positive_assets = trailing_total_returns[trailing_total_returns > 0].sort_values(ascending=False)
        if positive_assets.empty:
            return []
        volatilities = returns[positive_assets.index].std()
        low_vol_threshold = float(volatilities.median())
        selected = volatilities[volatilities <= low_vol_threshold].sort_values()
        if selected.empty:
            return [str(volatilities.idxmin())]
        return list(selected.index)
    if strategy_definition.strategy_type == "positive_momentum_high_volume_universe":
        if volume_history is None:
            raise ValueError("Volume history is required for the selected portfolio strategy.")
        total_returns = (1 + returns).prod() - 1
        positive_assets = total_returns[total_returns > 0].sort_values(ascending=False)
        if positive_assets.empty:
            return []
        positive_volumes = volume_history[positive_assets.index].dropna(how="all")
        if positive_volumes.empty:
            return list(positive_assets.index)
        recent_window = max(2, min(20, max(2, len(positive_volumes) // 4)))
        if len(positive_volumes) <= recent_window:
            baseline = positive_volumes.mean(axis=0)
        else:
            baseline = positive_volumes.iloc[:-recent_window].mean(axis=0)
        baseline = baseline.replace(0, np.nan)
        recent = positive_volumes.iloc[-recent_window:].mean(axis=0)
        volume_score = (recent / baseline).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        threshold = float(volume_score.median())
        selected = volume_score[volume_score >= threshold].sort_values(ascending=False)
        if selected.empty:
            return [str(volume_score.idxmax())]
        return list(selected.index)
    raise ValueError("Unsupported portfolio strategy.")


def compute_trailing_total_returns(returns: pd.DataFrame) -> pd.Series:
    lookback = min(len(returns), 252)
    trailing_returns = returns.iloc[-lookback:]
    return (1 + trailing_returns).prod() - 1


def expand_weights(
    universe_columns: pd.Index,
    selected_columns: pd.Index,
    selected_weights: np.ndarray,
) -> np.ndarray:
    weight_map = {str(asset): float(weight) for asset, weight in zip(selected_columns, selected_weights, strict=True)}
    return np.asarray([weight_map.get(str(asset), 0.0) for asset in universe_columns], dtype="float64")


def compute_portfolio_allocation(
    history_returns: pd.DataFrame,
    volume_history: pd.DataFrame | None,
    strategy_definition: PortfolioStrategyDefinition,
    model_definition: PortfolioModelDefinition,
    universe_columns: pd.Index,
    max_investment_ratio: float,
    max_weight: float | None,
    previous_weights: np.ndarray | None,
    transaction_cost: float,
) -> tuple[list[str], np.ndarray]:
    selected_assets = select_assets(history_returns, volume_history, strategy_definition)
    if not selected_assets:
        return [], np.zeros(len(universe_columns), dtype="float64")
    strategy_returns = filter_positive_variance_assets(history_returns[selected_assets])
    selected_assets = list(strategy_returns.columns)
    selected_previous_weights = None
    if previous_weights is not None:
        previous_weight_map = {
            str(asset): float(weight)
            for asset, weight in zip(universe_columns, previous_weights, strict=True)
        }
        selected_previous_weights = np.asarray(
            [previous_weight_map.get(asset, 0.0) for asset in selected_assets],
            dtype="float64",
        )
    weights = fit_portfolio_model(
        strategy_returns,
        model_definition,
        max_investment_ratio=max_investment_ratio,
        max_weight=max_weight,
        previous_weights=selected_previous_weights,
        transaction_cost=transaction_cost,
    ) * max_investment_ratio
    weights = apply_strategy_weight_tilt(
        weights=weights,
        history_returns=strategy_returns,
        strategy_definition=strategy_definition,
        max_investment_ratio=max_investment_ratio,
        max_weight=max_weight,
    )
    expanded_weights = expand_weights(
        universe_columns=universe_columns,
        selected_columns=strategy_returns.columns,
        selected_weights=weights,
    )
    return selected_assets, expanded_weights


def apply_strategy_weight_tilt(
    *,
    weights: np.ndarray,
    history_returns: pd.DataFrame,
    strategy_definition: PortfolioStrategyDefinition,
    max_investment_ratio: float,
    max_weight: float | None,
) -> np.ndarray:
    if strategy_definition.strategy_type != "full_universe_momentum_tilt":
        return weights

    trailing_returns = compute_trailing_total_returns(history_returns)
    percentile_ranks = trailing_returns.rank(method="average", pct=True)
    score_parameters = dict(strategy_definition.score_parameters)
    tilt_strength = float(score_parameters.get("tilt_strength", 0.5))
    tilt_shape = float(score_parameters.get("tilt_shape", 0.0))
    rank_values = percentile_ranks.to_numpy(dtype="float64")
    if tilt_shape == 1.0:
        tilt = np.where(rank_values >= 0.75, 1 + tilt_strength, 1.0)
    elif tilt_shape == 2.0:
        centered = rank_values - rank_values.mean()
        scaled = np.exp(np.clip(centered * tilt_strength * 2.0, -2.0, 2.0))
        tilt = scaled / scaled.mean()
    else:
        tilt = 1 + tilt_strength * (rank_values - 0.5)
    tilt = np.clip(tilt, 0.25, None)
    tilted_weights = weights * tilt
    tilted_weights_sum = tilted_weights.sum()
    if tilted_weights_sum <= 0:
        return weights
    tilted_weights = tilted_weights / tilted_weights_sum * weights.sum()
    if max_weight is not None:
        tilted_weights = np.minimum(tilted_weights, max_weight)
    return tilted_weights


def filter_positive_variance_assets(returns: pd.DataFrame) -> pd.DataFrame:
    positive_variance = returns.var(axis=0) > 1e-12
    filtered = returns.loc[:, positive_variance]
    if filtered.empty:
        raise ValueError("At least one asset with positive variance is required.")
    return filtered


def fit_portfolio_model(
    returns: pd.DataFrame,
    model_definition: PortfolioModelDefinition,
    *,
    max_investment_ratio: float,
    max_weight: float | None,
    previous_weights: np.ndarray | None,
    transaction_cost: float,
) -> np.ndarray:
    asset_count = len(returns.columns)
    if asset_count == 0:
        raise ValueError("At least one asset is required.")

    raw_max_weight = None
    if max_weight is not None:
        raw_max_weight = min(1.0, max_weight / max_investment_ratio)

    if model_definition.model_type == "equal_weight":
        weights = build_equal_weight_fallback(asset_count, raw_max_weight)
    elif model_definition.model_type == "risk_budgeting":
        try:
            estimator = RiskBudgeting(
                max_weights=raw_max_weight if raw_max_weight is not None else 1.0,
                transaction_costs=transaction_cost,
                previous_weights=previous_weights,
            )
            estimator.fit(returns)
            weights = estimator.weights_
        except Exception:
            weights = build_equal_weight_fallback(asset_count, raw_max_weight)
    elif model_definition.model_type == "minimum_variance":
        try:
            estimator = MeanRisk(
                max_weights=raw_max_weight if raw_max_weight is not None else 1.0,
                transaction_costs=transaction_cost,
                previous_weights=previous_weights,
            )
            estimator.fit(returns)
            weights = estimator.weights_
        except Exception:
            weights = build_equal_weight_fallback(asset_count, raw_max_weight)
    elif model_definition.model_type == "hierarchical_risk_parity":
        if asset_count <= 2:
            try:
                estimator = RiskBudgeting(
                    max_weights=raw_max_weight if raw_max_weight is not None else 1.0,
                    transaction_costs=transaction_cost,
                    previous_weights=previous_weights,
                )
                estimator.fit(returns)
                weights = estimator.weights_
            except Exception:
                weights = build_equal_weight_fallback(asset_count, raw_max_weight)
        else:
            try:
                estimator = HierarchicalRiskParity(
                    max_weights=raw_max_weight if raw_max_weight is not None else 1.0,
                    transaction_costs=transaction_cost,
                    previous_weights=previous_weights,
                )
                estimator.fit(returns)
                weights = estimator.weights_
            except Exception:
                weights = build_equal_weight_fallback(asset_count, raw_max_weight)
    else:
        raise ValueError("Unsupported portfolio model.")

    weights = np.asarray(weights, dtype="float64")
    weight_sum = weights.sum()
    if weight_sum <= 0:
        raise ValueError("Portfolio model returned invalid weights.")
    if raw_max_weight is not None and weight_sum < 0.999999:
        return weights
    return weights / weight_sum


def build_equal_weight_fallback(asset_count: int, raw_max_weight: float | None) -> np.ndarray:
    weights = np.repeat(1 / asset_count, asset_count)
    if raw_max_weight is not None:
        weights = np.minimum(weights, raw_max_weight)
    return weights


def run_portfolio_backtest(
    returns: pd.DataFrame,
    volumes: pd.DataFrame | None,
    split_index: int,
    split_ratio: float,
    strategy_definition: PortfolioStrategyDefinition,
    model_definition: PortfolioModelDefinition,
    initial_weights: np.ndarray,
    initial_selected_assets: list[str],
    max_investment_ratio: float,
    initial_capital: float,
    benchmark_returns: pd.Series,
    transaction_cost: float,
    max_weight: float | None,
    rebalance_frequency: str,
    portfolio_state: PortfolioState | None,
) -> dict:
    portfolio_equity = initial_capital
    benchmark_equity = initial_capital
    portfolio_returns: list[float] = []
    benchmark_return_values: list[float] = []
    series: list[dict] = []
    train_dates: list[str] = []
    test_dates: list[str] = []
    train_portfolio_returns: list[float] = []
    test_portfolio_returns: list[float] = []
    train_benchmark_returns: list[float] = []
    test_benchmark_returns: list[float] = []
    train_turnover = 0.0
    test_turnover = 0.0
    current_weights = initial_weights.copy()
    current_selected_assets = list(initial_selected_assets)
    latest_weights = initial_weights.copy()
    latest_selected_assets = list(initial_selected_assets)

    for index, (date, row) in enumerate(returns.iterrows()):
        trade_turnover = 0.0
        if index == 0:
            trade_turnover = float(np.abs(current_weights).sum())
        elif index > split_index and should_rebalance(
            previous_date=returns.index[index - 1],
            current_date=date,
            rebalance_frequency=rebalance_frequency,
        ):
            current_selected_assets, rebalanced_weights = compute_portfolio_allocation(
                history_returns=returns.iloc[:index],
                volume_history=volumes.iloc[:index] if volumes is not None else None,
                strategy_definition=strategy_definition,
                model_definition=model_definition,
                universe_columns=returns.columns,
                max_investment_ratio=max_investment_ratio,
                max_weight=max_weight,
                previous_weights=current_weights,
                transaction_cost=transaction_cost,
            )
            trade_turnover = float(np.abs(rebalanced_weights - current_weights).sum())
            current_weights = rebalanced_weights
            latest_weights = rebalanced_weights.copy()
            latest_selected_assets = list(current_selected_assets)

        portfolio_return = float(np.dot(row.to_numpy(dtype="float64"), current_weights))
        portfolio_return -= transaction_cost * trade_turnover
        benchmark_return = float(benchmark_returns.loc[date])

        portfolio_equity *= 1 + portfolio_return
        benchmark_equity *= 1 + benchmark_return
        portfolio_returns.append(portfolio_return)
        benchmark_return_values.append(benchmark_return)
        segment_dates = train_dates if index < split_index else test_dates
        segment_portfolio_returns = train_portfolio_returns if index < split_index else test_portfolio_returns
        segment_benchmark_returns = train_benchmark_returns if index < split_index else test_benchmark_returns
        segment_dates.append(str(date))
        segment_portfolio_returns.append(portfolio_return)
        segment_benchmark_returns.append(benchmark_return)
        if index < split_index:
            train_turnover += trade_turnover
        else:
            test_turnover += trade_turnover
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
            turnover=round((train_turnover + test_turnover) * 100, 2),
        ),
        "benchmark": summarize_portfolio_metrics(
            final_value=benchmark_equity,
            initial_value=initial_capital,
            periods=len(returns),
            returns=benchmark_return_values,
            series=series,
            equity_key="benchmarkEquity",
            turnover=0.0,
        ),
    }
    return {
        "summary": summary,
        "series": series,
        "latestWeights": latest_weights,
        "latestSelectedAssets": latest_selected_assets,
        "splitAnalysis": {
            "config": {"splitRatioPct": round(split_ratio * 100, 1)},
            "train": summarize_segment_from_returns(
                dates=train_dates,
                portfolio_returns=train_portfolio_returns,
                benchmark_returns=train_benchmark_returns,
                turnover=train_turnover,
            ),
            "test": summarize_segment_from_returns(
                dates=test_dates,
                portfolio_returns=test_portfolio_returns,
                benchmark_returns=test_benchmark_returns,
                turnover=test_turnover,
            ),
        },
    }


def summarize_portfolio_metrics(
    final_value: float,
    initial_value: float,
    periods: int,
    returns: list[float],
    series: list[dict],
    equity_key: str,
    turnover: float,
) -> dict:
    return {
        "totalReturnPct": round(percent_return(final_value, initial_value), 2),
        "cagrPct": round(cagr(final_value, initial_value, periods), 2),
        "sharpeRatio": round(sharpe_ratio(returns), 2),
        "maxDrawdownPct": round(max_drawdown(series, equity_key), 2),
        "turnoverPct": round(turnover, 2),
    }


def summarize_segment_from_returns(
    dates: list[str],
    portfolio_returns: list[float],
    benchmark_returns: list[float],
    turnover: float,
) -> dict:
    return {
        "startDate": dates[0],
        "endDate": dates[-1],
        "dayCount": len(dates),
        "portfolio": summarize_metrics_from_daily_returns(
            dates=dates,
            daily_returns=portfolio_returns,
            equity_key="portfolioEquity",
            turnover=turnover,
        ),
        "benchmark": summarize_metrics_from_daily_returns(
            dates=dates,
            daily_returns=benchmark_returns,
            equity_key="benchmarkEquity",
            turnover=0.0,
        ),
    }


def serialize_weights(columns: pd.Index, weights: np.ndarray, max_investment_ratio: float) -> list[dict]:
    weight_map = [
        {"asset": str(asset), "weightPct": round(float(weight) * 100, 2)}
        for asset, weight in zip(columns, weights, strict=True)
    ]
    invested_weight = float(np.sum(weights))
    cash_weight_pct = round(max(0.0, 1 - invested_weight) * 100, 2)
    if cash_weight_pct > 0:
        weight_map.append({"asset": "CASH", "weightPct": cash_weight_pct})
    weight_map.sort(key=lambda item: item["weightPct"], reverse=True)
    return weight_map


def compute_split_index(length: int, split_ratio: float) -> int:
    validate_split_ratio(split_ratio)
    split_index = int(length * split_ratio)
    split_index = min(max(split_index, 3), length - 3)
    return split_index


def summarize_metrics_from_daily_returns(
    dates: list[str],
    daily_returns: list[float],
    equity_key: str,
    turnover: float,
) -> dict:
    equity = 100.0
    series = []
    for date, daily_return in zip(dates, daily_returns, strict=True):
        equity *= 1 + daily_return
        series.append({"date": date, equity_key: round(equity, 2)})
    return summarize_portfolio_metrics(
        final_value=equity,
        initial_value=100.0,
        periods=len(daily_returns),
        returns=daily_returns,
        series=series,
        equity_key=equity_key,
        turnover=round(turnover * 100, 2),
    )


def should_rebalance(previous_date, current_date, rebalance_frequency: str) -> bool:
    if rebalance_frequency == "hold":
        return False

    previous_timestamp = pd.Timestamp(previous_date)
    current_timestamp = pd.Timestamp(current_date)
    if rebalance_frequency == "monthly":
        return previous_timestamp.month != current_timestamp.month or previous_timestamp.year != current_timestamp.year
    if rebalance_frequency == "quarterly":
        return previous_timestamp.quarter != current_timestamp.quarter or previous_timestamp.year != current_timestamp.year
    if rebalance_frequency == "annual":
        return previous_timestamp.year != current_timestamp.year
    raise ValueError("Unsupported rebalance frequency.")


def resolve_initial_weights(
    *,
    universe_columns: pd.Index,
    portfolio_state: PortfolioState | None,
) -> np.ndarray:
    if portfolio_state is None:
        return np.zeros(len(universe_columns), dtype="float64")
    return np.asarray(
        [portfolio_state.current_weights.get(str(asset), 0.0) for asset in universe_columns],
        dtype="float64",
    )
