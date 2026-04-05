from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from skfolio.optimization import HierarchicalRiskParity, MeanRisk, ObjectiveFunction, RiskBudgeting

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
    "mean_risk_utility",
    "mean_risk_utility_conservative",
}
PORTFOLIO_MODEL_LABELS = {
    "equal_weight": "等金額配分",
    "risk_budgeting": "リスク予算配分",
    "minimum_variance": "最小分散",
    "hierarchical_risk_parity": "HRP",
    "mean_risk_utility": "MeanRisk効用最大化",
    "mean_risk_utility_conservative": "MeanRisk効用最大化 弱",
}
SUPPORTED_REBALANCE_FREQUENCIES = {"hold", "monthly", "quarterly", "annual"}
SUPPORTED_PORTFOLIO_STRATEGIES = {
    "full_universe",
    "full_universe_momentum_tilt",
    "full_universe_momentum_low_vol_tilt",
    "full_universe_momentum_macro_tilt",
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
    "full_universe_momentum_low_vol_tilt": "全資産モメンタム低ボラ傾斜",
    "full_universe_momentum_macro_tilt": "全資産モメンタムマクロ傾斜",
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
    "momentum_low_vol": "12ヶ月モメンタム+低ボラ",
    "momentum_macro": "12ヶ月モメンタム+マクロproxy",
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
class InvestmentUniverseDefinition:
    key: str
    label: str
    tickers: tuple[str, ...]


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
class ExecutionPolicyDefinition:
    key: str
    label: str
    entry: str
    rebalance_frequency: str = "hold"


@dataclass(frozen=True)
class RiskControlsDefinition:
    max_investment_ratio: float
    max_weight: float | None = None


@dataclass(frozen=True)
class StrategyDefinition:
    strategy_id: str
    version: str
    label: str
    hypothesis: str
    description: str
    investment_universe_definition: InvestmentUniverseDefinition
    selection_definition: PortfolioStrategyDefinition
    portfolio_model_definition: PortfolioModelDefinition
    execution_policy_definition: ExecutionPolicyDefinition
    risk_controls_definition: RiskControlsDefinition
    extensions: tuple[tuple[str, str], ...] = ()

    @property
    def key(self) -> str:
        return self.strategy_id


@dataclass(frozen=True)
class PortfolioState:
    current_weights: dict[str, float]
    cash_weight: float = 0.0


@dataclass(frozen=True)
class AssetRankingDefinition:
    key: str
    label: str
    description: str
    investment_universe_definition: InvestmentUniverseDefinition
    strategy_definition: PortfolioStrategyDefinition
    source_strategy_keys: tuple[str, ...]
    source_strategy_labels: tuple[str, ...]


def build_investment_universe_definition(
    *,
    tickers: list[str] | tuple[str, ...],
    key: str,
    label: str,
) -> InvestmentUniverseDefinition:
    normalized_tickers = tuple(dict.fromkeys(ticker.strip().upper() for ticker in tickers if ticker.strip()))
    if len(normalized_tickers) < 2:
        raise ValueError("Investment universe must contain at least two tickers.")
    return InvestmentUniverseDefinition(
        key=key,
        label=label,
        tickers=normalized_tickers,
    )


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
        "full_universe_momentum_low_vol_tilt": "全ETFを候補にし、12ヶ月モメンタムと低ボラの複合スコアで重みを傾ける",
        "full_universe_momentum_macro_tilt": "全ETFを候補にし、12ヶ月モメンタムとマクロproxyの複合スコアで重みを傾ける",
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
        "full_universe_momentum_low_vol_tilt": ("close",),
        "full_universe_momentum_macro_tilt": ("close",),
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
        "full_universe_momentum_low_vol_tilt": UniversePolicyDefinition(
            "all_assets",
            UNIVERSE_POLICY_LABELS["all_assets"],
        ),
        "full_universe_momentum_macro_tilt": UniversePolicyDefinition(
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
        "full_universe_momentum_low_vol_tilt": ScoreModelDefinition(
            "momentum_low_vol",
            SCORE_MODEL_LABELS["momentum_low_vol"],
        ),
        "full_universe_momentum_macro_tilt": ScoreModelDefinition(
            "momentum_macro",
            SCORE_MODEL_LABELS["momentum_macro"],
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
        "full_universe_momentum_low_vol_tilt": {
            "tilt_strength": 0.25,
            "tilt_shape": 1.0,
            "momentum_weight": 0.7,
            "low_vol_weight": 0.3,
        },
        "full_universe_momentum_macro_tilt": {
            "tilt_strength": 0.25,
            "tilt_shape": 1.0,
            "momentum_weight": 0.85,
            "macro_weight": 0.15,
        },
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
        "full_universe_momentum_low_vol_tilt": (),
        "full_universe_momentum_macro_tilt": (),
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
        "full_universe_momentum_low_vol_tilt": FallbackRuleDefinition("none", FALLBACK_RULE_LABELS["none"]),
        "full_universe_momentum_macro_tilt": FallbackRuleDefinition("none", FALLBACK_RULE_LABELS["none"]),
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
        "mean_risk_utility": "期待リターン proxy とリスクを同時に見て効用最大化する",
        "mean_risk_utility_conservative": "期待リターン proxy を弱めに使い、リスクをより強く見る",
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


def build_execution_policy_definition(
    *,
    key: str,
    label: str,
    entry: str,
    rebalance_frequency: str = "hold",
) -> ExecutionPolicyDefinition:
    if rebalance_frequency not in SUPPORTED_REBALANCE_FREQUENCIES:
        raise ValueError("Unsupported rebalance frequency.")
    return ExecutionPolicyDefinition(
        key=key,
        label=label,
        entry=entry,
        rebalance_frequency=rebalance_frequency,
    )


def build_risk_controls_definition(
    *,
    max_investment_ratio: float,
    max_weight: float | None = None,
) -> RiskControlsDefinition:
    if max_investment_ratio <= 0 or max_investment_ratio > 1:
        raise ValueError("Max investment ratio must be between 0 and 1.")
    if max_weight is not None and (max_weight <= 0 or max_weight > 1):
        raise ValueError("Max weight must be between 0 and 1.")
    return RiskControlsDefinition(
        max_investment_ratio=max_investment_ratio,
        max_weight=max_weight,
    )


def build_strategy_definition(
    *,
    investment_universe_definition: InvestmentUniverseDefinition,
    selection_definition: PortfolioStrategyDefinition,
    portfolio_model_definition: PortfolioModelDefinition,
    execution_policy_definition: ExecutionPolicyDefinition,
    risk_controls_definition: RiskControlsDefinition,
    strategy_id: str | None = None,
    version: str = "v1",
    hypothesis: str | None = None,
    extensions: dict[str, str] | None = None,
    key: str | None = None,
    label: str | None = None,
    description: str | None = None,
) -> StrategyDefinition:
    resolved_strategy_id = strategy_id or key or "__".join(
        [
            selection_definition.key,
            portfolio_model_definition.key,
            execution_policy_definition.key,
        ]
    )
    if strategy_id is not None and key is not None and strategy_id != key:
        raise ValueError("strategy_id and key must match when both are provided.")
    strategy_label = label or " × ".join(
        [
            selection_definition.label,
            portfolio_model_definition.label,
            execution_policy_definition.label,
        ]
    )
    return StrategyDefinition(
        strategy_id=resolved_strategy_id,
        version=version,
        label=strategy_label,
        hypothesis=hypothesis or selection_definition.description,
        description=description or selection_definition.description,
        investment_universe_definition=investment_universe_definition,
        selection_definition=selection_definition,
        portfolio_model_definition=portfolio_model_definition,
        execution_policy_definition=execution_policy_definition,
        risk_controls_definition=risk_controls_definition,
        extensions=tuple(sorted((extensions or {}).items())),
    )


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


def serialize_execution_policy_definition(execution_policy_definition: ExecutionPolicyDefinition) -> dict:
    return {
        "key": execution_policy_definition.key,
        "label": execution_policy_definition.label,
        "entry": execution_policy_definition.entry,
        "rebalanceFrequency": execution_policy_definition.rebalance_frequency,
    }


def serialize_risk_controls_definition(risk_controls_definition: RiskControlsDefinition) -> dict:
    return {
        "maxInvestmentPct": round(risk_controls_definition.max_investment_ratio * 100, 1),
        "maxWeightPct": round(risk_controls_definition.max_weight * 100, 1)
        if risk_controls_definition.max_weight is not None
        else None,
    }


def serialize_strategy_definition(strategy_definition: StrategyDefinition) -> dict:
    ranking_model = (
        {
            "key": strategy_definition.selection_definition.score_model.key,
            "label": strategy_definition.selection_definition.score_model.label,
        }
        if strategy_definition.selection_definition.score_model.key != "none"
        else None
    )
    fallback_rule = (
        {
            "key": strategy_definition.selection_definition.fallback_rule.key,
            "label": strategy_definition.selection_definition.fallback_rule.label,
        }
        if strategy_definition.selection_definition.fallback_rule.key != "none"
        else None
    )
    return {
        "strategyId": strategy_definition.strategy_id,
        "version": strategy_definition.version,
        "key": strategy_definition.key,
        "label": strategy_definition.label,
        "hypothesis": strategy_definition.hypothesis,
        "description": strategy_definition.description,
        "components": {
            "core": {
                "investmentUniverse": {
                    "key": strategy_definition.investment_universe_definition.key,
                    "label": strategy_definition.investment_universe_definition.label,
                    "assetCount": len(strategy_definition.investment_universe_definition.tickers),
                    "tickers": list(strategy_definition.investment_universe_definition.tickers),
                },
                "portfolioModel": serialize_portfolio_model_definition(
                    strategy_definition.portfolio_model_definition
                ),
                "executionPolicy": serialize_execution_policy_definition(
                    strategy_definition.execution_policy_definition
                ),
            },
            "optional": {
                "assetRankingModel": ranking_model,
                "featureInputs": list(strategy_definition.selection_definition.feature_inputs),
                "filterRules": [
                    {
                        "key": filter_rule.key,
                        "label": filter_rule.label,
                    }
                    for filter_rule in strategy_definition.selection_definition.filter_rules
                ],
                "fallbackRule": fallback_rule,
                "riskControls": serialize_risk_controls_definition(
                    strategy_definition.risk_controls_definition
                ),
            },
        },
        "extensions": {
            key: value for key, value in strategy_definition.extensions
        },
        "selectionDefinition": serialize_portfolio_strategy_definition(
            strategy_definition.selection_definition
        ),
        "portfolioModel": serialize_portfolio_model_definition(
            strategy_definition.portfolio_model_definition
        ),
        "executionPolicy": serialize_execution_policy_definition(
            strategy_definition.execution_policy_definition
        ),
        "riskControls": serialize_risk_controls_definition(
            strategy_definition.risk_controls_definition
        ),
    }


def serialize_asset_ranking_definition(
    ranking_definition: AssetRankingDefinition,
) -> dict:
    strategy_definition = ranking_definition.strategy_definition
    return {
        "key": ranking_definition.key,
        "label": ranking_definition.label,
        "description": ranking_definition.description,
        "investmentUniverse": {
            "key": ranking_definition.investment_universe_definition.key,
            "label": ranking_definition.investment_universe_definition.label,
            "assetCount": len(ranking_definition.investment_universe_definition.tickers),
            "tickers": list(ranking_definition.investment_universe_definition.tickers),
        },
        "featureInputs": list(strategy_definition.feature_inputs),
        "universePolicy": {
            "key": strategy_definition.universe_policy.key,
            "label": strategy_definition.universe_policy.label,
        },
        "rankingModel": {
            "key": strategy_definition.score_model.key,
            "label": strategy_definition.score_model.label,
        },
        "scoreParameters": extract_ranking_score_parameters(strategy_definition),
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
        "sourceStrategyKeys": list(ranking_definition.source_strategy_keys),
        "sourceStrategyLabels": list(ranking_definition.source_strategy_labels),
    }


def build_asset_ranking_definitions(
    strategy_definitions: list[StrategyDefinition],
) -> list[AssetRankingDefinition]:
    grouped: dict[
        tuple[
            tuple[str, ...],
            str,
            str,
            tuple[str, ...],
            tuple[str, ...],
            str,
            tuple[tuple[str, float], ...],
        ],
        list[StrategyDefinition],
    ] = {}

    for strategy in strategy_definitions:
        selection_definition = strategy.selection_definition
        if selection_definition.score_model.key == "none":
            continue
        signature = (
            strategy.investment_universe_definition.tickers,
            selection_definition.strategy_type,
            selection_definition.score_model.key,
            tuple(selection_definition.feature_inputs),
            tuple(filter_rule.key for filter_rule in selection_definition.filter_rules),
            selection_definition.fallback_rule.key,
            tuple(sorted(extract_ranking_score_parameters(selection_definition).items())),
        )
        grouped.setdefault(signature, []).append(strategy)

    ranking_definitions: list[AssetRankingDefinition] = []
    for strategies in grouped.values():
        representative = strategies[0].selection_definition
        filter_label = " / ".join(filter_rule.label for filter_rule in representative.filter_rules)
        label_parts = [
            strategies[0].investment_universe_definition.label,
            representative.universe_policy.label,
            representative.score_model.label,
        ]
        ranking_score_parameters = extract_ranking_score_parameters(representative)
        if ranking_score_parameters:
            parameter_label = ", ".join(
                f"{key}={value:.2f}" for key, value in ranking_score_parameters.items()
            )
            label_parts.append(parameter_label)
        if filter_label:
            label_parts.append(filter_label)
        ranking_definitions.append(
            AssetRankingDefinition(
                key=f"ranking__{representative.key}",
                label=" / ".join(label_parts),
                description=representative.description,
                investment_universe_definition=strategies[0].investment_universe_definition,
                strategy_definition=representative,
                source_strategy_keys=tuple(strategy.key for strategy in strategies),
                source_strategy_labels=tuple(strategy.label for strategy in strategies),
            )
        )

    ranking_definitions.sort(key=lambda definition: definition.label)
    return ranking_definitions


def extract_ranking_score_parameters(
    strategy_definition: PortfolioStrategyDefinition,
) -> dict[str, float]:
    score_parameters = dict(strategy_definition.score_parameters)
    if strategy_definition.score_model.key == "momentum_low_vol":
        return {
            "momentumWeight": float(score_parameters.get("momentum_weight", 0.7)),
            "lowVolWeight": float(score_parameters.get("low_vol_weight", 0.3)),
        }
    if strategy_definition.score_model.key == "momentum_macro":
        return {
            "momentumWeight": float(score_parameters.get("momentum_weight", 0.85)),
            "macroWeight": float(score_parameters.get("macro_weight", 0.15)),
        }
    return {}


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
    strategy_definitions: list[StrategyDefinition],
    initial_capital: float,
    split_ratio: float,
    transaction_cost: float,
    portfolio_state: PortfolioState | None = None,
) -> list[dict]:
    if not strategy_definitions:
        raise ValueError("At least one strategy is required.")
    if initial_capital <= 0:
        raise ValueError("Initial capital must be positive.")
    if transaction_cost < 0 or transaction_cost >= 1:
        raise ValueError("Transaction cost must be between 0 and 1.")

    runs: list[dict] = []
    for strategy_definition in strategy_definitions:
        strategy_universe = [
            asset
            for asset in strategy_definition.investment_universe_definition.tickers
            if asset in closes.columns
        ]
        if len(strategy_universe) < 2:
            raise ValueError("Strategy investment universe must contain at least two available assets.")

        strategy_closes = closes[strategy_universe]
        strategy_volumes = volumes[strategy_universe] if volumes is not None else None
        returns = strategy_closes.pct_change().dropna()
        if len(returns) < 6:
            raise ValueError("At least 6 return rows are required for portfolio comparison.")

        split_index = compute_split_index(len(returns), split_ratio)
        train_returns = returns.iloc[:split_index]
        benchmark_weights = np.repeat(1.0 / len(returns.columns), len(returns.columns))
        benchmark_returns = pd.Series(
            returns.to_numpy(dtype="float64") @ benchmark_weights,
            index=returns.index,
            dtype="float64",
        )
        initial_portfolio_weights = resolve_initial_weights(
            universe_columns=returns.columns,
            portfolio_state=portfolio_state,
        )

        selection_definition = strategy_definition.selection_definition
        model_definition = strategy_definition.portfolio_model_definition
        execution_policy = strategy_definition.execution_policy_definition
        risk_controls = strategy_definition.risk_controls_definition
        initial_selected_assets, initial_weights = compute_portfolio_allocation(
            history_returns=train_returns,
            volume_history=strategy_volumes.loc[train_returns.index] if strategy_volumes is not None else None,
            strategy_definition=selection_definition,
            model_definition=model_definition,
            universe_columns=returns.columns,
            max_investment_ratio=risk_controls.max_investment_ratio,
            max_weight=risk_controls.max_weight,
            previous_weights=initial_portfolio_weights,
            transaction_cost=transaction_cost,
        )
        backtest = run_portfolio_backtest(
            returns=returns,
            volumes=strategy_volumes,
            split_index=split_index,
            split_ratio=split_ratio,
            strategy_definition=selection_definition,
            model_definition=model_definition,
            initial_weights=initial_weights,
            initial_selected_assets=initial_selected_assets,
            max_investment_ratio=risk_controls.max_investment_ratio,
            initial_capital=initial_capital,
            benchmark_returns=benchmark_returns,
            transaction_cost=transaction_cost,
            max_weight=risk_controls.max_weight,
            rebalance_frequency=execution_policy.rebalance_frequency,
            portfolio_state=portfolio_state,
        )

        runs.append(
            {
                "key": strategy_definition.key,
                "strategy": serialize_strategy_definition(strategy_definition),
                "weights": serialize_weights(
                    returns.columns,
                    backtest["latestWeights"],
                    risk_controls.max_investment_ratio,
                ),
                "selectedAssets": backtest["latestSelectedAssets"],
                "summary": backtest["summary"]["portfolio"],
                "benchmark": backtest["summary"]["benchmark"],
                "splitAnalysis": backtest["splitAnalysis"],
                "series": backtest["series"],
            }
        )

    return runs


def evaluate_strategy_run(
    closes: pd.DataFrame,
    volumes: pd.DataFrame | None,
    strategy_definition: StrategyDefinition,
    initial_capital: float,
    split_ratio: float,
    transaction_cost: float,
    portfolio_state: PortfolioState | None = None,
) -> dict:
    return compare_portfolio_runs(
        closes=closes,
        volumes=volumes,
        strategy_definitions=[strategy_definition],
        initial_capital=initial_capital,
        split_ratio=split_ratio,
        transaction_cost=transaction_cost,
        portfolio_state=portfolio_state,
    )[0]


def compare_portfolio_models(
    closes: pd.DataFrame,
    volumes: pd.DataFrame | None,
    model_definitions: list[PortfolioModelDefinition],
    initial_capital: float,
    split_ratio: float,
    transaction_cost: float,
    portfolio_state: PortfolioState | None = None,
) -> list[dict]:
    return compare_portfolio_runs(
        closes=closes,
        volumes=volumes,
        strategy_definitions=[
            build_strategy_definition(
                investment_universe_definition=build_investment_universe_definition(
                    tickers=list(closes.columns),
                    key="ad_hoc_universe",
                    label="Ad hoc universe",
                ),
                selection_definition=build_portfolio_strategy_definition("full_universe"),
                portfolio_model_definition=model_definition,
                execution_policy_definition=build_execution_policy_definition(
                    key="hold",
                    label="保有",
                    entry="hold",
                    rebalance_frequency="hold",
                ),
                risk_controls_definition=build_risk_controls_definition(
                    max_investment_ratio=1.0,
                    max_weight=None,
                ),
            )
            for model_definition in model_definitions
        ],
        initial_capital=initial_capital,
        split_ratio=split_ratio,
        transaction_cost=transaction_cost,
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
    if strategy_definition.strategy_type == "full_universe_momentum_low_vol_tilt":
        return list(returns.columns)
    if strategy_definition.strategy_type == "full_universe_momentum_macro_tilt":
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


def compute_strategy_score_series(
    returns: pd.DataFrame,
    volume_history: pd.DataFrame | None,
    strategy_definition: PortfolioStrategyDefinition,
) -> pd.Series | None:
    score_key = strategy_definition.score_model.key
    if score_key == "none":
        return None
    if score_key in {"momentum", "trailing_momentum_12m"}:
        return compute_trailing_total_returns(returns)
    if score_key == "momentum_low_vol":
        score_parameters = dict(strategy_definition.score_parameters)
        momentum_weight = float(score_parameters.get("momentum_weight", 0.7))
        low_vol_weight = float(score_parameters.get("low_vol_weight", 0.3))
        trailing_returns = compute_trailing_total_returns(returns)
        momentum_rank = trailing_returns.rank(method="average", pct=True)
        low_vol_rank = (-returns.std()).rank(method="average", pct=True)
        return momentum_weight * momentum_rank + low_vol_weight * low_vol_rank
    if score_key == "momentum_macro":
        score_parameters = dict(strategy_definition.score_parameters)
        momentum_weight = float(score_parameters.get("momentum_weight", 0.85))
        macro_weight = float(score_parameters.get("macro_weight", 0.15))
        trailing_returns = compute_trailing_total_returns(returns)
        momentum_rank = trailing_returns.rank(method="average", pct=True)
        macro_rank = compute_macro_proxy_rank(returns)
        return momentum_weight * momentum_rank + macro_weight * macro_rank
    if score_key == "volume_strength":
        if volume_history is None:
            raise ValueError("Volume history is required for the selected ranking model.")
        return compute_volume_strength(volume_history)
    if score_key == "volatility":
        return -returns.std()
    raise ValueError("Unsupported score model.")


def compute_trailing_total_returns(returns: pd.DataFrame) -> pd.Series:
    lookback = min(len(returns), 252)
    trailing_returns = returns.iloc[-lookback:]
    return (1 + trailing_returns).prod() - 1


def compute_volume_strength(volume_history: pd.DataFrame) -> pd.Series:
    recent_window = max(2, min(20, max(2, len(volume_history) // 4)))
    if len(volume_history) <= recent_window:
        baseline = volume_history.mean(axis=0)
    else:
        baseline = volume_history.iloc[:-recent_window].mean(axis=0)
    baseline = baseline.replace(0, np.nan)
    recent = volume_history.iloc[-recent_window:].mean(axis=0)
    return (recent / baseline).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def compute_macro_proxy_rank(returns: pd.DataFrame) -> pd.Series:
    trailing_returns = compute_trailing_total_returns(returns)
    risk_assets = [asset for asset in ["SPY", "QQQ", "IWM", "EFA", "EEM", "EWJ", "EWZ", "VNQ", "DBC", "USO", "BTC-USD", "ETH-USD"] if asset in trailing_returns.index]
    defensive_assets = [asset for asset in ["TLT", "IEF", "LQD", "HYG", "TIP", "GLD", "SLV", "UUP"] if asset in trailing_returns.index]
    if not risk_assets or not defensive_assets:
        return trailing_returns.rank(method="average", pct=True)

    risk_signal = float(trailing_returns.loc[risk_assets].mean())
    defensive_signal = float(trailing_returns.loc[defensive_assets].mean())
    regime_signal = risk_signal - defensive_signal

    macro_scores: dict[str, float] = {}
    for asset in trailing_returns.index:
        if asset in risk_assets:
            macro_scores[asset] = regime_signal
        elif asset in defensive_assets:
            macro_scores[asset] = -regime_signal
        else:
            macro_scores[asset] = 0.0
    return pd.Series(macro_scores, dtype="float64").rank(method="average", pct=True)


def evaluate_asset_ranking_definition(
    returns: pd.DataFrame,
    volumes: pd.DataFrame | None,
    split_ratio: float,
    ranking_definition: AssetRankingDefinition,
) -> dict:
    ranking_universe = [
        asset for asset in ranking_definition.investment_universe_definition.tickers if asset in returns.columns
    ]
    returns = returns[ranking_universe]
    volumes = volumes[ranking_universe] if volumes is not None else None
    split_index = compute_split_index(len(returns), split_ratio)
    observations: list[dict] = []
    latest_top_assets: list[str] = []

    for index in range(2, len(returns)):
        history_returns = returns.iloc[:index]
        history_volumes = volumes.iloc[:index] if volumes is not None else None
        strategy_definition = ranking_definition.strategy_definition
        selected_assets = select_assets(history_returns, history_volumes, strategy_definition)
        if len(selected_assets) < 2:
            continue

        score_series = compute_strategy_score_series(history_returns, history_volumes, strategy_definition)
        if score_series is None:
            continue

        selected_scores = score_series.loc[selected_assets].dropna()
        if len(selected_scores) < 2 or selected_scores.nunique() < 2:
            continue

        forward_returns = returns.iloc[index].loc[selected_scores.index].dropna()
        if len(forward_returns) < 2:
            continue

        aligned_scores = selected_scores.loc[forward_returns.index]
        if len(aligned_scores) < 2 or aligned_scores.nunique() < 2 or forward_returns.nunique() < 2:
            continue

        group_size = max(1, len(aligned_scores) // 3)
        ordered_scores = aligned_scores.sort_values(ascending=False)
        top_assets = list(ordered_scores.head(group_size).index)
        bottom_assets = list(ordered_scores.tail(group_size).index)
        top_return = float(forward_returns.loc[top_assets].mean())
        bottom_return = float(forward_returns.loc[bottom_assets].mean())
        rank_ic = aligned_scores.rank().corr(forward_returns.rank(), method="pearson")
        if pd.isna(rank_ic):
            continue

        latest_top_assets = top_assets
        observations.append(
            {
                "date": str(returns.index[index]),
                "rankIc": float(rank_ic),
                "topReturn": top_return,
                "bottomReturn": bottom_return,
                "topMinusBottom": top_return - bottom_return,
                "assetCount": len(aligned_scores),
                "segment": "train" if index < split_index else "test",
            }
        )

    return {
        "rankingDefinition": serialize_asset_ranking_definition(ranking_definition),
        "latestTopAssets": latest_top_assets,
        "overall": summarize_ranking_observations(observations),
        "train": summarize_ranking_observations(
            [observation for observation in observations if observation["segment"] == "train"]
        ),
        "test": summarize_ranking_observations(
            [observation for observation in observations if observation["segment"] == "test"]
        ),
    }


def summarize_ranking_observations(observations: list[dict]) -> dict:
    if not observations:
        return {
            "observationCount": 0,
            "meanRankIc": None,
            "meanTopReturnPct": None,
            "meanBottomReturnPct": None,
            "meanTopMinusBottomPct": None,
            "hitRatePct": None,
            "meanAssetCount": None,
        }

    top_minus_bottom_values = [observation["topMinusBottom"] for observation in observations]
    hit_count = sum(1 for value in top_minus_bottom_values if value > 0)
    return {
        "observationCount": len(observations),
        "meanRankIc": round(float(np.mean([observation["rankIc"] for observation in observations])), 4),
        "meanTopReturnPct": round(float(np.mean([observation["topReturn"] for observation in observations])) * 100, 2),
        "meanBottomReturnPct": round(
            float(np.mean([observation["bottomReturn"] for observation in observations])) * 100,
            2,
        ),
        "meanTopMinusBottomPct": round(float(np.mean(top_minus_bottom_values)) * 100, 2),
        "hitRatePct": round(hit_count / len(observations) * 100, 2),
        "meanAssetCount": round(float(np.mean([observation["assetCount"] for observation in observations])), 2),
    }


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
    expected_return_proxy = compute_expected_return_proxy(
        returns=strategy_returns,
        volume_history=volume_history[selected_assets] if volume_history is not None else None,
        strategy_definition=strategy_definition,
    )
    weights = fit_portfolio_model(
        strategy_returns,
        model_definition,
        max_investment_ratio=max_investment_ratio,
        max_weight=max_weight,
        previous_weights=selected_previous_weights,
        transaction_cost=transaction_cost,
        expected_return_proxy=expected_return_proxy,
    ) * max_investment_ratio
    if model_definition.model_type != "mean_risk_utility":
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
    score_parameters = dict(strategy_definition.score_parameters)
    if "tilt_strength" not in score_parameters:
        return weights

    score_series = compute_strategy_score_series(
        history_returns,
        None,
        strategy_definition,
    )
    if score_series is None:
        return weights
    percentile_ranks = score_series.rank(method="average", pct=True)
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


def compute_expected_return_proxy(
    *,
    returns: pd.DataFrame,
    volume_history: pd.DataFrame | None,
    strategy_definition: PortfolioStrategyDefinition,
) -> np.ndarray | None:
    score_series = compute_strategy_score_series(returns, volume_history, strategy_definition)
    if score_series is None:
        return None

    aligned_scores = score_series.reindex(returns.columns).replace([np.inf, -np.inf], np.nan)
    if aligned_scores.isna().all():
        return None

    aligned_scores = aligned_scores.fillna(aligned_scores.mean())
    if aligned_scores.nunique() < 2:
        return None

    standardized_scores = (aligned_scores - aligned_scores.mean()) / aligned_scores.std(ddof=0)
    standardized_scores = standardized_scores.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    historical_mean = returns.mean(axis=0)
    proxy_scale = float(max(historical_mean.std(ddof=0), 1e-4))
    proxy = standardized_scores.to_numpy(dtype="float64") * proxy_scale
    return np.clip(proxy, -0.05, 0.05)


def shrink_expected_return_proxy(
    expected_return_proxy: np.ndarray | None,
    strength: float,
) -> np.ndarray | None:
    if expected_return_proxy is None:
        return None
    return np.asarray(expected_return_proxy, dtype="float64") * float(strength)


def fit_portfolio_model(
    returns: pd.DataFrame,
    model_definition: PortfolioModelDefinition,
    *,
    max_investment_ratio: float,
    max_weight: float | None,
    previous_weights: np.ndarray | None,
    transaction_cost: float,
    expected_return_proxy: np.ndarray | None,
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
    elif model_definition.model_type == "mean_risk_utility":
        try:
            estimator = MeanRisk(
                objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
                risk_aversion=3.0,
                max_weights=raw_max_weight if raw_max_weight is not None else 1.0,
                transaction_costs=transaction_cost,
                previous_weights=previous_weights,
                overwrite_expected_return=(
                    None
                    if expected_return_proxy is None
                    else lambda w, proxy=expected_return_proxy: proxy @ w
                ),
            )
            estimator.fit(returns)
            weights = estimator.weights_
        except Exception:
            weights = build_equal_weight_fallback(asset_count, raw_max_weight)
    elif model_definition.model_type == "mean_risk_utility_conservative":
        try:
            conservative_proxy = shrink_expected_return_proxy(expected_return_proxy, 0.35)
            estimator = MeanRisk(
                objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
                risk_aversion=8.0,
                max_weights=raw_max_weight if raw_max_weight is not None else 1.0,
                transaction_costs=transaction_cost,
                previous_weights=previous_weights,
                overwrite_expected_return=(
                    None
                    if conservative_proxy is None
                    else lambda w, proxy=conservative_proxy: proxy @ w
                ),
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
