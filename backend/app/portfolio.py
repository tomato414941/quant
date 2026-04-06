from __future__ import annotations

from dataclasses import dataclass
import math
import statistics

import numpy as np
import pandas as pd
from skfolio.optimization import HierarchicalRiskParity, MeanRisk, ObjectiveFunction, RiskBudgeting
from app.timeframe_models import DEFAULT_DAILY_TIMEFRAME, TimeframeSpec


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
SUPPORTED_REBALANCE_SCHEDULES = {"hold", "every_bar", "month_end", "quarter_end", "year_end"}
PREDICTION_FEATURE_NAMES = ("score", "momentum", "lowVolRank", "macroRank", "volumeStrength")
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
    "trailing_momentum": "モメンタム",
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
class UniversePolicySpec:
    key: str
    label: str


@dataclass(frozen=True)
class InvestmentUniverseSpec:
    key: str
    label: str
    tickers: tuple[str, ...]


@dataclass(frozen=True)
class RankingModelSpec:
    kind: str
    label: str


@dataclass(frozen=True)
class FilterRuleSpec:
    key: str
    label: str


@dataclass(frozen=True)
class FallbackRuleSpec:
    key: str
    label: str


@dataclass(frozen=True)
class SelectionSpec:
    key: str
    strategy_type: str
    label: str
    description: str
    feature_inputs: tuple[str, ...]
    universe_policy: UniversePolicySpec
    score_model: RankingModelSpec
    score_parameters: tuple[tuple[str, object], ...]
    filter_rules: tuple[FilterRuleSpec, ...]
    fallback_rule: FallbackRuleSpec


@dataclass(frozen=True)
class PortfolioModelSpec:
    key: str
    model_type: str
    label: str
    description: str


@dataclass(frozen=True)
class ExecutionPolicySpec:
    key: str
    label: str
    entry: str
    rebalance_schedule: str


@dataclass(frozen=True)
class RiskControlsSpec:
    max_investment_ratio: float
    max_weight: float | None = None


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    version: str
    label: str
    hypothesis: str | None
    description: str
    timeframe: TimeframeSpec
    investment_universe: InvestmentUniverseSpec
    selection: SelectionSpec
    portfolio_model: PortfolioModelSpec
    execution_policy: ExecutionPolicySpec
    risk_controls: RiskControlsSpec
    extensions: tuple[tuple[str, str], ...] = ()

    @property
    def key(self) -> str:
        return self.strategy_id


@dataclass(frozen=True)
class PortfolioState:
    current_weights: dict[str, float]
    cash_weight: float = 0.0


@dataclass(frozen=True)
class AssetRankingSpec:
    key: str
    label: str
    description: str
    timeframe: TimeframeSpec
    investment_universe: InvestmentUniverseSpec
    selection: SelectionSpec
    source_strategy_keys: tuple[str, ...]
    source_strategy_labels: tuple[str, ...]


@dataclass(frozen=True)
class PredictionTargetSpec:
    key: str
    label: str
    kind: str
    horizon_spec: tuple[tuple[str, object], ...]
    baseline: str


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    label: str
    inputs: tuple[str, ...]
    parameters: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True)
class PredictionModelSpec:
    key: str
    kind: str
    label: str
    parameters: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True)
class PredictionSpec:
    key: str
    label: str
    description: str
    timeframe: TimeframeSpec
    investment_universe: InvestmentUniverseSpec
    feature_spec: FeatureSpec
    model_spec: PredictionModelSpec
    target_spec: PredictionTargetSpec
    selection: SelectionSpec
    source_strategy_keys: tuple[str, ...]
    source_strategy_labels: tuple[str, ...]


def validate_split_ratio(split_ratio: float) -> None:
    if split_ratio <= 0 or split_ratio >= 1:
        raise ValueError("Split ratio must be between 0 and 1.")


def percent_return(final_value: float, initial_value: float) -> float:
    return ((final_value / initial_value) - 1) * 100


def cagr(final_value: float, initial_value: float, periods: int, bars_per_year: float) -> float:
    years = max((periods - 1) / bars_per_year, 1 / bars_per_year)
    return (((final_value / initial_value) ** (1 / years)) - 1) * 100


def sharpe_ratio(returns: list[float], bars_per_year: float) -> float:
    if len(returns) < 2:
        return 0.0

    mean_return = statistics.fmean(returns)
    volatility = statistics.pstdev(returns)
    if volatility == 0:
        return 0.0
    return math.sqrt(bars_per_year) * (mean_return / volatility)


def max_drawdown(series: list[dict], equity_key: str) -> float:
    peak = series[0][equity_key]
    max_dd = 0.0
    for point in series:
        equity = point[equity_key]
        peak = max(peak, equity)
        drawdown = (equity / peak) - 1
        max_dd = min(max_dd, drawdown)
    return abs(max_dd) * 100


def build_investment_universe_spec(
    *,
    tickers: list[str] | tuple[str, ...],
    key: str,
    label: str,
) -> InvestmentUniverseSpec:
    normalized_tickers = tuple(dict.fromkeys(ticker.strip().upper() for ticker in tickers if ticker.strip()))
    if len(normalized_tickers) < 2:
        raise ValueError("Investment universe must contain at least two tickers.")
    return InvestmentUniverseSpec(
        key=key,
        label=label,
        tickers=normalized_tickers,
    )


def build_selection_spec(
    strategy_type: str,
    *,
    key: str | None = None,
    label: str | None = None,
    description: str | None = None,
    score_parameters: dict[str, object] | None = None,
) -> SelectionSpec:
    if strategy_type not in SUPPORTED_PORTFOLIO_STRATEGIES:
        raise ValueError("Unsupported portfolio strategy.")

    default_descriptions = {
        "full_universe": "全ETFを候補にする",
        "full_universe_momentum_tilt": "全ETFを候補にし、モメンタムで重みを傾ける",
        "full_universe_momentum_low_vol_tilt": "全ETFを候補にし、モメンタムと低ボラの複合スコアで重みを傾ける",
        "full_universe_momentum_macro_tilt": "全ETFを候補にし、モメンタムとマクロproxyの複合スコアで重みを傾ける",
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
        "full_universe": UniversePolicySpec("all_assets", UNIVERSE_POLICY_LABELS["all_assets"]),
        "full_universe_momentum_tilt": UniversePolicySpec(
            "all_assets",
            UNIVERSE_POLICY_LABELS["all_assets"],
        ),
        "full_universe_momentum_low_vol_tilt": UniversePolicySpec(
            "all_assets",
            UNIVERSE_POLICY_LABELS["all_assets"],
        ),
        "full_universe_momentum_macro_tilt": UniversePolicySpec(
            "all_assets",
            UNIVERSE_POLICY_LABELS["all_assets"],
        ),
        "momentum_top3": UniversePolicySpec("all_assets", UNIVERSE_POLICY_LABELS["all_assets"]),
        "dual_momentum_top3": UniversePolicySpec(
            "positive_assets_only",
            UNIVERSE_POLICY_LABELS["positive_assets_only"],
        ),
        "trailing_momentum_low_vol_universe": UniversePolicySpec(
            "all_assets",
            UNIVERSE_POLICY_LABELS["all_assets"],
        ),
        "positive_momentum_universe": UniversePolicySpec(
            "positive_assets_only",
            UNIVERSE_POLICY_LABELS["positive_assets_only"],
        ),
        "positive_momentum_low_vol_universe": UniversePolicySpec(
            "positive_assets_only",
            UNIVERSE_POLICY_LABELS["positive_assets_only"],
        ),
        "positive_momentum_high_volume_universe": UniversePolicySpec(
            "positive_assets_only",
            UNIVERSE_POLICY_LABELS["positive_assets_only"],
        ),
    }
    score_models = {
        "full_universe": RankingModelSpec("none", SCORE_MODEL_LABELS["none"]),
        "full_universe_momentum_tilt": RankingModelSpec(
            "trailing_momentum",
            SCORE_MODEL_LABELS["trailing_momentum"],
        ),
        "full_universe_momentum_low_vol_tilt": RankingModelSpec(
            "momentum_low_vol",
            SCORE_MODEL_LABELS["momentum_low_vol"],
        ),
        "full_universe_momentum_macro_tilt": RankingModelSpec(
            "momentum_macro",
            SCORE_MODEL_LABELS["momentum_macro"],
        ),
        "momentum_top3": RankingModelSpec("trailing_momentum", SCORE_MODEL_LABELS["trailing_momentum"]),
        "dual_momentum_top3": RankingModelSpec("trailing_momentum", SCORE_MODEL_LABELS["trailing_momentum"]),
        "trailing_momentum_low_vol_universe": RankingModelSpec(
            "trailing_momentum",
            SCORE_MODEL_LABELS["trailing_momentum"],
        ),
        "positive_momentum_universe": RankingModelSpec("trailing_momentum", SCORE_MODEL_LABELS["trailing_momentum"]),
        "positive_momentum_low_vol_universe": RankingModelSpec("trailing_momentum", SCORE_MODEL_LABELS["trailing_momentum"]),
        "positive_momentum_high_volume_universe": RankingModelSpec(
            "volume_strength",
            SCORE_MODEL_LABELS["volume_strength"],
        ),
    }
    default_score_parameters = {
        "full_universe": {},
        "full_universe_momentum_tilt": {
            "tilt_strength": 0.5,
            "windowSpec": {"unit": "bars", "value": 252},
        },
        "full_universe_momentum_low_vol_tilt": {
            "tilt_strength": 0.25,
            "tilt_shape": 1.0,
            "windowSpec": {"unit": "bars", "value": 252},
            "momentum_weight": 0.7,
            "low_vol_weight": 0.3,
        },
        "full_universe_momentum_macro_tilt": {
            "tilt_strength": 0.25,
            "tilt_shape": 1.0,
            "windowSpec": {"unit": "bars", "value": 252},
            "momentum_weight": 0.85,
            "macro_weight": 0.15,
        },
        "momentum_top3": {"windowSpec": {"unit": "bars", "value": 252}},
        "dual_momentum_top3": {"windowSpec": {"unit": "bars", "value": 252}},
        "trailing_momentum_low_vol_universe": {"windowSpec": {"unit": "bars", "value": 252}},
        "positive_momentum_universe": {"windowSpec": {"unit": "bars", "value": 252}},
        "positive_momentum_low_vol_universe": {"windowSpec": {"unit": "bars", "value": 252}},
        "positive_momentum_high_volume_universe": {"windowSpec": {"unit": "bars", "value": 252}},
    }
    filter_rules = {
        "full_universe": (),
        "full_universe_momentum_tilt": (),
        "full_universe_momentum_low_vol_tilt": (),
        "full_universe_momentum_macro_tilt": (),
        "momentum_top3": (
            FilterRuleSpec("top_3", FILTER_RULE_LABELS["top_3"]),
        ),
        "dual_momentum_top3": (
            FilterRuleSpec("positive_return", FILTER_RULE_LABELS["positive_return"]),
            FilterRuleSpec("top_3", FILTER_RULE_LABELS["top_3"]),
        ),
        "trailing_momentum_low_vol_universe": (
            FilterRuleSpec("positive_return", FILTER_RULE_LABELS["positive_return"]),
            FilterRuleSpec("low_volatility_half", FILTER_RULE_LABELS["low_volatility_half"]),
        ),
        "positive_momentum_universe": (
            FilterRuleSpec("positive_return", FILTER_RULE_LABELS["positive_return"]),
        ),
        "positive_momentum_low_vol_universe": (
            FilterRuleSpec("positive_return", FILTER_RULE_LABELS["positive_return"]),
            FilterRuleSpec("low_volatility_half", FILTER_RULE_LABELS["low_volatility_half"]),
        ),
        "positive_momentum_high_volume_universe": (
            FilterRuleSpec("positive_return", FILTER_RULE_LABELS["positive_return"]),
            FilterRuleSpec("high_volume_half", FILTER_RULE_LABELS["high_volume_half"]),
        ),
    }
    fallback_rules = {
        "full_universe": FallbackRuleSpec("none", FALLBACK_RULE_LABELS["none"]),
        "full_universe_momentum_tilt": FallbackRuleSpec("none", FALLBACK_RULE_LABELS["none"]),
        "full_universe_momentum_low_vol_tilt": FallbackRuleSpec("none", FALLBACK_RULE_LABELS["none"]),
        "full_universe_momentum_macro_tilt": FallbackRuleSpec("none", FALLBACK_RULE_LABELS["none"]),
        "momentum_top3": FallbackRuleSpec("none", FALLBACK_RULE_LABELS["none"]),
        "dual_momentum_top3": FallbackRuleSpec("cash_on_empty", FALLBACK_RULE_LABELS["cash_on_empty"]),
        "trailing_momentum_low_vol_universe": FallbackRuleSpec(
            "cash_on_empty",
            FALLBACK_RULE_LABELS["cash_on_empty"],
        ),
        "positive_momentum_universe": FallbackRuleSpec("cash_on_empty", FALLBACK_RULE_LABELS["cash_on_empty"]),
        "positive_momentum_low_vol_universe": FallbackRuleSpec("cash_on_empty", FALLBACK_RULE_LABELS["cash_on_empty"]),
        "positive_momentum_high_volume_universe": FallbackRuleSpec(
            "cash_on_empty",
            FALLBACK_RULE_LABELS["cash_on_empty"],
        ),
    }

    normalized_score_parameters = dict(default_score_parameters[strategy_type])
    if score_parameters is not None:
        normalized_score_parameters.update(score_parameters)
    if "window_bars" in normalized_score_parameters and "windowSpec" not in normalized_score_parameters:
        normalized_score_parameters["windowSpec"] = {
            "unit": "bars",
            "value": int(round(float(normalized_score_parameters.pop("window_bars")))),
        }

    return SelectionSpec(
        key=key or strategy_type,
        strategy_type=strategy_type,
        label=label or PORTFOLIO_STRATEGY_LABELS[strategy_type],
        description=description or default_descriptions[strategy_type],
        feature_inputs=feature_inputs[strategy_type],
        universe_policy=universe_policies[strategy_type],
        score_model=score_models[strategy_type],
        score_parameters=tuple(sorted(normalized_score_parameters.items())),
        filter_rules=filter_rules[strategy_type],
        fallback_rule=fallback_rules[strategy_type],
    )


def build_portfolio_model_spec(
    model_type: str,
    *,
    key: str | None = None,
    label: str | None = None,
    description: str | None = None,
) -> PortfolioModelSpec:
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

    return PortfolioModelSpec(
        key=key or model_type,
        model_type=model_type,
        label=label or PORTFOLIO_MODEL_LABELS[model_type],
        description=description or default_descriptions[model_type],
    )


def serialize_portfolio_model_spec(model: PortfolioModelSpec) -> dict:
    return {
        "key": model.key,
        "modelType": model.model_type,
        "label": model.label,
        "description": model.description,
    }


def build_execution_policy_spec(
    *,
    key: str,
    label: str,
    entry: str,
    rebalance_schedule: str,
) -> ExecutionPolicySpec:
    if rebalance_schedule not in SUPPORTED_REBALANCE_SCHEDULES:
        raise ValueError("Unsupported execution policy rebalance schedule.")
    return ExecutionPolicySpec(
        key=key,
        label=label,
        entry=entry,
        rebalance_schedule=rebalance_schedule,
    )


def serialize_execution_policy_spec(execution_policy: ExecutionPolicySpec) -> dict:
    return {
        "key": execution_policy.key,
        "label": execution_policy.label,
        "entry": execution_policy.entry,
        "rebalanceSchedule": execution_policy.rebalance_schedule,
    }


def build_risk_controls_spec(
    *,
    max_investment_ratio: float,
    max_weight: float | None = None,
) -> RiskControlsSpec:
    if max_investment_ratio <= 0 or max_investment_ratio > 1:
        raise ValueError("Max investment ratio must be between 0 and 1.")
    if max_weight is not None and (max_weight <= 0 or max_weight > 1):
        raise ValueError("Max weight must be between 0 and 1.")
    return RiskControlsSpec(
        max_investment_ratio=max_investment_ratio,
        max_weight=max_weight,
    )


def build_strategy_spec(
    *,
    timeframe: TimeframeSpec | None = None,
    investment_universe: InvestmentUniverseSpec,
    selection: SelectionSpec,
    portfolio_model: PortfolioModelSpec,
    execution_policy: ExecutionPolicySpec | None = None,
    risk_controls: RiskControlsSpec,
    strategy_id: str | None = None,
    version: str = "v1",
    hypothesis: str | None = None,
    extensions: dict[str, str] | None = None,
    key: str | None = None,
    label: str | None = None,
    description: str | None = None,
) -> StrategySpec:
    resolved_strategy_id = strategy_id or key or "__".join(
        [
            selection.key,
            portfolio_model.key,
        ]
    )
    if strategy_id is not None and key is not None and strategy_id != key:
        raise ValueError("strategy_id and key must match when both are provided.")
    strategy_label = label or " × ".join(
        [
            selection.label,
            portfolio_model.label,
        ]
    )
    return StrategySpec(
        strategy_id=resolved_strategy_id,
        version=version,
        label=strategy_label,
        hypothesis=hypothesis,
        description=description or selection.description,
        timeframe=timeframe or DEFAULT_DAILY_TIMEFRAME,
        investment_universe=investment_universe,
        selection=selection,
        portfolio_model=portfolio_model,
        execution_policy=execution_policy
        or build_execution_policy_spec(
            key="year_end",
            label="年次",
            entry="train_once_then_periodic_rebalance",
            rebalance_schedule="year_end",
        ),
        risk_controls=risk_controls,
        extensions=tuple(sorted((extensions or {}).items())),
    )


def serialize_selection_spec(selection: SelectionSpec) -> dict:
    return {
        "key": selection.key,
        "strategyType": selection.strategy_type,
        "label": selection.label,
        "description": selection.description,
        "featureInputs": list(selection.feature_inputs),
        "universePolicy": {
            "key": selection.universe_policy.key,
            "label": selection.universe_policy.label,
        },
        "scoreModel": {
            "kind": selection.score_model.kind,
            "label": selection.score_model.label,
        },
        "scoreParameters": {
            key: value for key, value in selection.score_parameters
        },
        "filterRules": [
            {
                "key": filter_rule.key,
                "label": filter_rule.label,
            }
            for filter_rule in selection.filter_rules
        ],
        "fallbackRule": {
            "key": selection.fallback_rule.key,
            "label": selection.fallback_rule.label,
        },
    }

def serialize_risk_controls_spec(risk_controls: RiskControlsSpec) -> dict:
    return {
        "maxInvestmentPct": round(risk_controls.max_investment_ratio * 100, 1),
        "maxWeightPct": round(risk_controls.max_weight * 100, 1)
        if risk_controls.max_weight is not None
        else None,
    }


def serialize_asset_ranking_model_parameters(strategy: StrategySpec) -> dict[str, object]:
    score_parameters = dict(strategy.selection.score_parameters)
    serialized: dict[str, object] = {}

    if "windowSpec" in score_parameters:
        serialized["windowSpec"] = dict(score_parameters["windowSpec"])
    if "momentum_weight" in score_parameters:
        serialized["momentumWeight"] = float(score_parameters["momentum_weight"])
    if "low_vol_weight" in score_parameters:
        serialized["lowVolWeight"] = float(score_parameters["low_vol_weight"])
    if "macro_weight" in score_parameters:
        serialized["macroWeight"] = float(score_parameters["macro_weight"])
    if "predictionSupplement" in score_parameters:
        serialized["predictionSupplement"] = dict(score_parameters["predictionSupplement"])

    return serialized


def serialize_tilt_rule(strategy: StrategySpec) -> dict | None:
    score_parameters = dict(strategy.selection.score_parameters)
    if "tilt_strength" not in score_parameters:
        return None

    parameters = {
        "strength": float(score_parameters["tilt_strength"]),
    }
    if "tilt_shape" in score_parameters:
        parameters["shape"] = float(score_parameters["tilt_shape"])

    return {
        "kind": "ranking_weight_tilt",
        "label": "ランキング連動ティルト",
        "parameters": parameters,
    }


def serialize_strategy_spec(strategy: StrategySpec) -> dict:
    ranking_model = (
        {
            "kind": strategy.selection.score_model.kind,
            "label": strategy.selection.score_model.label,
            "parameters": serialize_asset_ranking_model_parameters(strategy),
        }
        if strategy.selection.score_model.kind != "none"
        else None
    )
    fallback_rule = (
        {
            "key": strategy.selection.fallback_rule.key,
            "label": strategy.selection.fallback_rule.label,
        }
        if strategy.selection.fallback_rule.key != "none"
        else None
    )
    return {
        "kind": "strategy_spec",
        "schemaVersion": "v1",
        "strategyId": strategy.strategy_id,
        "version": strategy.version,
        "label": strategy.label,
        "hypothesis": strategy.hypothesis,
        "description": strategy.description,
        "components": {
            "core": {
                "dataResolution": {
                    "key": strategy.timeframe.key,
                    "label": strategy.timeframe.label,
                    "yfinanceInterval": strategy.timeframe.yfinance_interval,
                    "barsPerYear": strategy.timeframe.bars_per_year,
                    "barSeconds": strategy.timeframe.bar_seconds,
                },
                "investmentUniverse": {
                    "key": strategy.investment_universe.key,
                    "label": strategy.investment_universe.label,
                    "assetCount": len(strategy.investment_universe.tickers),
                    "tickers": list(strategy.investment_universe.tickers),
                },
                "portfolioModel": serialize_portfolio_model_spec(strategy.portfolio_model),
                "executionPolicy": serialize_execution_policy_spec(strategy.execution_policy),
            },
            "optional": {
                "assetRankingModel": ranking_model,
                "featureInputs": list(strategy.selection.feature_inputs),
                "filterRules": [
                    {
                        "key": filter_rule.key,
                        "label": filter_rule.label,
                    }
                    for filter_rule in strategy.selection.filter_rules
                ],
                "fallbackRule": fallback_rule,
                "riskControls": serialize_risk_controls_spec(strategy.risk_controls),
                "tiltRule": serialize_tilt_rule(strategy),
            },
        },
        "extensions": {
            key: value for key, value in strategy.extensions
        },
    }


def serialize_asset_ranking_spec(
    ranking_spec: AssetRankingSpec,
) -> dict:
    selection = ranking_spec.selection
    return {
        "kind": "asset_ranking_spec",
        "schemaVersion": "v1",
        "key": ranking_spec.key,
        "label": ranking_spec.label,
        "description": ranking_spec.description,
        "timeframe": {
            "key": ranking_spec.timeframe.key,
            "label": ranking_spec.timeframe.label,
            "yfinanceInterval": ranking_spec.timeframe.yfinance_interval,
            "barsPerYear": ranking_spec.timeframe.bars_per_year,
            "barSeconds": ranking_spec.timeframe.bar_seconds,
        },
        "investmentUniverse": {
            "key": ranking_spec.investment_universe.key,
            "label": ranking_spec.investment_universe.label,
            "assetCount": len(ranking_spec.investment_universe.tickers),
            "tickers": list(ranking_spec.investment_universe.tickers),
        },
        "featureInputs": list(selection.feature_inputs),
        "universePolicy": {
            "key": selection.universe_policy.key,
            "label": selection.universe_policy.label,
        },
        "rankingModel": {
            "kind": selection.score_model.kind,
            "label": selection.score_model.label,
        },
        "scoreParameters": extract_ranking_score_parameters(selection),
        "filterRules": [
            {
                "key": filter_rule.key,
                "label": filter_rule.label,
            }
            for filter_rule in selection.filter_rules
        ],
        "fallbackRule": {
            "key": selection.fallback_rule.key,
            "label": selection.fallback_rule.label,
        },
        "sourceStrategyKeys": list(ranking_spec.source_strategy_keys),
        "sourceStrategyLabels": list(ranking_spec.source_strategy_labels),
    }


def build_prediction_target_spec(
    *,
    key: str,
    label: str,
    kind: str,
    horizon_spec: dict[str, object],
    baseline: str = "cross_sectional_mean",
) -> PredictionTargetSpec:
    if kind not in {"forward_excess_return"}:
        raise ValueError("Unsupported prediction target kind.")
    return PredictionTargetSpec(
        key=key,
        label=label,
        kind=kind,
        horizon_spec=tuple(sorted(horizon_spec.items())),
        baseline=baseline,
    )


def serialize_prediction_target_spec(
    target_spec: PredictionTargetSpec,
) -> dict:
    return {
        "kind": "prediction_target_spec",
        "schemaVersion": "v1",
        "key": target_spec.key,
        "label": target_spec.label,
        "targetKind": target_spec.kind,
        "horizonSpec": {
            key: value for key, value in target_spec.horizon_spec
        },
        "baseline": target_spec.baseline,
    }


def build_feature_spec(
    *,
    key: str,
    label: str,
    inputs: tuple[str, ...],
    parameters: dict[str, object] | None = None,
) -> FeatureSpec:
    return FeatureSpec(
        key=key,
        label=label,
        inputs=inputs,
        parameters=tuple(sorted((parameters or {}).items())),
    )


def serialize_feature_spec(
    feature_spec: FeatureSpec,
) -> dict:
    return {
        "kind": "feature_spec",
        "schemaVersion": "v1",
        "key": feature_spec.key,
        "label": feature_spec.label,
        "inputs": list(feature_spec.inputs),
        "parameters": {
            key: value for key, value in feature_spec.parameters
        },
    }


def build_prediction_model_spec(
    *,
    key: str,
    kind: str,
    label: str,
    parameters: dict[str, object] | None = None,
) -> PredictionModelSpec:
    return PredictionModelSpec(
        key=key,
        kind=kind,
        label=label,
        parameters=tuple(sorted((parameters or {}).items())),
    )


def serialize_prediction_model_spec(
    model_spec: PredictionModelSpec,
) -> dict:
    return {
        "kind": "prediction_model_spec",
        "schemaVersion": "v1",
        "key": model_spec.key,
        "modelKind": model_spec.kind,
        "label": model_spec.label,
        "parameters": {
            key: value for key, value in model_spec.parameters
        },
    }


def freeze_parameter_value(value: object) -> object:
    if isinstance(value, dict):
        return tuple((str(key), freeze_parameter_value(nested_value)) for key, nested_value in sorted(value.items()))
    if isinstance(value, list):
        return tuple(freeze_parameter_value(item) for item in value)
    return value


def build_asset_ranking_specs(
    strategies: list[StrategySpec],
) -> list[AssetRankingSpec]:
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
        list[StrategySpec],
    ] = {}

    for strategy in strategies:
        selection = strategy.selection
        if selection.score_model.kind == "none":
            continue
        signature = (
            strategy.timeframe.key,
            strategy.investment_universe.tickers,
            selection.strategy_type,
            selection.score_model.kind,
            tuple(selection.feature_inputs),
            tuple(filter_rule.key for filter_rule in selection.filter_rules),
            selection.fallback_rule.key,
            tuple(
                (key, freeze_parameter_value(value))
                for key, value in sorted(extract_ranking_score_parameters(selection).items())
            ),
        )
        grouped.setdefault(signature, []).append(strategy)

    ranking_specs: list[AssetRankingSpec] = []
    for strategies in grouped.values():
        representative = strategies[0].selection
        filter_label = " / ".join(filter_rule.label for filter_rule in representative.filter_rules)
        label_parts = [
            strategies[0].timeframe.label,
            strategies[0].investment_universe.label,
            representative.universe_policy.label,
            representative.score_model.label,
        ]
        ranking_score_parameters = extract_ranking_score_parameters(representative)
        if ranking_score_parameters:
            parameter_label = ", ".join(
                (
                    f"{key}={value['value']} {value['unit']}"
                    if isinstance(value, dict) and "unit" in value and "value" in value
                    else f"{key}={float(value):.2f}"
                )
                for key, value in ranking_score_parameters.items()
            )
            label_parts.append(parameter_label)
        if filter_label:
            label_parts.append(filter_label)
        ranking_specs.append(
            AssetRankingSpec(
                key=f"ranking__{representative.key}",
                label=" / ".join(label_parts),
                description=representative.description,
                timeframe=strategies[0].timeframe,
                investment_universe=strategies[0].investment_universe,
                selection=representative,
                source_strategy_keys=tuple(strategy.key for strategy in strategies),
                source_strategy_labels=tuple(strategy.label for strategy in strategies),
            )
        )

    ranking_specs.sort(key=lambda ranking_spec: ranking_spec.label)
    return ranking_specs


def build_prediction_specs(
    ranking_specs: list[AssetRankingSpec],
    target_specs: list[PredictionTargetSpec],
) -> list[PredictionSpec]:
    prediction_specs: list[PredictionSpec] = []
    for ranking_spec in ranking_specs:
        ranking_parameters = extract_ranking_score_parameters(ranking_spec.selection)
        feature_spec = build_feature_spec(
            key=f"features__{ranking_spec.key}",
            label=f"{ranking_spec.selection.score_model.label} features",
            inputs=tuple(ranking_spec.selection.feature_inputs),
            parameters={
                **ranking_parameters,
                "featureKeys": ("score", "momentum", "lowVolRank", "macroRank", "volumeStrength"),
            },
        )
        model_specs = [
            build_prediction_model_spec(
                key=f"model__{ranking_spec.key}__signal",
                kind="ranking_signal_model",
                label=f"{ranking_spec.selection.score_model.label} signal",
                parameters={
                    "scoreModelKind": ranking_spec.selection.score_model.kind,
                    **ranking_parameters,
                },
            ),
            build_prediction_model_spec(
                key=f"model__{ranking_spec.key}__linear",
                kind="linear_regression",
                label=f"{ranking_spec.selection.score_model.label} linear",
                parameters={
                    "scoreModelKind": ranking_spec.selection.score_model.kind,
                    "fitMode": "expanding",
                    "minTrainSamples": 50,
                    **ranking_parameters,
                },
            ),
            build_prediction_model_spec(
                key=f"model__{ranking_spec.key}__blend",
                kind="blended_signal_model",
                label=f"{ranking_spec.selection.score_model.label} blend",
                parameters={
                    "scoreModelKind": ranking_spec.selection.score_model.kind,
                    "fitMode": "expanding",
                    "minTrainSamples": 50,
                    "signalWeight": 0.8,
                    "linearWeight": 0.2,
                    **ranking_parameters,
                },
            ),
        ]
        for target_spec in target_specs:
            for model_spec in model_specs:
                prediction_specs.append(
                    PredictionSpec(
                        key=f"prediction__{ranking_spec.key}__{model_spec.kind}__{target_spec.key}",
                        label=f"{ranking_spec.label} / {model_spec.label} -> {target_spec.label}",
                        description=ranking_spec.description,
                        timeframe=ranking_spec.timeframe,
                        investment_universe=ranking_spec.investment_universe,
                        feature_spec=feature_spec,
                        model_spec=model_spec,
                        target_spec=target_spec,
                        selection=ranking_spec.selection,
                        source_strategy_keys=ranking_spec.source_strategy_keys,
                        source_strategy_labels=ranking_spec.source_strategy_labels,
                    )
                )
    prediction_specs.sort(key=lambda prediction_spec: prediction_spec.label)
    return prediction_specs


def extract_ranking_score_parameters(
    selection: SelectionSpec,
) -> dict[str, object]:
    score_parameters = dict(selection.score_parameters)
    extracted: dict[str, object] = {}
    if "windowSpec" in score_parameters:
        extracted["windowSpec"] = dict(score_parameters["windowSpec"])
    if selection.score_model.kind == "momentum_low_vol":
        extracted["momentumWeight"] = float(score_parameters.get("momentum_weight", 0.7))
        extracted["lowVolWeight"] = float(score_parameters.get("low_vol_weight", 0.3))
        return extracted
    if selection.score_model.kind == "momentum_macro":
        extracted["momentumWeight"] = float(score_parameters.get("momentum_weight", 0.85))
        extracted["macroWeight"] = float(score_parameters.get("macro_weight", 0.15))
        return extracted
    return extracted


def serialize_prediction_spec(
    prediction_spec: PredictionSpec,
) -> dict:
    return {
        "kind": "prediction_spec",
        "schemaVersion": "v1",
        "key": prediction_spec.key,
        "label": prediction_spec.label,
        "description": prediction_spec.description,
        "timeframe": {
            "key": prediction_spec.timeframe.key,
            "label": prediction_spec.timeframe.label,
            "yfinanceInterval": prediction_spec.timeframe.yfinance_interval,
            "barsPerYear": prediction_spec.timeframe.bars_per_year,
            "barSeconds": prediction_spec.timeframe.bar_seconds,
        },
        "investmentUniverse": {
            "key": prediction_spec.investment_universe.key,
            "label": prediction_spec.investment_universe.label,
            "assetCount": len(prediction_spec.investment_universe.tickers),
            "tickers": list(prediction_spec.investment_universe.tickers),
        },
        "featureSpec": serialize_feature_spec(prediction_spec.feature_spec),
        "modelSpec": serialize_prediction_model_spec(prediction_spec.model_spec),
        "targetSpec": serialize_prediction_target_spec(prediction_spec.target_spec),
        "sourceStrategyKeys": list(prediction_spec.source_strategy_keys),
        "sourceStrategyLabels": list(prediction_spec.source_strategy_labels),
    }


def convert_window_spec_to_bars(
    window_spec: dict[str, object],
    *,
    bars_per_year: float,
) -> int:
    unit = str(window_spec.get("unit", "bars"))
    value = float(window_spec.get("value", bars_per_year))
    bars_per_trading_day = bars_per_year / 252
    bars_per_week = bars_per_year / 52
    bars_per_month = bars_per_year / 12

    if unit == "bars":
        configured_bars = value
    elif unit == "days":
        configured_bars = value * bars_per_trading_day
    elif unit == "weeks":
        configured_bars = value * bars_per_week
    elif unit == "months":
        configured_bars = value * bars_per_month
    elif unit == "years":
        configured_bars = value * bars_per_year
    else:
        raise ValueError("Unsupported windowSpec unit.")
    return max(1, int(round(configured_bars)))


def convert_horizon_spec_to_bars(
    horizon_spec: dict[str, object],
    *,
    bars_per_year: float,
) -> int:
    return convert_window_spec_to_bars(horizon_spec, bars_per_year=bars_per_year)


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


def build_flat_cost_model(*, commission_pct: float, slippage_pct: float = 0.0) -> dict:
    return {
        "kind": "flat_cost",
        "parameters": {
            "commissionPct": float(commission_pct),
            "slippagePct": float(slippage_pct),
        },
        "perAssetOverrides": {},
    }


def cost_rate_from_parameters(parameters: dict[str, float]) -> float:
    commission_pct = float(parameters.get("commissionPct", 0.0))
    slippage_pct = float(parameters.get("slippagePct", 0.0))
    return (commission_pct + slippage_pct) / 100


def impact_rate_from_parameters(parameters: dict[str, float]) -> float:
    return float(parameters.get("impactCoefficientPct", 0.0)) / 100


def resolve_cost_model_inputs(
    *,
    universe_columns: pd.Index,
    cost_model: dict,
) -> dict[str, np.ndarray | float | int]:
    kind = cost_model.get("kind", "flat_cost")
    parameters = cost_model.get("parameters", {})
    default_rate = cost_rate_from_parameters(parameters)
    default_impact_rate = impact_rate_from_parameters(parameters)
    linear_rates = np.repeat(default_rate, len(universe_columns)).astype("float64")
    impact_rates = np.repeat(default_impact_rate, len(universe_columns)).astype("float64")
    adv_window_bars = max(1, int(float(parameters.get("advWindowBars", 20))))
    min_adv_notional = float(parameters.get("minAdvNotional", 1_000_000.0))

    if kind == "flat_cost":
        return {
            "defaultLinearRate": default_rate,
            "linearRates": linear_rates,
            "impactRates": impact_rates,
            "advWindowBars": adv_window_bars,
            "minAdvNotional": min_adv_notional,
        }
    if kind not in {"asset_specific_linear_cost", "asset_specific_adv_cost"}:
        raise ValueError("Unsupported cost model.")

    overrides = cost_model.get("perAssetOverrides", {})
    for index, asset in enumerate(universe_columns):
        override = overrides.get(str(asset))
        if not override:
            continue
        linear_rates[index] = cost_rate_from_parameters(override)
        impact_rates[index] = impact_rate_from_parameters({**parameters, **override})

    return {
        "defaultLinearRate": default_rate,
        "linearRates": linear_rates,
        "impactRates": impact_rates,
        "advWindowBars": adv_window_bars,
        "minAdvNotional": min_adv_notional,
    }


def compute_trade_cost(
    *,
    weight_delta: np.ndarray,
    linear_cost_rates: np.ndarray,
    impact_cost_rates: np.ndarray,
    portfolio_equity: float,
    price_snapshot: pd.Series | None,
    volume_history: pd.DataFrame | None,
    adv_window_bars: int,
    min_adv_notional: float,
) -> float:
    linear_cost = float(np.dot(weight_delta, linear_cost_rates))
    if (
        portfolio_equity <= 0
        or price_snapshot is None
        or volume_history is None
        or len(volume_history) == 0
        or np.allclose(impact_cost_rates, 0.0)
    ):
        return linear_cost

    recent_volumes = volume_history.tail(adv_window_bars)
    if recent_volumes.empty:
        return linear_cost

    aligned_prices = price_snapshot.reindex(recent_volumes.columns).astype("float64")
    adv_shares = recent_volumes.mean(axis=0).astype("float64")
    adv_notional = np.maximum((adv_shares * aligned_prices).to_numpy(dtype="float64"), min_adv_notional)
    trade_notional = weight_delta * float(portfolio_equity)
    participation = np.clip(trade_notional / adv_notional, 0.0, None)
    impact_rates = impact_cost_rates * np.sqrt(participation)
    impact_cost = float(np.dot(weight_delta, impact_rates))
    return linear_cost + impact_cost


def compare_portfolio_runs(
    closes: pd.DataFrame,
    volumes: pd.DataFrame | None,
    strategies: list[StrategySpec],
    initial_capital: float,
    split_ratio: float,
    bars_per_year: float = 252.0,
    execution_assumptions: dict | None = None,
    cost_model: dict | None = None,
    transaction_cost: float | None = None,
    portfolio_state: PortfolioState | None = None,
) -> list[dict]:
    if not strategies:
        raise ValueError("At least one strategy is required.")
    if initial_capital <= 0:
        raise ValueError("Initial capital must be positive.")
    if execution_assumptions is None:
        execution_assumptions = {
            "kind": "close_execution_assumptions",
            "label": "終値約定",
            "parameters": {
                "fillPrice": "close",
            },
            "costModel": None,
        }
    if cost_model is None:
        cost_model = execution_assumptions.get("costModel")
    if cost_model is None:
        if transaction_cost is None:
            raise ValueError("Either cost_model or transaction_cost must be provided.")
        if transaction_cost < 0 or transaction_cost >= 1:
            raise ValueError("Transaction cost must be between 0 and 1.")
        cost_model = build_flat_cost_model(commission_pct=transaction_cost * 100, slippage_pct=0.0)

    runs: list[dict] = []
    for strategy in strategies:
        rebalance_schedule = strategy.execution_policy.rebalance_schedule
        strategy_universe = [
            asset
            for asset in strategy.investment_universe.tickers
            if asset in closes.columns
        ]
        if len(strategy_universe) < 2:
            raise ValueError("Strategy investment universe must contain at least two available assets.")

        strategy_closes = closes[strategy_universe]
        strategy_volumes = volumes[strategy_universe] if volumes is not None else None
        returns = strategy_closes.pct_change().dropna()
        if len(returns) < 6:
            raise ValueError("At least 6 return rows are required for portfolio comparison.")
        cost_inputs = resolve_cost_model_inputs(
            universe_columns=returns.columns,
            cost_model=cost_model,
        )
        default_transaction_cost = float(cost_inputs["defaultLinearRate"])
        asset_transaction_costs = np.asarray(cost_inputs["linearRates"], dtype="float64")
        asset_impact_costs = np.asarray(cost_inputs["impactRates"], dtype="float64")
        adv_window_bars = int(cost_inputs["advWindowBars"])
        min_adv_notional = float(cost_inputs["minAdvNotional"])

        split_index = compute_split_index(len(returns), split_ratio)
        train_returns = returns.iloc[:split_index]
        initial_portfolio_weights = resolve_initial_weights(
            universe_columns=returns.columns,
            portfolio_state=portfolio_state,
        )

        selection = strategy.selection
        portfolio_model = strategy.portfolio_model
        risk_controls = strategy.risk_controls
        initial_selected_assets, initial_weights = compute_portfolio_allocation(
            history_returns=train_returns,
            volume_history=strategy_volumes.loc[train_returns.index] if strategy_volumes is not None else None,
            selection=selection,
            portfolio_model=portfolio_model,
            bars_per_year=bars_per_year,
            universe_columns=returns.columns,
            max_investment_ratio=risk_controls.max_investment_ratio,
            max_weight=risk_controls.max_weight,
            previous_weights=initial_portfolio_weights,
            transaction_cost=default_transaction_cost,
        )
        backtest = run_portfolio_backtest(
            closes=strategy_closes,
            returns=returns,
            volumes=strategy_volumes,
            split_index=split_index,
            split_ratio=split_ratio,
            selection=selection,
            portfolio_model=portfolio_model,
            bars_per_year=bars_per_year,
            initial_weights=initial_weights,
            initial_selected_assets=initial_selected_assets,
            max_investment_ratio=risk_controls.max_investment_ratio,
            initial_capital=initial_capital,
            transaction_cost=default_transaction_cost,
            asset_transaction_costs=asset_transaction_costs,
            asset_impact_costs=asset_impact_costs,
            adv_window_bars=adv_window_bars,
            min_adv_notional=min_adv_notional,
            max_weight=risk_controls.max_weight,
            rebalance_schedule=rebalance_schedule,
        )

        runs.append(
            {
                "kind": "run_result",
                "schemaVersion": "v1",
                "key": strategy.key,
                "strategy": serialize_strategy_spec(strategy),
                "weights": serialize_weights(
                    returns.columns,
                    backtest["latestWeights"],
                    risk_controls.max_investment_ratio,
                ),
                "selectedAssets": backtest["latestSelectedAssets"],
                "summary": backtest["summary"],
                "splitAnalysis": backtest["splitAnalysis"],
                "series": backtest["series"],
            }
        )

    return runs


def evaluate_strategy_run(
    closes: pd.DataFrame,
    volumes: pd.DataFrame | None,
    strategy: StrategySpec,
    initial_capital: float,
    split_ratio: float,
    bars_per_year: float = 252.0,
    execution_assumptions: dict | None = None,
    cost_model: dict | None = None,
    transaction_cost: float | None = None,
    portfolio_state: PortfolioState | None = None,
) -> dict:
    return compare_portfolio_runs(
        closes=closes,
        volumes=volumes,
        strategies=[strategy],
        bars_per_year=bars_per_year,
        initial_capital=initial_capital,
        split_ratio=split_ratio,
        execution_assumptions=execution_assumptions,
        cost_model=cost_model,
        transaction_cost=transaction_cost,
        portfolio_state=portfolio_state,
    )[0]


def compare_portfolio_models(
    closes: pd.DataFrame,
    volumes: pd.DataFrame | None,
    portfolio_models: list[PortfolioModelSpec],
    initial_capital: float,
    split_ratio: float,
    transaction_cost: float,
    bars_per_year: float = 252.0,
    portfolio_state: PortfolioState | None = None,
) -> list[dict]:
    return compare_portfolio_runs(
        closes=closes,
        volumes=volumes,
        strategies=[
            build_strategy_spec(
                investment_universe=build_investment_universe_spec(
                    tickers=list(closes.columns),
                    key="ad_hoc_universe",
                    label="Ad hoc universe",
                ),
                selection=build_selection_spec("full_universe"),
                portfolio_model=portfolio_model,
                execution_policy=build_execution_policy_spec(
                    key="hold",
                    label="保有",
                    entry="hold",
                    rebalance_schedule="hold",
                ),
                risk_controls=build_risk_controls_spec(
                    max_investment_ratio=1.0,
                    max_weight=None,
                ),
            )
            for portfolio_model in portfolio_models
        ],
        bars_per_year=bars_per_year,
        initial_capital=initial_capital,
        split_ratio=split_ratio,
        execution_assumptions={
            "kind": "close_execution_assumptions",
            "label": "終値約定",
            "parameters": {
                "fillPrice": "close",
            },
            "costModel": build_flat_cost_model(
                commission_pct=transaction_cost * 100,
                slippage_pct=0.0,
            ),
        },
        transaction_cost=transaction_cost,
        portfolio_state=portfolio_state,
    )


def select_assets(
    returns: pd.DataFrame,
    volume_history: pd.DataFrame | None,
    selection: SelectionSpec,
    *,
    bars_per_year: float,
) -> list[str]:
    trailing_total_returns = compute_trailing_total_returns(
        returns,
        selection,
        bars_per_year=bars_per_year,
    )

    if selection.strategy_type == "full_universe":
        return list(returns.columns)
    if selection.strategy_type == "full_universe_momentum_tilt":
        return list(returns.columns)
    if selection.strategy_type == "full_universe_momentum_low_vol_tilt":
        return list(returns.columns)
    if selection.strategy_type == "full_universe_momentum_macro_tilt":
        return list(returns.columns)
    if selection.strategy_type == "momentum_top3":
        selected = trailing_total_returns.sort_values(ascending=False).head(min(3, len(trailing_total_returns)))
        return list(selected.index)
    if selection.strategy_type == "dual_momentum_top3":
        positive_returns = trailing_total_returns[trailing_total_returns > 0]
        if positive_returns.empty:
            return []
        selected = positive_returns.sort_values(ascending=False).head(min(3, len(positive_returns)))
        return list(selected.index)
    if selection.strategy_type == "trailing_momentum_low_vol_universe":
        positive_assets = trailing_total_returns[trailing_total_returns > 0].sort_values(ascending=False)
        if positive_assets.empty:
            return []
        volatilities = returns[positive_assets.index].std()
        low_vol_threshold = float(volatilities.quantile(0.6))
        selected = volatilities[volatilities <= low_vol_threshold].sort_values()
        if selected.empty:
            return [str(volatilities.idxmin())]
        return list(selected.index)
    if selection.strategy_type == "positive_momentum_universe":
        positive_returns = trailing_total_returns[trailing_total_returns > 0]
        if positive_returns.empty:
            return []
        return list(positive_returns.sort_values(ascending=False).index)
    if selection.strategy_type == "positive_momentum_low_vol_universe":
        positive_assets = trailing_total_returns[trailing_total_returns > 0].sort_values(ascending=False)
        if positive_assets.empty:
            return []
        volatilities = returns[positive_assets.index].std()
        low_vol_threshold = float(volatilities.median())
        selected = volatilities[volatilities <= low_vol_threshold].sort_values()
        if selected.empty:
            return [str(volatilities.idxmin())]
        return list(selected.index)
    if selection.strategy_type == "positive_momentum_high_volume_universe":
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


def compute_strategy_score_series_base(
    returns: pd.DataFrame,
    volume_history: pd.DataFrame | None,
    selection: SelectionSpec,
    *,
    bars_per_year: float,
) -> pd.Series | None:
    score_kind = selection.score_model.kind
    if score_kind == "none":
        return None
    if score_kind == "trailing_momentum":
        return compute_trailing_total_returns(returns, selection, bars_per_year=bars_per_year)
    if score_kind == "momentum_low_vol":
        score_parameters = dict(selection.score_parameters)
        momentum_weight = float(score_parameters.get("momentum_weight", 0.7))
        low_vol_weight = float(score_parameters.get("low_vol_weight", 0.3))
        trailing_returns = compute_trailing_total_returns(
            returns,
            selection,
            bars_per_year=bars_per_year,
        )
        momentum_rank = trailing_returns.rank(method="average", pct=True)
        low_vol_rank = (-returns.std()).rank(method="average", pct=True)
        return momentum_weight * momentum_rank + low_vol_weight * low_vol_rank
    if score_kind == "momentum_macro":
        score_parameters = dict(selection.score_parameters)
        momentum_weight = float(score_parameters.get("momentum_weight", 0.85))
        macro_weight = float(score_parameters.get("macro_weight", 0.15))
        trailing_returns = compute_trailing_total_returns(
            returns,
            selection,
            bars_per_year=bars_per_year,
        )
        momentum_rank = trailing_returns.rank(method="average", pct=True)
        macro_rank = compute_macro_proxy_rank(returns, selection, bars_per_year=bars_per_year)
        return momentum_weight * momentum_rank + macro_weight * macro_rank
    if score_kind == "volume_strength":
        if volume_history is None:
            raise ValueError("Volume history is required for the selected ranking model.")
        return compute_volume_strength(volume_history)
    if score_kind == "volatility":
        return -returns.std()
    raise ValueError("Unsupported score model.")


def extract_prediction_supplement(selection: SelectionSpec) -> dict[str, object] | None:
    prediction_supplement = dict(selection.score_parameters).get("predictionSupplement")
    if isinstance(prediction_supplement, dict):
        return prediction_supplement
    return None


def get_ranking_window_bars(selection: SelectionSpec, *, bars_per_year: float) -> int:
    score_parameters = dict(selection.score_parameters)
    window_spec = score_parameters.get("windowSpec")
    if isinstance(window_spec, dict):
        return convert_window_spec_to_bars(window_spec, bars_per_year=bars_per_year)
    return max(1, int(round(float(bars_per_year))))


def compute_trailing_total_returns(
    returns: pd.DataFrame,
    selection: SelectionSpec,
    *,
    bars_per_year: float,
) -> pd.Series:
    lookback = min(len(returns), get_ranking_window_bars(selection, bars_per_year=bars_per_year))
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


def compute_macro_proxy_rank(
    returns: pd.DataFrame,
    selection: SelectionSpec,
    *,
    bars_per_year: float,
) -> pd.Series:
    trailing_returns = compute_trailing_total_returns(
        returns,
        selection,
        bars_per_year=bars_per_year,
    )
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


def compute_prediction_feature_frame(
    returns: pd.DataFrame,
    volume_history: pd.DataFrame | None,
    selection: SelectionSpec,
    *,
    bars_per_year: float,
) -> pd.DataFrame:
    trailing_returns = compute_trailing_total_returns(
        returns,
        selection,
        bars_per_year=bars_per_year,
    )
    score_series = compute_strategy_score_series_base(
        returns,
        volume_history,
        selection,
        bars_per_year=bars_per_year,
    )
    if score_series is None:
        score_series = pd.Series(0.0, index=returns.columns, dtype="float64")
    vol_series = (-returns.std()).rank(method="average", pct=True)
    macro_series = compute_macro_proxy_rank(
        returns,
        selection,
        bars_per_year=bars_per_year,
    )
    if volume_history is not None:
        volume_series = compute_volume_strength(volume_history).rank(method="average", pct=True)
    else:
        volume_series = pd.Series(0.0, index=returns.columns, dtype="float64")

    feature_frame = pd.DataFrame(
        {
            "score": score_series.reindex(returns.columns).fillna(0.0),
            "momentum": trailing_returns.rank(method="average", pct=True).reindex(returns.columns).fillna(0.0),
            "lowVolRank": vol_series.reindex(returns.columns).fillna(0.0),
            "macroRank": macro_series.reindex(returns.columns).fillna(0.0),
            "volumeStrength": volume_series.reindex(returns.columns).fillna(0.0),
        }
    )
    return feature_frame.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def fit_linear_prediction_beta(
    returns: pd.DataFrame,
    volume_history: pd.DataFrame | None,
    selection: SelectionSpec,
    *,
    bars_per_year: float,
    horizon_spec: dict[str, object],
    min_train_samples: int,
) -> np.ndarray | None:
    target_horizon_bars = convert_horizon_spec_to_bars(horizon_spec, bars_per_year=bars_per_year)
    if len(returns) <= target_horizon_bars + 2:
        return None

    feature_count = len(PREDICTION_FEATURE_NAMES)
    xtx = np.zeros((feature_count + 1, feature_count + 1), dtype="float64")
    xty = np.zeros(feature_count + 1, dtype="float64")
    train_sample_count = 0

    for index in range(2, len(returns) - target_horizon_bars + 1):
        history_returns = returns.iloc[:index]
        history_volumes = volume_history.iloc[:index] if volume_history is not None else None
        selected_assets = select_assets(
            history_returns,
            history_volumes,
            selection,
            bars_per_year=bars_per_year,
        )
        if len(selected_assets) < 2:
            continue

        base_scores = compute_strategy_score_series_base(
            history_returns,
            history_volumes,
            selection,
            bars_per_year=bars_per_year,
        )
        if base_scores is None:
            continue

        selected_scores = base_scores.loc[selected_assets].dropna()
        if len(selected_scores) < 2 or selected_scores.nunique() < 2:
            continue

        feature_frame = compute_prediction_feature_frame(
            history_returns,
            history_volumes,
            selection,
            bars_per_year=bars_per_year,
        )
        aligned_features = feature_frame.loc[selected_scores.index, list(PREDICTION_FEATURE_NAMES)].astype("float64")

        forward_window = returns.iloc[index : index + target_horizon_bars]
        if len(forward_window) < target_horizon_bars:
            continue
        forward_returns = ((1 + forward_window).prod() - 1).loc[selected_scores.index].dropna()
        if len(forward_returns) < 2 or forward_returns.nunique() < 2:
            continue

        aligned_features = aligned_features.loc[forward_returns.index]
        if len(aligned_features) < 2:
            continue

        target_values = forward_returns - float(forward_returns.mean())
        if target_values.nunique() < 2:
            continue

        design_matrix = np.column_stack(
            [np.ones(len(aligned_features), dtype="float64"), aligned_features.to_numpy(dtype="float64")]
        )
        xtx += design_matrix.T @ design_matrix
        xty += design_matrix.T @ target_values.to_numpy(dtype="float64")
        train_sample_count += len(aligned_features)

    if train_sample_count < min_train_samples:
        return None
    return np.linalg.pinv(xtx) @ xty


def compute_prediction_supplemented_score_series(
    returns: pd.DataFrame,
    volume_history: pd.DataFrame | None,
    selection: SelectionSpec,
    *,
    bars_per_year: float,
    base_score_series: pd.Series,
) -> pd.Series:
    prediction_supplement = extract_prediction_supplement(selection)
    if prediction_supplement is None:
        return base_score_series

    horizon_spec = prediction_supplement.get("horizonSpec")
    if not isinstance(horizon_spec, dict):
        return base_score_series

    beta = fit_linear_prediction_beta(
        returns,
        volume_history,
        selection,
        bars_per_year=bars_per_year,
        horizon_spec=horizon_spec,
        min_train_samples=int(prediction_supplement.get("minTrainSamples", 50)),
    )
    if beta is None:
        return base_score_series

    feature_frame = compute_prediction_feature_frame(
        returns,
        volume_history,
        selection,
        bars_per_year=bars_per_year,
    )
    aligned_features = feature_frame.loc[base_score_series.index, list(PREDICTION_FEATURE_NAMES)].astype("float64")
    design_matrix = np.column_stack(
        [np.ones(len(aligned_features), dtype="float64"), aligned_features.to_numpy(dtype="float64")]
    )
    linear_prediction_values = pd.Series(
        design_matrix @ beta,
        index=aligned_features.index,
        dtype="float64",
    )
    if linear_prediction_values.nunique() < 2:
        return base_score_series

    signal_weight = float(prediction_supplement.get("signalWeight", 0.8))
    linear_weight = float(prediction_supplement.get("linearWeight", 0.2))
    blended_scores = (
        signal_weight * standardize_prediction_series(base_score_series)
        + linear_weight * standardize_prediction_series(linear_prediction_values)
    )
    if blended_scores.nunique() < 2:
        return base_score_series
    return blended_scores


def compute_strategy_score_series(
    returns: pd.DataFrame,
    volume_history: pd.DataFrame | None,
    selection: SelectionSpec,
    *,
    bars_per_year: float,
) -> pd.Series | None:
    base_score_series = compute_strategy_score_series_base(
        returns,
        volume_history,
        selection,
        bars_per_year=bars_per_year,
    )
    if base_score_series is None:
        return None
    return compute_prediction_supplemented_score_series(
        returns,
        volume_history,
        selection,
        bars_per_year=bars_per_year,
        base_score_series=base_score_series,
    )


def evaluate_asset_ranking_spec(
    returns: pd.DataFrame,
    volumes: pd.DataFrame | None,
    split_ratio: float,
    ranking_spec: AssetRankingSpec,
    *,
    bars_per_year: float,
) -> dict:
    ranking_universe = [
        asset for asset in ranking_spec.investment_universe.tickers if asset in returns.columns
    ]
    returns = returns[ranking_universe]
    volumes = volumes[ranking_universe] if volumes is not None else None
    split_index = compute_split_index(len(returns), split_ratio)
    observations: list[dict] = []
    latest_top_assets: list[str] = []

    for index in range(2, len(returns)):
        history_returns = returns.iloc[:index]
        history_volumes = volumes.iloc[:index] if volumes is not None else None
        selection = ranking_spec.selection
        selected_assets = select_assets(
            history_returns,
            history_volumes,
            selection,
            bars_per_year=bars_per_year,
        )
        if len(selected_assets) < 2:
            continue

        score_series = compute_strategy_score_series(
            history_returns,
            history_volumes,
            selection,
            bars_per_year=bars_per_year,
        )
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
        "rankingSpec": serialize_asset_ranking_spec(ranking_spec),
        "latestTopAssets": latest_top_assets,
        "overall": summarize_ranking_observations(observations),
        "train": summarize_ranking_observations(
            [observation for observation in observations if observation["segment"] == "train"]
        ),
        "test": summarize_ranking_observations(
            [observation for observation in observations if observation["segment"] == "test"]
        ),
    }


def evaluate_prediction_spec(
    returns: pd.DataFrame,
    volumes: pd.DataFrame | None,
    split_ratio: float,
    prediction_spec: PredictionSpec,
    *,
    bars_per_year: float,
) -> dict:
    ranking_spec = AssetRankingSpec(
        key=prediction_spec.key,
        label=prediction_spec.label,
        description=prediction_spec.description,
        timeframe=prediction_spec.timeframe,
        investment_universe=prediction_spec.investment_universe,
        selection=prediction_spec.selection,
        source_strategy_keys=prediction_spec.source_strategy_keys,
        source_strategy_labels=prediction_spec.source_strategy_labels,
    )
    target_spec = prediction_spec.target_spec
    ranking_universe = [
        asset for asset in ranking_spec.investment_universe.tickers if asset in returns.columns
    ]
    returns = returns[ranking_universe]
    volumes = volumes[ranking_universe] if volumes is not None else None
    split_index = compute_split_index(len(returns), split_ratio)
    target_horizon_bars = convert_horizon_spec_to_bars(
        {key: value for key, value in target_spec.horizon_spec},
        bars_per_year=bars_per_year,
    )
    observations: list[dict] = []
    latest_top_assets: list[str] = []
    latest_scores: list[dict[str, float | str]] = []
    feature_names = ("score", "momentum", "lowVolRank", "macroRank", "volumeStrength")
    feature_count = len(feature_names)
    train_sample_count = 0
    xtx = np.zeros((feature_count + 1, feature_count + 1), dtype="float64")
    xty = np.zeros(feature_count + 1, dtype="float64")
    model_parameters = {key: value for key, value in prediction_spec.model_spec.parameters}
    min_train_samples = int(model_parameters.get("minTrainSamples", 1))
    signal_weight = float(model_parameters.get("signalWeight", 0.8))
    linear_weight = float(model_parameters.get("linearWeight", 0.2))

    for index in range(2, len(returns) - target_horizon_bars + 1):
        history_returns = returns.iloc[:index]
        history_volumes = volumes.iloc[:index] if volumes is not None else None
        selection = ranking_spec.selection
        selected_assets = select_assets(
            history_returns,
            history_volumes,
            selection,
            bars_per_year=bars_per_year,
        )
        if len(selected_assets) < 2:
            continue

        score_series = compute_strategy_score_series(
            history_returns,
            history_volumes,
            selection,
            bars_per_year=bars_per_year,
        )
        if score_series is None:
            continue
        feature_frame = compute_prediction_feature_frame(
            history_returns,
            history_volumes,
            selection,
            bars_per_year=bars_per_year,
        )

        selected_scores = score_series.loc[selected_assets].dropna()
        if len(selected_scores) < 2 or selected_scores.nunique() < 2:
            continue

        forward_window = returns.iloc[index : index + target_horizon_bars]
        if len(forward_window) < target_horizon_bars:
            continue
        forward_returns = ((1 + forward_window).prod() - 1).loc[selected_scores.index].dropna()
        if len(forward_returns) < 2:
            continue

        aligned_scores = selected_scores.loc[forward_returns.index]
        aligned_features = feature_frame.loc[forward_returns.index, list(feature_names)].astype("float64")
        if len(aligned_scores) < 2 or aligned_scores.nunique() < 2 or forward_returns.nunique() < 2:
            continue

        if target_spec.baseline == "cross_sectional_mean":
            target_values = forward_returns - float(forward_returns.mean())
        else:
            raise ValueError("Unsupported prediction target baseline.")

        if target_values.nunique() < 2:
            continue

        if prediction_spec.model_spec.kind == "ranking_signal_model":
            prediction_values = aligned_scores
        elif prediction_spec.model_spec.kind in {"linear_regression", "blended_signal_model"}:
            design_matrix = np.column_stack(
                [np.ones(len(aligned_features), dtype="float64"), aligned_features.to_numpy(dtype="float64")]
            )
            if train_sample_count < min_train_samples:
                xtx += design_matrix.T @ design_matrix
                xty += design_matrix.T @ target_values.to_numpy(dtype="float64")
                train_sample_count += len(aligned_features)
                continue
            beta = np.linalg.pinv(xtx) @ xty
            prediction_array = design_matrix @ beta
            linear_prediction_values = pd.Series(
                prediction_array,
                index=aligned_features.index,
                dtype="float64",
            )
            if linear_prediction_values.nunique() < 2:
                xtx += design_matrix.T @ design_matrix
                xty += design_matrix.T @ target_values.to_numpy(dtype="float64")
                train_sample_count += len(aligned_features)
                continue
            if prediction_spec.model_spec.kind == "linear_regression":
                prediction_values = linear_prediction_values
            else:
                prediction_values = (
                    signal_weight * standardize_prediction_series(aligned_scores)
                    + linear_weight * standardize_prediction_series(linear_prediction_values)
                )
                if prediction_values.nunique() < 2:
                    xtx += design_matrix.T @ design_matrix
                    xty += design_matrix.T @ target_values.to_numpy(dtype="float64")
                    train_sample_count += len(aligned_features)
                    continue
        else:
            raise ValueError("Unsupported prediction model kind.")

        group_size = max(1, len(aligned_scores) // 3)
        ordered_scores = prediction_values.sort_values(ascending=False)
        top_assets = list(ordered_scores.head(group_size).index)
        bottom_assets = list(ordered_scores.tail(group_size).index)
        top_return = float(forward_returns.loc[top_assets].mean())
        bottom_return = float(forward_returns.loc[bottom_assets].mean())
        rank_ic = prediction_values.rank().corr(target_values.rank(), method="pearson")
        pearson_corr = prediction_values.corr(target_values, method="pearson")
        if pd.isna(rank_ic) or pd.isna(pearson_corr):
            continue

        latest_top_assets = top_assets
        latest_scores = [
            {"asset": str(asset), "score": round(float(score), 6)}
            for asset, score in ordered_scores.head(5).items()
        ]
        observations.append(
            {
                "date": str(returns.index[index]),
                "rankIc": float(rank_ic),
                "pearsonCorr": float(pearson_corr),
                "topReturn": top_return,
                "bottomReturn": bottom_return,
                "topMinusBottom": top_return - bottom_return,
                "assetCount": len(aligned_scores),
                "segment": "train" if index < split_index else "test",
            }
        )
        design_matrix = np.column_stack(
            [np.ones(len(aligned_features), dtype="float64"), aligned_features.to_numpy(dtype="float64")]
        )
        xtx += design_matrix.T @ design_matrix
        xty += design_matrix.T @ target_values.to_numpy(dtype="float64")
        train_sample_count += len(aligned_features)

    return {
        "predictionSpec": serialize_prediction_spec(prediction_spec),
        "latestTopAssets": latest_top_assets,
        "latestScores": latest_scores,
        "overall": summarize_prediction_observations(observations),
        "train": summarize_prediction_observations(
            [observation for observation in observations if observation["segment"] == "train"]
        ),
        "test": summarize_prediction_observations(
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


def summarize_prediction_observations(observations: list[dict]) -> dict:
    if not observations:
        return {
            "observationCount": 0,
            "meanRankIc": None,
            "meanPearsonCorr": None,
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
        "meanPearsonCorr": round(
            float(np.mean([observation["pearsonCorr"] for observation in observations])),
            4,
        ),
        "meanTopReturnPct": round(float(np.mean([observation["topReturn"] for observation in observations])) * 100, 2),
        "meanBottomReturnPct": round(
            float(np.mean([observation["bottomReturn"] for observation in observations])) * 100,
            2,
        ),
        "meanTopMinusBottomPct": round(float(np.mean(top_minus_bottom_values)) * 100, 2),
        "hitRatePct": round(hit_count / len(observations) * 100, 2),
        "meanAssetCount": round(float(np.mean([observation["assetCount"] for observation in observations])), 2),
    }


def standardize_prediction_series(series: pd.Series) -> pd.Series:
    centered = series - float(series.mean())
    std = float(series.std(ddof=0))
    if std <= 0:
        return pd.Series(0.0, index=series.index, dtype="float64")
    standardized = centered / std
    return standardized.replace([np.inf, -np.inf], np.nan).fillna(0.0)


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
    selection: SelectionSpec,
    portfolio_model: PortfolioModelSpec,
    bars_per_year: float,
    universe_columns: pd.Index,
    max_investment_ratio: float,
    max_weight: float | None,
    previous_weights: np.ndarray | None,
    transaction_cost: float,
) -> tuple[list[str], np.ndarray]:
    selected_assets = select_assets(
        history_returns,
        volume_history,
        selection,
        bars_per_year=bars_per_year,
    )
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
        selection=selection,
        bars_per_year=bars_per_year,
    )
    weights = fit_portfolio_model(
        strategy_returns,
        portfolio_model,
        max_investment_ratio=max_investment_ratio,
        max_weight=max_weight,
        previous_weights=selected_previous_weights,
        transaction_cost=transaction_cost,
        expected_return_proxy=expected_return_proxy,
    ) * max_investment_ratio
    if portfolio_model.model_type != "mean_risk_utility":
        weights = apply_strategy_weight_tilt(
            weights=weights,
            history_returns=strategy_returns,
            selection=selection,
            bars_per_year=bars_per_year,
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
    selection: SelectionSpec,
    bars_per_year: float,
    max_investment_ratio: float,
    max_weight: float | None,
) -> np.ndarray:
    score_parameters = dict(selection.score_parameters)
    if "tilt_strength" not in score_parameters:
        return weights

    score_series = compute_strategy_score_series(
        history_returns,
        None,
        selection,
        bars_per_year=bars_per_year,
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
    selection: SelectionSpec,
    bars_per_year: float,
) -> np.ndarray | None:
    score_series = compute_strategy_score_series(
        returns,
        volume_history,
        selection,
        bars_per_year=bars_per_year,
    )
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
    portfolio_model: PortfolioModelSpec,
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

    if portfolio_model.model_type == "equal_weight":
        weights = build_equal_weight_fallback(asset_count, raw_max_weight)
    elif portfolio_model.model_type == "risk_budgeting":
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
    elif portfolio_model.model_type == "minimum_variance":
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
    elif portfolio_model.model_type == "hierarchical_risk_parity":
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
    elif portfolio_model.model_type == "mean_risk_utility":
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
    elif portfolio_model.model_type == "mean_risk_utility_conservative":
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
    closes: pd.DataFrame,
    returns: pd.DataFrame,
    volumes: pd.DataFrame | None,
    bars_per_year: float,
    split_index: int,
    split_ratio: float,
    selection: SelectionSpec,
    portfolio_model: PortfolioModelSpec,
    initial_weights: np.ndarray,
    initial_selected_assets: list[str],
    max_investment_ratio: float,
    initial_capital: float,
    transaction_cost: float,
    asset_transaction_costs: np.ndarray,
    asset_impact_costs: np.ndarray,
    adv_window_bars: int,
    min_adv_notional: float,
    max_weight: float | None,
    rebalance_schedule: str,
) -> dict:
    portfolio_equity = initial_capital
    portfolio_returns: list[float] = []
    series: list[dict] = []
    train_dates: list[str] = []
    test_dates: list[str] = []
    train_portfolio_returns: list[float] = []
    test_portfolio_returns: list[float] = []
    train_turnover = 0.0
    test_turnover = 0.0
    current_weights = initial_weights.copy()
    current_selected_assets = list(initial_selected_assets)
    latest_weights = initial_weights.copy()
    latest_selected_assets = list(initial_selected_assets)

    for index, (date, row) in enumerate(returns.iterrows()):
        trade_turnover = 0.0
        trade_cost = 0.0
        if index == 0:
            trade_turnover = float(np.abs(current_weights).sum())
            trade_cost = compute_trade_cost(
                weight_delta=np.abs(current_weights),
                linear_cost_rates=asset_transaction_costs,
                impact_cost_rates=asset_impact_costs,
                portfolio_equity=portfolio_equity,
                price_snapshot=closes.iloc[index],
                volume_history=volumes.iloc[: index + 1] if volumes is not None else None,
                adv_window_bars=adv_window_bars,
                min_adv_notional=min_adv_notional,
            )
        elif index > split_index and should_rebalance(
            previous_date=returns.index[index - 1],
            current_date=date,
            rebalance_schedule=rebalance_schedule,
        ):
            current_selected_assets, rebalanced_weights = compute_portfolio_allocation(
                history_returns=returns.iloc[:index],
                volume_history=volumes.iloc[:index] if volumes is not None else None,
                selection=selection,
                portfolio_model=portfolio_model,
                bars_per_year=bars_per_year,
                universe_columns=returns.columns,
                max_investment_ratio=max_investment_ratio,
                max_weight=max_weight,
                previous_weights=current_weights,
                transaction_cost=transaction_cost,
            )
            weight_delta = np.abs(rebalanced_weights - current_weights)
            trade_turnover = float(weight_delta.sum())
            trade_cost = compute_trade_cost(
                weight_delta=weight_delta,
                linear_cost_rates=asset_transaction_costs,
                impact_cost_rates=asset_impact_costs,
                portfolio_equity=portfolio_equity,
                price_snapshot=closes.iloc[index - 1],
                volume_history=volumes.iloc[:index] if volumes is not None else None,
                adv_window_bars=adv_window_bars,
                min_adv_notional=min_adv_notional,
            )
            current_weights = rebalanced_weights
            latest_weights = rebalanced_weights.copy()
            latest_selected_assets = list(current_selected_assets)

        portfolio_return = float(np.dot(row.to_numpy(dtype="float64"), current_weights))
        portfolio_return -= trade_cost

        portfolio_equity *= 1 + portfolio_return
        portfolio_returns.append(portfolio_return)
        segment_dates = train_dates if index < split_index else test_dates
        segment_portfolio_returns = train_portfolio_returns if index < split_index else test_portfolio_returns
        segment_dates.append(str(date))
        segment_portfolio_returns.append(portfolio_return)
        if index < split_index:
            train_turnover += trade_turnover
        else:
            test_turnover += trade_turnover
        series.append(
            {
                "date": str(date),
                "portfolioEquity": round(portfolio_equity, 2),
                "portfolioReturnPct": round(portfolio_return * 100, 2),
            }
        )

    summary = summarize_portfolio_metrics(
        final_value=portfolio_equity,
        initial_value=initial_capital,
        periods=len(returns),
        returns=portfolio_returns,
        bars_per_year=bars_per_year,
        series=series,
        equity_key="portfolioEquity",
        turnover=round((train_turnover + test_turnover) * 100, 2),
    )
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
                bars_per_year=bars_per_year,
                turnover=train_turnover,
            ),
            "test": summarize_segment_from_returns(
                dates=test_dates,
                portfolio_returns=test_portfolio_returns,
                bars_per_year=bars_per_year,
                turnover=test_turnover,
            ),
        },
    }


def summarize_portfolio_metrics(
    final_value: float,
    initial_value: float,
    periods: int,
    returns: list[float],
    bars_per_year: float,
    series: list[dict],
    equity_key: str,
    turnover: float,
) -> dict:
    return {
        "totalReturnPct": round(percent_return(final_value, initial_value), 2),
        "cagrPct": round(cagr(final_value, initial_value, periods, bars_per_year), 2),
        "sharpeRatio": round(sharpe_ratio(returns, bars_per_year), 2),
        "maxDrawdownPct": round(max_drawdown(series, equity_key), 2),
        "turnoverPct": round(turnover, 2),
    }


def summarize_segment_from_returns(
    dates: list[str],
    portfolio_returns: list[float],
    bars_per_year: float,
    turnover: float,
) -> dict:
    return {
        "startDate": dates[0],
        "endDate": dates[-1],
        "barCount": len(dates),
        "portfolio": summarize_metrics_from_bar_returns(
            dates=dates,
            bar_returns=portfolio_returns,
            bars_per_year=bars_per_year,
            equity_key="portfolioEquity",
            turnover=turnover,
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


def summarize_metrics_from_bar_returns(
    dates: list[str],
    bar_returns: list[float],
    bars_per_year: float,
    equity_key: str,
    turnover: float,
) -> dict:
    equity = 100.0
    series = []
    for date, bar_return in zip(dates, bar_returns, strict=True):
        equity *= 1 + bar_return
        series.append({"date": date, equity_key: round(equity, 2)})
    return summarize_portfolio_metrics(
        final_value=equity,
        initial_value=100.0,
        periods=len(bar_returns),
        returns=bar_returns,
        bars_per_year=bars_per_year,
        series=series,
        equity_key=equity_key,
        turnover=round(turnover * 100, 2),
    )


def should_rebalance(previous_date, current_date, rebalance_schedule: str) -> bool:
    if rebalance_schedule == "hold":
        return False

    previous_timestamp = pd.Timestamp(previous_date)
    current_timestamp = pd.Timestamp(current_date)
    if rebalance_schedule == "every_bar":
        return previous_timestamp.normalize() != current_timestamp.normalize()
    if rebalance_schedule == "month_end":
        return previous_timestamp.month != current_timestamp.month or previous_timestamp.year != current_timestamp.year
    if rebalance_schedule == "quarter_end":
        return previous_timestamp.quarter != current_timestamp.quarter or previous_timestamp.year != current_timestamp.year
    if rebalance_schedule == "year_end":
        return previous_timestamp.year != current_timestamp.year
    raise ValueError("Unsupported rebalance schedule.")


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
