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
class ObservationSpec:
    key: str
    label: str
    tickers: tuple[str, ...]
    fields: tuple[str, ...]


@dataclass(frozen=True)
class RankingModelSpec:
    kind: str
    label: str


@dataclass(frozen=True)
class RankingSignalSpec:
    feature_inputs: tuple[str, ...]
    score_model: RankingModelSpec
    score_parameters: tuple[tuple[str, object], ...]


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
    ranking_signal: RankingSignalSpec
    universe_policy: UniversePolicySpec
    filter_rules: tuple[FilterRuleSpec, ...]
    fallback_rule: FallbackRuleSpec


@dataclass(frozen=True)
class RankingFeatureRecipeSpec:
    key: str
    strategy_type: str
    label: str
    description: str
    ranking_signal: RankingSignalSpec
    universe_policy: UniversePolicySpec
    filter_rules: tuple[FilterRuleSpec, ...]
    fallback_rule: FallbackRuleSpec


RankingSourceSpec = SelectionSpec | RankingFeatureRecipeSpec


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
class PredictorUseSpec:
    predictor_key: str
    signal_weight: float
    predictor_weight: float


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
    predictor_use: PredictorUseSpec | None = None
    extensions: tuple[tuple[str, str], ...] = ()

    @property
    def key(self) -> str:
        return self.strategy_id


@dataclass(frozen=True)
class PortfolioState:
    current_weights: dict[str, float]
    cash_weight: float = 0.0


@dataclass(frozen=True)
class CandidateSetSpec:
    key: str
    label: str
    description: str
    universe_policy: UniversePolicySpec
    filter_rules: tuple[FilterRuleSpec, ...]
    fallback_rule: FallbackRuleSpec


@dataclass(frozen=True)
class DecisionUseSpec:
    key: str
    label: str
    description: str
    kind: str
    candidate_set_spec: CandidateSetSpec


@dataclass(frozen=True)
class AssetRankingSpec:
    key: str
    label: str
    description: str
    timeframe: TimeframeSpec
    investment_universe: InvestmentUniverseSpec
    selection: SelectionSpec


@dataclass(frozen=True)
class PredictionTargetSpec:
    key: str
    label: str
    horizon_spec: tuple[tuple[str, object], ...]
    baseline: str
    transform: str


@dataclass(frozen=True)
class PredictedQuantitySpec:
    key: str
    label: str
    kind: str


@dataclass(frozen=True)
class FeatureInputSpec:
    key: str
    label: str


@dataclass(frozen=True)
class DerivedFeatureSpec:
    key: str
    label: str


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    label: str
    feature_inputs: tuple[FeatureInputSpec, ...]
    derived_features: tuple[DerivedFeatureSpec, ...]
    ranking_feature_recipe: RankingFeatureRecipeSpec | None = None


@dataclass(frozen=True)
class PredictionSignalSourceSpec:
    key: str
    label: str
    kind: str
    feature_key: str | None = None


@dataclass(frozen=True)
class PredictionLearnerSpec:
    key: str
    label: str
    kind: str
    ridge_alpha: float | None = None


@dataclass(frozen=True)
class PredictionCombinerSpec:
    key: str
    label: str
    kind: str
    signal_source_weight: float | None = None
    learner_weight: float | None = None


@dataclass(frozen=True)
class PredictionEngineSpec:
    key: str
    label: str
    signal_source_spec: PredictionSignalSourceSpec | None
    learner_spec: PredictionLearnerSpec | None
    combiner_spec: PredictionCombinerSpec


@dataclass(frozen=True)
class TrainingSpec:
    key: str
    label: str
    fit_mode: str
    min_train_samples: int


@dataclass(frozen=True)
class PredictionOutputSpec:
    key: str
    label: str
    kind: str


@dataclass(frozen=True)
class SignalSpec:
    key: str
    label: str
    observation_spec: ObservationSpec
    entity_kind: str
    entity_identifiers: tuple[str, ...]
    output_spec: PredictionOutputSpec
    decision_use_spec: DecisionUseSpec


@dataclass(frozen=True)
class PredictorSpec:
    key: str
    label: str
    description: str
    timeframe: TimeframeSpec
    signal_spec: SignalSpec
    predicted_quantity_spec: PredictedQuantitySpec
    target_spec: PredictionTargetSpec
    feature_spec: FeatureSpec
    engine_spec: PredictionEngineSpec
    training_spec: TrainingSpec


@dataclass(frozen=True)
class PredictorRunSpec:
    predictor_spec: PredictorSpec


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


def build_observation_spec(
    *,
    tickers: list[str] | tuple[str, ...],
    fields: list[str] | tuple[str, ...],
    key: str,
    label: str,
) -> ObservationSpec:
    normalized_tickers = tuple(dict.fromkeys(ticker.strip().upper() for ticker in tickers if ticker.strip()))
    normalized_fields = tuple(dict.fromkeys(field.strip() for field in fields if field.strip()))
    if len(normalized_tickers) < 1:
        raise ValueError("Observation spec must contain at least one ticker.")
    if len(normalized_fields) < 1:
        raise ValueError("Observation spec must contain at least one field.")
    return ObservationSpec(
        key=key,
        label=label,
        tickers=normalized_tickers,
        fields=normalized_fields,
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
        ranking_signal=RankingSignalSpec(
            feature_inputs=feature_inputs[strategy_type],
            score_model=score_models[strategy_type],
            score_parameters=tuple(sorted(normalized_score_parameters.items())),
        ),
        universe_policy=universe_policies[strategy_type],
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


def build_predictor_use_spec(
    *,
    predictor_key: str,
    signal_weight: float = 0.8,
    predictor_weight: float = 0.2,
) -> PredictorUseSpec:
    return PredictorUseSpec(
        predictor_key=predictor_key,
        signal_weight=float(signal_weight),
        predictor_weight=float(predictor_weight),
    )


def build_strategy_spec(
    *,
    timeframe: TimeframeSpec | None = None,
    investment_universe: InvestmentUniverseSpec,
    selection: SelectionSpec,
    portfolio_model: PortfolioModelSpec,
    execution_policy: ExecutionPolicySpec | None = None,
    risk_controls: RiskControlsSpec,
    predictor_use: PredictorUseSpec | None = None,
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
        predictor_use=predictor_use,
        extensions=tuple(sorted((extensions or {}).items())),
    )


def serialize_ranking_source_spec(ranking_source: RankingSourceSpec) -> dict:
    return {
        "key": ranking_source.key,
        "strategyType": ranking_source.strategy_type,
        "label": ranking_source.label,
        "description": ranking_source.description,
        "featureInputs": list(ranking_source.ranking_signal.feature_inputs),
        "universePolicy": {
            "key": ranking_source.universe_policy.key,
            "label": ranking_source.universe_policy.label,
        },
        "scoreModel": {
            "kind": ranking_source.ranking_signal.score_model.kind,
            "label": ranking_source.ranking_signal.score_model.label,
        },
        "scoreParameters": {
            key: value for key, value in ranking_source.ranking_signal.score_parameters
        },
        "filterRules": [
            {
                "key": filter_rule.key,
                "label": filter_rule.label,
            }
            for filter_rule in ranking_source.filter_rules
        ],
        "fallbackRule": {
            "key": ranking_source.fallback_rule.key,
            "label": ranking_source.fallback_rule.label,
        },
    }


def serialize_selection_spec(selection: SelectionSpec) -> dict:
    return serialize_ranking_source_spec(selection)


def build_candidate_set_spec(selection: SelectionSpec) -> CandidateSetSpec:
    filter_label = " / ".join(filter_rule.label for filter_rule in selection.filter_rules)
    candidate_label = selection.universe_policy.label if not filter_label else f"{selection.universe_policy.label} / {filter_label}"
    return CandidateSetSpec(
        key=f"candidate_set__{selection.key}",
        label=candidate_label,
        description=selection.description,
        universe_policy=selection.universe_policy,
        filter_rules=selection.filter_rules,
        fallback_rule=selection.fallback_rule,
    )


def serialize_candidate_set_spec(candidate_set_spec: CandidateSetSpec) -> dict:
    return {
        "kind": "candidate_set_spec",
        "schemaVersion": "v1",
        "key": candidate_set_spec.key,
        "label": candidate_set_spec.label,
        "description": candidate_set_spec.description,
        "universePolicy": {
            "key": candidate_set_spec.universe_policy.key,
            "label": candidate_set_spec.universe_policy.label,
        },
        "filterRules": [
            {
                "key": filter_rule.key,
                "label": filter_rule.label,
            }
            for filter_rule in candidate_set_spec.filter_rules
        ],
        "fallbackRule": {
            "key": candidate_set_spec.fallback_rule.key,
            "label": candidate_set_spec.fallback_rule.label,
        },
    }


def serialize_observation_spec(observation_spec: ObservationSpec) -> dict:
    return {
        "kind": "observation_spec",
        "schemaVersion": "v1",
        "key": observation_spec.key,
        "label": observation_spec.label,
        "assetCount": len(observation_spec.tickers),
        "tickers": list(observation_spec.tickers),
        "fields": list(observation_spec.fields),
    }


def build_signal_spec(
    *,
    key: str,
    label: str,
    observation_spec: ObservationSpec,
    entity_kind: str,
    entity_identifiers: list[str] | tuple[str, ...],
    output_spec: PredictionOutputSpec,
    decision_use_spec: DecisionUseSpec,
) -> SignalSpec:
    normalized_identifiers = tuple(
        dict.fromkeys(identifier.strip().upper() for identifier in entity_identifiers if identifier.strip())
    )
    if len(normalized_identifiers) < 1:
        raise ValueError("Signal spec must contain at least one entity identifier.")
    return SignalSpec(
        key=key,
        label=label,
        observation_spec=observation_spec,
        entity_kind=entity_kind,
        entity_identifiers=normalized_identifiers,
        output_spec=output_spec,
        decision_use_spec=decision_use_spec,
    )


def serialize_signal_spec(signal_spec: SignalSpec) -> dict:
    return {
        "kind": "signal_spec",
        "schemaVersion": "v1",
        "key": signal_spec.key,
        "label": signal_spec.label,
        "observationSpec": serialize_observation_spec(signal_spec.observation_spec),
        "entityKind": signal_spec.entity_kind,
        "entityIdentifiers": list(signal_spec.entity_identifiers),
        "outputSpec": serialize_prediction_output_spec(signal_spec.output_spec),
        "decisionUseSpec": serialize_decision_use_spec(signal_spec.decision_use_spec),
    }


def build_decision_use_spec(selection: SelectionSpec) -> DecisionUseSpec:
    candidate_set_spec = build_candidate_set_spec(selection)
    return DecisionUseSpec(
        key=f"decision_use__{selection.key}",
        label="Candidate ranking signal",
        description=selection.description,
        kind="candidate_ranking_signal",
        candidate_set_spec=candidate_set_spec,
    )


def serialize_decision_use_spec(
    decision_use_spec: DecisionUseSpec,
) -> dict:
    return {
        "kind": "decision_use_spec",
        "schemaVersion": "v1",
        "key": decision_use_spec.key,
        "label": decision_use_spec.label,
        "description": decision_use_spec.description,
        "useKind": decision_use_spec.kind,
        "candidateSet": serialize_candidate_set_spec(decision_use_spec.candidate_set_spec),
    }


def deserialize_ranking_feature_recipe(payload: dict[str, object]) -> RankingFeatureRecipeSpec:
    universe_policy = payload.get("universePolicy")
    score_model = payload.get("scoreModel")
    fallback_rule = payload.get("fallbackRule")
    if not isinstance(universe_policy, dict):
        raise ValueError("Ranking feature recipe must include a universePolicy object.")
    if not isinstance(score_model, dict):
        raise ValueError("Ranking feature recipe must include a scoreModel object.")
    if not isinstance(fallback_rule, dict):
        raise ValueError("Ranking feature recipe must include a fallbackRule object.")

    return RankingFeatureRecipeSpec(
        key=str(payload["key"]),
        strategy_type=str(payload["strategyType"]),
        label=str(payload["label"]),
        description=str(payload["description"]),
        ranking_signal=RankingSignalSpec(
            feature_inputs=tuple(str(feature_input) for feature_input in payload.get("featureInputs", ())),
            score_model=RankingModelSpec(
                kind=str(score_model["kind"]),
                label=str(score_model["label"]),
            ),
            score_parameters=tuple(sorted(dict(payload.get("scoreParameters", {})).items())),
        ),
        universe_policy=UniversePolicySpec(
            key=str(universe_policy["key"]),
            label=str(universe_policy["label"]),
        ),
        filter_rules=tuple(
            FilterRuleSpec(key=str(filter_rule["key"]), label=str(filter_rule["label"]))
            for filter_rule in payload.get("filterRules", ())
            if isinstance(filter_rule, dict)
        ),
        fallback_rule=FallbackRuleSpec(
            key=str(fallback_rule["key"]),
            label=str(fallback_rule["label"]),
        ),
    )


def build_ranking_feature_recipe_spec(selection: SelectionSpec) -> RankingFeatureRecipeSpec:
    return RankingFeatureRecipeSpec(
        key=selection.key,
        strategy_type=selection.strategy_type,
        label=selection.label,
        description=selection.description,
        ranking_signal=selection.ranking_signal,
        universe_policy=selection.universe_policy,
        filter_rules=selection.filter_rules,
        fallback_rule=selection.fallback_rule,
    )


def serialize_ranking_feature_recipe(recipe_spec: RankingFeatureRecipeSpec) -> dict:
    recipe = serialize_ranking_source_spec(recipe_spec)
    return {
        "kind": "ranking_feature_recipe",
        "schemaVersion": "v1",
        **recipe,
    }


def extract_ranking_feature_recipe(feature_spec: FeatureSpec) -> RankingFeatureRecipeSpec:
    if feature_spec.ranking_feature_recipe is None:
        raise ValueError("Feature spec must include a ranking feature recipe.")
    return feature_spec.ranking_feature_recipe


def extract_feature_keys(feature_spec: FeatureSpec) -> tuple[str, ...]:
    return tuple(derived_feature.key for derived_feature in feature_spec.derived_features)


def serialize_risk_controls_spec(risk_controls: RiskControlsSpec) -> dict:
    return {
        "maxInvestmentPct": round(risk_controls.max_investment_ratio * 100, 1),
        "maxWeightPct": round(risk_controls.max_weight * 100, 1)
        if risk_controls.max_weight is not None
        else None,
    }


def extract_ranking_score_parameters(
    selection: SelectionSpec,
) -> dict[str, object]:
    score_parameters = dict(selection.ranking_signal.score_parameters)
    extracted: dict[str, object] = {}
    if "windowSpec" in score_parameters:
        extracted["windowSpec"] = dict(score_parameters["windowSpec"])
    if selection.ranking_signal.score_model.kind == "momentum_low_vol":
        extracted["momentumWeight"] = float(score_parameters.get("momentum_weight", 0.7))
        extracted["lowVolWeight"] = float(score_parameters.get("low_vol_weight", 0.3))
        return extracted
    if selection.ranking_signal.score_model.kind == "momentum_macro":
        extracted["momentumWeight"] = float(score_parameters.get("momentum_weight", 0.85))
        extracted["macroWeight"] = float(score_parameters.get("macro_weight", 0.15))
        return extracted
    return extracted


def serialize_asset_ranking_model_parameters(strategy: StrategySpec) -> dict[str, object]:
    score_parameters = dict(strategy.selection.ranking_signal.score_parameters)
    serialized: dict[str, object] = {}

    if "windowSpec" in score_parameters:
        serialized["windowSpec"] = dict(score_parameters["windowSpec"])
    if "momentum_weight" in score_parameters:
        serialized["momentumWeight"] = float(score_parameters["momentum_weight"])
    if "low_vol_weight" in score_parameters:
        serialized["lowVolWeight"] = float(score_parameters["low_vol_weight"])
    if "macro_weight" in score_parameters:
        serialized["macroWeight"] = float(score_parameters["macro_weight"])

    return serialized


def serialize_predictor_use_spec(predictor_use: PredictorUseSpec) -> dict[str, object]:
    return {
        "predictorKey": predictor_use.predictor_key,
        "signalWeight": predictor_use.signal_weight,
        "predictorWeight": predictor_use.predictor_weight,
    }


def serialize_tilt_rule(strategy: StrategySpec) -> dict | None:
    score_parameters = dict(strategy.selection.ranking_signal.score_parameters)
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
            "kind": strategy.selection.ranking_signal.score_model.kind,
            "label": strategy.selection.ranking_signal.score_model.label,
            "parameters": serialize_asset_ranking_model_parameters(strategy),
        }
        if strategy.selection.ranking_signal.score_model.kind != "none"
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
                "predictor": None
                if strategy.predictor_use is None
                else serialize_predictor_use_spec(strategy.predictor_use),
                "featureInputs": list(strategy.selection.ranking_signal.feature_inputs),
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
        "featureInputs": list(selection.ranking_signal.feature_inputs),
        "universePolicy": {
            "key": selection.universe_policy.key,
            "label": selection.universe_policy.label,
        },
        "rankingModel": {
            "kind": selection.ranking_signal.score_model.kind,
            "label": selection.ranking_signal.score_model.label,
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
    }


def build_prediction_target_spec(
    *,
    key: str,
    label: str,
    horizon_spec: dict[str, object],
    baseline: str,
    transform: str,
) -> PredictionTargetSpec:
    if baseline not in {"cross_sectional_mean", "none"}:
        raise ValueError("Unsupported prediction target baseline.")
    if transform not in {"identity"}:
        raise ValueError("Unsupported prediction target transform.")
    return PredictionTargetSpec(
        key=key,
        label=label,
        horizon_spec=tuple(sorted(horizon_spec.items())),
        baseline=baseline,
        transform=transform,
    )


def serialize_prediction_target_spec(
    target_spec: PredictionTargetSpec,
) -> dict:
    return {
        "kind": "prediction_target_spec",
        "schemaVersion": "v1",
        "key": target_spec.key,
        "label": target_spec.label,
        "horizonSpec": {
            key: value for key, value in target_spec.horizon_spec
        },
        "baseline": target_spec.baseline,
        "transform": target_spec.transform,
    }


def build_predicted_quantity_spec(
    *,
    key: str,
    label: str,
    kind: str,
) -> PredictedQuantitySpec:
    if kind not in {"return", "market_regime"}:
        raise ValueError("Unsupported predicted quantity kind.")
    return PredictedQuantitySpec(
        key=key,
        label=label,
        kind=kind,
    )


def serialize_predicted_quantity_spec(
    predicted_quantity_spec: PredictedQuantitySpec,
) -> dict:
    return {
        "kind": "predicted_quantity_spec",
        "schemaVersion": "v1",
        "key": predicted_quantity_spec.key,
        "label": predicted_quantity_spec.label,
        "quantityKind": predicted_quantity_spec.kind,
    }


def apply_prediction_target_transform(
    target_values: pd.Series,
    target_spec: PredictionTargetSpec,
) -> pd.Series:
    if target_spec.transform == "identity":
        return target_values
    raise ValueError("Unsupported prediction target transform.")


FEATURE_INPUT_LABELS = {
    "close": "Price",
    "volume": "Volume",
}

DERIVED_FEATURE_LABELS = {
    "score": "Score",
    "momentum": "Momentum",
    "lowVolRank": "Low Vol Rank",
    "macroRank": "Macro Rank",
    "volumeStrength": "Volume Strength",
}


def build_feature_input_spec(
    *,
    key: str,
    label: str | None = None,
) -> FeatureInputSpec:
    return FeatureInputSpec(
        key=key,
        label=label or FEATURE_INPUT_LABELS.get(key, key),
    )


def build_derived_feature_spec(
    *,
    key: str,
    label: str | None = None,
) -> DerivedFeatureSpec:
    return DerivedFeatureSpec(
        key=key,
        label=label or DERIVED_FEATURE_LABELS.get(key, key),
    )


def build_feature_spec(
    *,
    key: str,
    label: str,
    feature_inputs: tuple[FeatureInputSpec, ...],
    derived_features: tuple[DerivedFeatureSpec, ...],
    ranking_feature_recipe: RankingFeatureRecipeSpec | None = None,
) -> FeatureSpec:
    return FeatureSpec(
        key=key,
        label=label,
        feature_inputs=feature_inputs,
        derived_features=derived_features,
        ranking_feature_recipe=ranking_feature_recipe,
    )


def serialize_feature_spec(
    feature_spec: FeatureSpec,
) -> dict:
    return {
        "kind": "feature_spec",
        "schemaVersion": "v1",
        "key": feature_spec.key,
        "label": feature_spec.label,
        "featureInputs": [
            {
                "key": feature_input.key,
                "label": feature_input.label,
            }
            for feature_input in feature_spec.feature_inputs
        ],
        "derivedFeatures": [
            {
                "key": derived_feature.key,
                "label": derived_feature.label,
            }
            for derived_feature in feature_spec.derived_features
        ],
        "rankingFeatureRecipe": serialize_ranking_feature_recipe(feature_spec.ranking_feature_recipe)
        if feature_spec.ranking_feature_recipe is not None
        else None,
    }


def build_prediction_signal_source_spec(
    *,
    key: str,
    label: str,
    kind: str = "ranking_signal",
    feature_key: str | None = None,
) -> PredictionSignalSourceSpec:
    if kind not in {"ranking_signal", "derived_feature"}:
        raise ValueError("Unsupported prediction signal source kind.")
    if kind == "derived_feature":
        if feature_key not in PREDICTION_FEATURE_NAMES:
            raise ValueError("Unsupported prediction signal source feature key.")
    elif feature_key is not None:
        raise ValueError("feature_key is only supported for derived_feature signal sources.")
    return PredictionSignalSourceSpec(
        key=key,
        label=label,
        kind=kind,
        feature_key=feature_key,
    )


def serialize_prediction_signal_source_spec(
    signal_source_spec: PredictionSignalSourceSpec,
) -> dict:
    return {
        "kind": "prediction_signal_source_spec",
        "schemaVersion": "v1",
        "key": signal_source_spec.key,
        "label": signal_source_spec.label,
        "signalSourceKind": signal_source_spec.kind,
        "featureKey": signal_source_spec.feature_key,
    }


def build_prediction_learner_spec(
    *,
    key: str,
    label: str,
    kind: str,
    ridge_alpha: float | None = None,
) -> PredictionLearnerSpec:
    if kind not in {"linear_regression", "ridge_regression"}:
        raise ValueError("Unsupported prediction learner kind.")
    if kind == "ridge_regression" and ridge_alpha is None:
        raise ValueError("ridge_alpha is required for ridge_regression.")
    return PredictionLearnerSpec(
        key=key,
        label=label,
        kind=kind,
        ridge_alpha=ridge_alpha,
    )


def serialize_prediction_learner_spec(
    learner_spec: PredictionLearnerSpec,
) -> dict:
    return {
        "kind": "prediction_learner_spec",
        "schemaVersion": "v1",
        "key": learner_spec.key,
        "label": learner_spec.label,
        "learnerKind": learner_spec.kind,
        "ridgeAlpha": learner_spec.ridge_alpha,
    }


def build_prediction_combiner_spec(
    *,
    key: str,
    label: str,
    kind: str,
    signal_source_weight: float | None = None,
    learner_weight: float | None = None,
) -> PredictionCombinerSpec:
    if kind not in {"baseline_signal_only", "learner_only", "weighted_blend"}:
        raise ValueError("Unsupported prediction combiner kind.")
    if kind == "weighted_blend":
        if signal_source_weight is None or learner_weight is None:
            raise ValueError(
                "signal_source_weight and learner_weight are required for weighted_blend."
            )
    return PredictionCombinerSpec(
        key=key,
        label=label,
        kind=kind,
        signal_source_weight=signal_source_weight,
        learner_weight=learner_weight,
    )


def serialize_prediction_combiner_spec(
    combiner_spec: PredictionCombinerSpec,
) -> dict:
    return {
        "kind": "prediction_combiner_spec",
        "schemaVersion": "v1",
        "key": combiner_spec.key,
        "label": combiner_spec.label,
        "combinerKind": combiner_spec.kind,
        "signalSourceWeight": combiner_spec.signal_source_weight,
        "learnerWeight": combiner_spec.learner_weight,
    }


def build_prediction_engine_spec(
    *,
    key: str,
    label: str,
    signal_source_spec: PredictionSignalSourceSpec | None,
    learner_spec: PredictionLearnerSpec | None,
    combiner_spec: PredictionCombinerSpec,
) -> PredictionEngineSpec:
    if combiner_spec.kind == "baseline_signal_only":
        if signal_source_spec is None or learner_spec is not None:
            raise ValueError(
                "baseline_signal_only combiner requires a signal source and no learner."
            )
    if combiner_spec.kind == "learner_only":
        if signal_source_spec is not None or learner_spec is None:
            raise ValueError(
                "learner_only combiner requires a learner and no signal source."
            )
    if combiner_spec.kind == "weighted_blend":
        if signal_source_spec is None or learner_spec is None:
            raise ValueError(
                "weighted_blend combiner requires both a signal source and a learner."
            )
    return PredictionEngineSpec(
        key=key,
        label=label,
        signal_source_spec=signal_source_spec,
        learner_spec=learner_spec,
        combiner_spec=combiner_spec,
    )


def build_training_spec(
    *,
    key: str,
    label: str,
    fit_mode: str,
    min_train_samples: int,
) -> TrainingSpec:
    return TrainingSpec(
        key=key,
        label=label,
        fit_mode=fit_mode,
        min_train_samples=min_train_samples,
    )


def build_prediction_output_spec(
    *,
    key: str,
    label: str,
    kind: str,
) -> PredictionOutputSpec:
    if kind not in {
        "score",
        "probability",
        "return_estimate",
        "state_label",
        "state_probability_vector",
    }:
        raise ValueError("Unsupported prediction output kind.")
    return PredictionOutputSpec(
        key=key,
        label=label,
        kind=kind,
    )


def serialize_prediction_output_spec(
    output_spec: PredictionOutputSpec,
) -> dict:
    return {
        "kind": "prediction_output_spec",
        "schemaVersion": "v1",
        "key": output_spec.key,
        "label": output_spec.label,
        "outputKind": output_spec.kind,
    }


def build_predictor_spec(
    *,
    key: str,
    label: str,
    description: str,
    timeframe: TimeframeSpec,
    signal_spec: SignalSpec,
    predicted_quantity_spec: PredictedQuantitySpec,
    target_spec: PredictionTargetSpec,
    feature_spec: FeatureSpec,
    engine_spec: PredictionEngineSpec,
    training_spec: TrainingSpec,
) -> PredictorSpec:
    return PredictorSpec(
        key=key,
        label=label,
        description=description,
        timeframe=timeframe,
        signal_spec=signal_spec,
        predicted_quantity_spec=predicted_quantity_spec,
        target_spec=target_spec,
        feature_spec=feature_spec,
        engine_spec=engine_spec,
        training_spec=training_spec,
    )


def serialize_prediction_engine_spec(
    engine_spec: PredictionEngineSpec,
) -> dict:
    return {
        "kind": "prediction_engine_spec",
        "schemaVersion": "v1",
        "key": engine_spec.key,
        "label": engine_spec.label,
        "signalSourceSpec": (
            serialize_prediction_signal_source_spec(engine_spec.signal_source_spec)
            if engine_spec.signal_source_spec is not None
            else None
        ),
        "learnerSpec": (
            serialize_prediction_learner_spec(engine_spec.learner_spec)
            if engine_spec.learner_spec is not None
            else None
        ),
        "combinerSpec": serialize_prediction_combiner_spec(engine_spec.combiner_spec),
    }


def serialize_training_spec(
    training_spec: TrainingSpec,
) -> dict:
    return {
        "kind": "training_spec",
        "schemaVersion": "v1",
        "key": training_spec.key,
        "label": training_spec.label,
        "fitMode": training_spec.fit_mode,
        "minTrainSamples": training_spec.min_train_samples,
    }
