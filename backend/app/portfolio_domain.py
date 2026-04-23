from __future__ import annotations

from dataclasses import dataclass
import json
import math
import statistics

import numpy as np
import pandas as pd
from app.timeframe_models import (
    DEFAULT_DAILY_TIMEFRAME,
    DEFAULT_MONTHLY_TIMEFRAME,
    DEFAULT_WEEKLY_TIMEFRAME,
    TimeframeSpec,
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
SUPPORTED_REBALANCE_SCHEDULES = {"hold", "every_bar", "month_end", "quarter_end", "year_end"}
DIRECT_EXECUTION_MULTI_SELECTION_STRATEGY_TYPES = {
    "full_universe_momentum_tilt",
    "full_universe_momentum_low_vol_tilt",
    "full_universe_momentum_macro_tilt",
    "momentum_top3",
    "dual_momentum_top3",
    "trailing_momentum_low_vol_universe",
    "positive_momentum_universe",
    "positive_momentum_low_vol_universe",
    "positive_momentum_high_volume_universe",
    "positive_trend_short_reversal",
    "risk_regime_positive_momentum",
}
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
    "positive_trend_short_reversal",
    "risk_regime_positive_momentum",
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
    "positive_trend_short_reversal": "上昇トレンド短期リバーサル",
    "risk_regime_positive_momentum": "リスク局面別上昇資産",
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
class StrategyDataSourceSpec:
    key: str
    label: str
    kind: str
    observation_spec: ObservationSpec


@dataclass(frozen=True)
class StrategyFeatureDefinitionSpec:
    key: str
    label: str
    source_field_keys: tuple[str, ...]
    derived_feature_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class AlignmentPolicySpec:
    key: str
    label: str
    method: str
    parameters: tuple[tuple[str, object], ...] = ()


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
class StrategyExecutionPlanSpec:
    key: str
    label: str
    decision_schedule: str
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
class StrategySignalSpec:
    key: str
    label: str
    description: str
    observation_spec: ObservationSpec
    data_source_spec: StrategyDataSourceSpec | None
    feature_definition_spec: StrategyFeatureDefinitionSpec | None
    alignment_policy: AlignmentPolicySpec | None
    data_timeframe: TimeframeSpec
    signal_timeframe: TimeframeSpec
    source_kind: str
    weight: float
    signal_parameters: tuple[tuple[str, object], ...]
    predictor_key: str | None = None


@dataclass(frozen=True)
class EvaluatorStrategySpec:
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
    signal_execution_contexts: tuple[dict[str, object], ...] = ()
    predictor_signal_execution_context: dict[str, object] | None = None
    decision_schedule: str | None = None
    execution_mode: str | None = None
    extensions: tuple[tuple[str, str], ...] = ()

    @property
    def key(self) -> str:
        return self.strategy_id


@dataclass(frozen=True)
class StrategyDefinition:
    strategy_id: str
    version: str
    label: str
    hypothesis: str | None
    description: str
    investment_universe: InvestmentUniverseSpec
    signals: tuple[StrategySignalSpec, ...]
    portfolio_model: PortfolioModelSpec
    execution_plan: StrategyExecutionPlanSpec
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
    if len(normalized_tickers) < 1:
        raise ValueError("Investment universe must contain at least one ticker.")
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
        "positive_trend_short_reversal": "長期上昇中で直近短期に売られたETFを候補にする",
        "risk_regime_positive_momentum": "リスク資産が弱い局面では防御資産へ絞り、それ以外は上昇ETFを候補にする",
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
        "positive_trend_short_reversal": ("close",),
        "risk_regime_positive_momentum": ("close",),
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
        "positive_trend_short_reversal": UniversePolicySpec(
            "positive_assets_only",
            UNIVERSE_POLICY_LABELS["positive_assets_only"],
        ),
        "risk_regime_positive_momentum": UniversePolicySpec(
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
        "positive_trend_short_reversal": RankingModelSpec(
            "trailing_momentum",
            SCORE_MODEL_LABELS["trailing_momentum"],
        ),
        "risk_regime_positive_momentum": RankingModelSpec(
            "trailing_momentum",
            SCORE_MODEL_LABELS["trailing_momentum"],
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
        "positive_trend_short_reversal": {
            "trendWindowSpec": {"unit": "days", "value": 60},
            "reversalWindowSpec": {"unit": "days", "value": 5},
            "assetCount": 5,
        },
        "risk_regime_positive_momentum": {
            "windowSpec": {"unit": "bars", "value": 252},
            "riskWindowSpec": {"unit": "days", "value": 60},
            "riskProxyAssets": ("SPY", "QQQ"),
            "defensiveAssetClasses": ("bond_etf", "commodity_etf", "currency_etf"),
        },
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
        "positive_trend_short_reversal": (
            FilterRuleSpec("positive_return", FILTER_RULE_LABELS["positive_return"]),
        ),
        "risk_regime_positive_momentum": (
            FilterRuleSpec("positive_return", FILTER_RULE_LABELS["positive_return"]),
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
        "positive_trend_short_reversal": FallbackRuleSpec("cash_on_empty", FALLBACK_RULE_LABELS["cash_on_empty"]),
        "risk_regime_positive_momentum": FallbackRuleSpec("cash_on_empty", FALLBACK_RULE_LABELS["cash_on_empty"]),
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
        "mean_risk_utility": "校正済みforecastがない場合は等ウェイトfallbackで配分する",
        "mean_risk_utility_conservative": "校正済みforecastがない場合は保守的に等ウェイトfallbackで配分する",
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


def build_strategy_execution_plan_spec(
    *,
    key: str,
    label: str,
    decision_schedule: str,
    rebalance_schedule: str,
) -> StrategyExecutionPlanSpec:
    if decision_schedule not in SUPPORTED_REBALANCE_SCHEDULES:
        raise ValueError("Unsupported decision schedule.")
    if rebalance_schedule not in SUPPORTED_REBALANCE_SCHEDULES:
        raise ValueError("Unsupported execution policy rebalance schedule.")
    return StrategyExecutionPlanSpec(
        key=key,
        label=label,
        decision_schedule=decision_schedule,
        rebalance_schedule=rebalance_schedule,
    )


def serialize_strategy_execution_plan_spec(execution_plan: StrategyExecutionPlanSpec) -> dict:
    return {
        "key": execution_plan.key,
        "label": execution_plan.label,
        "decisionSchedule": execution_plan.decision_schedule,
        "rebalanceSchedule": execution_plan.rebalance_schedule,
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


def freeze_strategy_parameter_value(value: object) -> object:
    if isinstance(value, dict):
        return tuple(
            (str(key), freeze_strategy_parameter_value(nested_value))
            for key, nested_value in sorted(value.items())
        )
    if isinstance(value, list):
        return tuple(freeze_strategy_parameter_value(item) for item in value)
    return value


def build_strategy_data_source_spec(
    *,
    key: str,
    label: str,
    kind: str,
    observation_spec: ObservationSpec,
) -> StrategyDataSourceSpec:
    if not key.strip():
        raise ValueError("Strategy data source key is required.")
    if not kind.strip():
        raise ValueError("Strategy data source kind is required.")
    return StrategyDataSourceSpec(
        key=key,
        label=label,
        kind=kind,
        observation_spec=observation_spec,
    )



def build_strategy_feature_definition_spec(
    *,
    key: str,
    label: str,
    source_field_keys: list[str] | tuple[str, ...],
    derived_feature_keys: list[str] | tuple[str, ...] = (),
) -> StrategyFeatureDefinitionSpec:
    if not key.strip():
        raise ValueError("Strategy feature definition key is required.")
    return StrategyFeatureDefinitionSpec(
        key=key,
        label=label,
        source_field_keys=tuple(str(field_key) for field_key in source_field_keys),
        derived_feature_keys=tuple(str(feature_key) for feature_key in derived_feature_keys),
    )



def build_alignment_policy_spec(
    *,
    key: str,
    label: str,
    method: str,
    parameters: dict[str, object] | None = None,
) -> AlignmentPolicySpec:
    if not key.strip():
        raise ValueError("Alignment policy key is required.")
    if not method.strip():
        raise ValueError("Alignment policy method is required.")
    return AlignmentPolicySpec(
        key=key,
        label=label,
        method=method,
        parameters=tuple(
            (str(parameter_key), freeze_strategy_parameter_value(parameter_value))
            for parameter_key, parameter_value in sorted((parameters or {}).items())
        ),
    )


def build_strategy_signal_spec(
    *,
    key: str,
    label: str,
    description: str,
    observation_spec: ObservationSpec,
    data_timeframe: TimeframeSpec,
    signal_timeframe: TimeframeSpec | None = None,
    source_kind: str,
    data_source_spec: StrategyDataSourceSpec | None = None,
    feature_definition_spec: StrategyFeatureDefinitionSpec | None = None,
    alignment_policy: AlignmentPolicySpec | None = None,
    weight: float = 1.0,
    signal_parameters: dict[str, object] | None = None,
    predictor_key: str | None = None,
) -> StrategySignalSpec:
    if not key.strip():
        raise ValueError("Strategy signal key is required.")
    if not source_kind.strip():
        raise ValueError("Strategy signal source kind is required.")
    if weight <= 0:
        raise ValueError("Strategy signal weight must be positive.")
    resolved_signal_timeframe = signal_timeframe or data_timeframe
    resolved_data_source_spec = data_source_spec or build_strategy_data_source_spec(
        key=f"data_source__{key}",
        label=f"{label} data source",
        kind="market_observation",
        observation_spec=observation_spec,
    )
    resolved_feature_definition_spec = feature_definition_spec or build_strategy_feature_definition_spec(
        key=f"feature_definition__{key}",
        label=f"{label} features",
        source_field_keys=observation_spec.fields,
    )
    resolved_alignment_policy = alignment_policy
    if resolved_alignment_policy is None and resolved_signal_timeframe.key != data_timeframe.key:
        resolved_alignment_policy = build_alignment_policy_spec(
            key=f"alignment_policy__{key}",
            label=f"{label} alignment",
            method="asof_last",
            parameters={
                "fromTimeframe": data_timeframe.key,
                "toTimeframe": resolved_signal_timeframe.key,
            },
        )
    return StrategySignalSpec(
        key=key,
        label=label,
        description=description,
        observation_spec=observation_spec,
        data_source_spec=resolved_data_source_spec,
        feature_definition_spec=resolved_feature_definition_spec,
        alignment_policy=resolved_alignment_policy,
        data_timeframe=data_timeframe,
        signal_timeframe=resolved_signal_timeframe,
        source_kind=source_kind,
        weight=float(weight),
        signal_parameters=tuple(
            (str(parameter_key), freeze_strategy_parameter_value(parameter_value))
            for parameter_key, parameter_value in sorted((signal_parameters or {}).items())
        ),
        predictor_key=predictor_key,
    )


def serialize_strategy_signal_spec(signal: StrategySignalSpec) -> dict:
    return {
        "key": signal.key,
        "label": signal.label,
        "description": signal.description,
        "sourceKind": signal.source_kind,
        "weight": signal.weight,
        "predictorKey": signal.predictor_key,
        "signalParameters": {
            key: thaw_strategy_parameter_value(value)
            for key, value in signal.signal_parameters
        },
        "observationSpec": serialize_observation_spec(signal.observation_spec),
        "dataSource": None if signal.data_source_spec is None else {
            "key": signal.data_source_spec.key,
            "label": signal.data_source_spec.label,
            "kind": signal.data_source_spec.kind,
            "observationSpec": serialize_observation_spec(signal.data_source_spec.observation_spec),
        },
        "featureDefinition": None if signal.feature_definition_spec is None else {
            "key": signal.feature_definition_spec.key,
            "label": signal.feature_definition_spec.label,
            "sourceFieldKeys": list(signal.feature_definition_spec.source_field_keys),
            "derivedFeatureKeys": list(signal.feature_definition_spec.derived_feature_keys),
        },
        "alignmentPolicy": None if signal.alignment_policy is None else {
            "key": signal.alignment_policy.key,
            "label": signal.alignment_policy.label,
            "method": signal.alignment_policy.method,
            "parameters": {key: value for key, value in signal.alignment_policy.parameters},
        },
        "dataTimeframe": {
            "key": signal.data_timeframe.key,
            "label": signal.data_timeframe.label,
            "yfinanceInterval": signal.data_timeframe.yfinance_interval,
            "barsPerYear": signal.data_timeframe.bars_per_year,
            "barSeconds": signal.data_timeframe.bar_seconds,
        },
        "signalTimeframe": {
            "key": signal.signal_timeframe.key,
            "label": signal.signal_timeframe.label,
            "yfinanceInterval": signal.signal_timeframe.yfinance_interval,
            "barsPerYear": signal.signal_timeframe.bars_per_year,
            "barSeconds": signal.signal_timeframe.bar_seconds,
        },
    }


def build_strategy_definition(
    *,
    investment_universe: InvestmentUniverseSpec,
    signals: list[StrategySignalSpec] | tuple[StrategySignalSpec, ...],
    portfolio_model: PortfolioModelSpec,
    execution_plan: StrategyExecutionPlanSpec,
    risk_controls: RiskControlsSpec,
    strategy_id: str | None = None,
    version: str = "v1",
    hypothesis: str | None = None,
    extensions: dict[str, str] | None = None,
    key: str | None = None,
    label: str | None = None,
    description: str | None = None,
) -> StrategyDefinition:
    resolved_strategy_id = strategy_id or key or portfolio_model.key
    if strategy_id is not None and key is not None and strategy_id != key:
        raise ValueError("strategy_id and key must match when both are provided.")
    resolved_signals = tuple(signals)
    if len(resolved_signals) < 1:
        raise ValueError("Strategy definition must contain at least one signal.")
    universe_tickers = set(investment_universe.tickers)
    for signal in resolved_signals:
        unknown_tickers = set(signal.observation_spec.tickers) - universe_tickers
        if unknown_tickers:
            raise ValueError("Strategy signal observation tickers must be contained in the investment universe.")
    strategy_label = label or " + ".join(signal.label for signal in resolved_signals)
    strategy_description = description or " / ".join(signal.description for signal in resolved_signals)
    return StrategyDefinition(
        strategy_id=resolved_strategy_id,
        version=version,
        label=strategy_label,
        hypothesis=hypothesis,
        description=strategy_description,
        investment_universe=investment_universe,
        signals=resolved_signals,
        portfolio_model=portfolio_model,
        execution_plan=execution_plan,
        risk_controls=risk_controls,
        extensions=tuple(sorted((extensions or {}).items())),
    )


def serialize_strategy_definition(strategy: StrategyDefinition) -> dict:
    direct_execution_issues = get_direct_execution_strategy_definition_compatibility_issues(strategy)
    return {
        "kind": "strategy_definition",
        "schemaVersion": "v1",
        "strategyId": strategy.strategy_id,
        "version": strategy.version,
        "label": strategy.label,
        "hypothesis": strategy.hypothesis,
        "description": strategy.description,
        "components": {
            "core": {
                "investmentUniverse": {
                    "key": strategy.investment_universe.key,
                    "label": strategy.investment_universe.label,
                    "assetCount": len(strategy.investment_universe.tickers),
                    "tickers": list(strategy.investment_universe.tickers),
                },
                "portfolioModel": serialize_portfolio_model_spec(strategy.portfolio_model),
                "executionPlan": serialize_strategy_execution_plan_spec(strategy.execution_plan),
            },
            "optional": {
                "signals": [serialize_strategy_signal_spec(signal) for signal in strategy.signals],
                "riskControls": serialize_risk_controls_spec(strategy.risk_controls),
            },
        },
        "executionSupport": {
            "directExecutionCompatible": not direct_execution_issues,
            "directExecutionIssues": direct_execution_issues,
        },
        "extensions": {key: value for key, value in strategy.extensions},
    }


def resolve_timeframe_spec_from_key(timeframe_key: str, fallback: TimeframeSpec) -> TimeframeSpec:
    if timeframe_key == fallback.key:
        return fallback
    if timeframe_key == DEFAULT_DAILY_TIMEFRAME.key:
        return DEFAULT_DAILY_TIMEFRAME
    if timeframe_key == DEFAULT_WEEKLY_TIMEFRAME.key:
        return DEFAULT_WEEKLY_TIMEFRAME
    if timeframe_key == DEFAULT_MONTHLY_TIMEFRAME.key:
        return DEFAULT_MONTHLY_TIMEFRAME
    raise ValueError(f"Unsupported timeframe key: {timeframe_key}")


def build_alignment_policy_spec_from_payload(
    alignment_policy_payload: dict[str, object] | None,
) -> AlignmentPolicySpec | None:
    if alignment_policy_payload is None:
        return None
    parameters = alignment_policy_payload.get("parameters")
    return build_alignment_policy_spec(
        key=str(alignment_policy_payload.get("key") or alignment_policy_payload.get("method") or "alignment_policy"),
        label=str(alignment_policy_payload.get("label") or alignment_policy_payload.get("method") or "Alignment policy"),
        method=str(alignment_policy_payload.get("method") or "asof_last"),
        parameters=parameters if isinstance(parameters, dict) else None,
    )


def build_strategy_definition_from_evaluator_strategy_spec(strategy: EvaluatorStrategySpec) -> StrategyDefinition:
    selection_contexts = [dict(context) for context in strategy.signal_execution_contexts]
    predictor_context = (
        None
        if strategy.predictor_signal_execution_context is None
        else dict(strategy.predictor_signal_execution_context)
    )

    if not selection_contexts:
        selection_contexts = [{
            "signalKey": f"signal__{strategy.strategy_id}__selection",
            "signalLabel": strategy.selection.label,
            "description": strategy.selection.description,
            "sourceKind": "selection_signal",
            "selectionKey": strategy.selection.key,
            "strategyType": strategy.selection.strategy_type,
            "scoreParameters": thaw_strategy_parameter_value(dict(strategy.selection.ranking_signal.score_parameters)),
            "dataTimeframe": strategy.timeframe.key,
            "signalTimeframe": strategy.timeframe.key,
            "alignmentPolicy": None,
            "weight": 1.0 if strategy.predictor_use is None else strategy.predictor_use.signal_weight,
        }]
    if predictor_context is None and strategy.predictor_use is not None:
        predictor_context = {
            "signalKey": f"signal__{strategy.strategy_id}__predictor",
            "signalLabel": f"{strategy.label} predictor overlay",
            "description": "Legacy predictor overlay translated from predictor_use.",
            "sourceKind": "predictor_overlay",
            "predictorKey": strategy.predictor_use.predictor_key,
            "signalWeight": strategy.predictor_use.signal_weight,
            "predictorWeight": strategy.predictor_use.predictor_weight,
            "dataTimeframe": strategy.timeframe.key,
            "signalTimeframe": strategy.timeframe.key,
            "alignmentPolicy": None,
            "weight": strategy.predictor_use.predictor_weight,
        }

    signals = []
    selection_observation_spec = build_observation_spec(
        key=f"observation__{strategy.strategy_id}__selection",
        label=f"{strategy.selection.label} observation",
        tickers=strategy.investment_universe.tickers,
        fields=strategy.selection.ranking_signal.feature_inputs,
    )
    for index, selection_context in enumerate(selection_contexts):
        selection_parameters = {
            "selectionKey": str(selection_context.get("selectionKey") or strategy.selection.key),
            "strategyType": str(selection_context.get("strategyType") or strategy.selection.strategy_type),
            "scoreModelKind": strategy.selection.ranking_signal.score_model.kind,
            "scoreParameters": thaw_strategy_parameter_value(
                selection_context.get("scoreParameters")
                if selection_context.get("scoreParameters") is not None
                else dict(strategy.selection.ranking_signal.score_parameters)
            ),
            "featureInputs": list(strategy.selection.ranking_signal.feature_inputs),
            "universePolicyKey": strategy.selection.universe_policy.key,
            "filterRuleKeys": [filter_rule.key for filter_rule in strategy.selection.filter_rules],
            "fallbackRuleKey": strategy.selection.fallback_rule.key,
        }
        signals.append(
            build_strategy_signal_spec(
                key=str(selection_context.get("signalKey") or f"signal__{strategy.strategy_id}__selection__{index}"),
                label=str(selection_context.get("signalLabel") or strategy.selection.label),
                description=str(selection_context.get("description") or strategy.selection.description),
                observation_spec=selection_observation_spec,
                data_timeframe=resolve_timeframe_spec_from_key(
                    str(selection_context.get("dataTimeframe") or strategy.timeframe.key),
                    strategy.timeframe,
                ),
                signal_timeframe=resolve_timeframe_spec_from_key(
                    str(selection_context.get("signalTimeframe") or strategy.timeframe.key),
                    strategy.timeframe,
                ),
                source_kind="selection_signal",
                weight=float(selection_context.get("weight", 1.0 if strategy.predictor_use is None else strategy.predictor_use.signal_weight)),
                alignment_policy=build_alignment_policy_spec_from_payload(selection_context.get("alignmentPolicy")),
                signal_parameters=selection_parameters,
            )
        )

    if strategy.predictor_use is not None or predictor_context is not None:
        predictor_payload = predictor_context or {}
        signals.append(
            build_strategy_signal_spec(
                key=str(predictor_payload.get("signalKey") or f"signal__{strategy.strategy_id}__predictor"),
                label=str(predictor_payload.get("signalLabel") or f"{strategy.label} predictor overlay"),
                description=str(predictor_payload.get("description") or "Legacy predictor overlay translated from predictor_use."),
                observation_spec=build_observation_spec(
                    key=f"observation__{strategy.strategy_id}__predictor",
                    label=f"{strategy.label} predictor observation",
                    tickers=strategy.investment_universe.tickers,
                    fields=strategy.selection.ranking_signal.feature_inputs,
                ),
                data_timeframe=resolve_timeframe_spec_from_key(
                    str(predictor_payload.get("dataTimeframe") or strategy.timeframe.key),
                    strategy.timeframe,
                ),
                signal_timeframe=resolve_timeframe_spec_from_key(
                    str(predictor_payload.get("signalTimeframe") or strategy.timeframe.key),
                    strategy.timeframe,
                ),
                source_kind="predictor_overlay",
                weight=float(predictor_payload.get("weight", strategy.predictor_use.predictor_weight if strategy.predictor_use is not None else 1.0)),
                alignment_policy=build_alignment_policy_spec_from_payload(predictor_payload.get("alignmentPolicy")),
                signal_parameters=serialize_predictor_use_spec(
                    strategy.predictor_use
                    or build_predictor_use_spec(
                        predictor_key=str(predictor_payload.get("predictorKey")),
                        signal_weight=float(predictor_payload.get("signalWeight", 1.0)),
                        predictor_weight=float(predictor_payload.get("predictorWeight", predictor_payload.get("weight", 1.0))),
                    )
                ),
                predictor_key=(
                    None
                    if predictor_payload.get("predictorKey") is None
                    else str(predictor_payload.get("predictorKey"))
                ) or (None if strategy.predictor_use is None else strategy.predictor_use.predictor_key),
            )
        )
    return build_strategy_definition(
        strategy_id=strategy.strategy_id,
        version=strategy.version,
        hypothesis=strategy.hypothesis,
        label=strategy.label,
        description=strategy.description,
        investment_universe=strategy.investment_universe,
        signals=signals,
        portfolio_model=strategy.portfolio_model,
        execution_plan=build_strategy_execution_plan_spec(
            key=f"execution_plan__{strategy.execution_policy.key}",
            label=strategy.execution_policy.label,
            decision_schedule=strategy.decision_schedule or strategy.execution_policy.rebalance_schedule,
            rebalance_schedule=strategy.execution_policy.rebalance_schedule,
        ),
        risk_controls=strategy.risk_controls,
        extensions=dict(strategy.extensions),
    )


def serialize_alignment_policy_spec(
    alignment_policy: AlignmentPolicySpec | None,
) -> dict[str, object] | None:
    if alignment_policy is None:
        return None
    return {
        "key": alignment_policy.key,
        "label": alignment_policy.label,
        "method": alignment_policy.method,
        "parameters": {key: value for key, value in alignment_policy.parameters},
    }


def build_strategy_signal_execution_contexts_from_definition(
    strategy: StrategyDefinition,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    selection_contexts: list[dict[str, object]] = []
    predictor_context: dict[str, object] | None = None
    for signal in strategy.signals:
        signal_parameters = thaw_strategy_parameter_value(dict(signal.signal_parameters))
        base_context = {
            "signalKey": signal.key,
            "signalLabel": signal.label,
            "description": signal.description,
            "dataTimeframe": signal.data_timeframe.key,
            "signalTimeframe": signal.signal_timeframe.key,
            "alignmentPolicy": serialize_alignment_policy_spec(signal.alignment_policy),
            "weight": float(signal.weight),
        }
        if signal.source_kind == "selection_signal":
            selection_contexts.append(
                {
                    **base_context,
                    "sourceKind": signal.source_kind,
                    "selectionKey": str(signal_parameters.get("selectionKey") or signal_parameters.get("strategyType")),
                    "strategyType": str(signal_parameters.get("strategyType")),
                    "scoreParameters": signal_parameters.get("scoreParameters"),
                }
            )
        elif signal.source_kind == "predictor_overlay":
            predictor_context = {
                **base_context,
                "sourceKind": signal.source_kind,
                "predictorKey": str(signal.predictor_key or signal_parameters.get("predictorKey")),
                "signalWeight": float(signal_parameters.get("signalWeight", 1.0)),
                "predictorWeight": float(signal_parameters.get("predictorWeight", signal.weight)),
            }
    return selection_contexts, predictor_context


def thaw_strategy_parameter_value(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: thaw_strategy_parameter_value(nested_value)
            for key, nested_value in value.items()
        }
    if isinstance(value, tuple):
        if all(
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], str)
            for item in value
        ):
            return {
                key: thaw_strategy_parameter_value(nested_value)
                for key, nested_value in value
            }
        return [thaw_strategy_parameter_value(item) for item in value]
    return value


def get_direct_execution_strategy_definition_compatibility_issues(
    strategy: StrategyDefinition,
) -> list[str]:
    issues: list[str] = []
    selection_signals = [
        signal for signal in strategy.signals if signal.source_kind == "selection_signal"
    ]
    predictor_signals = [
        signal for signal in strategy.signals if signal.source_kind == "predictor_overlay"
    ]
    if not selection_signals:
        issues.append("direct execution requires at least one selection_signal")
    if len(predictor_signals) > 1:
        issues.append("direct execution supports at most one predictor_overlay")
    if strategy.execution_plan.decision_schedule not in {
        strategy.execution_plan.rebalance_schedule,
        "every_bar",
    }:
        issues.append("direct execution requires decision_schedule to match rebalance_schedule or be every_bar")
    if not selection_signals:
        return issues

    selection_signal = selection_signals[0]
    selection_parameters = thaw_strategy_parameter_value(dict(selection_signal.signal_parameters))
    strategy_type = str(selection_parameters.get("strategyType"))
    if selection_signal.signal_timeframe.bar_seconds < selection_signal.data_timeframe.bar_seconds:
        issues.append("direct execution requires signal_timeframe to be coarser than or equal to data_timeframe")

    score_parameters = selection_parameters.get("scoreParameters")
    if not isinstance(score_parameters, dict):
        issues.append("direct execution requires scoreParameters in the selection signal")

    if selection_signal.alignment_policy is not None and selection_signal.alignment_policy.method not in {
        "asof_last",
        "end_of_period",
        "calendar_resample",
    }:
        issues.append("direct execution only supports asof_last, end_of_period, or calendar_resample alignment")

    if len(selection_signals) > 1:
        if strategy_type not in DIRECT_EXECUTION_MULTI_SELECTION_STRATEGY_TYPES:
            issues.append("direct execution only supports multi-selection blending for full_universe scoring strategies")
        for additional_signal in selection_signals[1:]:
            additional_parameters = thaw_strategy_parameter_value(dict(additional_signal.signal_parameters))
            additional_strategy_type = str(additional_parameters.get("strategyType"))
            if additional_strategy_type != strategy_type:
                issues.append("direct execution requires blended selection signals to share the same strategyType")
            additional_score_parameters = additional_parameters.get("scoreParameters")
            if not isinstance(additional_score_parameters, dict):
                issues.append("direct execution requires scoreParameters in all blended selection signals")
            if additional_signal.data_timeframe.key != selection_signal.data_timeframe.key:
                issues.append("direct execution requires blended selection signals to share the same data timeframe")
            if additional_signal.signal_timeframe.key != selection_signal.signal_timeframe.key:
                issues.append("direct execution requires blended selection signals to share the same signal timeframe")
            if additional_signal.alignment_policy is not None and additional_signal.alignment_policy.method not in {
                "asof_last",
                "end_of_period",
                "calendar_resample",
            }:
                issues.append("direct execution only supports asof_last, end_of_period, or calendar_resample alignment")

    if predictor_signals:
        predictor_signal = predictor_signals[0]
        if predictor_signal.data_timeframe.key != selection_signal.data_timeframe.key:
            issues.append("direct execution requires predictor and selection signals to share the same data timeframe")
        if predictor_signal.signal_timeframe.key != selection_signal.signal_timeframe.key:
            issues.append("direct execution requires predictor and selection signals to share the same signal timeframe")
        predictor_parameters = thaw_strategy_parameter_value(dict(predictor_signal.signal_parameters))
        predictor_key = predictor_signal.predictor_key or predictor_parameters.get("predictorKey")
        if predictor_key is None:
            issues.append("direct execution requires predictorKey for predictor overlays")

    return issues


def is_direct_execution_compatible_strategy_definition(
    strategy: StrategyDefinition,
) -> bool:
    return not get_direct_execution_strategy_definition_compatibility_issues(strategy)


def _build_direct_execution_evaluator_strategy_spec_from_definition(
    strategy: StrategyDefinition,
) -> EvaluatorStrategySpec:
    issues = get_direct_execution_strategy_definition_compatibility_issues(strategy)
    if issues:
        raise ValueError(
            "Direct execution incompatibilities: " + "; ".join(issues)
        )

    selection_signals = [signal for signal in strategy.signals if signal.source_kind == "selection_signal"]
    selection_signal = selection_signals[0]
    selection_contexts, predictor_context = build_strategy_signal_execution_contexts_from_definition(strategy)
    primary_selection_context = selection_contexts[0]
    strategy_type = str(primary_selection_context["strategyType"])
    score_parameters = primary_selection_context["scoreParameters"]

    predictor_use = None
    if predictor_context is not None:
        predictor_use = build_predictor_use_spec(
            predictor_key=str(predictor_context["predictorKey"]),
            signal_weight=float(predictor_context["signalWeight"]),
            predictor_weight=float(predictor_context["predictorWeight"]),
        )

    return build_evaluator_strategy_spec(
        strategy_id=strategy.strategy_id,
        version=strategy.version,
        hypothesis=strategy.hypothesis,
        label=strategy.label,
        description=strategy.description,
        timeframe=selection_signal.signal_timeframe,
        investment_universe=strategy.investment_universe,
        selection=build_selection_spec(
            strategy_type,
            key=str(primary_selection_context["selectionKey"]),
            label=selection_signal.label,
            description=selection_signal.description,
            score_parameters=score_parameters,
        ),
        portfolio_model=strategy.portfolio_model,
        execution_policy=build_execution_policy_spec(
            key=strategy.execution_plan.key,
            label=strategy.execution_plan.label,
            entry="train_once_then_periodic_rebalance",
            rebalance_schedule=strategy.execution_plan.rebalance_schedule,
        ),
        risk_controls=strategy.risk_controls,
        predictor_use=predictor_use,
        signal_execution_contexts=selection_contexts,
        predictor_signal_execution_context=predictor_context,
        decision_schedule=strategy.execution_plan.decision_schedule,
        execution_mode="direct_signal_timeframe",
        extensions=dict(strategy.extensions),
    )


def build_executable_evaluator_strategy_spec_from_definition(strategy: StrategyDefinition) -> EvaluatorStrategySpec:
    return _build_direct_execution_evaluator_strategy_spec_from_definition(strategy)


def build_evaluator_strategy_spec(
    *,
    timeframe: TimeframeSpec | None = None,
    investment_universe: InvestmentUniverseSpec,
    selection: SelectionSpec,
    portfolio_model: PortfolioModelSpec,
    execution_policy: ExecutionPolicySpec | None = None,
    risk_controls: RiskControlsSpec,
    predictor_use: PredictorUseSpec | None = None,
    signal_execution_contexts: list[dict[str, object]] | tuple[dict[str, object], ...] | None = None,
    predictor_signal_execution_context: dict[str, object] | None = None,
    decision_schedule: str | None = None,
    execution_mode: str | None = None,
    strategy_id: str | None = None,
    version: str = "v1",
    hypothesis: str | None = None,
    extensions: dict[str, str] | None = None,
    key: str | None = None,
    label: str | None = None,
    description: str | None = None,
) -> EvaluatorStrategySpec:
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
    return EvaluatorStrategySpec(
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
        signal_execution_contexts=tuple(signal_execution_contexts or ()),
        predictor_signal_execution_context=predictor_signal_execution_context,
        decision_schedule=decision_schedule,
        execution_mode=execution_mode,
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


def serialize_asset_ranking_model_parameters(strategy: EvaluatorStrategySpec) -> dict[str, object]:
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


def serialize_strategy_execution_context_payload(context: dict[str, object]) -> dict[str, object]:
    payload = {
        "sourceKind": str(context.get("sourceKind", context.get("source_kind", ""))),
        "dataTimeframe": str(context.get("dataTimeframe", context.get("data_timeframe_key", ""))),
        "signalTimeframe": str(context.get("signalTimeframe", context.get("signal_timeframe_key", ""))),
        "alignmentPolicy": context.get("alignmentPolicy", context.get("alignment_policy")),
    }
    if "selectionKey" in context:
        payload["selectionKey"] = str(context["selectionKey"])
    if "strategyType" in context:
        payload["strategyType"] = str(context["strategyType"])
    if "scoreParameters" in context:
        payload["scoreParameters"] = context["scoreParameters"]
    if "weight" in context:
        payload["weight"] = float(context["weight"])
    if "predictorKey" in context:
        payload["predictorKey"] = str(context["predictorKey"])
    if "signalWeight" in context:
        payload["signalWeight"] = float(context["signalWeight"])
    if "predictorWeight" in context:
        payload["predictorWeight"] = float(context["predictorWeight"])
    return payload


def serialize_predictor_use_spec(predictor_use: PredictorUseSpec) -> dict[str, object]:
    return {
        "predictorKey": predictor_use.predictor_key,
        "signalWeight": predictor_use.signal_weight,
        "predictorWeight": predictor_use.predictor_weight,
    }


def serialize_tilt_rule(strategy: EvaluatorStrategySpec) -> dict | None:
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


def serialize_evaluator_strategy_spec(strategy: EvaluatorStrategySpec) -> dict:
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
        "kind": "evaluator_strategy_spec",
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
                "decisionSchedule": strategy.decision_schedule or strategy.execution_policy.rebalance_schedule,
                "executionMode": strategy.execution_mode,
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
                "decisionPolicy": {
                    "key": dict(strategy.extensions).get("decision_policy", "direct_score_to_weight"),
                },
                "signalExecutionContexts": {
                    "selectionSignals": [
                        serialize_strategy_execution_context_payload(context)
                        for context in strategy.signal_execution_contexts
                    ],
                    "predictorSignal": None
                    if strategy.predictor_signal_execution_context is None
                    else serialize_strategy_execution_context_payload(
                        strategy.predictor_signal_execution_context
                    ),
                },
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
