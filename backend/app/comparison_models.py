from __future__ import annotations

from dataclasses import dataclass, field

from app.portfolio import PortfolioState, StrategySpec


@dataclass
class TimeframeSpec:
    key: str
    label: str
    yfinance_interval: str
    bar_seconds: int
    bars_per_year: float


@dataclass
class MarketDataSpec:
    period: str
    timeframe: TimeframeSpec
    fields: list[str] = field(default_factory=lambda: ["close", "volume"])
    sanity_periods: list[str] = field(default_factory=list)


@dataclass
class CostModelSpec:
    kind: str
    parameters: dict[str, float]
    per_asset_overrides: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class ExecutionAssumptionsSpec:
    kind: str
    label: str
    parameters: dict[str, str | float | bool]
    cost_model: CostModelSpec


def build_cost_model_spec(
    *,
    kind: str = "flat_cost",
    commission_pct: float,
    slippage_pct: float = 0.0,
    per_asset_overrides: dict[str, dict[str, float]] | None = None,
) -> CostModelSpec:
    return CostModelSpec(
        kind=kind,
        parameters={
            "commissionPct": commission_pct,
            "slippagePct": slippage_pct,
        },
        per_asset_overrides=per_asset_overrides or {},
    )


def build_asset_specific_linear_cost_model_spec(
    *,
    default_commission_pct: float,
    default_slippage_pct: float = 0.0,
    per_asset_overrides: dict[str, dict[str, float]] | None = None,
) -> CostModelSpec:
    return build_cost_model_spec(
        kind="asset_specific_linear_cost",
        commission_pct=default_commission_pct,
        slippage_pct=default_slippage_pct,
        per_asset_overrides=per_asset_overrides,
    )


def build_asset_specific_adv_cost_model_spec(
    *,
    default_commission_pct: float,
    default_slippage_pct: float = 0.0,
    default_impact_coefficient_pct: float = 0.0,
    adv_window_days: int = 20,
    min_adv_notional: float = 1_000_000.0,
    per_asset_overrides: dict[str, dict[str, float]] | None = None,
) -> CostModelSpec:
    return CostModelSpec(
        kind="asset_specific_adv_cost",
        parameters={
            "commissionPct": float(default_commission_pct),
            "slippagePct": float(default_slippage_pct),
            "impactCoefficientPct": float(default_impact_coefficient_pct),
            "advWindowDays": float(adv_window_days),
            "minAdvNotional": float(min_adv_notional),
        },
        per_asset_overrides=per_asset_overrides or {},
    )


def build_execution_assumptions_spec(
    *,
    kind: str = "close_execution_assumptions",
    label: str,
    parameters: dict[str, str | float | bool] | None = None,
    cost_model: CostModelSpec,
) -> ExecutionAssumptionsSpec:
    return ExecutionAssumptionsSpec(
        kind=kind,
        label=label,
        parameters=parameters or {},
        cost_model=cost_model,
    )


@dataclass
class EvaluationSettings:
    split_ratio: float


@dataclass
class EvaluationSpec:
    evaluation_settings: EvaluationSettings


@dataclass
class RunInputSpec:
    portfolio_state: PortfolioState
    capital_base: float


@dataclass
class SelectionPolicy:
    primary_metric: str
    secondary_metric: str
    tertiary_metric: str


@dataclass
class ConditionVariant:
    key: str
    label: str
    commission_pct: float
    max_investment_ratio: float
    max_weight: float | None = None


@dataclass
class ComparisonSpec:
    comparison_id: str
    title: str
    question: str
    market_data: MarketDataSpec
    run_input: RunInputSpec
    execution_assumptions: ExecutionAssumptionsSpec
    evaluation: EvaluationSpec
    selection_policy: SelectionPolicy
    candidate_strategies: list[StrategySpec]
    reference_strategies: list[StrategySpec] = field(default_factory=list)
    condition_variants: list[ConditionVariant] = field(default_factory=list)
    result_store_dir: str = "backend/data/run_results"


def build_timeframe_spec(
    *,
    key: str,
    label: str,
    yfinance_interval: str,
    bar_seconds: int,
    bars_per_year: float,
) -> TimeframeSpec:
    return TimeframeSpec(
        key=key,
        label=label,
        yfinance_interval=yfinance_interval,
        bar_seconds=bar_seconds,
        bars_per_year=float(bars_per_year),
    )
