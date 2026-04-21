from __future__ import annotations

from dataclasses import dataclass, field

from app.portfolio import PortfolioState, StrategyDefinition


@dataclass
class MarketSliceSpec:
    period: str
    sanity_periods: list[str] = field(default_factory=list)
    start_date: str | None = None
    end_date: str | None = None


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
    adv_window_bars: int = 20,
    min_adv_notional: float = 1_000_000.0,
    per_asset_overrides: dict[str, dict[str, float]] | None = None,
) -> CostModelSpec:
    return CostModelSpec(
        kind="asset_specific_adv_cost",
        parameters={
            "commissionPct": float(default_commission_pct),
            "slippagePct": float(default_slippage_pct),
            "impactCoefficientPct": float(default_impact_coefficient_pct),
            "advWindowBars": float(adv_window_bars),
            "minAdvNotional": float(min_adv_notional),
        },
        per_asset_overrides=per_asset_overrides or {},
    )


COST_SCALABLE_PARAMETER_KEYS = {"commissionPct", "slippagePct", "impactCoefficientPct"}


def scale_cost_model_spec(cost_model: CostModelSpec, multiplier: float) -> CostModelSpec:
    return CostModelSpec(
        kind=cost_model.kind,
        parameters={
            key: (float(value) * multiplier if key in COST_SCALABLE_PARAMETER_KEYS else float(value))
            for key, value in cost_model.parameters.items()
        },
        per_asset_overrides={
            asset: {
                key: (float(value) * multiplier if key in COST_SCALABLE_PARAMETER_KEYS else float(value))
                for key, value in overrides.items()
            }
            for asset, overrides in cost_model.per_asset_overrides.items()
        },
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
class RunSpec:
    market_slice: MarketSliceSpec
    portfolio_state: PortfolioState
    capital_base: float
    execution_assumptions: ExecutionAssumptionsSpec
    evaluation: EvaluationSpec


@dataclass
class SelectionPolicy:
    primary_metric: str
    secondary_metric: str
    tertiary_metric: str


@dataclass
class ConditionVariant:
    key: str
    label: str
    cost_multiplier: float
    max_investment_ratio: float
    max_weight: float | None = None


@dataclass
class ComparisonSpec:
    comparison_id: str
    title: str
    question: str
    run_spec: RunSpec
    selection_policy: SelectionPolicy
    candidate_strategies: list[StrategyDefinition]
    reference_strategies: list[StrategyDefinition] = field(default_factory=list)
    condition_variants: list[ConditionVariant] = field(default_factory=list)
    result_store_dir: str = "backend/data/run_results"
