from __future__ import annotations

from dataclasses import dataclass, field

from app.portfolio import PortfolioState, StrategySpec


@dataclass
class DatasetSpec:
    period: str
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
    initial_capital: float


@dataclass
class EvaluationSpec:
    evaluation_settings: EvaluationSettings


@dataclass
class RunInputSpec:
    portfolio_state: PortfolioState


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
    dataset: DatasetSpec
    run_input: RunInputSpec
    execution_assumptions: ExecutionAssumptionsSpec
    evaluation: EvaluationSpec
    selection_policy: SelectionPolicy
    candidate_strategies: list[StrategySpec]
    reference_strategies: list[StrategySpec] = field(default_factory=list)
    condition_variants: list[ConditionVariant] = field(default_factory=list)
    result_store_dir: str = "backend/data/run_results"
