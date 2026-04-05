from __future__ import annotations

from dataclasses import dataclass, field

from app.portfolio import PortfolioState, StrategyDefinition


@dataclass
class DatasetSpec:
    period: str
    sanity_periods: list[str] = field(default_factory=list)


@dataclass
class CostAssumptions:
    commission_pct: float
    slippage_pct: float = 0.0


@dataclass
class EvaluationSettings:
    split_ratio: float
    initial_capital: float


@dataclass
class EvaluationContext:
    evaluation_settings: EvaluationSettings
    cost_assumptions: CostAssumptions


@dataclass
class RunInputDefinition:
    portfolio_state: PortfolioState
    evaluation_context: EvaluationContext


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
class ComparisonDefinition:
    comparison_id: str
    title: str
    question: str
    dataset_spec: DatasetSpec
    run_input: RunInputDefinition
    selection_policy: SelectionPolicy
    candidate_strategy_definitions: list[StrategyDefinition]
    reference_strategy_definitions: list[StrategyDefinition] = field(default_factory=list)
    condition_variants: list[ConditionVariant] = field(default_factory=list)
    result_store_dir: str = "backend/data/run_results"
