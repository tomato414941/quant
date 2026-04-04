from __future__ import annotations

from dataclasses import dataclass, field

from app.portfolio import PortfolioState, StrategyDefinition


@dataclass
class DatasetSpec:
    tickers: list[str]
    period: str
    frequency: str
    sanity_periods: list[str] = field(default_factory=list)


@dataclass
class CostAssumptions:
    commission_pct: float
    slippage_pct: float = 0.0


@dataclass
class EvaluationAssumptions:
    split_ratio: float
    initial_capital: float
    benchmark: str
    cost_assumptions: CostAssumptions


@dataclass
class ConditionVariant:
    key: str
    label: str
    commission_pct: float
    max_investment_ratio: float
    max_weight: float | None = None


@dataclass
class StudyDefinition:
    study_id: str
    title: str
    question: str
    dataset_spec: DatasetSpec
    evaluation_assumptions: EvaluationAssumptions
    portfolio_state: PortfolioState
    strategy_definitions: list[StrategyDefinition]
    condition_variants: list[ConditionVariant] = field(default_factory=list)
    result_store_dir: str = "backend/data/run_results"
