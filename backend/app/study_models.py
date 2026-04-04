from __future__ import annotations

from dataclasses import dataclass, field

from app.portfolio import PortfolioCandidateDefinition, PortfolioState


@dataclass
class DatasetSpec:
    tickers: list[str]
    period: str
    frequency: str
    sanity_periods: list[str] = field(default_factory=list)


@dataclass
class ExecutionModelConfig:
    entry: str
    commission_pct: float
    slippage_pct: float
    rebalance_frequency: str = "hold"


@dataclass
class BacktestConfig:
    split_ratio: float
    initial_capital: float
    max_investment_ratio: float
    benchmark: str


@dataclass
class StudyDefinition:
    study_id: str
    title: str
    question: str
    dataset_spec: DatasetSpec
    execution_model: ExecutionModelConfig
    backtest_config: BacktestConfig
    portfolio_state: PortfolioState
    candidate_definitions: list[PortfolioCandidateDefinition]
    result_store_dir: str = "backend/data/run_results"
