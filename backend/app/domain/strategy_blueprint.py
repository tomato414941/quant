from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategySignalBlueprint:
    key: str
    label: str
    description: str
    source_kind: str
    data_timeframe_key: str
    signal_timeframe_key: str
    observation_tickers: tuple[str, ...]
    observation_fields: tuple[str, ...]
    signal_parameters: tuple[tuple[str, object], ...]
    weight: float
    predictor_key: str | None = None
    data_source_key: str | None = None
    data_source_label: str | None = None
    data_source_kind: str | None = None
    feature_definition_key: str | None = None
    feature_definition_label: str | None = None
    feature_definition_source_fields: tuple[str, ...] = ()
    feature_definition_derived_fields: tuple[str, ...] = ()
    alignment_policy_key: str | None = None
    alignment_policy_label: str | None = None
    alignment_policy_method: str | None = None
    alignment_policy_parameters: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True)
class ForecastSourceBlueprint:
    kind: str
    predictor_key: str | None
    signal_weight: float | None
    predictor_weight: float | None


@dataclass(frozen=True)
class PortfolioConstructionBlueprint:
    model_key: str
    model_type: str
    max_investment_ratio: float
    max_weight: float | None


@dataclass(frozen=True)
class DecisionPolicyBlueprint:
    key: str
    execution_plan_key: str
    execution_plan_label: str
    decision_schedule: str
    rebalance_schedule: str
    execution_mode: str | None


@dataclass(frozen=True)
class ExecutionScheduleBlueprint:
    market_data_timeframe_key: str
    signal_timeframe_keys: tuple[str, ...]


@dataclass(frozen=True)
class StrategyBlueprint:
    key: str
    version: str
    label: str
    hypothesis: str | None
    description: str
    investment_universe_key: str
    investment_universe_label: str
    investment_universe_tickers: tuple[str, ...]
    signals: tuple[StrategySignalBlueprint, ...]
    portfolio_construction: PortfolioConstructionBlueprint
    decision_policy: DecisionPolicyBlueprint
    execution_schedule: ExecutionScheduleBlueprint
    forecast_source: ForecastSourceBlueprint | None
    extensions: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class RunContext:
    initial_capital: float
    split_ratio: float
    bars_per_year: float
    execution_assumptions: dict[str, object]
    availability_policy: dict[str, object]
    portfolio_state: dict[str, object] | None = None
