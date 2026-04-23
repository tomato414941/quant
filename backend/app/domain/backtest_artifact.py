from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageMetadata:
    name: str
    applicable: bool
    reason: str | None = None


@dataclass(frozen=True)
class SelectionSnapshot:
    as_of_date: str | None
    selected_assets: tuple[str, ...]
    scores: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class ForecastSnapshotArtifact:
    as_of_date: str | None
    expected_return_by_asset: tuple[tuple[str, float], ...] = ()
    confidence_by_asset: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class TargetPortfolioSnapshot:
    as_of_date: str | None
    selected_assets: tuple[str, ...]
    weights: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class ExecutionDecisionArtifact:
    as_of_date: str | None
    action: str
    reason: str
    policy: str
    turnover: float
    estimated_cost_pct: float
    estimated_edge_pct: float | None
    average_confidence: float | None


@dataclass(frozen=True)
class ExecutionEventArtifact:
    as_of_date: str | None
    action: str
    selected_assets: tuple[str, ...]
    weights: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class PortfolioPathPoint:
    date: str
    equity: float


@dataclass(frozen=True)
class PerformanceSummary:
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float


@dataclass(frozen=True)
class BacktestArtifact:
    blueprint: object
    run_context: object
    run_result: dict[str, object]
    stage_metadata: tuple[StageMetadata, ...]
    selection_snapshots: tuple[SelectionSnapshot, ...]
    forecast_snapshots: tuple[ForecastSnapshotArtifact, ...]
    target_portfolios: tuple[TargetPortfolioSnapshot, ...]
    decision_events: tuple[ExecutionDecisionArtifact, ...]
    execution_events: tuple[ExecutionEventArtifact, ...]
    portfolio_path: tuple[PortfolioPathPoint, ...]
    performance_summary: PerformanceSummary | None
