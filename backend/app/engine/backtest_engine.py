from __future__ import annotations

from app.domain import (
    BacktestArtifact,
    ExecutionDecisionArtifact,
    ExecutionEventArtifact,
    PerformanceSummary,
    PortfolioPathPoint,
    RunContext,
    SelectionSnapshot,
    StageMetadata,
    StrategyBlueprint,
    TargetPortfolioSnapshot,
)
from app.spec import build_strategy_definition_from_blueprint


def run_strategy_backtest(
    strategy_blueprint: StrategyBlueprint,
    market_bundle: dict[str, object],
    run_context: RunContext,
    *,
    predictor_panel=None,
) -> BacktestArtifact:
    from app.portfolio import evaluate_strategy_definition_run

    run_result = evaluate_strategy_definition_run(
        closes=market_bundle['closes'],
        volumes=market_bundle.get('volumes'),
        strategy_definition=build_strategy_definition_from_blueprint(strategy_blueprint),
        initial_capital=run_context.initial_capital,
        split_ratio=run_context.split_ratio,
        bars_per_year=run_context.bars_per_year,
        execution_assumptions=run_context.execution_assumptions,
        portfolio_state=_deserialize_portfolio_state(run_context.portfolio_state),
        predictor_panel=predictor_panel,
        availability_policy=run_context.availability_policy,
    )
    return build_backtest_artifact(
        strategy_blueprint=strategy_blueprint,
        run_context=run_context,
        run_result=run_result,
    )


def build_backtest_artifact(
    *,
    strategy_blueprint: StrategyBlueprint,
    run_context: RunContext,
    run_result: dict[str, object],
) -> BacktestArtifact:
    decision_events = tuple(
        ExecutionDecisionArtifact(
            as_of_date=event.get('date'),
            action=str(event.get('action')),
            reason=str(event.get('reason')),
            policy=str(event.get('policy')),
            turnover=float(event.get('turnover', 0.0) or 0.0),
            estimated_cost_pct=float(event.get('estimatedCostPct', 0.0) or 0.0),
            estimated_edge_pct=_optional_float(event.get('estimatedEdgePct')),
            average_confidence=_optional_float(event.get('averageConfidence')),
        )
        for event in run_result.get('decisionEvents', [])
    )
    execution_events = tuple(
        ExecutionEventArtifact(
            as_of_date=event.get('date'),
            action=str(event.get('action')),
            selected_assets=tuple(event.get('selectedAssets', [])),
            weights=tuple((str(weight['asset']), float(weight['weight'])) for weight in event.get('weights', [])),
        )
        for event in run_result.get('executionTrace', [])
    )
    selection_snapshots = tuple(
        SelectionSnapshot(
            as_of_date=event.as_of_date,
            selected_assets=event.selected_assets,
        )
        for event in execution_events
    )
    target_portfolios = tuple(
        TargetPortfolioSnapshot(
            as_of_date=event.as_of_date,
            selected_assets=event.selected_assets,
            weights=event.weights,
        )
        for event in execution_events
    )
    portfolio_path = tuple(
        PortfolioPathPoint(
            date=str(point['date']),
            equity=float(point.get('portfolioEquity', point.get('equity', 0.0))),
        )
        for point in run_result.get('series', [])
    )
    summary = run_result.get('summary')
    performance_summary = None
    if isinstance(summary, dict):
        performance_summary = PerformanceSummary(
            total_return_pct=float(summary.get('totalReturnPct', 0.0)),
            sharpe_ratio=float(summary.get('sharpeRatio', 0.0)),
            max_drawdown_pct=float(summary.get('maxDrawdownPct', 0.0)),
        )
    return BacktestArtifact(
        blueprint=strategy_blueprint,
        run_context=run_context,
        run_result=run_result,
        stage_metadata=(
            StageMetadata(name='selection_stage', applicable=True),
            StageMetadata(name='portfolio_stage', applicable=True),
            StageMetadata(name='decision_stage', applicable=True),
            StageMetadata(name='execution_stage', applicable=True),
        ),
        selection_snapshots=selection_snapshots,
        forecast_snapshots=(),
        target_portfolios=target_portfolios,
        decision_events=decision_events,
        execution_events=execution_events,
        portfolio_path=portfolio_path,
        performance_summary=performance_summary,
    )


def _deserialize_portfolio_state(portfolio_state_payload: dict[str, object] | None):
    if portfolio_state_payload is None:
        return None
    from app.portfolio import build_portfolio_state

    return build_portfolio_state(
        current_weights=portfolio_state_payload.get('currentWeights') or {},
        cash_weight=float(portfolio_state_payload.get('cashWeight', 0.0) or 0.0),
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
