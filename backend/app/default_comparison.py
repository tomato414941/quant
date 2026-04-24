from __future__ import annotations

from app.comparison_models import (
    ComparisonSpec,
    EvaluationSpec,
    EvaluationSettings,
    MarketSliceSpec,
    RunSpec,
    SelectionPolicy,
)
from app.portfolio import build_portfolio_state
from app.execution_defaults import (
    DEFAULT_COMPARISON_CONDITION_VARIANTS,
    DEFAULT_EXECUTION_ASSUMPTIONS,
)
from app.instrument_registry import ETF_TICKERS
from app.strategy_registry import (
    CANONICAL_CANDIDATE_DEFINITIONS,
    REFERENCE_EQUAL_WEIGHT_WITH_CASH,
)

DEFAULT_COMPARISON_SPEC = ComparisonSpec(
    comparison_id="etf_portfolio_models_2015_2025",
    title="有望Strategyの探索",
    question="共通の評価前提で Strategy を比較し、現時点で最も有望な構成を見つける",
    run_spec=RunSpec(
        market_slice=MarketSliceSpec(
            period="2015_2025",
            sanity_periods=["3y"],
            start_date="2015-01-01",
            end_date="2025-12-31",
        ),
        portfolio_state=build_portfolio_state(
            current_weights={
                ticker: 0.85 / len(ETF_TICKERS)
                for ticker in ETF_TICKERS
            },
            cash_weight=0.15,
        ),
        capital_base=10_000,
        execution_assumptions=DEFAULT_EXECUTION_ASSUMPTIONS,
        evaluation=EvaluationSpec(
            evaluation_settings=EvaluationSettings(
                split_ratio=0.7,
            ),
        ),
    ),
    selection_policy=SelectionPolicy(
        primary_metric="sharpe_ratio",
        secondary_metric="total_return",
        tertiary_metric="max_drawdown",
    ),
    candidate_strategies=CANONICAL_CANDIDATE_DEFINITIONS,
    reference_strategies=[REFERENCE_EQUAL_WEIGHT_WITH_CASH],
    condition_variants=DEFAULT_COMPARISON_CONDITION_VARIANTS,
)

__all__ = ["DEFAULT_COMPARISON_SPEC"]
