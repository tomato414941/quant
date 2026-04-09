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
from app.predictor_registry import DEFAULT_PREDICTION_TARGET_SPECS
from app.strategy_registry import (
    CANONICAL_CANDIDATE_STRATEGIES,
    REFERENCE_EQUAL_WEIGHT_WITH_CASH,
)

DEFAULT_COMPARISON_SPEC = ComparisonSpec(
    comparison_id="etf_portfolio_models_10y",
    title="有望Strategyの探索",
    question="共通の評価前提で Strategy を比較し、現時点で最も有望な構成を見つける",
    run_spec=RunSpec(
        market_slice=MarketSliceSpec(
            period="10y",
            sanity_periods=["3y"],
        ),
        portfolio_state=build_portfolio_state(
            current_weights={
                "SPY": 0.0425,
                "QQQ": 0.0425,
                "IWM": 0.0425,
                "EFA": 0.0425,
                "EEM": 0.0425,
                "EWJ": 0.0425,
                "EWZ": 0.0425,
                "VNQ": 0.0425,
                "TLT": 0.0425,
                "IEF": 0.0425,
                "LQD": 0.0425,
                "HYG": 0.0425,
                "TIP": 0.0425,
                "GLD": 0.0425,
                "SLV": 0.0425,
                "DBC": 0.0425,
                "USO": 0.0425,
                "UUP": 0.0425,
                "BTC-USD": 0.0425,
                "ETH-USD": 0.0425,
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
    candidate_strategies=CANONICAL_CANDIDATE_STRATEGIES,
    reference_strategies=[REFERENCE_EQUAL_WEIGHT_WITH_CASH],
    condition_variants=DEFAULT_COMPARISON_CONDITION_VARIANTS,
)

__all__ = ["DEFAULT_COMPARISON_SPEC", "DEFAULT_PREDICTION_TARGET_SPECS"]
