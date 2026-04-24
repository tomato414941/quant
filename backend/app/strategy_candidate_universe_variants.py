from __future__ import annotations

from app.strategy_definition_builder import (
    ExecutionVariantDefinition,
    PortfolioModelVariantDefinition,
    SelectionVariantDefinition,
    build_selection_strategy_definition_product,
)
from app.strategy_presets import (
    DEFAULT_ANNUAL_EXECUTION_POLICY,
    DEFAULT_DAILY_TIMEFRAME,
    DEFAULT_RISK_CONTROLS,
    ETF_INVESTMENT_UNIVERSE,
    FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP,
    HIERARCHICAL_RISK_PARITY,
)


UNIVERSE_VARIANT_CANDIDATE_DEFINITIONS = build_selection_strategy_definition_product(
    strategy_id_pattern="{selection}",
    selection_variants=[
        SelectionVariantDefinition(
            key="stg-etf-momo12-top035-hrp",
            selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP,
            label="ETF限定モメンタム傾斜 上位優遇 12ヶ月 × HRP",
            hypothesis="ETFユニバースでも、上位優遇型のモメンタム傾斜が有効に働く可能性がある",
            description="ETFに限定して、12ヶ月モメンタムの上位優遇傾斜をHRPに載せる",
        ),
    ],
    portfolio_model_variants=[
        PortfolioModelVariantDefinition(key="hrp", portfolio_model=HIERARCHICAL_RISK_PARITY),
    ],
    execution_variants=[
        ExecutionVariantDefinition(
            key="annual",
            timeframe=DEFAULT_DAILY_TIMEFRAME,
            execution_policy=DEFAULT_ANNUAL_EXECUTION_POLICY,
        ),
    ],
    investment_universe=ETF_INVESTMENT_UNIVERSE,
    risk_controls=DEFAULT_RISK_CONTROLS,
)



__all__ = [
    "UNIVERSE_VARIANT_CANDIDATE_DEFINITIONS",
]
