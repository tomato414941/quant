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
    DEFAULT_INVESTMENT_UNIVERSE,
    DEFAULT_RISK_CONTROLS,
    DUAL_MOMENTUM_TOP3,
    EQUAL_WEIGHT,
    HIERARCHICAL_RISK_PARITY,
    MINIMUM_VARIANCE,
    MOMENTUM_TOP3,
    POSITIVE_MOMENTUM_HIGH_VOLUME_UNIVERSE,
    POSITIVE_MOMENTUM_LOW_VOL_UNIVERSE,
    RISK_BUDGETING,
    TRAILING_MOMENTUM_LOW_VOL_UNIVERSE,
)


FILTERED_CANDIDATE_DEFINITIONS = [
    *build_selection_strategy_definition_product(
        strategy_id_pattern="stg-top3-{portfolio_model}",
        selection_variants=[
            SelectionVariantDefinition(
                key="top3",
                selection=MOMENTUM_TOP3,
            ),
        ],
        portfolio_model_variants=[
            PortfolioModelVariantDefinition(key="eq", portfolio_model=EQUAL_WEIGHT),
            PortfolioModelVariantDefinition(key="rb", portfolio_model=RISK_BUDGETING),
            PortfolioModelVariantDefinition(key="minvar", portfolio_model=MINIMUM_VARIANCE),
            PortfolioModelVariantDefinition(key="hrp", portfolio_model=HIERARCHICAL_RISK_PARITY),
        ],
        execution_variants=[
            ExecutionVariantDefinition(
                key="annual",
                timeframe=DEFAULT_DAILY_TIMEFRAME,
                execution_policy=DEFAULT_ANNUAL_EXECUTION_POLICY,
            ),
        ],
        investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
        risk_controls=DEFAULT_RISK_CONTROLS,
    ),
    *build_selection_strategy_definition_product(
        strategy_id_pattern="{selection}",
        selection_variants=[
            SelectionVariantDefinition(key="stg-dualtop3-hrp", selection=DUAL_MOMENTUM_TOP3),
            SelectionVariantDefinition(key="stg-poslowvol-hrp", selection=POSITIVE_MOMENTUM_LOW_VOL_UNIVERSE),
            SelectionVariantDefinition(key="stg-trailmomlowvol-hrp", selection=TRAILING_MOMENTUM_LOW_VOL_UNIVERSE),
            SelectionVariantDefinition(key="stg-posvol-hrp", selection=POSITIVE_MOMENTUM_HIGH_VOLUME_UNIVERSE),
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
        investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
        risk_controls=DEFAULT_RISK_CONTROLS,
    ),
]



__all__ = [
    "FILTERED_CANDIDATE_DEFINITIONS",
]
