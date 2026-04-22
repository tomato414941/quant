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
    DEFAULT_MONTH_END_EXECUTION_POLICY,
    DEFAULT_INVESTMENT_UNIVERSE,
    DEFAULT_RISK_CONTROLS,
    DUAL_MOMENTUM_TOP3,
    EQUAL_WEIGHT,
    HIERARCHICAL_RISK_PARITY,
    MINIMUM_VARIANCE,
    MOMENTUM_TOP3,
    POSITIVE_MOMENTUM_HIGH_VOLUME_UNIVERSE,
    POSITIVE_MOMENTUM_LOW_VOL_UNIVERSE,
    POSITIVE_MOMENTUM_UNIVERSE,
    POSITIVE_TREND_SHORT_REVERSAL,
    RISK_BUDGETING,
    RISK_REGIME_POSITIVE_MOMENTUM,
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
    *build_selection_strategy_definition_product(
        strategy_id_pattern="{selection}",
        selection_variants=[
            SelectionVariantDefinition(
                key="stg-posmom-hrp-month",
                selection=POSITIVE_MOMENTUM_UNIVERSE,
                hypothesis="正の中期モメンタム資産だけを持ち、全滅時はCASHへ逃げると広い下落局面の損失を抑えやすい",
            ),
            SelectionVariantDefinition(
                key="stg-posrev5-trend60-hrp-month",
                selection=POSITIVE_TREND_SHORT_REVERSAL,
                hypothesis="長期上昇中の短期押し目は、単純モメンタムとは独立した日次寄りの収益源になりうる",
            ),
            SelectionVariantDefinition(
                key="stg-riskoff-posmom-hrp-month",
                selection=RISK_REGIME_POSITIVE_MOMENTUM,
                hypothesis="株式リスクが崩れている局面では、防御資産へ絞ることで損失を抑えやすい",
            ),
        ],
        portfolio_model_variants=[
            PortfolioModelVariantDefinition(key="hrp", portfolio_model=HIERARCHICAL_RISK_PARITY),
        ],
        execution_variants=[
            ExecutionVariantDefinition(
                key="month",
                timeframe=DEFAULT_DAILY_TIMEFRAME,
                execution_policy=DEFAULT_MONTH_END_EXECUTION_POLICY,
            ),
        ],
        investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
        risk_controls=DEFAULT_RISK_CONTROLS,
    ),
]



__all__ = [
    "FILTERED_CANDIDATE_DEFINITIONS",
]
