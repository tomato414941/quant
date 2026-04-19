from __future__ import annotations

from app.strategy_blueprint_builder import (
    ExecutionVariantDefinition,
    PortfolioModelVariantDefinition,
    SelectionVariantDefinition,
    build_selection_strategy_blueprint_product,
)
from app.strategy_presets import (
    DEFAULT_ANNUAL_EXECUTION_POLICY,
    DEFAULT_DAILY_TIMEFRAME,
    DEFAULT_INVESTMENT_UNIVERSE,
    DEFAULT_RISK_CONTROLS,
    EQUAL_WEIGHT,
    FULL_UNIVERSE,
    FULL_UNIVERSE_MOMENTUM_LOW_VOL_TILT_LIGHT_TOP,
    FULL_UNIVERSE_MOMENTUM_LOW_VOL_TILT_WEAK_TOP,
    FULL_UNIVERSE_MOMENTUM_MACRO_TILT_LIGHT_TOP,
    FULL_UNIVERSE_MOMENTUM_TILT,
    FULL_UNIVERSE_MOMENTUM_TILT_STRONG,
    FULL_UNIVERSE_MOMENTUM_TILT_WEAK,
    FULL_UNIVERSE_MOMENTUM_TILT_WEAK_SOFTMAX,
    FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP,
    FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_10M,
    FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_11M,
    FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_15M,
    FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_3M,
    FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_6M,
    FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_8M,
    FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_9M,
    HIERARCHICAL_RISK_PARITY,
    MEAN_RISK_UTILITY,
    MEAN_RISK_UTILITY_CONSERVATIVE,
    MINIMUM_VARIANCE,
    RISK_BUDGETING,
)


FULL_UNIVERSE_CANDIDATE_BLUEPRINTS = [
    *build_selection_strategy_blueprint_product(
        strategy_id_pattern="stg-fu-{portfolio_model}",
        selection_variants=[
            SelectionVariantDefinition(key="fu", selection=FULL_UNIVERSE),
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
    *build_selection_strategy_blueprint_product(
        strategy_id_pattern="stg-fu-{selection}-hrp",
        selection_variants=[
            SelectionVariantDefinition(
                key="momo12-lin025",
                selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK,
                hypothesis="全資産を残した弱いモメンタム傾斜は、分散を保ちながら成績を改善しやすい",
            ),
            SelectionVariantDefinition(
                key="momo12-top035",
                selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP,
                hypothesis="全資産を残した上位優遇型のモメンタム傾斜は、分散を保ちながらSharpeを改善しやすい",
            ),
            SelectionVariantDefinition(
                key="momo6-top035",
                selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_6M,
                hypothesis="全資産を残した6ヶ月モメンタムの上位優遇傾斜は、中期の強さを取り込みやすい",
            ),
            SelectionVariantDefinition(
                key="momo9-top035",
                selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_9M,
                hypothesis="全資産を残した9ヶ月モメンタムの上位優遇傾斜は、中長期の強さを取り込みやすい",
            ),
            SelectionVariantDefinition(
                key="momo8-top035",
                selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_8M,
                hypothesis="全資産を残した8ヶ月モメンタムの上位優遇傾斜は、中期寄りの強さを取り込みやすい",
            ),
            SelectionVariantDefinition(
                key="momo10-top035",
                selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_10M,
                hypothesis="全資産を残した10ヶ月モメンタムの上位優遇傾斜は、中長期の強さを取り込みやすい",
            ),
            SelectionVariantDefinition(
                key="momo11-top035",
                selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_11M,
                hypothesis="全資産を残した11ヶ月モメンタムの上位優遇傾斜は、中長期の強さを取り込みやすい",
            ),
            SelectionVariantDefinition(
                key="momo3-top035",
                selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_3M,
                hypothesis="全資産を残した3ヶ月モメンタムの上位優遇傾斜は、短期の強さを取り込みやすい",
            ),
            SelectionVariantDefinition(
                key="momo15-top035",
                selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_15M,
                hypothesis="全資産を残した15ヶ月モメンタムの上位優遇傾斜は、より長いトレンドを取り込みやすい",
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
        investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
        risk_controls=DEFAULT_RISK_CONTROLS,
    ),
    *build_selection_strategy_blueprint_product(
        strategy_id_pattern="{selection}",
        selection_variants=[
            SelectionVariantDefinition(
                key="stg-fu-momolv7030-top025-hrp",
                selection=FULL_UNIVERSE_MOMENTUM_LOW_VOL_TILT_WEAK_TOP,
            ),
            SelectionVariantDefinition(
                key="stg-fu-momolv8515-top025-hrp",
                selection=FULL_UNIVERSE_MOMENTUM_LOW_VOL_TILT_LIGHT_TOP,
            ),
            SelectionVariantDefinition(
                key="stg-fu-momomac8515-top025-hrp",
                selection=FULL_UNIVERSE_MOMENTUM_MACRO_TILT_LIGHT_TOP,
            ),
            SelectionVariantDefinition(
                key="stg-fu-momo12-soft025-hrp",
                selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_SOFTMAX,
            ),
            SelectionVariantDefinition(
                key="stg-fu-momo12-lin050-hrp",
                selection=FULL_UNIVERSE_MOMENTUM_TILT,
            ),
            SelectionVariantDefinition(
                key="stg-fu-momo12-lin100-hrp",
                selection=FULL_UNIVERSE_MOMENTUM_TILT_STRONG,
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
        investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
        risk_controls=DEFAULT_RISK_CONTROLS,
    ),
    *build_selection_strategy_blueprint_product(
        strategy_id_pattern="stg-fu-momo12-top035-{portfolio_model}",
        selection_variants=[
            SelectionVariantDefinition(
                key="momo12-top035",
                selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP,
            ),
        ],
        portfolio_model_variants=[
            PortfolioModelVariantDefinition(key="mru", portfolio_model=MEAN_RISK_UTILITY),
            PortfolioModelVariantDefinition(key="mruc", portfolio_model=MEAN_RISK_UTILITY_CONSERVATIVE),
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
    "FULL_UNIVERSE_CANDIDATE_BLUEPRINTS",
]
