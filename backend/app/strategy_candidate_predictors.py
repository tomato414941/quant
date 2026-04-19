from __future__ import annotations

from app.strategy_blueprint_builder import (
    ExecutionVariantDefinition,
    PortfolioModelVariantDefinition,
    PredictorVariantDefinition,
    SelectionVariantDefinition,
    build_predictor_strategy_blueprint_product,
)
from app.strategy_presets import (
    DEFAULT_DAILY_TIMEFRAME,
    DEFAULT_EVERY_BAR_EXECUTION_POLICY,
    DEFAULT_INVESTMENT_UNIVERSE,
    DEFAULT_MONTH_END_EXECUTION_POLICY,
    DEFAULT_RISK_CONTROLS,
    FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M,
    HIERARCHICAL_RISK_PARITY,
)


PREDICTOR_CANDIDATE_BLUEPRINTS = [
    *build_predictor_strategy_blueprint_product(
        strategy_id_pattern="stg-fu-{selection}-{predictor}-{portfolio_model}-{execution}",
        selection_variants=[
            SelectionVariantDefinition(
                key="momo2-top035",
                selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M,
            ),
        ],
        predictor_variants=[
            PredictorVariantDefinition(
                key="pred10mom9010",
                predictor_key="pred-fu-momo2-supplement-10bar-linear-momentum-weighted_blend-9010",
                signal_weight=0.8,
                predictor_weight=0.2,
                label="全資産モメンタム傾斜 上位優遇 2ヶ月 + 10bar予測補助 90/10 × HRP",
                hypothesis="2ヶ月モメンタムを主役に10bar予測を 90/10 で薄く混ぜると、中期アルファを取り込みやすい",
                description="全ETFを候補に残しつつ、2ヶ月モメンタムに10bar予測補助 90/10 を薄く混ぜてHRPに反映する",
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
                label="月次",
                hypothesis="2ヶ月モメンタムを主役に10bar予測を 90/10 で薄く混ぜると、月次更新でも中期アルファを取り込みやすい",
                description="全ETFを候補に残しつつ、2ヶ月モメンタムに10bar予測補助 90/10 を薄く混ぜて月次でHRPに反映する",
            ),
        ],
        investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
        risk_controls=DEFAULT_RISK_CONTROLS,
    ),
    *build_predictor_strategy_blueprint_product(
        strategy_id_pattern="stg-fu-{selection}-{predictor}-{portfolio_model}-{execution}",
        selection_variants=[
            SelectionVariantDefinition(
                key="momo2-top035",
                selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M,
            ),
        ],
        predictor_variants=[
            PredictorVariantDefinition(
                key="pred10mom5050-pw40",
                predictor_key="pred-fu-momo2-supplement-10bar-linear-momentum-weighted_blend-5050",
                signal_weight=0.6,
                predictor_weight=0.4,
                label="全資産モメンタム傾斜 上位優遇 2ヶ月 + 10bar予測補助 50/50 40% × HRP",
                hypothesis="2ヶ月モメンタムを主役に10bar予測 50/50 を 40% 混ぜると、直近の変化を取り込みやすい",
                description="全ETFを候補に残しつつ、2ヶ月モメンタムに10bar予測補助 50/50 を 40% 混ぜてHRPに反映する",
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
                label="月次",
                hypothesis="2ヶ月モメンタムを主役に10bar予測 50/50 を 40% 混ぜると、月次更新でも直近の変化を取り込みやすい",
                description="全ETFを候補に残しつつ、2ヶ月モメンタムに10bar予測補助 50/50 を 40% 混ぜて月次でHRPに反映する",
            ),
            ExecutionVariantDefinition(
                key="daily",
                timeframe=DEFAULT_DAILY_TIMEFRAME,
                execution_policy=DEFAULT_EVERY_BAR_EXECUTION_POLICY,
                label="毎バー",
                hypothesis="2ヶ月モメンタムに10bar予測補助 50/50 を 40% 混ぜて毎バー更新すると、月次より速く直近変化を反映できる",
                description="全ETFを候補に残しつつ、2ヶ月モメンタムに10bar予測補助 50/50 を 40% 混ぜて毎バーでHRPに反映する",
            ),
        ],
        investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
        risk_controls=DEFAULT_RISK_CONTROLS,
    ),
]



__all__ = [
    "PREDICTOR_CANDIDATE_BLUEPRINTS",
]
