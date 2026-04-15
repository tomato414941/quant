from __future__ import annotations

from app.portfolio import build_strategy_spec
from app.strategy_presets import (
    DEFAULT_EVERY_BAR_EXECUTION_POLICY,
    DEFAULT_INVESTMENT_UNIVERSE,
    DEFAULT_MONTH_END_EXECUTION_POLICY,
    DEFAULT_MONTHLY_TIMEFRAME,
    DEFAULT_RISK_CONTROLS,
    DEFAULT_WEEKLY_TIMEFRAME,
    FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M,
    FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_9M,
    HIERARCHICAL_RISK_PARITY,
)


TIMEFRAME_VARIANT_CANDIDATE_STRATEGIES = [
    build_strategy_spec(
        strategy_id="stg-fu-momo9-top035-hrp-1w",
        timeframe=DEFAULT_WEEKLY_TIMEFRAME,
        investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
        selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_9M,
        portfolio_model=HIERARCHICAL_RISK_PARITY,
        execution_policy=DEFAULT_EVERY_BAR_EXECUTION_POLICY,
        risk_controls=DEFAULT_RISK_CONTROLS,
        label="全資産モメンタム傾斜 上位優遇 9ヶ月 × HRP × 毎バー × 週次",
        hypothesis="全資産を残した9ヶ月モメンタムを週次バーごとに反映すると、日次より低回転でトレンドを取り込みやすい",
        description="全ETFを候補に残しつつ、9ヶ月モメンタムの上位優遇傾斜を週次バーごとにHRPへ反映する",
    ),
    build_strategy_spec(
        strategy_id="stg-fu-momo9-top035-hrp-1mo",
        timeframe=DEFAULT_MONTHLY_TIMEFRAME,
        investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
        selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_9M,
        portfolio_model=HIERARCHICAL_RISK_PARITY,
        execution_policy=DEFAULT_EVERY_BAR_EXECUTION_POLICY,
        risk_controls=DEFAULT_RISK_CONTROLS,
        label="全資産モメンタム傾斜 上位優遇 9ヶ月 × HRP × 毎バー × 月次",
        hypothesis="全資産を残した9ヶ月モメンタムを月次バーごとに反映すると、さらに低回転で中長期トレンドを取り込みやすい",
        description="全ETFを候補に残しつつ、9ヶ月モメンタムの上位優遇傾斜を月次バーごとにHRPへ反映する",
    ),
    build_strategy_spec(
        strategy_id="stg-fu-momo2-top035-hrp-month",
        investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
        selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M,
        portfolio_model=HIERARCHICAL_RISK_PARITY,
        execution_policy=DEFAULT_MONTH_END_EXECUTION_POLICY,
        risk_controls=DEFAULT_RISK_CONTROLS,
        label="全資産モメンタム傾斜 上位優遇 2ヶ月 × HRP × 月次",
        hypothesis="短期モメンタムを月次で反映すると、回転を抑えつつSharpeを改善しやすい",
        description="全ETFを候補に残しつつ、2ヶ月モメンタムの上位優遇傾斜を月次でHRPに反映する",
    ),
]


__all__ = ["TIMEFRAME_VARIANT_CANDIDATE_STRATEGIES"]
