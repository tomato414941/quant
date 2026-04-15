from __future__ import annotations

from app.portfolio import build_strategy_spec
from app.strategy_presets import (
    DEFAULT_RISK_CONTROLS,
    ETF_ONLY_INVESTMENT_UNIVERSE,
    FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP,
    HIERARCHICAL_RISK_PARITY,
)


UNIVERSE_VARIANT_CANDIDATE_STRATEGIES = [
    build_strategy_spec(
        investment_universe=ETF_ONLY_INVESTMENT_UNIVERSE,
        selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP,
        portfolio_model=HIERARCHICAL_RISK_PARITY,
        risk_controls=DEFAULT_RISK_CONTROLS,
        strategy_id="stg-etf-momo12-top035-hrp",
        label="ETF限定モメンタム傾斜 上位優遇 12ヶ月 × HRP",
        hypothesis="暗号資産を外したETFユニバースでも、上位優遇型のモメンタム傾斜が有効に働く可能性がある",
        description="18資産ETFに限定して、12ヶ月モメンタムの上位優遇傾斜をHRPに載せる",
    ),
]


__all__ = ["UNIVERSE_VARIANT_CANDIDATE_STRATEGIES"]
