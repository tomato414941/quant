from __future__ import annotations

from app.portfolio import build_risk_controls_spec, build_strategy_definition_from_evaluator_strategy_spec, build_evaluator_strategy_spec
from app.strategy_presets import (
    DEFAULT_INVESTMENT_UNIVERSE,
    EQUAL_WEIGHT,
    FULL_UNIVERSE,
    REFERENCE_HOLD_EXECUTION_POLICY,
)


_REFERENCE_EQUAL_WEIGHT_WITH_CASH_SPEC = build_evaluator_strategy_spec(
    strategy_id="ref-fu-eq-cash",
    investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
    selection=FULL_UNIVERSE,
    portfolio_model=EQUAL_WEIGHT,
    execution_policy=REFERENCE_HOLD_EXECUTION_POLICY,
    risk_controls=build_risk_controls_spec(
        max_investment_ratio=0.85,
        max_weight=None,
    ),
    label="等金額買い持ち + CASH",
    description="全資産を等金額で買い持ちし、15% を CASH に残す参照用 Strategy",
)


REFERENCE_EQUAL_WEIGHT_WITH_CASH = build_strategy_definition_from_evaluator_strategy_spec(
    _REFERENCE_EQUAL_WEIGHT_WITH_CASH_SPEC
)


__all__ = ["REFERENCE_EQUAL_WEIGHT_WITH_CASH"]
