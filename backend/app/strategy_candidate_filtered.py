from __future__ import annotations

from app.portfolio import build_strategy_spec
from app.strategy_presets import (
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


FILTERED_CANDIDATE_STRATEGIES = [
    build_strategy_spec(
        strategy_id="stg-top3-eq",
        investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
        selection=MOMENTUM_TOP3,
        portfolio_model=EQUAL_WEIGHT,
        risk_controls=DEFAULT_RISK_CONTROLS,
    ),
    build_strategy_spec(
        strategy_id="stg-top3-rb",
        investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
        selection=MOMENTUM_TOP3,
        portfolio_model=RISK_BUDGETING,
        risk_controls=DEFAULT_RISK_CONTROLS,
    ),
    build_strategy_spec(
        strategy_id="stg-top3-minvar",
        investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
        selection=MOMENTUM_TOP3,
        portfolio_model=MINIMUM_VARIANCE,
        risk_controls=DEFAULT_RISK_CONTROLS,
    ),
    build_strategy_spec(
        strategy_id="stg-top3-hrp",
        investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
        selection=MOMENTUM_TOP3,
        portfolio_model=HIERARCHICAL_RISK_PARITY,
        risk_controls=DEFAULT_RISK_CONTROLS,
    ),
    build_strategy_spec(
        strategy_id="stg-dualtop3-hrp",
        investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
        selection=DUAL_MOMENTUM_TOP3,
        portfolio_model=HIERARCHICAL_RISK_PARITY,
        risk_controls=DEFAULT_RISK_CONTROLS,
    ),
    build_strategy_spec(
        strategy_id="stg-poslowvol-hrp",
        investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
        selection=POSITIVE_MOMENTUM_LOW_VOL_UNIVERSE,
        portfolio_model=HIERARCHICAL_RISK_PARITY,
        risk_controls=DEFAULT_RISK_CONTROLS,
    ),
    build_strategy_spec(
        strategy_id="stg-trailmomlowvol-hrp",
        investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
        selection=TRAILING_MOMENTUM_LOW_VOL_UNIVERSE,
        portfolio_model=HIERARCHICAL_RISK_PARITY,
        risk_controls=DEFAULT_RISK_CONTROLS,
    ),
    build_strategy_spec(
        strategy_id="stg-posvol-hrp",
        investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
        selection=POSITIVE_MOMENTUM_HIGH_VOLUME_UNIVERSE,
        portfolio_model=HIERARCHICAL_RISK_PARITY,
        risk_controls=DEFAULT_RISK_CONTROLS,
    ),
]


__all__ = ["FILTERED_CANDIDATE_STRATEGIES"]
