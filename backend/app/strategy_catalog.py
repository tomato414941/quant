from __future__ import annotations

from app.strategy_blueprint_builder import build_strategy_specs_from_blueprints
from app.strategy_candidate_baselines import BASELINE_CANDIDATE_BLUEPRINTS
from app.strategy_candidate_predictors import PREDICTOR_CANDIDATE_BLUEPRINTS
from app.strategy_candidate_timeframes import TIMEFRAME_VARIANT_CANDIDATE_BLUEPRINTS
from app.strategy_references import REFERENCE_EQUAL_WEIGHT_WITH_CASH


CANONICAL_CANDIDATE_BLUEPRINTS = [
    *BASELINE_CANDIDATE_BLUEPRINTS,
    *TIMEFRAME_VARIANT_CANDIDATE_BLUEPRINTS,
    *PREDICTOR_CANDIDATE_BLUEPRINTS,
]



__all__ = [
    "BASELINE_CANDIDATE_BLUEPRINTS",
    "CANONICAL_CANDIDATE_BLUEPRINTS",
    "PREDICTOR_CANDIDATE_BLUEPRINTS",
    "REFERENCE_EQUAL_WEIGHT_WITH_CASH",
    "TIMEFRAME_VARIANT_CANDIDATE_BLUEPRINTS",
]
