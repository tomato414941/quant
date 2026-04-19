from __future__ import annotations

from app.strategy_definition_builder import build_strategy_specs_from_definitions
from app.strategy_candidate_baselines import BASELINE_CANDIDATE_DEFINITIONS
from app.strategy_candidate_predictors import PREDICTOR_CANDIDATE_DEFINITIONS
from app.strategy_candidate_timeframes import TIMEFRAME_VARIANT_CANDIDATE_DEFINITIONS
from app.strategy_references import REFERENCE_EQUAL_WEIGHT_WITH_CASH


CANONICAL_CANDIDATE_DEFINITIONS = [
    *BASELINE_CANDIDATE_DEFINITIONS,
    *TIMEFRAME_VARIANT_CANDIDATE_DEFINITIONS,
    *PREDICTOR_CANDIDATE_DEFINITIONS,
]



__all__ = [
    "BASELINE_CANDIDATE_DEFINITIONS",
    "CANONICAL_CANDIDATE_DEFINITIONS",
    "PREDICTOR_CANDIDATE_DEFINITIONS",
    "REFERENCE_EQUAL_WEIGHT_WITH_CASH",
    "TIMEFRAME_VARIANT_CANDIDATE_DEFINITIONS",
]
