from __future__ import annotations

from app.strategy_definition_builder import build_strategy_specs_from_definitions
from app.strategy_candidate_filtered import FILTERED_CANDIDATE_DEFINITIONS
from app.strategy_candidate_full_universe import FULL_UNIVERSE_CANDIDATE_DEFINITIONS
from app.strategy_candidate_universe_variants import UNIVERSE_VARIANT_CANDIDATE_DEFINITIONS


BASELINE_CANDIDATE_DEFINITIONS = [
    *FULL_UNIVERSE_CANDIDATE_DEFINITIONS,
    *FILTERED_CANDIDATE_DEFINITIONS,
    *UNIVERSE_VARIANT_CANDIDATE_DEFINITIONS,
]



__all__ = [
    "BASELINE_CANDIDATE_DEFINITIONS",
    "FILTERED_CANDIDATE_DEFINITIONS",
    "FULL_UNIVERSE_CANDIDATE_DEFINITIONS",
    "UNIVERSE_VARIANT_CANDIDATE_DEFINITIONS",
]
