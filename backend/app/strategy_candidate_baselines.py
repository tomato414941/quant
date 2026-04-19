from __future__ import annotations

from app.strategy_blueprint_builder import build_strategy_specs_from_blueprints
from app.strategy_candidate_filtered import FILTERED_CANDIDATE_BLUEPRINTS
from app.strategy_candidate_full_universe import FULL_UNIVERSE_CANDIDATE_BLUEPRINTS
from app.strategy_candidate_universe_variants import UNIVERSE_VARIANT_CANDIDATE_BLUEPRINTS


BASELINE_CANDIDATE_BLUEPRINTS = [
    *FULL_UNIVERSE_CANDIDATE_BLUEPRINTS,
    *FILTERED_CANDIDATE_BLUEPRINTS,
    *UNIVERSE_VARIANT_CANDIDATE_BLUEPRINTS,
]



__all__ = [
    "BASELINE_CANDIDATE_BLUEPRINTS",
    "FILTERED_CANDIDATE_BLUEPRINTS",
    "FULL_UNIVERSE_CANDIDATE_BLUEPRINTS",
    "UNIVERSE_VARIANT_CANDIDATE_BLUEPRINTS",
]
