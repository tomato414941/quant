from __future__ import annotations

from dataclasses import dataclass


EVALUATION_PROFILE_SCHEMA_VERSION = "v1"


@dataclass(frozen=True)
class EvaluationProfile:
    profile_id: str
    label: str
    universe_key: str
    period_key: str
    evaluation_kind: str
    cost_multiplier: float
    max_weight: float | None
    walk_forward_start_year: int | None
    walk_forward_end_year: int | None
    purpose: str


EVALUATION_PROFILES = (
    EvaluationProfile(
        profile_id="etf_2015_2025",
        label="ETF 2015-2025",
        universe_key="etf",
        period_key="2015_2025",
        evaluation_kind="single_run",
        cost_multiplier=1.0,
        max_weight=0.45,
        walk_forward_start_year=None,
        walk_forward_end_year=None,
        purpose="Base ETF strategy evaluation over the full configured period.",
    ),
    EvaluationProfile(
        profile_id="etf_cost_2x_2015_2025",
        label="ETF 2015-2025 cost x2",
        universe_key="etf",
        period_key="2015_2025",
        evaluation_kind="single_run",
        cost_multiplier=2.0,
        max_weight=0.45,
        walk_forward_start_year=None,
        walk_forward_end_year=None,
        purpose="ETF strategy cost sensitivity check.",
    ),
    EvaluationProfile(
        profile_id="etf_walk_forward_2020_2025",
        label="ETF walk-forward 2020-2025",
        universe_key="etf",
        period_key="2015_2025",
        evaluation_kind="walk_forward",
        cost_multiplier=1.0,
        max_weight=0.45,
        walk_forward_start_year=2020,
        walk_forward_end_year=2025,
        purpose="ETF strategy time-stability check across walk-forward windows.",
    ),
)

EVALUATION_PROFILES_BY_ID = {
    profile.profile_id: profile
    for profile in EVALUATION_PROFILES
}


def serialize_evaluation_profile(profile: EvaluationProfile) -> dict:
    return {
        "profileId": profile.profile_id,
        "label": profile.label,
        "universeKey": profile.universe_key,
        "periodKey": profile.period_key,
        "evaluationKind": profile.evaluation_kind,
        "costMultiplier": profile.cost_multiplier,
        "maxWeight": profile.max_weight,
        "walkForwardStartYear": profile.walk_forward_start_year,
        "walkForwardEndYear": profile.walk_forward_end_year,
        "purpose": profile.purpose,
    }


def validate_evaluation_profiles(profiles: tuple[EvaluationProfile, ...] = EVALUATION_PROFILES) -> None:
    profile_ids = [profile.profile_id for profile in profiles]
    duplicate_ids = sorted({profile_id for profile_id in profile_ids if profile_ids.count(profile_id) > 1})
    if duplicate_ids:
        raise ValueError(f"Duplicate evaluation profile id(s): {', '.join(duplicate_ids)}")
    for profile in profiles:
        if profile.evaluation_kind not in {"single_run", "walk_forward"}:
            raise ValueError(f"Unsupported evaluation profile kind for {profile.profile_id}: {profile.evaluation_kind}")
        if profile.universe_key != "etf":
            raise ValueError(f"Unsupported evaluation profile universe for {profile.profile_id}: {profile.universe_key}")


validate_evaluation_profiles()


__all__ = [
    "EVALUATION_PROFILE_SCHEMA_VERSION",
    "EVALUATION_PROFILES",
    "EVALUATION_PROFILES_BY_ID",
    "EvaluationProfile",
    "serialize_evaluation_profile",
    "validate_evaluation_profiles",
]
