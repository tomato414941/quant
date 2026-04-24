from app.evaluation_profiles import EVALUATION_PROFILES, serialize_evaluation_profile


def test_evaluation_profiles_are_unique() -> None:
    profile_ids = [profile.profile_id for profile in EVALUATION_PROFILES]

    assert len(set(profile_ids)) == len(profile_ids)


def test_evaluation_profiles_define_required_fields() -> None:
    payloads = [serialize_evaluation_profile(profile) for profile in EVALUATION_PROFILES]

    assert {payload["profileId"] for payload in payloads} == {
        "etf_2015_2025",
        "etf_cost_2x_2015_2025",
        "etf_walk_forward_2020_2025",
    }
    assert all(payload["universeKey"] == "etf" for payload in payloads)
    assert all(payload["periodKey"] == "2015_2025" for payload in payloads)
