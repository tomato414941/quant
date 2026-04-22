import copy

from app import robustness_service
from app.default_comparison import DEFAULT_COMPARISON_SPEC


def test_build_robustness_scenarios_uses_smoke_profile() -> None:
    scenarios = robustness_service.build_robustness_scenarios(profile="smoke")

    assert len(scenarios) == 1
    scenario = scenarios[0]
    assert scenario["periodKey"] == "2020_2025"
    assert scenario["universe"] == "crypto_included"
    assert scenario["costMultiplier"] == 1.0
    assert scenario["maxWeight"] == 0.35


def test_build_robustness_scenarios_uses_quick_profile_by_default() -> None:
    scenarios = robustness_service.build_robustness_scenarios()

    assert len(scenarios) == 4
    assert {scenario["periodKey"] for scenario in scenarios} == {"2020_2025"}
    assert {scenario["universe"] for scenario in scenarios} == {"crypto_included", "no_crypto"}
    assert {scenario["costMultiplier"] for scenario in scenarios} == {1.0, 3.0}
    assert {scenario["maxWeight"] for scenario in scenarios} == {0.35}


def test_build_robustness_scenarios_standard_profile_uses_full_matrix() -> None:
    scenarios = robustness_service.build_robustness_scenarios(profile="standard")

    assert len(scenarios) == 81


def test_build_robustness_scenarios_accepts_explicit_filters() -> None:
    scenarios = robustness_service.build_robustness_scenarios(
        profile="standard",
        period_keys=("2015_2025",),
        universes=("btc_only",),
        cost_multipliers=(2.0,),
        max_weights=(0.25,),
    )

    assert len(scenarios) == 1
    scenario = scenarios[0]
    assert scenario["periodKey"] == "2015_2025"
    assert scenario["universe"] == "btc_only"
    assert scenario["costMultiplier"] == 2.0
    assert scenario["maxWeight"] == 0.25


def test_classify_robustness_decision_prioritizes_invalid() -> None:
    decision = robustness_service.classify_robustness_decision(
        scenario_count=10,
        worst_sharpe=1.0,
        top5_count=10,
        diagnostic_flags=["actionable availability warning", "mixed calendar"],
        crypto_sensitivity=0.0,
        cost_sensitivity=0.0,
    )

    assert decision == "INVALID"


def test_classify_robustness_decision_fails_fragile_result() -> None:
    decision = robustness_service.classify_robustness_decision(
        scenario_count=10,
        worst_sharpe=-0.1,
        top5_count=10,
        diagnostic_flags=[],
        crypto_sensitivity=0.0,
        cost_sensitivity=0.0,
    )

    assert decision == "FAIL"


def test_classify_robustness_decision_watches_condition_sensitive_result() -> None:
    decision = robustness_service.classify_robustness_decision(
        scenario_count=10,
        worst_sharpe=0.1,
        top5_count=5,
        diagnostic_flags=[],
        crypto_sensitivity=0.0,
        cost_sensitivity=-0.3,
    )

    assert decision == "WATCH"


def test_classify_robustness_decision_passes_stable_result() -> None:
    decision = robustness_service.classify_robustness_decision(
        scenario_count=10,
        worst_sharpe=0.1,
        top5_count=5,
        diagnostic_flags=[],
        crypto_sensitivity=0.0,
        cost_sensitivity=0.0,
    )

    assert decision == "PASS"


def test_classify_robustness_decision_does_not_penalize_mixed_calendar_only() -> None:
    decision = robustness_service.classify_robustness_decision(
        scenario_count=10,
        worst_sharpe=0.1,
        top5_count=5,
        diagnostic_flags=["mixed calendar"],
        crypto_sensitivity=0.0,
        cost_sensitivity=0.0,
    )

    assert decision == "PASS"


def test_build_robustness_decision_reports_reasons() -> None:
    decision = robustness_service.build_robustness_decision(
        scenario_count=10,
        worst_sharpe=0.1,
        top5_count=5,
        diagnostic_flags=["mixed calendar"],
        crypto_sensitivity=0.0,
        cost_sensitivity=-0.3,
    )

    assert decision == {
        "decision": "WATCH",
        "reasons": [
            "mixed calendar diagnostic only",
            "cost sensitivity below -0.30",
        ],
    }


def test_collect_scenario_diagnostics_normalizes_events() -> None:
    payload = {
        "comparison": {
            "runSpec": {
                "evaluation": {
                    "availabilityDiagnostics": {
                        "actionableWarnings": [
                            {
                                "kind": "requested_asset_unavailable",
                                "message": "1d data has no usable rows for MISSING.",
                                "asset": "MISSING",
                                "timeframe": "1d",
                            }
                        ],
                        "calendarBoundaryWarnings": [
                            {
                                "kind": "aligned_start_after_requested_start",
                                "message": "Calendar boundary moved start",
                                "timeframe": "1d",
                            }
                        ],
                        "assetLifecycleWarnings": [
                            {
                                "kind": "asset_available_after_aligned_start",
                                "message": "BTC starts after the aligned start",
                                "ticker": "BTC-USD",
                                "timeframe": "1d",
                                "alignedStartDate": "2015-01-01",
                                "firstValidDate": "2015-01-05",
                            }
                        ],
                    },
                    "instrumentDiagnostics": {
                        "mixedMarketCalendar": True,
                        "marketCalendars": {"24_7": 1, "nyse": 1},
                        "assetClassCounts": {"crypto": 1, "equity_etf": 1},
                        "unknownSymbols": ["UNKNOWN"],
                    },
                }
            }
        }
    }

    diagnostics = robustness_service.collect_scenario_diagnostics(payload)

    assert diagnostics["actionableWarningCount"] == 1
    assert diagnostics["calendarBoundaryWarningCount"] == 1
    assert diagnostics["assetLifecycleWarningCount"] == 1
    assert diagnostics["flags"] == [
        "actionable availability warning",
        "mixed calendar",
        "unknown symbols",
    ]
    assert diagnostics["diagnosticSummary"]["severityCounts"] == {
        "info": 3,
        "invalidating": 2,
    }
    lifecycle_events = [
        event
        for event in diagnostics["diagnosticEvents"]
        if event["scope"] == "asset_lifecycle"
    ]
    assert lifecycle_events[0]["severity"] == "info"
    assert lifecycle_events[0]["symbol"] == "BTC-USD"
    representative = diagnostics["representativeDiagnostic"]
    assert representative["severity"] == "invalidating"
    assert representative["category"] == "availability"
    assert representative["symbol"] == "MISSING"


def test_build_robustness_decision_uses_invalidating_events() -> None:
    decision = robustness_service.build_robustness_decision(
        scenario_count=10,
        worst_sharpe=1.0,
        top5_count=10,
        diagnostic_flags=[],
        crypto_sensitivity=0.0,
        cost_sensitivity=0.0,
        diagnostic_events=[
            {
                "kind": "requested_asset_unavailable",
                "category": "availability",
                "severity": "invalidating",
            }
        ],
    )

    assert decision == {
        "decision": "INVALID",
        "reasons": ["actionable availability warning"],
    }


def test_build_robustness_decision_does_not_invalidate_lifecycle_events() -> None:
    decision = robustness_service.build_robustness_decision(
        scenario_count=10,
        worst_sharpe=-0.1,
        top5_count=10,
        diagnostic_flags=[],
        crypto_sensitivity=0.0,
        cost_sensitivity=0.0,
        diagnostic_events=[
            {
                "kind": "asset_available_after_aligned_start",
                "category": "availability",
                "severity": "info",
                "scope": "asset_lifecycle",
            }
        ],
    )

    assert decision == {
        "decision": "FAIL",
        "reasons": ["worst Sharpe below 0.0"],
    }


def test_build_scenario_comparison_applies_period_cost_and_risk_controls() -> None:
    comparison = copy.deepcopy(DEFAULT_COMPARISON_SPEC)
    scenario = {
        "key": "test",
        "period": "2019-2021",
        "periodKey": "2019_2021",
        "startDate": "2019-01-01",
        "endDate": "2021-12-31",
        "walkForwardStartYear": 2020,
        "walkForwardEndYear": 2020,
        "universe": "crypto_included",
        "costMultiplier": 2.0,
        "maxInvestmentRatio": 1.0,
        "maxWeight": 0.35,
        "maxWeightPct": 35.0,
    }

    scenario_comparison = robustness_service.build_scenario_comparison(
        comparison,
        scenario=scenario,
        apply_universe_variant=lambda comparison_spec, _universe: comparison_spec,
    )

    assert scenario_comparison.run_spec.market_slice.start_date == "2019-01-01"
    assert scenario_comparison.run_spec.market_slice.end_date == "2021-12-31"
    assert (
        scenario_comparison.run_spec.execution_assumptions.cost_model.parameters["commissionPct"]
        == comparison.run_spec.execution_assumptions.cost_model.parameters["commissionPct"] * 2
    )
    assert scenario_comparison.candidate_strategies[0].risk_controls.max_weight == 0.35
    assert scenario_comparison.reference_strategies[0].risk_controls.max_weight == 0.35
    assert scenario_comparison.condition_variants[0].cost_multiplier == 2.0
