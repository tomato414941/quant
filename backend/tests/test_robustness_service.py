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


def test_build_scenario_strategy_result_preserves_compact_window_drilldown() -> None:
    result = {
        "strategyKey": "stg-test",
        "strategyLabel": "Test Strategy",
        "averageSharpeRatio": 0.3,
        "minimumSharpeRatio": -0.4,
        "averageTotalReturnPct": 4.0,
        "averageMaxDrawdownPct": -8.0,
        "averageTurnoverPct": 12.0,
        "positiveReturnWindowCount": 1,
        "windowCount": 2,
        "windows": [
            {
                "year": 2020,
                "testStartDate": "2020-01-01",
                "testEndDate": "2020-12-31",
                "test": {
                    "sharpeRatio": 0.2,
                    "totalReturnPct": 5.0,
                    "maxDrawdownPct": -6.0,
                    "turnoverPct": 10.0,
                    "cagrPct": 5.0,
                },
                "train": {
                    "sharpeRatio": 0.8,
                    "totalReturnPct": 15.0,
                    "maxDrawdownPct": -4.0,
                    "turnoverPct": 9.0,
                },
                "testAvailability": {
                    "barCount": 252,
                    "minAvailableAssetCount": 4,
                    "maxAvailableAssetCount": 6,
                    "minEligibleAssetCount": 3,
                    "maxEligibleAssetCount": 5,
                    "newlyEligibleAssetCount": 1,
                    "removedAssetCount": 0,
                    "newlyEligibleAssets": ["ETH-USD"],
                    "removedAssets": [],
                },
                "trainAvailability": {
                    "barCount": 500,
                    "minAvailableAssetCount": 4,
                    "maxAvailableAssetCount": 4,
                    "minEligibleAssetCount": 3,
                    "maxEligibleAssetCount": 3,
                    "newlyEligibleAssetCount": 0,
                    "removedAssetCount": 0,
                    "newlyEligibleAssets": [],
                    "removedAssets": [],
                },
            },
            {
                "year": 2021,
                "testStartDate": "2021-01-01",
                "testEndDate": "2021-12-31",
                "test": {
                    "sharpeRatio": -0.4,
                    "totalReturnPct": -7.0,
                    "maxDrawdownPct": -12.0,
                    "turnoverPct": 14.0,
                    "cagrPct": -7.0,
                },
                "train": {
                    "sharpeRatio": 0.1,
                    "totalReturnPct": 2.0,
                    "maxDrawdownPct": -8.0,
                    "turnoverPct": 11.0,
                },
                "testAvailability": {
                    "barCount": 252,
                    "minAvailableAssetCount": 5,
                    "maxAvailableAssetCount": 5,
                    "minEligibleAssetCount": 2,
                    "maxEligibleAssetCount": 4,
                    "newlyEligibleAssetCount": 0,
                    "removedAssetCount": 1,
                    "newlyEligibleAssets": [],
                    "removedAssets": ["AAA"],
                },
                "trainAvailability": {
                    "barCount": 500,
                    "minAvailableAssetCount": 4,
                    "maxAvailableAssetCount": 5,
                    "minEligibleAssetCount": 3,
                    "maxEligibleAssetCount": 4,
                    "newlyEligibleAssetCount": 1,
                    "removedAssetCount": 0,
                    "newlyEligibleAssets": ["BBB"],
                    "removedAssets": [],
                },
            },
        ],
    }

    projected = robustness_service.build_scenario_strategy_result(
        result,
        rank=3,
        diagnostics={"flags": []},
    )

    assert len(projected["windows"]) == 2
    assert projected["windows"][0]["test"] == {
        "sharpeRatio": 0.2,
        "totalReturnPct": 5.0,
        "maxDrawdownPct": -6.0,
        "turnoverPct": 10.0,
    }
    assert "cagrPct" not in projected["windows"][0]["test"]
    assert projected["worstWindow"]["year"] == 2021
    assert projected["worstWindow"]["testAvailability"]["minEligibleAssetCount"] == 2


def test_build_scenario_strategy_result_summarizes_weights() -> None:
    result = {
        "strategyKey": "stg-test",
        "strategyLabel": "Test Strategy",
        "averageSharpeRatio": 1.0,
        "minimumSharpeRatio": 0.5,
        "averageTotalReturnPct": 8.0,
        "averageMaxDrawdownPct": 3.0,
        "averageTurnoverPct": 12.0,
        "positiveReturnWindowCount": 1,
        "windowCount": 1,
        "windows": [
            {
                "year": 2020,
                "testStartDate": "2020-01-01",
                "testEndDate": "2020-12-31",
                "test": {
                    "sharpeRatio": 1.0,
                    "totalReturnPct": 8.0,
                    "maxDrawdownPct": 3.0,
                    "turnoverPct": 12.0,
                },
                "train": {
                    "sharpeRatio": 0.8,
                    "totalReturnPct": 6.0,
                    "maxDrawdownPct": 2.0,
                    "turnoverPct": 10.0,
                },
                "weights": [
                    {"asset": "SPY", "weightPct": 30.0},
                    {"asset": "QQQ", "weightPct": 25.0},
                    {"asset": "TLT", "weightPct": 20.0},
                    {"asset": "CASH", "weightPct": 25.0},
                ],
                "selectedAssets": ["SPY", "QQQ", "TLT"],
                "executionDecisionSummary": {
                    "decisionCount": 2,
                    "rebalanceCount": 1,
                    "noTradeCount": 1,
                    "policyCounts": {"cost_aware_no_trade": 2},
                    "reasonCounts": {"edge_below_cost": 1, "edge_after_cost": 1},
                    "edgeSourceCounts": {"signal_return_proxy": 2},
                    "averageTurnoverPct": 12.0,
                    "averageEstimatedCostPct": 0.2,
                    "averageEstimatedEdgePct": 0.5,
                    "averageConfidence": 0.7,
                    "estimatedEdgePctDistribution": {
                        "count": 2,
                        "minimum": 0.1,
                        "median": 0.5,
                        "maximum": 0.9,
                    },
                    "estimatedCostPctDistribution": {
                        "count": 2,
                        "minimum": 0.2,
                        "median": 0.3,
                        "maximum": 0.4,
                    },
                    "estimatedEdgeAfterCostPctDistribution": {
                        "count": 2,
                        "minimum": -0.1,
                        "median": 0.2,
                        "maximum": 0.5,
                    },
                    "confidenceDistribution": {
                        "count": 2,
                        "minimum": 0.6,
                        "median": 0.7,
                        "maximum": 0.8,
                    },
                },
                "testAvailability": {},
                "trainAvailability": {},
            }
        ],
        "executionDecisionSummary": {
            "decisionCount": 2,
            "rebalanceCount": 1,
            "noTradeCount": 1,
            "policyCounts": {"cost_aware_no_trade": 2},
            "reasonCounts": {"edge_below_cost": 1, "edge_after_cost": 1},
            "edgeSourceCounts": {"signal_return_proxy": 2},
            "averageTurnoverPct": 12.0,
            "averageEstimatedCostPct": 0.2,
            "averageEstimatedEdgePct": 0.5,
            "averageConfidence": 0.7,
            "estimatedEdgePctDistribution": {
                "count": 2,
                "minimum": 0.1,
                "median": 0.5,
                "maximum": 0.9,
            },
            "estimatedCostPctDistribution": {
                "count": 2,
                "minimum": 0.2,
                "median": 0.3,
                "maximum": 0.4,
            },
            "estimatedEdgeAfterCostPctDistribution": {
                "count": 2,
                "minimum": -0.1,
                "median": 0.2,
                "maximum": 0.5,
            },
            "confidenceDistribution": {
                "count": 2,
                "minimum": 0.6,
                "median": 0.7,
                "maximum": 0.8,
            },
        },
    }

    projected = robustness_service.build_scenario_strategy_result(
        result,
        rank=1,
        diagnostics={"flags": []},
    )

    assert projected["diversificationSummary"]["averageHoldingCount"] == 3.0
    assert projected["diversificationSummary"]["averageTop5WeightPct"] == 75.0
    assert projected["exposureSummary"]["averageCashWeightPct"] == 25.0
    assert projected["executionDecisionSummary"]["noTradeCount"] == 1
    assert projected["executionDecisionSummary"]["reasonCounts"]["edge_below_cost"] == 1
    assert projected["executionDecisionSummary"]["edgeSourceCounts"] == {"signal_return_proxy": 2}
    assert projected["executionDecisionSummary"]["estimatedEdgeAfterCostPctDistribution"]["median"] == 0.2
    assert projected["windows"][0]["weights"][0] == {"asset": "SPY", "weightPct": 30.0}
    assert projected["windows"][0]["executionDecisionSummary"]["decisionCount"] == 2
    assert projected["windows"][0]["executionDecisionSummary"]["confidenceDistribution"]["count"] == 2
    assert projected["windows"][0]["executionDecisionSummary"]["edgeSourceCounts"] == {"signal_return_proxy": 2}


def test_summarize_strategy_robustness_aggregates_execution_decisions() -> None:
    group = {
        "strategyKey": "stg-test",
        "strategyLabel": "Test Strategy",
        "scenarioResults": [
            {
                "rank": 1,
                "averageSharpeRatio": 1.0,
                "minimumSharpeRatio": 0.8,
                "averageTotalReturnPct": 10.0,
                "averageMaxDrawdownPct": -5.0,
                "averageTurnoverPct": 4.0,
                "positiveReturnWindowCount": 1,
                "windowCount": 1,
                "diagnostics": {"flags": [], "diagnosticEvents": []},
                "scenario": {
                    "key": "scenario-a",
                    "period": "2020-2021",
                    "universe": "crypto_included",
                    "costMultiplier": 1.0,
                    "maxWeightPct": 35.0,
                },
                "diversificationSummary": robustness_service.empty_diversification_summary(),
                "exposureSummary": robustness_service.empty_exposure_summary(),
                "executionDecisionSummary": {
                    "decisionCount": 2,
                    "rebalanceCount": 1,
                    "noTradeCount": 1,
                    "policyCounts": {"cost_aware_no_trade": 2},
                    "reasonCounts": {"edge_below_cost": 1, "edge_after_cost": 1},
                    "edgeSourceCounts": {"signal_return_proxy": 2},
                    "averageTurnoverPct": 10.0,
                    "averageEstimatedCostPct": 0.2,
                    "averageEstimatedEdgePct": 0.6,
                    "averageConfidence": 0.8,
                    "estimatedEdgePctDistribution": {
                        "count": 2,
                        "minimum": 0.1,
                        "median": 0.5,
                        "maximum": 0.9,
                    },
                    "estimatedCostPctDistribution": {
                        "count": 2,
                        "minimum": 0.2,
                        "median": 0.3,
                        "maximum": 0.4,
                    },
                    "estimatedEdgeAfterCostPctDistribution": {
                        "count": 2,
                        "minimum": -0.1,
                        "median": 0.2,
                        "maximum": 0.5,
                    },
                    "confidenceDistribution": {
                        "count": 2,
                        "minimum": 0.7,
                        "median": 0.8,
                        "maximum": 0.9,
                    },
                },
                "windows": [
                    {
                        "year": 2020,
                        "test": {
                            "sharpeRatio": 0.8,
                            "totalReturnPct": 10.0,
                            "maxDrawdownPct": -5.0,
                        },
                    }
                ],
            },
            {
                "rank": 2,
                "averageSharpeRatio": 0.6,
                "minimumSharpeRatio": 0.2,
                "averageTotalReturnPct": 4.0,
                "averageMaxDrawdownPct": -8.0,
                "averageTurnoverPct": 6.0,
                "positiveReturnWindowCount": 1,
                "windowCount": 1,
                "diagnostics": {"flags": [], "diagnosticEvents": []},
                "scenario": {
                    "key": "scenario-b",
                    "period": "2020-2021",
                    "universe": "no_crypto",
                    "costMultiplier": 3.0,
                    "maxWeightPct": 35.0,
                },
                "diversificationSummary": robustness_service.empty_diversification_summary(),
                "exposureSummary": robustness_service.empty_exposure_summary(),
                "executionDecisionSummary": {
                    "decisionCount": 1,
                    "rebalanceCount": 0,
                    "noTradeCount": 1,
                    "policyCounts": {"cost_aware_no_trade": 1},
                    "reasonCounts": {"edge_below_cost": 1},
                    "edgeSourceCounts": {"signal_return_proxy": 1},
                    "averageTurnoverPct": 4.0,
                    "averageEstimatedCostPct": 0.1,
                    "averageEstimatedEdgePct": 0.3,
                    "averageConfidence": 0.5,
                    "estimatedEdgePctDistribution": {
                        "count": 1,
                        "minimum": 0.3,
                        "median": 0.3,
                        "maximum": 0.3,
                    },
                    "estimatedCostPctDistribution": {
                        "count": 1,
                        "minimum": 0.1,
                        "median": 0.1,
                        "maximum": 0.1,
                    },
                    "estimatedEdgeAfterCostPctDistribution": {
                        "count": 1,
                        "minimum": 0.2,
                        "median": 0.2,
                        "maximum": 0.2,
                    },
                    "confidenceDistribution": {
                        "count": 1,
                        "minimum": 0.5,
                        "median": 0.5,
                        "maximum": 0.5,
                    },
                },
                "windows": [
                    {
                        "year": 2021,
                        "test": {
                            "sharpeRatio": 0.2,
                            "totalReturnPct": 4.0,
                            "maxDrawdownPct": -8.0,
                        },
                    }
                ],
            },
        ],
    }

    summary = robustness_service.summarize_strategy_robustness(group, scenario_count=2)

    execution_summary = summary["executionDecisionSummary"]
    assert execution_summary["decisionCount"] == 3
    assert execution_summary["noTradeCount"] == 2
    assert execution_summary["reasonCounts"] == {"edge_below_cost": 2, "edge_after_cost": 1}
    assert execution_summary["edgeSourceCounts"] == {"signal_return_proxy": 3}
    assert execution_summary["averageEstimatedEdgePct"] == 0.5
    assert execution_summary["estimatedEdgePctDistribution"] == {
        "count": 3,
        "minimum": 0.1,
        "median": 0.433333,
        "maximum": 0.9,
    }
    assert execution_summary["estimatedEdgeAfterCostPctDistribution"] == {
        "count": 3,
        "minimum": -0.1,
        "median": 0.2,
        "maximum": 0.5,
    }


def test_attach_delta_vs_baseline_reports_zero_for_baseline() -> None:
    baseline = {
        "strategyKey": "ref-fu-eq-cash",
        "averageSharpeRatio": 1.2,
        "worstSharpeRatio": 0.8,
        "averageTotalReturnPct": 10.0,
        "worstMaxDrawdownPct": 5.0,
        "averageTurnoverPct": 50.0,
        "maxTurnoverPct": 90.0,
        "cryptoSensitivity": 0.1,
        "costSensitivity": -0.05,
    }

    attached = robustness_service.attach_delta_vs_baseline([baseline], baseline)

    assert attached[0]["deltaVsBaseline"] == {
        "averageSharpeRatio": 0.0,
        "worstSharpeRatio": 0.0,
        "averageTotalReturnPct": 0.0,
        "worstMaxDrawdownPct": 0.0,
        "averageTurnoverPct": 0.0,
        "maxTurnoverPct": 0.0,
        "cryptoSensitivity": 0.0,
        "costSensitivity": 0.0,
    }

def test_build_strategy_worst_window_summary_crosses_scenarios() -> None:
    scenario_a = {
        "key": "scenario-a",
        "period": "2020-2021",
        "universe": "crypto_included",
        "costMultiplier": 1.0,
        "maxWeightPct": 35.0,
    }
    scenario_b = {
        "key": "scenario-b",
        "period": "2020-2021",
        "universe": "no_crypto",
        "costMultiplier": 3.0,
        "maxWeightPct": 25.0,
    }
    results = [
        {
            "rank": 1,
            "scenario": scenario_a,
            "windows": [
                {
                    "year": 2020,
                    "testStartDate": "2020-01-01",
                    "testEndDate": "2020-12-31",
                    "test": {
                        "sharpeRatio": -0.5,
                        "totalReturnPct": -4.0,
                        "maxDrawdownPct": -9.0,
                    },
                }
            ],
        },
        {
            "rank": 2,
            "scenario": scenario_b,
            "windows": [
                {
                    "year": 2021,
                    "testStartDate": "2021-01-01",
                    "testEndDate": "2021-12-31",
                    "test": {
                        "sharpeRatio": -0.5,
                        "totalReturnPct": -6.0,
                        "maxDrawdownPct": -8.0,
                    },
                }
            ],
        },
    ]

    worst = robustness_service.build_strategy_worst_window_summary(results)

    assert worst["scenarioKey"] == "scenario-b"
    assert worst["rank"] == 2
    assert worst["window"]["year"] == 2021


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
