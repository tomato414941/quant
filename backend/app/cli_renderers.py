from __future__ import annotations

import json
from pathlib import Path

from app import robustness_service
from app.diagnostics_service import summarize_diagnostic_events


def format_percent(value: float) -> str:
    return f"{value:+.2f}%"


def format_optional_percent(value: object) -> str:
    if value is None:
        return "n/a"
    return format_percent(float(value))


def format_optional_decimal(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def format_optional_signed_decimal(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.3f}"


def format_optional_ratio_percent(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def format_percent_distribution(distribution: dict | None) -> str:
    if not distribution or int(distribution.get("count", 0)) <= 0:
        return "n/a"
    return (
        f"min {format_optional_percent(distribution.get('minimum'))} / "
        f"med {format_optional_percent(distribution.get('median'))} / "
        f"max {format_optional_percent(distribution.get('maximum'))}"
    )


def format_decimal_distribution(distribution: dict | None) -> str:
    if not distribution or int(distribution.get("count", 0)) <= 0:
        return "n/a"
    return (
        f"min {format_optional_decimal(distribution.get('minimum'))} / "
        f"med {format_optional_decimal(distribution.get('median'))} / "
        f"max {format_optional_decimal(distribution.get('maximum'))}"
    )


def format_count_map(counts: dict, *, limit: int = 3) -> str:
    if not counts:
        return "none"
    pairs = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[:limit]
    return ", ".join(f"{key}={value}" for key, value in pairs)


def render_availability_diagnostics(lines: list[str], diagnostics: dict | None) -> None:
    if not diagnostics:
        return
    actionable_warnings = diagnostics.get("actionableWarnings") or []
    calendar_boundary_warning_count = int(diagnostics.get("calendarBoundaryWarningCount") or 0)
    asset_lifecycle_warning_count = int(diagnostics.get("assetLifecycleWarningCount") or 0)
    if actionable_warnings:
        lines.append("Warnings:")
        for warning in actionable_warnings:
            lines.append(f"- {warning['message']}")
    if calendar_boundary_warning_count:
        if actionable_warnings:
            lines.append("")
        lines.append(
            "Calendar boundary differences: "
            f"{calendar_boundary_warning_count} classified as non-actionable."
        )
    if asset_lifecycle_warning_count:
        if actionable_warnings or calendar_boundary_warning_count:
            lines.append("")
        lines.append(
            "Asset lifecycle differences: "
            f"{asset_lifecycle_warning_count} classified as non-actionable."
        )
    if actionable_warnings or calendar_boundary_warning_count or asset_lifecycle_warning_count:
        lines.append("")


def render_instrument_diagnostics(lines: list[str], diagnostics: dict | None) -> None:
    if not diagnostics or not diagnostics.get("mixedMarketCalendar"):
        return
    calendars = ", ".join((diagnostics.get("marketCalendars") or {}).keys())
    lines.append(f"Calendar diagnostics: mixed market calendars {calendars}.")
    lines.append("")


def portfolio_segment_summary(run: dict, segment: str) -> dict:
    return run["splitAnalysis"][segment]["portfolio"]


def sort_candidate_runs(candidate_runs: list[dict]) -> list[dict]:
    return sorted(
        candidate_runs,
        key=lambda run: (
            -portfolio_segment_summary(run, "test")["sharpeRatio"],
            -portfolio_segment_summary(run, "test")["totalReturnPct"],
            portfolio_segment_summary(run, "test")["maxDrawdownPct"],
        ),
    )

def render_benchmark_decomposition(payload: dict, *, top: int) -> str:
    walk_forward = payload["walkForward"]
    benchmarks = payload["benchmarks"]
    lines = [
        "Benchmark decomposition",
        f"Baseline: {payload['baselineKey']}",
        (
            "Walk-forward: "
            f"{walk_forward['startYear']}-{walk_forward['endYear']} "
            f"({walk_forward['windowCount']} windows)"
        ),
        (
            "Run store: "
            f"cached={payload['runStoreSummary']['cachedRunCount']} "
            f"computed={payload['runStoreSummary']['computedRunCount']}"
        ),
        "",
        f"Top {min(top, len(benchmarks))} benchmarks by walk-forward performance:",
    ]
    for index, benchmark in enumerate(benchmarks[:top], start=1):
        summary = benchmark["summary"]
        delta = benchmark.get("deltaVsBaseline", {})
        lines.append(f"{index}. {benchmark['label']} [{benchmark['strategyKey']}]")
        lines.append(
            "   "
            f"Avg Sharpe {summary['averageSharpeRatio']:.3f} | "
            f"Min Sharpe {summary['minimumSharpeRatio']:.3f} | "
            f"Avg Return {format_percent(summary['averageTotalReturnPct'])} | "
            f"Avg MDD {format_percent(summary['averageMaxDrawdownPct'])} | "
            f"Avg Turnover {format_percent(summary['averageTurnoverPct'])}"
        )
        if delta:
            lines.append(
                "   "
                "Delta vs ref-fu-eq-cash-15: "
                f"Sharpe {delta['averageSharpeRatio']:+.3f} | "
                f"Min Sharpe {delta['minimumSharpeRatio']:+.3f} | "
                f"Return {format_percent(delta['averageTotalReturnPct'])} | "
                f"Turnover {format_percent(delta['averageTurnoverPct'])}"
            )
        worst_window = benchmark.get("worstWindow")
        if worst_window:
            test = worst_window["test"]
            lines.append(
                "   "
                f"Worst window {worst_window['year']} | "
                f"Sharpe {test['sharpeRatio']:.3f} | "
                f"Return {format_percent(test['totalReturnPct'])} | "
                f"MDD {format_percent(test['maxDrawdownPct'])}"
            )
    return "\n".join(lines)


def render_comparison_summary(payload: dict, *, top: int) -> str:
    comparison = payload["comparison"]
    candidate_runs = sort_candidate_runs(payload["candidateRuns"])
    lines = [
        f"Comparison: {comparison['title']} ({comparison['comparisonId']})",
        f"Question: {comparison['question']}",
        (
            "Selection policy: "
            f"{comparison['selectionPolicy']['primaryMetric']} / "
            f"{comparison['selectionPolicy']['secondaryMetric']} / "
            f"{comparison['selectionPolicy']['tertiaryMetric']}"
        ),
        (
            "Run store: "
            f"cached={payload['runStoreSummary']['cachedRunCount']} "
            f"computed={payload['runStoreSummary']['computedRunCount']}"
        ),
        (
            "Counts: "
            f"candidate={len(payload['candidateRuns'])} "
            f"reference={len(payload['referenceRuns'])} "
            f"predictor={len(payload['predictorRuns'])}"
        ),
        "",
    ]

    evaluation = comparison.get("runSpec", {}).get("evaluation", {})
    render_instrument_diagnostics(lines, evaluation.get("instrumentDiagnostics"))
    render_availability_diagnostics(lines, evaluation.get("availabilityDiagnostics"))

    lines.append(f"Top {min(top, len(candidate_runs))} candidate runs by test performance:")

    for index, run in enumerate(candidate_runs[:top], start=1):
        test_summary = portfolio_segment_summary(run, "test")
        train_summary = portfolio_segment_summary(run, "train")
        overall_summary = run["summary"]
        lines.append(f"{index}. {run['strategy']['label']} [{run['key']}]")
        lines.append(
            "   "
            f"Test Sharpe {test_summary['sharpeRatio']:.3f} | "
            f"Return {format_percent(test_summary['totalReturnPct'])} | "
            f"MDD {format_percent(test_summary['maxDrawdownPct'])} | "
            f"Turnover {format_percent(test_summary['turnoverPct'])}"
        )
        lines.append(
            "   "
            f"Train Sharpe {train_summary['sharpeRatio']:.3f} | "
            f"Summary Sharpe {overall_summary['sharpeRatio']:.3f}"
        )

    return "\n".join(lines)



def format_count_range(min_count: int, max_count: int) -> str:
    if min_count == max_count:
        return str(min_count)
    return f"{min_count}-{max_count}"


def render_walk_forward_summary(payload: dict, *, top: int) -> str:
    comparison = payload["comparison"]
    walk_forward = payload["walkForward"]
    candidate_results = payload["candidateResults"]
    evaluation = comparison.get("runSpec", {}).get("evaluation", {})
    lines = [
        f"Comparison: {comparison['title']} ({comparison['comparisonId']})",
        f"Question: {comparison['question']}",
        (
            "Walk-forward: "
            f"{walk_forward['startYear']}-{walk_forward['endYear']} "
            f"({walk_forward['windowCount']} windows)"
        ),
        (
            "Ranking policy: average test Sharpe / minimum test Sharpe / "
            "average test return / average test max drawdown"
        ),
        (
            "Run store: "
            f"cached={payload['runStoreSummary']['cachedRunCount']} "
            f"computed={payload['runStoreSummary']['computedRunCount']}"
        ),
        "",
    ]

    availability_policy = evaluation.get("availabilityPolicy")
    if availability_policy:
        lines.append(
            "Availability policy: "
            f"minHistoryBars={availability_policy.get('minHistoryBars')} "
            f"maxStaleBars={availability_policy.get('maxStaleBars')} "
            f"delistedAssetPolicy={availability_policy.get('delistedAssetPolicy')}"
        )
    availability_summary = evaluation.get("availabilitySummary")
    if availability_summary:
        min_available = int(availability_summary.get("minAvailableAssetCount", 0))
        max_available = int(availability_summary.get("maxAvailableAssetCount", 0))
        lines.append(
            "Market availability: "
            f"available assets {format_count_range(min_available, max_available)}"
        )
    if availability_policy or availability_summary:
        lines.append("")

    render_instrument_diagnostics(lines, evaluation.get("instrumentDiagnostics"))
    render_availability_diagnostics(lines, evaluation.get("availabilityDiagnostics"))

    lines.append(f"Top {min(top, len(candidate_results))} candidate strategies by walk-forward test performance:")
    for index, result in enumerate(candidate_results[:top], start=1):
        lines.append(f"{index}. {result['strategyLabel']} [{result['strategyKey']}]")
        lines.append(
            "   "
            f"Avg Sharpe {result['averageSharpeRatio']:.3f} | "
            f"Min Sharpe {result['minimumSharpeRatio']:.3f} | "
            f"Avg Return {format_percent(result['averageTotalReturnPct'])} | "
            f"Avg MDD {format_percent(result['averageMaxDrawdownPct'])} | "
            f"Positive Years {result['positiveReturnWindowCount']}/{result['windowCount']}"
        )
        lines.append(
            "   "
            f"Avg Turnover {format_percent(result['averageTurnoverPct'])}"
        )
        if "minTestEligibleAssetCount" in result:
            lines.append(
                "   "
                "Test eligible assets "
                f"{format_count_range(int(result['minTestEligibleAssetCount']), int(result['maxTestEligibleAssetCount']))} | "
                f"Newly eligible {result['testNewlyEligibleAssetCount']} | "
                f"Removed {result['testRemovedAssetCount']}"
            )

    return "\n".join(lines)


def render_edge_attribution(payload: dict) -> str:
    walk_forward = payload["walkForward"]
    lines = [
        f"Edge attribution: {payload['strategyLabel']} [{payload['strategyKey']}]",
        f"Period: {payload['period']} | universe {payload['universe']}",
        (
            "Walk-forward: "
            f"{walk_forward['startYear']}-{walk_forward['endYear']} "
            f"({walk_forward['windowCount']} windows)"
        ),
        f"Baseline: {payload['baselineKey']}",
        f"Decision policy: {payload.get('decisionPolicyKey')}",
        (
            "Run store: "
            f"cached={payload['runStoreSummary']['cachedRunCount']} "
            f"computed={payload['runStoreSummary']['computedRunCount']}"
        ),
        "",
        "Components:",
    ]
    for component in payload.get("components") or []:
        summary = component["summary"]
        delta = component.get("deltaVsBaseline") or {}
        lines.append(f"- {component['label']} [{component['componentKey']}]")
        lines.append(
            "   "
            f"Avg Sharpe {summary['averageSharpeRatio']:.3f} | "
            f"Min Sharpe {summary['minimumSharpeRatio']:.3f} | "
            f"Avg Return {format_percent(summary['averageTotalReturnPct'])} | "
            f"Avg CAGR {format_percent(summary['averageCagrPct'])} | "
            f"Avg MDD {format_percent(summary['averageMaxDrawdownPct'])} | "
            f"Avg Turnover {format_percent(summary['averageTurnoverPct'])}"
        )
        if delta:
            lines.append(
                "   "
                "Delta vs universe: "
                f"Sharpe {delta['averageSharpeRatio']:+.3f} | "
                f"Return {format_percent(delta['averageTotalReturnPct'])} | "
                f"CAGR {format_percent(delta['averageCagrPct'])} | "
                f"Turnover {format_percent(delta['averageTurnoverPct'])}"
            )
        allocation = component.get("allocationSummary") or {}
        decisions = component.get("executionDecisionSummary") or {}
        trace = component.get("executionTraceSummary") or {}
        if allocation:
            lines.append(
                "   "
                f"Holdings avg {allocation['averageHoldingCount']:.1f} | "
                f"Top5 weight {format_percent(allocation['averageTop5WeightPct'])} | "
                f"Max asset {format_percent(allocation['maximumSingleAssetWeightPct'])} | "
                f"Cash {format_percent(allocation['averageCashWeightPct'])}"
            )
        if decisions.get("decisionCount"):
            lines.append(
                "   "
                f"Decisions {decisions['decisionCount']} | "
                f"avg decision cost {format_optional_percent(decisions.get('averageEstimatedCostPct'))}"
            )
        if trace.get("eventCount"):
            lines.append(
                "   "
                f"Trace events {trace['eventCount']} | "
                f"decisions {trace['decisionEventCount']} | "
                f"rebalances {trace['rebalanceEventCount']} | "
                f"forced changes {trace['forcedUniverseChangeEventCount']} | "
                f"no-trade {trace['noTradeCount']} | "
                f"avg trace cost {format_optional_percent(trace.get('averageEstimatedCostPct'))}"
            )
    omitted_components = [
        entry
        for entry in payload.get("componentApplicability") or []
        if not entry.get("applicable", True)
    ]
    if omitted_components:
        lines.extend(["", "Omitted components:"])
        for entry in omitted_components:
            lines.append(
                f"- {entry['label']} [{entry['componentKey']}] | reason {entry.get('omittedReason')}"
            )
    diagnosis = payload.get("diagnosis") or {}
    if diagnosis:
        selection_baseline_label = diagnosis.get("selectionBaselineLabel") or "Selection baseline"
        lines.extend([
            "",
            f"Diagnosis: {diagnosis['primaryFinding']}",
            (
                "Stage presence: "
                f"pure selection={'yes' if diagnosis.get('hasPureSelectionStage') else 'no'} | "
                f"tilt={'yes' if diagnosis.get('hasTiltStage') else 'no'} | "
                f"no-trade path={'yes' if diagnosis.get('hasDecisionNoTradePath') else 'no'}"
            ),
            (
                "Pure selection effect: "
                f"Return {format_optional_percent(diagnosis.get('pureSelectionEffectReturnPct'))} | "
                f"Sharpe {format_optional_signed_decimal(diagnosis.get('pureSelectionEffectSharpe'))}"
            ),
            (
                "Tilt effect: "
                f"Return {format_optional_percent(diagnosis.get('tiltEffectReturnPct'))} | "
                f"Sharpe {format_optional_signed_decimal(diagnosis.get('tiltEffectSharpe'))}"
            ),
            (
                "Portfolio model effect: "
                f"Return {format_optional_percent(diagnosis.get('portfolioModelEffectReturnPct'))} | "
                f"Sharpe {format_optional_signed_decimal(diagnosis.get('portfolioModelEffectSharpe'))}"
            ),
            (
                f"Full vs {selection_baseline_label}: "
                f"Return {format_optional_percent(diagnosis.get('fullVsSelectionBaselineEffectReturnPct'))} | "
                f"Sharpe {format_optional_signed_decimal(diagnosis.get('fullVsSelectionBaselineEffectSharpe'))} | "
                f"Turnover {format_optional_percent(diagnosis.get('turnoverIncreasePct'))} | "
                f"Cost {format_optional_percent(diagnosis.get('estimatedCostIncreasePct'))}"
            ),
            f"Likely causes: {', '.join(diagnosis.get('likelyCauses') or [])}",
        ])
    effect = payload.get("effectSummary") or {}
    full_vs_universe = effect.get("fullEffectVsUniverse")
    full_vs_cash = effect.get("fullEffectVsCash")
    if full_vs_universe and full_vs_cash:
        lines.extend([
            "",
            (
                "Full effect vs universe: "
                f"Sharpe {full_vs_universe['averageSharpeRatio']:+.3f} | "
                f"Return {format_percent(full_vs_universe['averageTotalReturnPct'])} | "
                f"CAGR {format_percent(full_vs_universe['averageCagrPct'])}"
            ),
            (
                "Full effect vs cash: "
                f"Sharpe {full_vs_cash['averageSharpeRatio']:+.3f} | "
                f"Return {format_percent(full_vs_cash['averageTotalReturnPct'])} | "
                f"CAGR {format_percent(full_vs_cash['averageCagrPct'])}"
            ),
        ])
    return "\n".join(lines)


def render_signal_diagnostics(payload: dict) -> str:
    lines = [
        (
            f"Signal diagnostics: {payload['strategyCount']} strategies | "
            f"period {payload['period']} | universe {payload['universe']} | "
            f"horizons {', '.join(payload.get('horizons') or [])} | "
            f"observations {payload.get('observationSchedule')}"
        )
    ]
    for result in payload.get("strategyResults") or []:
        lines.append(f"{result['strategyLabel']} [{result['strategyKey']}]")
        diagnosis = result.get("diagnosis") or {}
        if diagnosis:
            lines.append(
                "   "
                f"diagnosis {diagnosis.get('primaryFinding')} | "
                f"flags {', '.join(diagnosis.get('flags') or [])} | "
                f"positive horizons {diagnosis.get('positiveHorizonCount')}/{diagnosis.get('validHorizonCount')} | "
                f"positive years {diagnosis.get('positiveYearCount')}/{diagnosis.get('validYearCount')}"
            )
        for horizon_result in result.get("horizonResults") or []:
            lines.append(
                "   "
                f"{horizon_result['horizon']} | "
                f"samples {horizon_result['sampleCount']} | "
                f"rank IC {format_optional_decimal(horizon_result.get('rankIc'))} | "
                f"spread {format_optional_percent(horizon_result.get('topMinusBottomForwardReturnPct'))} | "
                f"top {format_optional_percent(horizon_result.get('topBucketForwardReturnPct'))} | "
                f"bottom {format_optional_percent(horizon_result.get('bottomBucketForwardReturnPct'))} | "
                f"hit {format_optional_ratio_percent(horizon_result.get('hitRate'))}"
            )
        for horizon in payload.get("horizons") or []:
            yearly_results = [
                item for item in result.get("yearlyResults") or []
                if item.get("horizon") == horizon and item.get("rankIc") is not None
            ]
            if yearly_results:
                positive_year_count = sum(
                    1
                    for item in yearly_results
                    if (
                        item.get("topMinusBottomForwardReturnPct") is not None
                        and float(item["topMinusBottomForwardReturnPct"]) > 0
                    )
                )
                worst_year = min(
                    yearly_results,
                    key=lambda item: float(item.get("topMinusBottomForwardReturnPct") or 0.0),
                )
                lines.append(
                    "   "
                    f"yearly {horizon} | "
                    f"positive spread years {positive_year_count}/{len(yearly_results)} | "
                    f"worst {worst_year['year']} "
                    f"spread {format_optional_percent(worst_year.get('topMinusBottomForwardReturnPct'))}"
                )
            asset_class_results = [
                item for item in result.get("assetClassResults") or []
                if item.get("horizon") == horizon and item.get("rankIc") is not None
            ]
            if asset_class_results:
                parts = [
                    (
                        f"{item['assetClass']} "
                        f"IC {format_optional_decimal(item.get('rankIc'))} "
                        f"spread {format_optional_percent(item.get('topMinusBottomForwardReturnPct'))}"
                    )
                    for item in asset_class_results
                ]
                lines.append(f"   asset classes {horizon} | " + "; ".join(parts))
    return "\n".join(lines)


def render_robustness_summary(payload: dict, *, top: int) -> str:
    matrix = payload["matrix"]
    strategy_results = payload["strategyResults"]
    periods = ", ".join(period["label"] for period in matrix["periods"])
    universes = ", ".join(matrix["universes"])
    cost_multipliers = ", ".join(f"x{value:.1f}" for value in matrix["costMultipliers"])
    max_weights = ", ".join(f"{value:.0f}%" for value in matrix["maxWeightPcts"])
    decision_counts = payload.get("decisionSummary", {}).get("counts", {})
    decision_summary = ", ".join(
        f"{decision}={decision_counts.get(decision, 0)}"
        for decision in robustness_service.DECISION_PRIORITY
    )
    lines = [
        f"Robustness summary: {payload['scenarioCount']} scenarios ({payload.get('profile', 'unknown')} profile)",
        f"Decisions: {decision_summary}",
        f"Periods: {periods}",
        f"Universes: {universes}",
        f"Costs: {cost_multipliers} | Max weights: {max_weights}",
        (
            "Ranking policy: decision / worst Sharpe / top-5 stability / "
            "average Sharpe"
        ),
        (
            "Run store: "
            f"cached={payload['runStoreSummary']['cachedRunCount']} "
            f"computed={payload['runStoreSummary']['computedRunCount']}"
        ),
        "",
        f"Top {min(top, len(strategy_results))} strategies by robustness:",
    ]
    for index, result in enumerate(strategy_results[:top], start=1):
        risks = ", ".join(result["diagnosticFlags"]) if result["diagnosticFlags"] else "none"
        reasons = ", ".join(result.get("decisionReasons") or []) or "none"
        lines.append(f"{index}. {result['strategyLabel']} [{result['strategyKey']}]")
        lines.append(
            "   "
            f"Decision {result['decision']} | "
            f"Reasons: {reasons} | "
            f"Avg Sharpe {result['averageSharpeRatio']:.3f} | "
            f"Worst Sharpe {result['worstSharpeRatio']:.3f} | "
            f"Top5 {result['top5ScenarioCount']}/{result['scenarioCount']}"
        )
        lines.append(
            "   "
            f"Avg Return {format_percent(result['averageTotalReturnPct'])} | "
            f"Worst MDD {format_percent(result['worstMaxDrawdownPct'])} | "
            f"Avg Turnover {format_percent(result['averageTurnoverPct'])} | "
            f"Max Turnover {format_percent(result['maxTurnoverPct'])}"
        )
        diversification = result.get("diversificationSummary") or {}
        exposure = result.get("exposureSummary") or {}
        if diversification:
            lines.append(
                "   "
                f"Holdings avg {diversification['averageHoldingCount']:.1f} | "
                f"min {diversification['minimumHoldingCount']} | "
                f"Top5 weight {format_percent(diversification['averageTop5WeightPct'])} | "
                f"Max asset {format_percent(diversification['maximumSingleAssetWeightPct'])}"
            )
        if exposure:
            lines.append(
                "   "
                f"Invested {format_percent(exposure['averageInvestedWeightPct'])} | "
                f"Cash {format_percent(exposure['averageCashWeightPct'])}"
            )
        execution_decisions = result.get("executionDecisionSummary") or {}
        if int(execution_decisions.get("decisionCount", 0)) > 0:
            lines.append(
                "   "
                f"Exec decisions {execution_decisions['decisionCount']} | "
                f"rebalance {execution_decisions['rebalanceCount']} | "
                f"no-trade {execution_decisions['noTradeCount']} | "
                f"avg edge {format_optional_percent(execution_decisions.get('averageEstimatedEdgePct'))} | "
                f"avg cost {format_optional_percent(execution_decisions.get('averageEstimatedCostPct'))}"
            )
            lines.append(
                "   "
                f"Exec reasons {format_count_map(execution_decisions.get('reasonCounts') or {})}"
            )
            lines.append(
                "   "
                f"Exec edge sources {format_count_map(execution_decisions.get('edgeSourceCounts') or {})}"
            )
            lines.append(
                "   "
                "Exec edge "
                f"{format_percent_distribution(execution_decisions.get('estimatedEdgePctDistribution'))} | "
                "cost "
                f"{format_percent_distribution(execution_decisions.get('estimatedCostPctDistribution'))} | "
                "edge-cost "
                f"{format_percent_distribution(execution_decisions.get('estimatedEdgeAfterCostPctDistribution'))}"
            )
            lines.append(
                "   "
                "Exec realized edge "
                f"{format_percent_distribution(execution_decisions.get('realizedEdgePctDistribution'))} | "
                "realized edge-cost "
                f"{format_percent_distribution(execution_decisions.get('realizedEdgeAfterCostPctDistribution'))}"
            )
            lines.append(
                "   "
                "Exec edge hit "
                f"{format_optional_ratio_percent(execution_decisions.get('edgeHitRate'))} | "
                f"corr {format_optional_decimal(execution_decisions.get('estimatedVsRealizedEdgeCorrelation'))} | "
                f"samples {execution_decisions.get('edgeHitSampleCount', 0)}"
            )
            lines.append(
                "   "
                "Exec confidence "
                f"{format_decimal_distribution(execution_decisions.get('confidenceDistribution'))}"
            )
        delta = result.get("deltaVsBaseline")
        if delta:
            lines.append(
                "   "
                f"Delta vs {payload['baselineKey']}: "
                f"Sharpe {delta['averageSharpeRatio']:+.3f} | "
                f"Worst Sharpe {delta['worstSharpeRatio']:+.3f} | "
                f"Return {format_percent(delta['averageTotalReturnPct'])} | "
                f"MDD {format_percent(delta['worstMaxDrawdownPct'])} | "
                f"Turnover {format_percent(delta['averageTurnoverPct'])}"
            )
        worst = result["worstScenario"]
        lines.append(
            "   "
            f"Crypto sensitivity {result['cryptoSensitivity']:+.3f} | "
            f"Cost sensitivity {result['costSensitivity']:+.3f} | "
            f"Risks: {risks}"
        )
        diagnostic_summary = result.get("diagnosticSummary") or {}
        if diagnostic_summary.get("eventCount"):
            severity_counts = diagnostic_summary.get("severityCounts") or {}
            lines.append(
                "   "
                f"Diagnostics {diagnostic_summary['eventCount']} events | "
                f"invalidating={severity_counts.get('invalidating', 0)} "
                f"warning={severity_counts.get('warning', 0)} "
                f"info={severity_counts.get('info', 0)}"
            )
        representative = result.get("representativeDiagnostic")
        if representative:
            lines.append(
                "   "
                "Representative diagnostic "
                f"{representative.get('severity')}/"
                f"{representative.get('category')}/"
                f"{representative.get('kind')}: "
                f"{representative.get('reason')}"
            )
        lines.append(
            "   "
            f"Worst scenario {worst['scenarioKey']} | "
            f"Min Sharpe {worst['minimumSharpeRatio']:.3f} | "
            f"Rank {worst['rank']}"
        )
        worst_window = result.get("worstWindow") or {}
        window = worst_window.get("window") or {}
        test_metrics = window.get("test") or {}
        test_availability = window.get("testAvailability") or {}
        if window:
            eligible_min = test_availability.get("minEligibleAssetCount", "?")
            eligible_max = test_availability.get("maxEligibleAssetCount", "?")
            lines.append(
                "   "
                f"Worst window {worst_window['scenarioKey']} / {window['year']} "
                f"{window['testStartDate']}..{window['testEndDate']} | "
                f"Sharpe {test_metrics['sharpeRatio']:.3f} | "
                f"Return {format_percent(test_metrics['totalReturnPct'])} | "
                f"MDD {format_percent(test_metrics['maxDrawdownPct'])} | "
                f"Eligible {eligible_min}-{eligible_max}"
            )
    return "\n".join(lines)


def render_robustness_progress(event: dict) -> str:
    scenario = event["scenario"]
    prefix = f"scenario {event['scenarioIndex']}/{event['scenarioCount']}"
    if event["kind"] == "scenario_started":
        return f"{prefix} started: {scenario['key']}"
    run_store_summary = event["runStoreSummary"]
    return (
        f"{prefix} done: {scenario['key']} | "
        f"cached={run_store_summary['cachedRunCount']} "
        f"computed={run_store_summary['computedRunCount']} "
        f"elapsed={event['elapsedSeconds']:.3f}s"
    )


def write_json_payload(path: str, payload: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def render_comparison_run_spec(payload: dict) -> str:
    lines = [
        f"Comparison: {payload['title']} ({payload['comparisonId']})",
        (
            "Selection policy: "
            f"{payload['selectionPolicy']['primaryMetric']} / "
            f"{payload['selectionPolicy']['secondaryMetric']} / "
            f"{payload['selectionPolicy']['tertiaryMetric']}"
        ),
        f"Comparison fingerprint: {payload['comparisonFingerprint']}",
        f"Run spec fingerprint: {payload['runSpecFingerprint']}",
        (
            "Strategy counts: "
            f"candidate={payload['candidateStrategyCount']} "
            f"reference={payload['referenceStrategyCount']}"
        ),
        "",
        json.dumps(payload['runSpec'], ensure_ascii=False, indent=2),
    ]
    return "\n".join(lines)


def render_latest_run(payload: dict) -> str:
    record = payload["record"]
    lines = [
        f"Comparison: {payload['comparisonId']}",
        f"Run kind: {payload['runKind'] or '-'}",
    ]
    if record is None:
        lines.append("Record: not found")
        return "\n".join(lines)
    lines.extend(
        [
            f"Run key: {record['runKey']}",
            f"Strategy: {record.get('strategyLabel')}",
            f"Logic version: {record.get('logicVersion')}",
        ]
    )
    return "\n".join(lines)


def render_run_catalog(payload: dict) -> str:
    lines = [
        f"Comparison: {payload['comparisonId']}",
        f"Records: {payload['recordCount']}",
    ]
    for index, record in enumerate(payload["records"], start=1):
        run_spec = record["runSpec"]
        strategy = run_spec["strategy"]
        generation = run_spec.get("generation", {})
        lines.append(
            (
                f"{index}. {record['runKey']} | "
                f"runKind={run_spec['runKind']} | "
                f"strategy={strategy['label']} | "
                f"generation={generation.get('method', '-')}"
            )
        )
    return "\n".join(lines)
