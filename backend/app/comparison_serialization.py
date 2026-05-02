from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd

from app.portfolio import (
    StrategyDefinition,
    build_alignment_policy_spec,
    build_default_availability_policy,
    build_asset_ranking_specs_from_strategy_definitions,
    build_investment_universe_spec,
    build_observation_spec,
    build_portfolio_model_spec,
    build_portfolio_state,
    build_risk_controls_spec,
    build_selection_spec,
    build_strategy_definition,
    build_strategy_data_source_spec,
    build_strategy_execution_plan_spec,
    build_strategy_feature_definition_spec,
    build_strategy_signal_spec,
    compute_predictor_panel,
    deserialize_predictor_panel,
    evaluate_asset_ranking_spec,
    evaluate_predictor_spec,
    get_strategy_definition_signal_execution_contexts,
    serialize_asset_ranking_spec,
    serialize_predictor_spec,
    serialize_predictor_panel,
    serialize_portfolio_state,
    serialize_strategy_definition as serialize_canonical_strategy_definition,
)
from app.domain import RunContext
from app.engine import run_strategy_backtest
from app.spec import build_strategy_blueprint_from_definition
from app.predictor_registry import REGISTERED_PREDICTOR_SPECS_BY_KEY
from app.comparison_models import (
    ComparisonSpec,
    ConditionVariant,
    CostModelSpec,
    EvaluationSettings,
    EvaluationSpec,
    ExecutionAssumptionsSpec,
    MarketSliceSpec,
    RunSpec,
    SelectionPolicy,
    scale_cost_model_spec,
)
from app.run_store import FileRunResultStore, RunStoreSummary, build_run_fingerprint, build_run_spec
from app.diagnostics_service import build_evaluation_diagnostic_events
from app.instrument_registry import build_instrument_diagnostics
from app.timeframe_models import (
    DEFAULT_DAILY_TIMEFRAME,
    DEFAULT_MONTHLY_TIMEFRAME,
    DEFAULT_WEEKLY_TIMEFRAME,
    TimeframeSpec,
)

from app.comparison_market_context import (
    collect_comparison_tickers,
    collect_required_market_fields,
    collect_strategy_predictor_specs,
    resolve_timeframe_spec_by_key,
)


def serialize_strategy_definition_payload(strategy_definition: StrategyDefinition) -> dict:
    if not isinstance(strategy_definition, StrategyDefinition):
        raise ValueError("StrategyDefinition is required; evaluator DTO payloads are not supported.")
    return serialize_canonical_strategy_definition(strategy_definition)


def serialize_reproducible_strategy_definition(strategy_definition: StrategyDefinition) -> dict:
    return serialize_strategy_definition_payload(strategy_definition)


def deserialize_portfolio_state(payload: dict[str, object]) -> object:
    rows = payload.get("weights", ())
    if not isinstance(rows, list):
        raise ValueError("Portfolio state must include a weights list.")
    current_weights: dict[str, float] = {}
    cash_weight = 0.0
    for row in rows:
        if not isinstance(row, dict):
            continue
        asset = str(row.get("asset", "")).strip().upper()
        weight_pct = float(row.get("weightPct", 0.0))
        if asset == "CASH":
            cash_weight = weight_pct / 100
            continue
        if asset:
            current_weights[asset] = weight_pct / 100
    return build_portfolio_state(
        current_weights=current_weights,
        cash_weight=cash_weight,
    )


def deserialize_cost_model_spec(payload: dict[str, object]) -> CostModelSpec:
    return CostModelSpec(
        kind=str(payload["kind"]),
        parameters={
            str(key): float(value)
            for key, value in dict(payload.get("parameters", {})).items()
        },
        per_asset_overrides={
            str(asset): {
                str(parameter_key): float(parameter_value)
                for parameter_key, parameter_value in dict(overrides).items()
            }
            for asset, overrides in dict(payload.get("perAssetOverrides", {})).items()
        },
    )


def deserialize_execution_assumptions_spec(
    payload: dict[str, object],
) -> ExecutionAssumptionsSpec:
    cost_model = payload.get("costModel")
    if not isinstance(cost_model, dict):
        raise ValueError("Execution assumptions must include a costModel object.")
    return ExecutionAssumptionsSpec(
        kind=str(payload["kind"]),
        label=str(payload["label"]),
        parameters={
            str(key): value
            for key, value in dict(payload.get("parameters", {})).items()
        },
        cost_model=deserialize_cost_model_spec(cost_model),
    )


def deserialize_evaluation_spec(payload: dict[str, object]) -> EvaluationSpec:
    evaluation_settings = payload.get("evaluationSettings")
    if not isinstance(evaluation_settings, dict):
        raise ValueError("Evaluation payload must include evaluationSettings.")
    return EvaluationSpec(
        evaluation_settings=EvaluationSettings(
            split_ratio=float(evaluation_settings["splitRatioPct"]) / 100,
        )
    )


def deserialize_condition_variant(payload: dict[str, object]) -> ConditionVariant:
    max_weight_pct = payload.get("maxWeightPct")
    cost_multiplier = payload.get("costMultiplier", 1.0)
    return ConditionVariant(
        key=str(payload["key"]),
        label=str(payload["label"]),
        cost_multiplier=float(cost_multiplier),
        max_investment_ratio=float(payload["maxInvestmentPct"]) / 100,
        max_weight=None if max_weight_pct is None else float(max_weight_pct) / 100,
    )


def deserialize_strategy_timeframe(payload: dict[str, object]) -> TimeframeSpec:
    return resolve_timeframe_spec_by_key(str(payload["key"]))


def deserialize_observation_spec_payload(payload: dict[str, object]):
    return build_observation_spec(
        key=str(payload["key"]),
        label=str(payload["label"]),
        tickers=[str(ticker) for ticker in payload.get("tickers", ())],
        fields=[str(field) for field in payload.get("fields", ())],
    )


def deserialize_strategy_data_source_spec_payload(payload: dict[str, object] | None):
    if payload is None:
        return None
    observation_spec = payload.get("observationSpec")
    if not isinstance(observation_spec, dict):
        raise ValueError("Strategy data source must include observationSpec.")
    return build_strategy_data_source_spec(
        key=str(payload["key"]),
        label=str(payload["label"]),
        kind=str(payload["kind"]),
        observation_spec=deserialize_observation_spec_payload(observation_spec),
    )


def deserialize_strategy_feature_definition_spec_payload(payload: dict[str, object] | None):
    if payload is None:
        return None
    return build_strategy_feature_definition_spec(
        key=str(payload["key"]),
        label=str(payload["label"]),
        source_field_keys=[str(key) for key in payload.get("sourceFieldKeys", ())],
        derived_feature_keys=[str(key) for key in payload.get("derivedFeatureKeys", ())],
    )


def deserialize_alignment_policy_spec_payload(payload: dict[str, object] | None):
    if payload is None:
        return None
    return build_alignment_policy_spec(
        key=str(payload["key"]),
        label=str(payload["label"]),
        method=str(payload["method"]),
        parameters=dict(payload.get("parameters", {})),
    )


def deserialize_strategy_signal_spec_payload(payload: dict[str, object]):
    observation_spec = payload.get("observationSpec")
    if not isinstance(observation_spec, dict):
        raise ValueError("Strategy signal payload must include observationSpec.")
    return build_strategy_signal_spec(
        key=str(payload["key"]),
        label=str(payload["label"]),
        description=str(payload["description"]),
        observation_spec=deserialize_observation_spec_payload(observation_spec),
        data_timeframe=deserialize_strategy_timeframe(dict(payload["dataTimeframe"])),
        signal_timeframe=deserialize_strategy_timeframe(dict(payload["signalTimeframe"])),
        source_kind=str(payload["sourceKind"]),
        data_source_spec=deserialize_strategy_data_source_spec_payload(payload.get("dataSource")),
        feature_definition_spec=deserialize_strategy_feature_definition_spec_payload(
            payload.get("featureDefinition")
        ),
        alignment_policy=deserialize_alignment_policy_spec_payload(payload.get("alignmentPolicy")),
        weight=float(payload.get("weight", 1.0)),
        signal_parameters=dict(payload.get("signalParameters", {})),
        predictor_key=(
            None
            if payload.get("predictorKey") is None
            else str(payload.get("predictorKey"))
        ),
    )


def deserialize_strategy_definition_payload(payload: dict[str, object]):
    if str(payload.get("kind")) != "strategy_definition":
        raise ValueError(
            "Only strategy_definition payloads are supported. Regenerate comparison-run-spec with the latest CLI."
        )
    components = payload.get("components")
    if not isinstance(components, dict):
        raise ValueError("Strategy definition payload must include components.")
    core = components.get("core")
    optional = components.get("optional")
    if not isinstance(core, dict) or not isinstance(optional, dict):
        raise ValueError("Strategy definition components must include core and optional objects.")
    investment_universe = core.get("investmentUniverse")
    portfolio_model = core.get("portfolioModel")
    execution_plan = core.get("executionPlan")
    signals = optional.get("signals")
    risk_controls = optional.get("riskControls")
    if not isinstance(investment_universe, dict):
        raise ValueError("Strategy definition must include investmentUniverse.")
    if not isinstance(portfolio_model, dict):
        raise ValueError("Strategy definition must include portfolioModel.")
    if not isinstance(execution_plan, dict):
        raise ValueError("Strategy definition must include executionPlan.")
    if not isinstance(signals, list):
        raise ValueError("Strategy definition must include signals.")
    if not isinstance(risk_controls, dict):
        raise ValueError("Strategy definition must include riskControls.")

    return build_strategy_definition(
        strategy_id=str(payload["strategyId"]),
        version=str(payload["version"]),
        label=str(payload["label"]),
        hypothesis=None if payload.get("hypothesis") is None else str(payload["hypothesis"]),
        description=str(payload["description"]),
        investment_universe=build_investment_universe_spec(
            key=str(investment_universe["key"]),
            label=str(investment_universe["label"]),
            tickers=[str(ticker) for ticker in investment_universe.get("tickers", ())],
        ),
        signals=[
            deserialize_strategy_signal_spec_payload(signal_payload)
            for signal_payload in signals
            if isinstance(signal_payload, dict)
        ],
        portfolio_model=build_portfolio_model_spec(
            str(portfolio_model["modelType"]),
            key=str(portfolio_model["key"]),
            label=str(portfolio_model["label"]),
            description=str(portfolio_model["description"]),
        ),
        execution_plan=build_strategy_execution_plan_spec(
            key=str(execution_plan["key"]),
            label=str(execution_plan["label"]),
            decision_schedule=str(execution_plan["decisionSchedule"]),
            rebalance_schedule=str(execution_plan["rebalanceSchedule"]),
        ),
        risk_controls=build_risk_controls_spec(
            max_investment_ratio=float(risk_controls["maxInvestmentPct"]) / 100,
            max_weight=(
                None
                if risk_controls.get("maxWeightPct") is None
                else float(risk_controls["maxWeightPct"]) / 100
            ),
        ),
        extensions={
            str(key): str(value)
            for key, value in dict(payload.get("extensions", {})).items()
        },
    )


def validate_comparison_run_spec_payload(payload: dict[str, object]) -> None:
    run_spec_payload = payload.get("runSpec")
    if not isinstance(run_spec_payload, dict):
        raise ValueError("Comparison run spec payload must include runSpec.")

    expected_run_spec_fingerprint = build_run_fingerprint(run_spec_payload)
    actual_run_spec_fingerprint = str(payload.get("runSpecFingerprint", ""))
    if actual_run_spec_fingerprint != expected_run_spec_fingerprint:
        raise ValueError("runSpecFingerprint does not match the embedded runSpec payload.")

    comparison_fingerprint_payload = {
        "comparisonId": payload.get("comparisonId"),
        "selectionPolicy": payload.get("selectionPolicy"),
        "candidateStrategies": payload.get("candidateStrategies", []),
        "referenceStrategies": payload.get("referenceStrategies", []),
        "conditionVariants": payload.get("conditionVariants", []),
        "runSpec": run_spec_payload,
    }
    expected_comparison_fingerprint = build_run_fingerprint(comparison_fingerprint_payload)
    actual_comparison_fingerprint = str(payload.get("comparisonFingerprint", ""))
    if actual_comparison_fingerprint != expected_comparison_fingerprint:
        raise ValueError("comparisonFingerprint does not match the embedded comparison payload.")


def deserialize_comparison_run_spec_payload(payload: dict[str, object]) -> ComparisonSpec:
    if str(payload.get("kind")) != "comparison_run_spec_payload":
        raise ValueError("Unsupported comparison run spec payload kind.")
    validate_comparison_run_spec_payload(payload)
    run_spec_payload = payload.get("runSpec")
    selection_policy_payload = payload.get("selectionPolicy")
    if not isinstance(run_spec_payload, dict):
        raise ValueError("Comparison run spec payload must include runSpec.")
    if not isinstance(selection_policy_payload, dict):
        raise ValueError("Comparison run spec payload must include selectionPolicy.")
    market_slice_payload = run_spec_payload.get("marketSlice")
    execution_assumptions_payload = run_spec_payload.get("executionAssumptions")
    evaluation_payload = run_spec_payload.get("evaluation")
    portfolio_state_payload = run_spec_payload.get("portfolioState")
    if not isinstance(market_slice_payload, dict):
        raise ValueError("Run spec must include marketSlice.")
    if not isinstance(execution_assumptions_payload, dict):
        raise ValueError("Run spec must include executionAssumptions.")
    if not isinstance(evaluation_payload, dict):
        raise ValueError("Run spec must include evaluation.")
    if not isinstance(portfolio_state_payload, dict):
        raise ValueError("Run spec must include portfolioState.")

    candidate_strategies = [
        deserialize_strategy_definition_payload(strategy_payload)
        for strategy_payload in payload.get("candidateStrategies", ())
        if isinstance(strategy_payload, dict)
    ]
    reference_strategies = [
        deserialize_strategy_definition_payload(strategy_payload)
        for strategy_payload in payload.get("referenceStrategies", ())
        if isinstance(strategy_payload, dict)
    ]

    return ComparisonSpec(
        comparison_id=str(payload["comparisonId"]),
        title=str(payload["title"]),
        question=str(payload["question"]),
        run_spec=RunSpec(
            market_slice=MarketSliceSpec(
                period=str(market_slice_payload["period"]),
                sanity_periods=[str(period) for period in market_slice_payload.get("sanityPeriods", ())],
                start_date=(
                    str(market_slice_payload["startDate"])
                    if market_slice_payload.get("startDate") is not None
                    else None
                ),
                end_date=(
                    str(market_slice_payload["endDate"])
                    if market_slice_payload.get("endDate") is not None
                    else None
                ),
            ),
            portfolio_state=deserialize_portfolio_state(portfolio_state_payload),
            capital_base=float(run_spec_payload["capitalBase"]),
            execution_assumptions=deserialize_execution_assumptions_spec(
                execution_assumptions_payload
            ),
            evaluation=deserialize_evaluation_spec(evaluation_payload),
        ),
        selection_policy=SelectionPolicy(
            primary_metric=str(selection_policy_payload["primaryMetric"]),
            secondary_metric=str(selection_policy_payload["secondaryMetric"]),
            tertiary_metric=str(selection_policy_payload["tertiaryMetric"]),
        ),
        candidate_strategies=candidate_strategies,
        reference_strategies=reference_strategies,
        condition_variants=[
            deserialize_condition_variant(condition_payload)
            for condition_payload in payload.get("conditionVariants", ())
            if isinstance(condition_payload, dict)
        ],
        result_store_dir=str(payload.get("resultStoreDir", "backend/data/run_results")),
    )


def serialize_timeframe(timeframe: TimeframeSpec) -> dict:
    return {
        "key": timeframe.key,
        "label": timeframe.label,
        "barsPerYear": timeframe.bars_per_year,
        "barSeconds": timeframe.bar_seconds,
    }


def serialize_market_slice_run_spec(
    comparison: ComparisonSpec,
    timeframes: list[TimeframeSpec],
    fields: list[str],
) -> dict:
    payload = {
        "period": comparison.run_spec.market_slice.period,
        "sanityPeriods": comparison.run_spec.market_slice.sanity_periods,
        "timeframes": [serialize_timeframe(timeframe) for timeframe in timeframes],
        "fields": fields,
    }
    add_market_slice_dates(payload, comparison)
    return payload


def should_include_market_slice_dates(
    comparison: ComparisonSpec,
    period_override: str | None = None,
) -> bool:
    return period_override is None or period_override == comparison.run_spec.market_slice.period


def add_market_slice_dates(
    payload: dict,
    comparison: ComparisonSpec,
    period_override: str | None = None,
) -> None:
    if not should_include_market_slice_dates(comparison, period_override):
        return
    if comparison.run_spec.market_slice.start_date is not None:
        payload["startDate"] = comparison.run_spec.market_slice.start_date
    if comparison.run_spec.market_slice.end_date is not None:
        payload["endDate"] = comparison.run_spec.market_slice.end_date


def build_market_data_warnings(
    comparison: ComparisonSpec,
    metadata_by_timeframe: dict[str, dict[str, object]],
    *,
    period_override: str | None = None,
) -> list[dict[str, object]]:
    if not should_include_market_slice_dates(comparison, period_override):
        return []

    warnings: list[dict[str, object]] = []
    requested_start = comparison.run_spec.market_slice.start_date
    requested_end = comparison.run_spec.market_slice.end_date
    for timeframe_key, metadata in sorted(metadata_by_timeframe.items()):
        aligned_start = str(metadata.get("aligned_start_date", ""))
        aligned_end = str(metadata.get("aligned_end_date", ""))
        if requested_start is not None and aligned_start and aligned_start > requested_start:
            warnings.append(
                {
                    "kind": "aligned_start_after_requested_start",
                    "timeframe": timeframe_key,
                    "requestedStartDate": requested_start,
                    "alignedStartDate": aligned_start,
                    "message": (
                        f"{timeframe_key} data starts at {aligned_start}, "
                        f"after requested start {requested_start}."
                    ),
                }
            )
        if requested_end is not None and aligned_end and aligned_end < requested_end:
            warnings.append(
                {
                    "kind": "aligned_end_before_requested_end",
                    "timeframe": timeframe_key,
                    "requestedEndDate": requested_end,
                    "alignedEndDate": aligned_end,
                    "message": (
                        f"{timeframe_key} data ends at {aligned_end}, "
                        f"before requested end {requested_end}."
                    ),
                }
            )
    for timeframe_key, metadata in sorted(metadata_by_timeframe.items()):
        aligned_start = metadata.get("aligned_start_date")
        aligned_end = metadata.get("aligned_end_date")
        for asset, availability in (metadata.get("assetAvailability") or {}).items():
            if availability.get("available") is False:
                warnings.append(
                    {
                        "kind": "requested_asset_unavailable",
                        "timeframe": timeframe_key,
                        "asset": asset,
                        "failedReason": availability.get("failedReason"),
                        "message": f"{timeframe_key} data has no usable rows for {asset}.",
                    }
                )
                continue
            first_valid = availability.get("firstValidDate")
            last_valid = availability.get("lastValidDate")
            if aligned_start and first_valid and str(first_valid) > str(aligned_start):
                warnings.append(
                    {
                        "kind": "asset_available_after_aligned_start",
                        "timeframe": timeframe_key,
                        "asset": asset,
                        "alignedStartDate": aligned_start,
                        "firstValidDate": first_valid,
                        "message": f"{asset} becomes available on {first_valid}, after aligned start {aligned_start}.",
                    }
                )
            if aligned_end and last_valid and str(last_valid) < str(aligned_end):
                warnings.append(
                    {
                        "kind": "asset_unavailable_before_aligned_end",
                        "timeframe": timeframe_key,
                        "asset": asset,
                        "alignedEndDate": aligned_end,
                        "lastValidDate": last_valid,
                        "message": f"{asset} last valid data is {last_valid}, before aligned end {aligned_end}.",
                    }
                )
    return warnings


def parse_iso_date(value: object) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def timeframe_calendar_boundary_days(timeframe_key: object) -> int:
    if timeframe_key in {"1wk", "1w"}:
        return 7
    if timeframe_key == "1mo":
        return 31
    return 1


def is_small_calendar_gap(
    warning: dict[str, object],
    availability_policy: dict[str, object] | None,
    left_date_key: str,
    right_date_key: str,
) -> bool:
    left_date = parse_iso_date(warning.get(left_date_key))
    right_date = parse_iso_date(warning.get(right_date_key))
    if left_date is None or right_date is None:
        return False
    max_stale_bars = int((availability_policy or {}).get("maxStaleBars") or 0)
    boundary_days = timeframe_calendar_boundary_days(warning.get("timeframe"))
    max_calendar_gap_days = max(max_stale_bars, boundary_days)
    return abs((right_date - left_date).days) <= max_calendar_gap_days


def classify_availability_warning(
    warning: dict[str, object],
    availability_policy: dict[str, object] | None,
) -> str:
    kind = warning.get("kind")
    if kind == "requested_asset_unavailable":
        return "actionable"
    if kind == "aligned_start_after_requested_start":
        if is_small_calendar_gap(warning, availability_policy, "requestedStartDate", "alignedStartDate"):
            return "calendar_boundary"
        return "actionable"
    if kind == "aligned_end_before_requested_end":
        if is_small_calendar_gap(warning, availability_policy, "requestedEndDate", "alignedEndDate"):
            return "calendar_boundary"
        return "actionable"
    if kind in {"asset_available_after_aligned_start", "asset_unavailable_before_aligned_end"}:
        return "asset_lifecycle"
    return "actionable"


def build_availability_diagnostics(
    warnings: list[dict[str, object]],
    availability_policy: dict[str, object] | None,
) -> dict[str, object]:
    actionable_warnings = []
    calendar_boundary_warnings = []
    asset_lifecycle_warnings = []
    for warning in warnings:
        classification = classify_availability_warning(warning, availability_policy)
        if classification == "calendar_boundary":
            calendar_boundary_warnings.append(warning)
        elif classification == "asset_lifecycle":
            asset_lifecycle_warnings.append(warning)
        else:
            actionable_warnings.append(warning)
    return {
        "warningCount": len(warnings),
        "actionableWarningCount": len(actionable_warnings),
        "calendarBoundaryWarningCount": len(calendar_boundary_warnings),
        "assetLifecycleWarningCount": len(asset_lifecycle_warnings),
        "actionableWarnings": actionable_warnings,
        "calendarBoundaryWarnings": calendar_boundary_warnings,
        "assetLifecycleWarnings": asset_lifecycle_warnings,
    }


def build_availability_summary(metadata_by_timeframe: dict[str, dict[str, object]]) -> dict[str, object]:
    summaries = []
    for timeframe_key, metadata in sorted(metadata_by_timeframe.items()):
        availability = metadata.get("assetAvailability") or {}
        requested_tickers = metadata.get("requested_tickers", [])
        available_tickers = metadata.get("tickers", [])
        requested_count = len(availability) or len(requested_tickers) or len(available_tickers)
        if availability:
            available_count = sum(1 for item in availability.values() if item.get("available") is not False)
        else:
            available_count = len(available_tickers) or requested_count
        summaries.append(
            {
                "timeframe": timeframe_key,
                "requestedAssetCount": requested_count,
                "availableAssetCount": available_count,
                "unavailableAssetCount": max(0, requested_count - available_count),
            }
        )
    if not summaries:
        return {"timeframes": []}
    return {
        "timeframes": summaries,
        "minAvailableAssetCount": min(summary["availableAssetCount"] for summary in summaries),
        "maxAvailableAssetCount": max(summary["availableAssetCount"] for summary in summaries),
    }


def serialize_market_slice_context(
    *,
    comparison: ComparisonSpec,
    timeframe: TimeframeSpec,
    dataset_metadata: dict[str, object],
    fields: list[str],
    period_override: str | None = None,
) -> dict:
    payload = {
        "period": period_override or comparison.run_spec.market_slice.period,
        "sanityPeriods": comparison.run_spec.market_slice.sanity_periods,
        "timeframe": serialize_timeframe(timeframe),
        "fields": fields,
        "source": dataset_metadata["source"],
        "alignedStartDate": dataset_metadata["aligned_start_date"],
        "alignedEndDate": dataset_metadata["aligned_end_date"],
        "rowCount": dataset_metadata["row_count"],
    }
    add_market_slice_dates(payload, comparison, period_override)
    failed_tickers = dataset_metadata.get("failed_tickers")
    if failed_tickers:
        payload["failedTickers"] = failed_tickers
    asset_availability = dataset_metadata.get("assetAvailability")
    if asset_availability:
        payload["assetAvailability"] = asset_availability
    dataset_snapshot = dataset_metadata.get("datasetSnapshot")
    if dataset_snapshot:
        payload["datasetSnapshot"] = dataset_snapshot
    return payload


def serialize_cost_model_spec(cost_model) -> dict:
    return {
        "kind": cost_model.kind,
        "parameters": {
            key: round(value, 3)
            for key, value in cost_model.parameters.items()
        },
        "perAssetOverrides": cost_model.per_asset_overrides,
    }


def serialize_execution_assumptions(comparison: ComparisonSpec) -> dict:
    return {
        "kind": comparison.run_spec.execution_assumptions.kind,
        "label": comparison.run_spec.execution_assumptions.label,
        "parameters": {
            key: value
            for key, value in comparison.run_spec.execution_assumptions.parameters.items()
        },
        "costModel": serialize_cost_model_spec(
            comparison.run_spec.execution_assumptions.cost_model
        ),
    }


def serialize_evaluation_settings(evaluation: EvaluationSpec) -> dict:
    return {
        "splitRatioPct": round(evaluation.evaluation_settings.split_ratio * 100, 1),
    }


def serialize_signal_market_data_context(
    *,
    strategy_id: str,
    signal_spec,
    dataset_metadata: dict[str, object],
    period: str,
    sanity_periods: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    payload = {
        "strategyId": strategy_id,
        "signalKey": signal_spec.key,
        "signalLabel": signal_spec.label,
        "sourceKind": signal_spec.source_kind,
        "fields": list(signal_spec.observation_spec.fields),
        "dataTimeframe": serialize_timeframe(signal_spec.data_timeframe),
        "signalTimeframe": serialize_timeframe(signal_spec.signal_timeframe),
        "period": period,
        "sanityPeriods": sanity_periods,
        "source": dataset_metadata["source"],
        "alignedStartDate": dataset_metadata["aligned_start_date"],
        "alignedEndDate": dataset_metadata["aligned_end_date"],
        "rowCount": dataset_metadata["row_count"],
    }
    if start_date is not None:
        payload["startDate"] = start_date
    if end_date is not None:
        payload["endDate"] = end_date
    if signal_spec.alignment_policy is not None:
        payload["alignmentPolicy"] = {
            "key": signal_spec.alignment_policy.key,
            "label": signal_spec.alignment_policy.label,
            "method": signal_spec.alignment_policy.method,
            "parameters": {
                key: value for key, value in signal_spec.alignment_policy.parameters
            },
        }
    failed_tickers = dataset_metadata.get("failed_tickers")
    if failed_tickers:
        payload["failedTickers"] = failed_tickers
    asset_availability = dataset_metadata.get("assetAvailability")
    if asset_availability:
        payload["assetAvailability"] = asset_availability
    return payload


def serialize_signal_market_data_contexts(
    strategy_definitions: list | None,
    metadata_by_timeframe: dict[str, dict[str, object]],
    *,
    period: str,
    sanity_periods: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict]:
    contexts: list[dict] = []
    for strategy_definition in strategy_definitions or []:
        if not isinstance(strategy_definition, StrategyDefinition):
            continue
        for signal_spec in strategy_definition.signals:
            dataset_metadata = metadata_by_timeframe.get(signal_spec.data_timeframe.key)
            if dataset_metadata is None:
                continue
            contexts.append(
                serialize_signal_market_data_context(
                    strategy_id=strategy_definition.strategy_id,
                    signal_spec=signal_spec,
                    dataset_metadata=dataset_metadata,
                    period=period,
                    sanity_periods=sanity_periods,
                    start_date=start_date,
                    end_date=end_date,
                )
            )
    return contexts


def collect_instrument_diagnostic_symbols(
    metadata_by_timeframe: dict[str, dict[str, object]],
) -> tuple[str, ...]:
    symbols: dict[str, None] = {}
    for metadata in metadata_by_timeframe.values():
        tickers = metadata.get("requested_tickers") or metadata.get("tickers") or []
        for ticker in tickers:
            symbols.setdefault(str(ticker), None)
    return tuple(symbols)


def serialize_evaluation(
    comparison: ComparisonSpec,
    metadata_by_timeframe: dict[str, dict[str, object]],
    timeframes: list[TimeframeSpec],
    *,
    fields: list[str],
    period_override: str | None = None,
    strategy_definitions: list | None = None,
) -> dict:
    availability_policy = build_default_availability_policy()
    warnings = build_market_data_warnings(
        comparison,
        metadata_by_timeframe,
        period_override=period_override,
    )
    availability_diagnostics = build_availability_diagnostics(warnings, availability_policy)
    instrument_diagnostics = build_instrument_diagnostics(
        collect_instrument_diagnostic_symbols(metadata_by_timeframe),
        cost_profile_key=str(
            comparison.run_spec.execution_assumptions.parameters.get(
                "costProfileKey",
                "unknown",
            )
        ),
    )
    payload = {
        "kind": "evaluation_spec",
        "schemaVersion": "v1",
        "availabilityPolicy": availability_policy,
        "availabilitySummary": build_availability_summary(metadata_by_timeframe),
        "availabilityDiagnostics": availability_diagnostics,
        "instrumentDiagnostics": instrument_diagnostics,
        "diagnosticEvents": build_evaluation_diagnostic_events(
            availability_diagnostics=availability_diagnostics,
            instrument_diagnostics=instrument_diagnostics,
        ),
        "marketDataContexts": [
            serialize_market_slice_context(
                comparison=comparison,
                timeframe=timeframe,
                dataset_metadata=metadata_by_timeframe[timeframe.key],
                fields=fields,
                period_override=period_override,
            )
            for timeframe in timeframes
            if timeframe.key in metadata_by_timeframe
        ],
        "evaluationSettings": serialize_evaluation_settings(comparison.run_spec.evaluation),
    }
    signal_market_data_contexts = serialize_signal_market_data_contexts(
        strategy_definitions,
        metadata_by_timeframe,
        period=period_override or comparison.run_spec.market_slice.period,
        sanity_periods=comparison.run_spec.market_slice.sanity_periods,
        start_date=(
            comparison.run_spec.market_slice.start_date
            if should_include_market_slice_dates(comparison, period_override)
            else None
        ),
        end_date=(
            comparison.run_spec.market_slice.end_date
            if should_include_market_slice_dates(comparison, period_override)
            else None
        ),
    )
    if signal_market_data_contexts:
        payload["signalMarketDataContexts"] = signal_market_data_contexts
    if warnings:
        payload["warnings"] = warnings
    return payload


def serialize_run_spec(
    comparison: ComparisonSpec,
    metadata_by_timeframe: dict[str, dict[str, object]],
    timeframes: list[TimeframeSpec],
    *,
    strategy_definitions: list | None = None,
) -> dict:
    strategy_definitions = strategy_definitions or (
        comparison.candidate_strategies + comparison.reference_strategies
    )
    predictor_specs = collect_strategy_predictor_specs(strategy_definitions)
    fields = collect_required_market_fields(
        comparison,
        predictor_specs,
        strategy_definitions=strategy_definitions,
    )
    return {
        "kind": "comparison_run_spec",
        "schemaVersion": "v1",
        "marketSlice": serialize_market_slice_run_spec(comparison, timeframes, fields),
        "portfolioState": serialize_portfolio_state(comparison.run_spec.portfolio_state),
        "capitalBase": round(comparison.run_spec.capital_base, 2),
        "executionAssumptions": serialize_execution_assumptions(comparison),
        "evaluation": serialize_evaluation(
            comparison,
            metadata_by_timeframe,
            timeframes,
            fields=fields,
            strategy_definitions=strategy_definitions,
        ),
    }


def serialize_comparison(
    comparison: ComparisonSpec,
    metadata_by_timeframe: dict[str, dict[str, object]],
    timeframes: list[TimeframeSpec],
    *,
    candidate_strategy_definitions: list | None = None,
    reference_strategy_definitions: list | None = None,
) -> dict:
    strategy_definitions = (candidate_strategy_definitions or comparison.candidate_strategies) + (
        reference_strategy_definitions or comparison.reference_strategies
    )
    comparison_tickers = collect_comparison_tickers(
        comparison,
        strategy_definitions=strategy_definitions,
    )
    return {
        "kind": "strategy_comparison",
        "schemaVersion": "v1",
        "comparisonId": comparison.comparison_id,
        "title": comparison.title,
        "question": comparison.question,
        "selectionPolicy": {
            "primaryMetric": comparison.selection_policy.primary_metric,
            "secondaryMetric": comparison.selection_policy.secondary_metric,
            "tertiaryMetric": comparison.selection_policy.tertiary_metric,
        },
        "marketUniverse": {
            "assetCount": len(comparison_tickers),
            "tickers": comparison_tickers,
        },
        "runSpec": serialize_run_spec(
            comparison,
            metadata_by_timeframe,
            timeframes,
            strategy_definitions=(candidate_strategy_definitions or comparison.candidate_strategies)
            + (reference_strategy_definitions or comparison.reference_strategies),
        ),
        "candidateStrategies": [
            serialize_strategy_definition_payload(strategy_definition)
            for strategy_definition in (candidate_strategy_definitions or comparison.candidate_strategies)
        ],
        "referenceStrategies": [
            serialize_strategy_definition_payload(strategy_definition)
            for strategy_definition in (reference_strategy_definitions or comparison.reference_strategies)
        ],
        "conditionVariants": [
            serialize_condition_variant(condition_variant)
            for condition_variant in comparison.condition_variants
        ],
    }


def serialize_condition_variant(condition_variant: ConditionVariant) -> dict:
    return {
        "key": condition_variant.key,
        "label": condition_variant.label,
        "costMultiplier": round(condition_variant.cost_multiplier, 3),
        "maxInvestmentPct": round(condition_variant.max_investment_ratio * 100, 1),
        "maxWeightPct": round(condition_variant.max_weight * 100, 1)
        if condition_variant.max_weight is not None
        else None,
    }


def sort_predictor_run_records(records: list[dict], *, sort_by: str) -> list[dict]:
    metric_key_by_sort = {
        "test_rank_ic": "testRankIc",
        "overall_rank_ic": "overallRankIc",
        "test_top_minus_bottom": "testTopMinusBottomPct",
        "overall_top_minus_bottom": "overallTopMinusBottomPct",
        "saved_at": "savedAtUtc",
    }
    if sort_by not in metric_key_by_sort:
        raise ValueError(f"Unsupported predictor run sort: {sort_by}")

    metric_key = metric_key_by_sort[sort_by]
    if sort_by == "saved_at":
        return sorted(
            records,
            key=lambda record: (record.get(metric_key) or "", record["runKey"]),
            reverse=True,
        )

    return sorted(
        records,
        key=lambda record: (
            record.get(metric_key) if record.get(metric_key) is not None else float("-inf"),
            record.get("testTopMinusBottomPct")
            if record.get("testTopMinusBottomPct") is not None
            else float("-inf"),
            record.get("overallRankIc") if record.get("overallRankIc") is not None else float("-inf"),
            record["runKey"],
        ),
        reverse=True,
    )


def _metric_sort_value(record: dict, key: str) -> float:
    value = record.get(key)
    if value is None:
        return float("-inf")
    return float(value)


def summarize_predictor_record_groups(
    records: list[dict],
    *,
    group_fields: tuple[str, ...],
) -> list[dict]:
    grouped: dict[tuple[object, ...], list[dict]] = {}
    for record in records:
        group_key = tuple(record.get(field) for field in group_fields)
        grouped.setdefault(group_key, []).append(record)

    summaries: list[dict] = []
    for group_key, group_records in grouped.items():
        best_record = max(
            group_records,
            key=lambda record: (
                _metric_sort_value(record, "testRankIc"),
                _metric_sort_value(record, "testTopMinusBottomPct"),
                _metric_sort_value(record, "overallRankIc"),
                record["runKey"],
            ),
        )
        summary = {
            field: value for field, value in zip(group_fields, group_key, strict=False)
        }
        summary.update(
            {
                "recordCount": len(group_records),
                "bestRunKey": best_record["runKey"],
                "bestPredictorKey": best_record["predictorKey"],
                "bestPredictorLabel": best_record["predictorLabel"],
                "bestTestRankIc": best_record.get("testRankIc"),
                "bestTestTopMinusBottomPct": best_record.get("testTopMinusBottomPct"),
                "bestTestHitRatePct": best_record.get("testHitRatePct"),
                "bestHorizonValue": best_record.get("horizonValue"),
            }
        )
        summaries.append(summary)

    return sorted(
        summaries,
        key=lambda summary: (
            _metric_sort_value(summary, "bestTestRankIc"),
            _metric_sort_value(summary, "bestTestTopMinusBottomPct"),
            summary["bestRunKey"],
        ),
        reverse=True,
    )


def compact_predictor_run_record(record: dict) -> dict:
    run_spec = record["runSpec"]
    result = record["result"]
    fingerprints = run_spec.get("fingerprints", {})
    predictor = run_spec.get("strategy", {}).get("predictor", {})
    signal = predictor.get("signalSpec", {})
    observation = signal.get("observationSpec", {})
    predicted_quantity = predictor.get("predictedQuantitySpec", {})
    target = predictor.get("targetSpec", {})
    output = signal.get("outputSpec", {})
    horizon = target.get("horizonSpec", {})
    feature = predictor.get("featureSpec", {})
    training = predictor.get("trainingSpec", {})
    engine = predictor.get("engineSpec", {})
    decision_use = signal.get("decisionUseSpec", {})
    signal_source = engine.get("signalSourceSpec") or {}
    learner = engine.get("learnerSpec") or {}
    combiner = engine.get("combinerSpec") or {}
    overall = result.get("overall", {})
    test = result.get("test", {})

    return {
        "runKey": record["runKey"],
        "savedAtUtc": record.get("savedAtUtc"),
        "runKind": run_spec.get("runKind"),
        "logicVersion": run_spec.get("logicVersion"),
        "strategyDefinitionFingerprint": fingerprints.get("strategyDefinition"),
        "marketDataFingerprint": fingerprints.get("marketData"),
        "evaluationFingerprint": fingerprints.get("evaluation"),
        "predictorKey": predictor.get("key"),
        "predictorLabel": predictor.get("label"),
        "observationKey": observation.get("key"),
        "observationLabel": observation.get("label"),
        "observationAssetCount": observation.get("assetCount"),
        "observationFields": observation.get("fields"),
        "signalEntityKind": signal.get("entityKind"),
        "signalEntityCount": len(signal.get("entityIdentifiers") or []),
        "decisionUseKind": decision_use.get("useKind"),
        "featureKey": feature.get("key"),
        "signalSourceKind": signal_source.get("signalSourceKind"),
        "signalSourceFeatureKey": signal_source.get("featureKey"),
        "learnerKind": learner.get("learnerKind"),
        "combinerKind": combiner.get("combinerKind"),
        "trainingFitMode": training.get("fitMode"),
        "trainingMinSamples": training.get("minTrainSamples"),
        "timeframe": predictor.get("timeframe", {}).get("key"),
        "targetKey": target.get("key"),
        "predictedQuantityKind": predicted_quantity.get("quantityKind"),
        "targetBaseline": target.get("baseline"),
        "targetTransform": target.get("transform"),
        "outputKind": output.get("outputKind"),
        "horizonUnit": horizon.get("unit"),
        "horizonValue": horizon.get("value"),
        "period": run_spec.get("marketSlice", {}).get("period"),
        "observationCount": overall.get("observationCount"),
        "testObservationCount": test.get("observationCount"),
        "overallRankIc": overall.get("meanRankIc"),
        "testRankIc": test.get("meanRankIc"),
        "overallTopMinusBottomPct": overall.get("meanTopMinusBottomPct"),
        "testTopMinusBottomPct": test.get("meanTopMinusBottomPct"),
        "testHitRatePct": test.get("hitRatePct"),
    }
