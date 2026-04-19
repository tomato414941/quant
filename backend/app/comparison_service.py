from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from app.portfolio import (
    StrategyDefinition,
    build_alignment_policy_spec,
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
    evaluate_strategy_definition_run,
    get_strategy_definition_signal_execution_contexts,
    serialize_asset_ranking_spec,
    serialize_predictor_spec,
    serialize_predictor_panel,
    serialize_portfolio_state,
    serialize_strategy_definition as serialize_canonical_strategy_definition,
)
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
)
from app.run_store import FileRunResultStore, RunStoreSummary, build_run_fingerprint, build_run_spec
from app.timeframe_models import (
    DEFAULT_DAILY_TIMEFRAME,
    DEFAULT_MONTHLY_TIMEFRAME,
    DEFAULT_WEEKLY_TIMEFRAME,
    TimeframeSpec,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


TIMEFRAMES_BY_KEY = {
    DEFAULT_DAILY_TIMEFRAME.key: DEFAULT_DAILY_TIMEFRAME,
    DEFAULT_WEEKLY_TIMEFRAME.key: DEFAULT_WEEKLY_TIMEFRAME,
    DEFAULT_MONTHLY_TIMEFRAME.key: DEFAULT_MONTHLY_TIMEFRAME,
}


def resolve_timeframe_spec_by_key(timeframe_key: str) -> TimeframeSpec:
    timeframe = TIMEFRAMES_BY_KEY.get(timeframe_key)
    if timeframe is None:
        raise ValueError(f"Unknown timeframe key: {timeframe_key}")
    return timeframe


def resolve_strategy_market_data_timeframe(strategy_definition) -> TimeframeSpec:
    selection_contexts, predictor_context = get_strategy_definition_signal_execution_contexts(strategy_definition)
    if selection_contexts:
        return resolve_timeframe_spec_by_key(str(selection_contexts[0]["dataTimeframe"]))
    if predictor_context is not None:
        return resolve_timeframe_spec_by_key(str(predictor_context["dataTimeframe"]))
    return resolve_timeframe_spec_by_key(strategy_definition.timeframe.key)


def serialize_strategy_definition_payload(strategy_definition: StrategyDefinition) -> dict:
    if not isinstance(strategy_definition, StrategyDefinition):
        raise ValueError("StrategyDefinition is required; evaluator DTO payloads are not supported.")
    return serialize_canonical_strategy_definition(strategy_definition)


def serialize_reproducible_strategy_definition(strategy_definition: StrategyDefinition) -> dict:
    return serialize_strategy_definition_payload(strategy_definition)


def collect_comparison_tickers(
    comparison: ComparisonSpec,
    predictor_specs: list | None = None,
    strategy_definitions: list | None = None,
) -> list[str]:
    seen: dict[str, None] = {}
    for strategy in (strategy_definitions or (comparison.candidate_strategies + comparison.reference_strategies)):
        for ticker in strategy.investment_universe.tickers:
            seen.setdefault(ticker, None)
    for predictor_spec in predictor_specs or []:
        for ticker in predictor_spec.signal_spec.observation_spec.tickers:
            seen.setdefault(ticker, None)
    return list(seen.keys())


def collect_comparison_timeframes(
    comparison: ComparisonSpec,
    strategy_definitions: list | None = None,
) -> list[TimeframeSpec]:
    seen: dict[str, TimeframeSpec] = {}
    for strategy_definition in (strategy_definitions or (comparison.candidate_strategies + comparison.reference_strategies)):
        if isinstance(strategy_definition, StrategyDefinition):
            for signal_spec in strategy_definition.signals:
                seen.setdefault(signal_spec.signal_timeframe.key, signal_spec.signal_timeframe)
                seen.setdefault(signal_spec.data_timeframe.key, signal_spec.data_timeframe)
            continue
        seen.setdefault(strategy_definition.timeframe.key, strategy_definition.timeframe)
        market_data_timeframe = resolve_strategy_market_data_timeframe(strategy_definition)
        seen.setdefault(market_data_timeframe.key, market_data_timeframe)
    return sorted(
        seen.values(),
        key=lambda timeframe: (timeframe.bar_seconds, timeframe.key),
    )


def collect_required_market_fields(
    comparison: ComparisonSpec,
    predictor_specs: list | None = None,
    strategy_definitions: list | None = None,
) -> list[str]:
    fields: dict[str, None] = {"close": None}
    for strategy_definition in (strategy_definitions or (comparison.candidate_strategies + comparison.reference_strategies)):
        if isinstance(strategy_definition, StrategyDefinition):
            for signal_spec in strategy_definition.signals:
                for field in signal_spec.observation_spec.fields:
                    fields.setdefault(field, None)
            continue
        for field in strategy_definition.selection.ranking_signal.feature_inputs:
            fields.setdefault(field, None)
    for predictor_spec in predictor_specs or []:
        for field in predictor_spec.signal_spec.observation_spec.fields:
            fields.setdefault(field, None)
    if comparison.run_spec.execution_assumptions.cost_model.kind == "asset_specific_adv_cost":
        fields.setdefault("volume", None)
    return list(fields.keys())


def collect_strategy_predictor_specs(strategy_definitions: list) -> list:
    predictor_specs = []
    seen_keys: set[str] = set()

    for strategy_definition in strategy_definitions:
        _selection_contexts, predictor_context = get_strategy_definition_signal_execution_contexts(
            strategy_definition
        )
        if predictor_context is None:
            continue
        predictor_key = str(predictor_context["predictorKey"])
        predictor_spec = REGISTERED_PREDICTOR_SPECS_BY_KEY.get(predictor_key)
        if predictor_spec is None:
            raise ValueError(f"Unknown predictor key: {predictor_key}")
        expected_predictor_timeframe_key = str(predictor_context["dataTimeframe"])
        if predictor_spec.timeframe.key != expected_predictor_timeframe_key:
            raise ValueError(
                "Supplemental predictor timeframe must match the predictor signal data timeframe."
            )
        if predictor_key in seen_keys:
            continue
        seen_keys.add(predictor_key)
        predictor_specs.append(
            align_predictor_spec_to_strategy_definition(
                predictor_spec,
                strategy_definition,
                predictor_key,
            )
        )

    return predictor_specs


def align_predictor_spec_to_strategy_definition(
    predictor_spec,
    strategy_definition: StrategyDefinition,
    predictor_key: str,
):
    predictor_signal = next(
        (
            signal
            for signal in strategy_definition.signals
            if signal.predictor_key == predictor_key
        ),
        None,
    )
    if predictor_signal is None:
        return predictor_spec
    return replace(
        predictor_spec,
        signal_spec=replace(
            predictor_spec.signal_spec,
            observation_spec=predictor_signal.observation_spec,
            entity_identifiers=predictor_signal.observation_spec.tickers,
        ),
    )


def collect_strategy_predictor_source_definitions(
    strategy_definitions: list[StrategyDefinition],
) -> dict[str, StrategyDefinition]:
    source_definitions: dict[str, StrategyDefinition] = {}
    for strategy_definition in strategy_definitions:
        _selection_contexts, predictor_context = get_strategy_definition_signal_execution_contexts(
            strategy_definition
        )
        if predictor_context is None:
            continue
        predictor_key = str(predictor_context["predictorKey"])
        source_definitions.setdefault(predictor_key, strategy_definition)
    return source_definitions


def collect_ranking_source_definitions(
    strategy_definitions: list[StrategyDefinition],
) -> dict[str, StrategyDefinition]:
    source_definitions: dict[str, StrategyDefinition] = {}
    for strategy_definition in strategy_definitions:
        ranking_specs = build_asset_ranking_specs_from_strategy_definitions([strategy_definition])
        for ranking_spec in ranking_specs:
            source_definitions.setdefault(ranking_spec.key, strategy_definition)
    return source_definitions


def build_parameter_sweep_strategy_definition(
    *,
    base_strategy: StrategyDefinition,
    selection_spec,
    market_data_timeframe: TimeframeSpec,
    max_weight: float,
) -> StrategyDefinition:
    strategy_id = "__".join([selection_spec.key, base_strategy.portfolio_model.key])
    selection_signal = build_strategy_signal_spec(
        key=f"signal__{strategy_id}__selection",
        label=selection_spec.label,
        description=selection_spec.description,
        observation_spec=build_observation_spec(
            key=f"observation__{strategy_id}__selection",
            label=f"{selection_spec.label} observation",
            tickers=base_strategy.investment_universe.tickers,
            fields=selection_spec.ranking_signal.feature_inputs,
        ),
        data_timeframe=market_data_timeframe,
        signal_timeframe=market_data_timeframe,
        source_kind="selection_signal",
        signal_parameters={
            "selectionKey": selection_spec.key,
            "strategyType": selection_spec.strategy_type,
            "scoreModelKind": selection_spec.ranking_signal.score_model.kind,
            "scoreParameters": dict(selection_spec.ranking_signal.score_parameters),
            "featureInputs": list(selection_spec.ranking_signal.feature_inputs),
            "universePolicyKey": selection_spec.universe_policy.key,
            "filterRuleKeys": [filter_rule.key for filter_rule in selection_spec.filter_rules],
            "fallbackRuleKey": selection_spec.fallback_rule.key,
        },
    )
    return build_strategy_definition(
        strategy_id=strategy_id,
        label=" × ".join([selection_spec.label, base_strategy.portfolio_model.label]),
        description=selection_spec.description,
        investment_universe=base_strategy.investment_universe,
        signals=[selection_signal],
        portfolio_model=base_strategy.portfolio_model,
        execution_plan=build_strategy_execution_plan_spec(
            key=f"execution_plan__{base_strategy.execution_plan.key}",
            label=base_strategy.execution_plan.label,
            decision_schedule=base_strategy.execution_plan.decision_schedule,
            rebalance_schedule=base_strategy.execution_plan.rebalance_schedule,
        ),
        risk_controls=build_risk_controls_spec(
            max_investment_ratio=1.0,
            max_weight=max_weight,
        ),
    )


def fetch_market_data_by_timeframe(
    comparison: ComparisonSpec,
    *,
    period: str,
    predictor_specs: list | None = None,
    strategy_definitions: list | None = None,
    fetch_market_universe_bundle,
) -> tuple[dict[str, dict], dict[str, dict], list[str], list[TimeframeSpec]]:
    comparison_tickers = collect_comparison_tickers(
        comparison,
        predictor_specs,
        strategy_definitions=strategy_definitions,
    )
    timeframes = collect_comparison_timeframes(
        comparison,
        strategy_definitions=strategy_definitions,
    )
    bundles_by_timeframe: dict[str, dict] = {}
    metadata_by_timeframe: dict[str, dict] = {}

    start_date = (
        comparison.run_spec.market_slice.start_date
        if period == comparison.run_spec.market_slice.period
        else None
    )
    end_date = (
        comparison.run_spec.market_slice.end_date
        if period == comparison.run_spec.market_slice.period
        else None
    )

    for timeframe in timeframes:
        market_bundle, metadata = fetch_market_universe_bundle(
            tickers=comparison_tickers,
            period=period,
            timeframe=timeframe.yfinance_interval,
            start_date=start_date,
            end_date=end_date,
        )
        bundles_by_timeframe[timeframe.key] = market_bundle
        metadata_by_timeframe[timeframe.key] = metadata

    return bundles_by_timeframe, metadata_by_timeframe, comparison_tickers, timeframes


def build_comparison_run_spec_payload(
    comparison: ComparisonSpec,
    *,
    fetch_market_universe_bundle,
) -> dict:
    original_candidate_definitions = list(comparison.candidate_strategies)
    original_reference_definitions = list(comparison.reference_strategies)
    strategy_definitions = original_candidate_definitions + original_reference_definitions
    predictor_specs = collect_strategy_predictor_specs(strategy_definitions)
    (
        _market_bundles_by_timeframe,
        metadata_by_timeframe,
        _comparison_tickers,
        comparison_timeframes,
    ) = fetch_market_data_by_timeframe(
        comparison,
        period=comparison.run_spec.market_slice.period,
        predictor_specs=predictor_specs,
        strategy_definitions=strategy_definitions,
        fetch_market_universe_bundle=fetch_market_universe_bundle,
    )
    run_spec = serialize_run_spec(
        comparison,
        metadata_by_timeframe,
        comparison_timeframes,
        strategy_definitions=original_candidate_definitions + original_reference_definitions,
    )
    payload = {
        "kind": "comparison_run_spec_payload",
        "schemaVersion": "v1",
        "comparisonId": comparison.comparison_id,
        "title": comparison.title,
        "question": comparison.question,
        "resultStoreDir": comparison.result_store_dir,
        "selectionPolicy": {
            "primaryMetric": comparison.selection_policy.primary_metric,
            "secondaryMetric": comparison.selection_policy.secondary_metric,
            "tertiaryMetric": comparison.selection_policy.tertiary_metric,
        },
        "candidateStrategyCount": len(original_candidate_definitions),
        "referenceStrategyCount": len(original_reference_definitions),
        "candidateStrategies": [
            serialize_reproducible_strategy_definition(strategy_definition)
            for strategy_definition in original_candidate_definitions
        ],
        "referenceStrategies": [
            serialize_reproducible_strategy_definition(strategy_definition)
            for strategy_definition in original_reference_definitions
        ],
        "conditionVariants": [
            serialize_condition_variant(condition_variant)
            for condition_variant in comparison.condition_variants
        ],
        "runSpecFingerprint": build_run_fingerprint(run_spec),
        "runSpec": run_spec,
    }
    payload["comparisonFingerprint"] = build_run_fingerprint(
        {
            "comparisonId": payload["comparisonId"],
            "selectionPolicy": payload["selectionPolicy"],
            "candidateStrategies": payload["candidateStrategies"],
            "referenceStrategies": payload["referenceStrategies"],
            "conditionVariants": payload["conditionVariants"],
            "runSpec": payload["runSpec"],
        }
    )
    return payload


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
    return ConditionVariant(
        key=str(payload["key"]),
        label=str(payload["label"]),
        commission_pct=float(payload["commissionPct"]),
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


def build_comparison_payload_from_run_spec_payload(
    payload: dict[str, object],
    *,
    fetch_market_universe_bundle,
) -> dict:
    comparison = deserialize_comparison_run_spec_payload(payload)
    return build_comparison_payload(
        comparison,
        fetch_market_universe_bundle=fetch_market_universe_bundle,
    )


def build_comparison_payload(
    comparison: ComparisonSpec,
    *,
    fetch_market_universe_bundle,
) -> dict:
    original_candidate_definitions = list(comparison.candidate_strategies)
    original_reference_definitions = list(comparison.reference_strategies)
    run_store = build_run_result_store(comparison)
    strategy_definitions = comparison.candidate_strategies + comparison.reference_strategies
    predictor_specs = collect_strategy_predictor_specs(strategy_definitions)
    (
        market_bundles_by_timeframe,
        metadata_by_timeframe,
        comparison_tickers,
        comparison_timeframes,
    ) = fetch_market_data_by_timeframe(
        comparison,
        period=comparison.run_spec.market_slice.period,
        predictor_specs=predictor_specs,
        strategy_definitions=strategy_definitions,
        fetch_market_universe_bundle=fetch_market_universe_bundle,
    )
    predictor_runs, predictor_panels_by_key, predictor_run_store_summary = build_predictor_runs(
        comparison=comparison,
        predictor_specs=predictor_specs,
        market_bundles_by_timeframe=market_bundles_by_timeframe,
        market_data_period=comparison.run_spec.market_slice.period,
        metadata_by_timeframe=metadata_by_timeframe,
        run_store=run_store,
    )
    candidate_runs, candidate_run_store_summary = build_strategy_runs(
        comparison=comparison,
        strategy_definitions=original_candidate_definitions,
        market_bundles_by_timeframe=market_bundles_by_timeframe,
        market_data_period=comparison.run_spec.market_slice.period,
        metadata_by_timeframe=metadata_by_timeframe,
        run_store=run_store,
        predictor_panels_by_key=predictor_panels_by_key,
    )
    reference_runs, reference_run_store_summary = build_strategy_runs(
        comparison=comparison,
        strategy_definitions=original_reference_definitions,
        market_bundles_by_timeframe=market_bundles_by_timeframe,
        market_data_period=comparison.run_spec.market_slice.period,
        metadata_by_timeframe=metadata_by_timeframe,
        run_store=run_store,
        predictor_panels_by_key=predictor_panels_by_key,
    )
    sanity_checks = []
    required_fields = collect_required_market_fields(
        comparison,
        predictor_specs,
        strategy_definitions=strategy_definitions,
    )
    total_cached_runs = (
        predictor_run_store_summary.cached_run_count
        + candidate_run_store_summary.cached_run_count
        + reference_run_store_summary.cached_run_count
    )
    total_computed_runs = (
        predictor_run_store_summary.computed_run_count
        + candidate_run_store_summary.computed_run_count
        + reference_run_store_summary.computed_run_count
    )
    for period in comparison.run_spec.market_slice.sanity_periods:
        (
            sanity_bundles_by_timeframe,
            sanity_metadata_by_timeframe,
            _sanity_tickers,
            sanity_timeframes,
        ) = fetch_market_data_by_timeframe(
            comparison,
            period=period,
            predictor_specs=predictor_specs,
            strategy_definitions=strategy_definitions,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
        )
        (
            sanity_predictor_runs,
            sanity_predictor_panels_by_key,
            sanity_predictor_run_store_summary,
        ) = build_predictor_runs(
            comparison=comparison,
            predictor_specs=predictor_specs,
            market_bundles_by_timeframe=sanity_bundles_by_timeframe,
            market_data_period=period,
            metadata_by_timeframe=sanity_metadata_by_timeframe,
            run_store=run_store,
        )
        sanity_candidate_runs, sanity_candidate_run_store_summary = build_strategy_runs(
            comparison=comparison,
            strategy_definitions=original_candidate_definitions,
            market_bundles_by_timeframe=sanity_bundles_by_timeframe,
            market_data_period=period,
            metadata_by_timeframe=sanity_metadata_by_timeframe,
            run_store=run_store,
            predictor_panels_by_key=sanity_predictor_panels_by_key,
        )
        sanity_reference_runs, sanity_reference_run_store_summary = build_strategy_runs(
            comparison=comparison,
            strategy_definitions=original_reference_definitions,
            market_bundles_by_timeframe=sanity_bundles_by_timeframe,
            market_data_period=period,
            metadata_by_timeframe=sanity_metadata_by_timeframe,
            run_store=run_store,
            predictor_panels_by_key=sanity_predictor_panels_by_key,
        )
        total_cached_runs += (
            sanity_predictor_run_store_summary.cached_run_count
            + sanity_candidate_run_store_summary.cached_run_count
            + sanity_reference_run_store_summary.cached_run_count
        )
        total_computed_runs += (
            sanity_predictor_run_store_summary.computed_run_count
            + sanity_candidate_run_store_summary.computed_run_count
            + sanity_reference_run_store_summary.computed_run_count
        )
        sanity_checks.append(
            {
                "period": period,
                "evaluation": serialize_evaluation(
                    comparison,
                    sanity_metadata_by_timeframe,
                    sanity_timeframes,
                    fields=required_fields,
                    period_override=period,
                    strategy_definitions=strategy_definitions,
                ),
                "runStoreSummary": {
                    "cachedRunCount": (
                        sanity_predictor_run_store_summary.cached_run_count
                        + sanity_candidate_run_store_summary.cached_run_count
                        + sanity_reference_run_store_summary.cached_run_count
                    ),
                    "computedRunCount": (
                        sanity_predictor_run_store_summary.computed_run_count
                        + sanity_candidate_run_store_summary.computed_run_count
                        + sanity_reference_run_store_summary.computed_run_count
                    ),
                },
                "predictorRuns": sanity_predictor_runs,
                "candidateRuns": sanity_candidate_runs,
                "referenceRuns": sanity_reference_runs,
            }
        )

    return {
        "comparison": serialize_comparison(
            comparison,
            metadata_by_timeframe,
            comparison_timeframes,
            candidate_strategy_definitions=original_candidate_definitions,
            reference_strategy_definitions=original_reference_definitions,
        ),
        "predictorRuns": predictor_runs,
        "candidateRuns": candidate_runs,
        "referenceRuns": reference_runs,
        "runStoreSummary": {
            "cachedRunCount": total_cached_runs,
            "computedRunCount": total_computed_runs,
        },
        "sanityChecks": sanity_checks,
    }


def build_predictor_runs_payload(
    comparison: ComparisonSpec,
    *,
    predictor_specs: list,
    fetch_market_universe_bundle,
) -> dict:
    run_store = build_run_result_store(comparison)
    strategy_definitions = comparison.candidate_strategies + comparison.reference_strategies
    predictor_source_definitions = collect_strategy_predictor_source_definitions(strategy_definitions)
    predictor_specs = [
        predictor_spec for predictor_spec in predictor_specs
        if predictor_spec.key in predictor_source_definitions
    ]
    (
        market_bundles_by_timeframe,
        metadata_by_timeframe,
        _comparison_tickers,
        comparison_timeframes,
    ) = fetch_market_data_by_timeframe(
        comparison,
        period=comparison.run_spec.market_slice.period,
        predictor_specs=predictor_specs,
        strategy_definitions=strategy_definitions,
        fetch_market_universe_bundle=fetch_market_universe_bundle,
    )
    predictor_runs, _predictor_panels_by_key, predictor_run_store_summary = build_predictor_runs(
        comparison=comparison,
        predictor_specs=predictor_specs,
        market_bundles_by_timeframe=market_bundles_by_timeframe,
        market_data_period=comparison.run_spec.market_slice.period,
        metadata_by_timeframe=metadata_by_timeframe,
        run_store=run_store,
    )
    required_fields = collect_required_market_fields(
        comparison,
        predictor_specs,
        strategy_definitions=strategy_definitions,
    )
    sanity_checks = []
    total_cached_runs = predictor_run_store_summary.cached_run_count
    total_computed_runs = predictor_run_store_summary.computed_run_count

    for period in comparison.run_spec.market_slice.sanity_periods:
        (
            sanity_bundles_by_timeframe,
            sanity_metadata_by_timeframe,
            _sanity_tickers,
            sanity_timeframes,
        ) = fetch_market_data_by_timeframe(
            comparison,
            period=period,
            predictor_specs=predictor_specs,
            strategy_definitions=strategy_definitions,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
        )
        sanity_predictor_runs, _sanity_panels_by_key, sanity_run_store_summary = build_predictor_runs(
            comparison=comparison,
            predictor_specs=predictor_specs,
            market_bundles_by_timeframe=sanity_bundles_by_timeframe,
            market_data_period=period,
            metadata_by_timeframe=sanity_metadata_by_timeframe,
            run_store=run_store,
        )
        total_cached_runs += sanity_run_store_summary.cached_run_count
        total_computed_runs += sanity_run_store_summary.computed_run_count
        sanity_checks.append(
            {
                "period": period,
                "evaluation": serialize_evaluation(
                    comparison,
                    sanity_metadata_by_timeframe,
                    sanity_timeframes,
                    fields=required_fields,
                    period_override=period,
                    strategy_definitions=strategy_definitions,
                ),
                "runStoreSummary": sanity_run_store_summary.to_payload(),
                "predictorRuns": sanity_predictor_runs,
            }
        )

    return {
        "kind": "predictor_run_collection",
        "schemaVersion": "v1",
        "comparisonId": comparison.comparison_id,
        "runSpec": serialize_run_spec(
            comparison,
            metadata_by_timeframe,
            comparison_timeframes,
            strategy_definitions=strategy_definitions,
        ),
        "predictorSpecs": [serialize_predictor_spec(predictor_spec) for predictor_spec in predictor_specs],
        "resultCount": len(predictor_runs),
        "runStoreSummary": {
            "cachedRunCount": total_cached_runs,
            "computedRunCount": total_computed_runs,
        },
        "predictorRuns": predictor_runs,
        "sanityChecks": sanity_checks,
    }


def build_strategy_runs_payload(
    comparison: ComparisonSpec,
    *,
    fetch_market_universe_bundle,
) -> dict:
    original_candidate_definitions = list(comparison.candidate_strategies)
    original_reference_definitions = list(comparison.reference_strategies)
    run_store = build_run_result_store(comparison)
    strategy_definitions = original_candidate_definitions + original_reference_definitions
    predictor_specs = collect_strategy_predictor_specs(strategy_definitions)
    (
        market_bundles_by_timeframe,
        metadata_by_timeframe,
        _comparison_tickers,
        comparison_timeframes,
    ) = fetch_market_data_by_timeframe(
        comparison,
        period=comparison.run_spec.market_slice.period,
        predictor_specs=predictor_specs,
        strategy_definitions=strategy_definitions,
        fetch_market_universe_bundle=fetch_market_universe_bundle,
    )
    predictor_runs, predictor_panels_by_key, predictor_run_store_summary = build_predictor_runs(
        comparison=comparison,
        predictor_specs=predictor_specs,
        market_bundles_by_timeframe=market_bundles_by_timeframe,
        market_data_period=comparison.run_spec.market_slice.period,
        metadata_by_timeframe=metadata_by_timeframe,
        run_store=run_store,
    )
    candidate_runs, candidate_run_store_summary = build_strategy_runs(
        comparison=comparison,
        strategy_definitions=original_candidate_definitions,
        market_bundles_by_timeframe=market_bundles_by_timeframe,
        market_data_period=comparison.run_spec.market_slice.period,
        metadata_by_timeframe=metadata_by_timeframe,
        run_store=run_store,
        predictor_panels_by_key=predictor_panels_by_key,
    )
    reference_runs, reference_run_store_summary = build_strategy_runs(
        comparison=comparison,
        strategy_definitions=original_reference_definitions,
        market_bundles_by_timeframe=market_bundles_by_timeframe,
        market_data_period=comparison.run_spec.market_slice.period,
        metadata_by_timeframe=metadata_by_timeframe,
        run_store=run_store,
        predictor_panels_by_key=predictor_panels_by_key,
    )
    required_fields = collect_required_market_fields(
        comparison,
        predictor_specs,
        strategy_definitions=strategy_definitions,
    )
    sanity_checks = []
    total_cached_runs = (
        predictor_run_store_summary.cached_run_count
        + candidate_run_store_summary.cached_run_count
        + reference_run_store_summary.cached_run_count
    )
    total_computed_runs = (
        predictor_run_store_summary.computed_run_count
        + candidate_run_store_summary.computed_run_count
        + reference_run_store_summary.computed_run_count
    )

    for period in comparison.run_spec.market_slice.sanity_periods:
        (
            sanity_bundles_by_timeframe,
            sanity_metadata_by_timeframe,
            _sanity_tickers,
            sanity_timeframes,
        ) = fetch_market_data_by_timeframe(
            comparison,
            period=period,
            predictor_specs=predictor_specs,
            strategy_definitions=strategy_definitions,
            fetch_market_universe_bundle=fetch_market_universe_bundle,
        )
        (
            sanity_predictor_runs,
            sanity_predictor_panels_by_key,
            sanity_predictor_run_store_summary,
        ) = build_predictor_runs(
            comparison=comparison,
            predictor_specs=predictor_specs,
            market_bundles_by_timeframe=sanity_bundles_by_timeframe,
            market_data_period=period,
            metadata_by_timeframe=sanity_metadata_by_timeframe,
            run_store=run_store,
        )
        sanity_candidate_runs, sanity_candidate_run_store_summary = build_strategy_runs(
            comparison=comparison,
            strategy_definitions=original_candidate_definitions,
            market_bundles_by_timeframe=sanity_bundles_by_timeframe,
            market_data_period=period,
            metadata_by_timeframe=sanity_metadata_by_timeframe,
            run_store=run_store,
            predictor_panels_by_key=sanity_predictor_panels_by_key,
        )
        sanity_reference_runs, sanity_reference_run_store_summary = build_strategy_runs(
            comparison=comparison,
            strategy_definitions=original_reference_definitions,
            market_bundles_by_timeframe=sanity_bundles_by_timeframe,
            market_data_period=period,
            metadata_by_timeframe=sanity_metadata_by_timeframe,
            run_store=run_store,
            predictor_panels_by_key=sanity_predictor_panels_by_key,
        )
        total_cached_runs += (
            sanity_predictor_run_store_summary.cached_run_count
            + sanity_candidate_run_store_summary.cached_run_count
            + sanity_reference_run_store_summary.cached_run_count
        )
        total_computed_runs += (
            sanity_predictor_run_store_summary.computed_run_count
            + sanity_candidate_run_store_summary.computed_run_count
            + sanity_reference_run_store_summary.computed_run_count
        )
        sanity_checks.append(
            {
                "period": period,
                "evaluation": serialize_evaluation(
                    comparison,
                    sanity_metadata_by_timeframe,
                    sanity_timeframes,
                    fields=required_fields,
                    period_override=period,
                    strategy_definitions=strategy_definitions,
                ),
                "runStoreSummary": {
                    "cachedRunCount": (
                        sanity_predictor_run_store_summary.cached_run_count
                        + sanity_candidate_run_store_summary.cached_run_count
                        + sanity_reference_run_store_summary.cached_run_count
                    ),
                    "computedRunCount": (
                        sanity_predictor_run_store_summary.computed_run_count
                        + sanity_candidate_run_store_summary.computed_run_count
                        + sanity_reference_run_store_summary.computed_run_count
                    ),
                },
                "predictorRuns": sanity_predictor_runs,
                "candidateRuns": sanity_candidate_runs,
                "referenceRuns": sanity_reference_runs,
            }
        )

    return {
        "kind": "strategy_run_collection",
        "schemaVersion": "v1",
        "comparisonId": comparison.comparison_id,
        "runSpec": serialize_run_spec(
            comparison,
            metadata_by_timeframe,
            comparison_timeframes,
            strategy_definitions=original_candidate_definitions + original_reference_definitions,
        ),
        "candidateStrategies": [
            serialize_strategy_definition_payload(strategy_definition)
            for strategy_definition in original_candidate_definitions
        ],
        "referenceStrategies": [
            serialize_strategy_definition_payload(strategy_definition)
            for strategy_definition in original_reference_definitions
        ],
        "predictorRuns": predictor_runs,
        "candidateRuns": candidate_runs,
        "referenceRuns": reference_runs,
        "runStoreSummary": {
            "cachedRunCount": total_cached_runs,
            "computedRunCount": total_computed_runs,
        },
        "sanityChecks": sanity_checks,
    }


def build_condition_sweep_payload(
    comparison: ComparisonSpec,
    *,
    fetch_market_universe_bundle,
) -> dict:
    original_candidate_definitions = list(comparison.candidate_strategies)
    original_reference_definitions = list(comparison.reference_strategies)
    run_store = build_run_result_store(comparison)
    strategy_definitions = original_candidate_definitions + original_reference_definitions
    predictor_specs = collect_strategy_predictor_specs(original_candidate_definitions)
    (
        market_bundles_by_timeframe,
        metadata_by_timeframe,
        _comparison_tickers,
        comparison_timeframes,
    ) = fetch_market_data_by_timeframe(
        comparison,
        period=comparison.run_spec.market_slice.period,
        predictor_specs=predictor_specs,
        strategy_definitions=strategy_definitions,
        fetch_market_universe_bundle=fetch_market_universe_bundle,
    )
    results, run_store_summary = build_condition_sweep_runs(
        comparison=comparison,
        market_bundles_by_timeframe=market_bundles_by_timeframe,
        market_data_period=comparison.run_spec.market_slice.period,
        metadata_by_timeframe=metadata_by_timeframe,
        run_store=run_store,
        strategy_definitions=original_candidate_definitions,
    )
    return {
        "comparison": serialize_comparison(
            comparison,
            metadata_by_timeframe,
            comparison_timeframes,
            candidate_strategy_definitions=original_candidate_definitions,
            reference_strategy_definitions=original_reference_definitions,
        ),
        "conditionVariants": [
            serialize_condition_variant(condition_variant)
            for condition_variant in comparison.condition_variants
        ],
        "resultCount": len(results),
        "runStoreSummary": run_store_summary.to_payload(),
        "results": results,
    }


def build_ranking_evaluation_payload(
    comparison: ComparisonSpec,
    *,
    fetch_market_universe_bundle,
) -> dict:
    original_candidate_definitions = list(comparison.candidate_strategies)
    original_reference_definitions = list(comparison.reference_strategies)
    run_store = build_run_result_store(comparison)
    (
        market_bundles_by_timeframe,
        metadata_by_timeframe,
        _comparison_tickers,
        comparison_timeframes,
    ) = fetch_market_data_by_timeframe(
        comparison,
        period=comparison.run_spec.market_slice.period,
        fetch_market_universe_bundle=fetch_market_universe_bundle,
    )
    results, run_store_summary = build_ranking_evaluation_runs(
        comparison=comparison,
        market_bundles_by_timeframe=market_bundles_by_timeframe,
        market_data_period=comparison.run_spec.market_slice.period,
        metadata_by_timeframe=metadata_by_timeframe,
        run_store=run_store,
    )
    return {
        "comparison": serialize_comparison(
            comparison,
            metadata_by_timeframe,
            comparison_timeframes,
            candidate_strategy_definitions=original_candidate_definitions,
            reference_strategy_definitions=original_reference_definitions,
        ),
        "resultCount": len(results),
        "runStoreSummary": run_store_summary.to_payload(),
        "results": results,
    }


def build_latest_run_payload(
    comparison: ComparisonSpec,
    *,
    run_kind: str | None = None,
    generation_method: str | None = None,
    strategy_definition_fingerprint: str | None = None,
    market_data_fingerprint: str | None = None,
    evaluation_fingerprint: str | None = None,
) -> dict:
    if (
        strategy_definition_fingerprint is None
        and market_data_fingerprint is None
        and evaluation_fingerprint is None
    ):
        raise ValueError("At least one fingerprint filter is required for latest run lookup.")

    run_store = build_run_result_store(comparison)
    record = run_store.find_latest_compact_record(
        run_kind=run_kind,
        generation_method=generation_method,
        strategy_definition_fingerprint=strategy_definition_fingerprint,
        market_data_fingerprint=market_data_fingerprint,
        evaluation_fingerprint=evaluation_fingerprint,
        view="generic",
    )
    return {
        "comparisonId": comparison.comparison_id,
        "runKind": run_kind,
        "generationMethod": generation_method,
        "filters": {
            "strategyDefinitionFingerprint": strategy_definition_fingerprint,
            "marketDataFingerprint": market_data_fingerprint,
            "evaluationFingerprint": evaluation_fingerprint,
        },
        "record": record,
    }


def build_run_catalog_payload(
    comparison: ComparisonSpec,
    *,
    limit: int = 50,
    run_kind: str | None = None,
    generation_method: str | None = None,
    strategy_definition_fingerprint: str | None = None,
    market_data_fingerprint: str | None = None,
    evaluation_fingerprint: str | None = None,
) -> dict:
    run_store = build_run_result_store(comparison)
    records = run_store.list_compact_records(
        run_kind=run_kind,
        generation_method=generation_method,
        strategy_definition_fingerprint=strategy_definition_fingerprint,
        market_data_fingerprint=market_data_fingerprint,
        evaluation_fingerprint=evaluation_fingerprint,
        limit=limit,
        view="generic",
    )
    return {
        "comparisonId": comparison.comparison_id,
        "limit": limit,
        "runKind": run_kind,
        "generationMethod": generation_method,
        "filters": {
            "strategyDefinitionFingerprint": strategy_definition_fingerprint,
            "marketDataFingerprint": market_data_fingerprint,
            "evaluationFingerprint": evaluation_fingerprint,
        },
        "recordCount": len(records),
        "records": records,
    }


def build_predictor_run_index_payload(
    comparison: ComparisonSpec,
    *,
    limit: int = 50,
    learner_kind: str | None = None,
    combiner_kind: str | None = None,
    signal_source_kind: str | None = None,
    signal_source_feature_key: str | None = None,
    horizon_value: int | None = None,
    strategy_definition_fingerprint: str | None = None,
    market_data_fingerprint: str | None = None,
    evaluation_fingerprint: str | None = None,
    sort_by: str = "test_rank_ic",
) -> dict:
    run_store = build_run_result_store(comparison)
    all_records = run_store.list_compact_records(
        run_kind="predictor_run",
        strategy_definition_fingerprint=strategy_definition_fingerprint,
        market_data_fingerprint=market_data_fingerprint,
        evaluation_fingerprint=evaluation_fingerprint,
        view="predictor",
    )
    if learner_kind is not None:
        all_records = [
            record for record in all_records
            if record["learnerKind"] == learner_kind
        ]
    if combiner_kind is not None:
        all_records = [
            record for record in all_records
            if record["combinerKind"] == combiner_kind
        ]
    if signal_source_kind is not None:
        all_records = [
            record for record in all_records
            if record["signalSourceKind"] == signal_source_kind
        ]
    if signal_source_feature_key is not None:
        all_records = [
            record for record in all_records
            if record["signalSourceFeatureKey"] == signal_source_feature_key
        ]
    if horizon_value is not None:
        all_records = [
            record for record in all_records
            if record["horizonValue"] == horizon_value
        ]
    sorted_records = sort_predictor_run_records(all_records, sort_by=sort_by)
    records = sorted_records[:limit]
    return {
        "kind": "predictor_run_index",
        "schemaVersion": "v1",
        "comparisonId": comparison.comparison_id,
        "limit": limit,
        "sortBy": sort_by,
        "filters": {
            "learnerKind": learner_kind,
            "combinerKind": combiner_kind,
            "signalSourceKind": signal_source_kind,
            "signalSourceFeatureKey": signal_source_feature_key,
            "horizonValue": horizon_value,
            "strategyDefinitionFingerprint": strategy_definition_fingerprint,
            "marketDataFingerprint": market_data_fingerprint,
            "evaluationFingerprint": evaluation_fingerprint,
        },
        "totalCount": len(all_records),
        "recordCount": len(records),
        "records": records,
        "groupedSummaries": {
            "bestBySignalSource": summarize_predictor_record_groups(
                all_records,
                group_fields=("signalSourceKind", "signalSourceFeatureKey"),
            ),
            "bestBySignalSourceAndHorizon": summarize_predictor_record_groups(
                all_records,
                group_fields=("signalSourceKind", "signalSourceFeatureKey", "horizonValue"),
            ),
            "bestBySignalSourceAndPeriod": summarize_predictor_record_groups(
                all_records,
                group_fields=("signalSourceKind", "signalSourceFeatureKey", "period"),
            ),
            "bestBySourceLearnerCombiner": summarize_predictor_record_groups(
                all_records,
                group_fields=(
                    "signalSourceKind",
                    "signalSourceFeatureKey",
                    "learnerKind",
                    "combinerKind",
                ),
            ),
        },
    }


def build_predictor_run_detail_payload(
    comparison: ComparisonSpec,
    *,
    run_key: str,
) -> dict:
    run_store = build_run_result_store(comparison)
    record = run_store.get_record(run_key)
    if record is None or record["runSpec"].get("runKind") != "predictor_run":
        raise ValueError(f"Predictor run not found: {run_key}")
    return {
        "kind": "predictor_run_detail",
        "schemaVersion": "v1",
        "comparisonId": comparison.comparison_id,
        "record": record,
    }


def build_strategy_run_index_payload(
    comparison: ComparisonSpec,
    *,
    limit: int = 50,
    strategy_definition_fingerprint: str | None = None,
    market_data_fingerprint: str | None = None,
    evaluation_fingerprint: str | None = None,
) -> dict:
    run_store = build_run_result_store(comparison)
    total_count = len(
        run_store.list_compact_records(
            run_kind="strategy_run",
            strategy_definition_fingerprint=strategy_definition_fingerprint,
            market_data_fingerprint=market_data_fingerprint,
            evaluation_fingerprint=evaluation_fingerprint,
            view="generic",
        )
    )
    records = run_store.list_compact_records(
        run_kind="strategy_run",
        strategy_definition_fingerprint=strategy_definition_fingerprint,
        market_data_fingerprint=market_data_fingerprint,
        evaluation_fingerprint=evaluation_fingerprint,
        limit=limit,
        view="generic",
    )
    return {
        "kind": "strategy_run_index",
        "schemaVersion": "v1",
        "comparisonId": comparison.comparison_id,
        "limit": limit,
        "filters": {
            "strategyDefinitionFingerprint": strategy_definition_fingerprint,
            "marketDataFingerprint": market_data_fingerprint,
            "evaluationFingerprint": evaluation_fingerprint,
        },
        "totalCount": total_count,
        "recordCount": len(records),
        "records": records,
    }


def build_strategy_run_detail_payload(
    comparison: ComparisonSpec,
    *,
    run_key: str,
) -> dict:
    run_store = build_run_result_store(comparison)
    record = run_store.get_record(run_key)
    if record is None or record["runSpec"].get("runKind") != "strategy_run":
        raise ValueError(f"Strategy run not found: {run_key}")
    return {
        "kind": "strategy_run_detail",
        "schemaVersion": "v1",
        "comparisonId": comparison.comparison_id,
        "record": record,
    }


def generate_parameter_sweep_runs_payload(
    comparison: ComparisonSpec,
    *,
    fetch_market_universe_bundle,
) -> dict:
    original_candidate_definitions = list(comparison.candidate_strategies)
    original_reference_definitions = list(comparison.reference_strategies)
    run_store = build_run_result_store(comparison)
    strategy_definitions = original_candidate_definitions + original_reference_definitions
    predictor_specs = collect_strategy_predictor_specs(strategy_definitions)
    (
        market_bundles_by_timeframe,
        metadata_by_timeframe,
        _comparison_tickers,
        comparison_timeframes,
    ) = fetch_market_data_by_timeframe(
        comparison,
        period=comparison.run_spec.market_slice.period,
        predictor_specs=predictor_specs,
        strategy_definitions=strategy_definitions,
        fetch_market_universe_bundle=fetch_market_universe_bundle,
    )
    results, run_store_summary = build_parameter_sweep_runs(
        comparison=comparison,
        market_bundles_by_timeframe=market_bundles_by_timeframe,
        market_data_period=comparison.run_spec.market_slice.period,
        metadata_by_timeframe=metadata_by_timeframe,
        run_store=run_store,
    )
    return {
        "comparison": serialize_comparison(
            comparison,
            metadata_by_timeframe,
            comparison_timeframes,
            candidate_strategy_definitions=original_candidate_definitions,
            reference_strategy_definitions=original_reference_definitions,
        ),
        "generation": {
            "method": "parameter_sweep",
            "batchKey": "local_tilt_search_9m_v1",
            "spec": build_parameter_sweep_generation_spec(),
        },
        "resultCount": len(results),
        "runStoreSummary": run_store_summary.to_payload(),
        "results": results,
    }


def build_run_result_store(comparison: ComparisonSpec) -> FileRunResultStore:
    root_dir = Path(comparison.result_store_dir)
    if not root_dir.is_absolute():
        root_dir = PROJECT_ROOT / root_dir
    return FileRunResultStore(root_dir)


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
    return warnings


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


def serialize_evaluation(
    comparison: ComparisonSpec,
    metadata_by_timeframe: dict[str, dict[str, str]],
    timeframes: list[TimeframeSpec],
    *,
    fields: list[str],
    period_override: str | None = None,
    strategy_definitions: list | None = None,
) -> dict:
    payload = {
        "kind": "evaluation_spec",
        "schemaVersion": "v1",
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
    warnings = build_market_data_warnings(
        comparison,
        metadata_by_timeframe,
        period_override=period_override,
    )
    if warnings:
        payload["warnings"] = warnings
    return payload


def serialize_run_spec(
    comparison: ComparisonSpec,
    metadata_by_timeframe: dict[str, dict[str, str]],
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
    metadata_by_timeframe: dict[str, dict[str, str]],
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
        "commissionPct": round(condition_variant.commission_pct, 3),
        "maxInvestmentPct": round(condition_variant.max_investment_ratio * 100, 1),
        "maxWeightPct": round(condition_variant.max_weight * 100, 1)
        if condition_variant.max_weight is not None
        else None,
    }


def compact_run_record(record: dict) -> dict:
    run_spec = record["runSpec"]
    result = record["result"]
    strategy = run_spec.get("strategy", {})
    execution_assumptions = run_spec.get("executionAssumptions", {})
    market_slice = run_spec.get("marketSlice", {})
    evaluation = run_spec.get("evaluation", {})
    fingerprints = run_spec.get("fingerprints", {})
    summary = result.get("summary", {})
    portfolio_summary = summary.get("portfolio", summary)

    return {
        "runKey": record["runKey"],
        "savedAtUtc": record.get("savedAtUtc"),
        "runKind": run_spec.get("runKind"),
        "logicVersion": run_spec.get("logicVersion"),
        "generationMethod": run_spec.get("generation", {}).get("method"),
        "generationBatchKey": run_spec.get("generation", {}).get("batchKey"),
        "strategyId": strategy.get("strategyId"),
        "strategyVersion": strategy.get("version"),
        "strategyLabel": strategy.get("label"),
        "strategyHypothesis": strategy.get("hypothesis"),
        "investmentUniverseLabel": strategy.get("components", {}).get("core", {}).get("investmentUniverse", {}).get("label"),
        "investmentUniverseAssetCount": strategy.get("components", {}).get("core", {}).get("investmentUniverse", {}).get("assetCount"),
        "portfolioModelLabel": strategy.get("components", {}).get("core", {}).get("portfolioModel", {}).get("label"),
        "executionLabel": strategy.get("components", {}).get("core", {}).get("executionPolicy", {}).get("label"),
        "period": market_slice.get("period"),
        "timeframe": strategy.get("components", {}).get("core", {}).get("dataResolution", {}).get("key")
        or market_slice.get("timeframe", {}).get("key"),
        "maxInvestmentPct": strategy.get("components", {}).get("optional", {}).get("riskControls", {}).get("maxInvestmentPct"),
        "maxWeightPct": strategy.get("components", {}).get("optional", {}).get("riskControls", {}).get("maxWeightPct"),
        "costModelKind": execution_assumptions.get("costModel", {}).get("kind"),
        "commissionPct": execution_assumptions.get("costModel", {}).get("parameters", {}).get("commissionPct"),
        "capitalBase": run_spec.get("capitalBase"),
        "splitRatioPct": evaluation.get("evaluationSettings", {}).get("splitRatioPct"),
        "strategyDefinitionFingerprint": fingerprints.get("strategyDefinition"),
        "marketDataFingerprint": fingerprints.get("marketData"),
        "evaluationFingerprint": fingerprints.get("evaluation"),
        "sharpeRatio": portfolio_summary.get("sharpeRatio"),
        "totalReturnPct": portfolio_summary.get("totalReturnPct"),
        "maxDrawdownPct": portfolio_summary.get("maxDrawdownPct"),
    }


def compact_strategy_run_record(record: dict) -> dict:
    return compact_run_record(record)


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


def build_strategy_runs(
    *,
    comparison: ComparisonSpec,
    strategy_definitions: list[StrategyDefinition],
    market_bundles_by_timeframe: dict[str, dict],
    market_data_period: str,
    metadata_by_timeframe: dict[str, dict[str, str]],
    run_store: FileRunResultStore,
    predictor_panels_by_key: dict[str, object] | None = None,
) -> tuple[list[dict], RunStoreSummary]:
    serialized_execution_assumptions = serialize_execution_assumptions(comparison)
    predictor_specs = collect_strategy_predictor_specs(strategy_definitions)
    required_fields = collect_required_market_fields(
        comparison,
        predictor_specs,
        strategy_definitions=strategy_definitions,
    )
    runs: list[dict] = []
    cached_run_count = 0
    computed_run_count = 0

    for strategy_definition in strategy_definitions:
        market_data_timeframe = resolve_strategy_market_data_timeframe(strategy_definition)
        timeframe_key = market_data_timeframe.key
        dataset_metadata = metadata_by_timeframe[timeframe_key]
        serialized_strategy_definition = serialize_strategy_definition_payload(strategy_definition)
        _selection_contexts, predictor_context = get_strategy_definition_signal_execution_contexts(strategy_definition)
        predictor_panel = None
        if predictor_context is not None and predictor_panels_by_key is not None:
            predictor_panel = predictor_panels_by_key.get(str(predictor_context["predictorKey"]))
        market_bundle = market_bundles_by_timeframe[timeframe_key]
        strategy_market_slice = {
            "period": market_data_period,
            "timeframe": serialize_timeframe(market_data_timeframe),
            "fields": required_fields,
        }
        run_spec = build_run_spec(
            run_kind="strategy_run",
            market_slice=strategy_market_slice,
            evaluation=serialize_evaluation(
                comparison,
                {timeframe_key: dataset_metadata},
                [market_data_timeframe],
                fields=required_fields,
                period_override=market_data_period,
            ),
            execution_assumptions=serialized_execution_assumptions,
            portfolio_state=serialize_portfolio_state(comparison.run_spec.portfolio_state),
            capital_base=comparison.run_spec.capital_base,
            strategy_definition=serialized_strategy_definition,
        )
        cached_run = run_store.load(run_spec)
        if cached_run is not None:
            cached_run_count += 1
            runs.append(cached_run)
            continue

        run = evaluate_strategy_definition_run(
            closes=market_bundle["closes"],
            volumes=market_bundle["volumes"],
            strategy_definition=strategy_definition,
            bars_per_year=market_data_timeframe.bars_per_year,
            initial_capital=comparison.run_spec.capital_base,
            split_ratio=comparison.run_spec.evaluation.evaluation_settings.split_ratio,
            execution_assumptions=serialized_execution_assumptions,
            portfolio_state=comparison.run_spec.portfolio_state,
            predictor_panel=predictor_panel,
        )
        run_store.save(run_spec, run)
        computed_run_count += 1
        runs.append(run)

    return runs, RunStoreSummary(
        cached_run_count=cached_run_count,
        computed_run_count=computed_run_count,
    )

def build_predictor_runs(
    *,
    comparison: ComparisonSpec,
    predictor_specs: list,
    market_bundles_by_timeframe: dict[str, dict],
    market_data_period: str,
    metadata_by_timeframe: dict[str, dict[str, str]],
    run_store: FileRunResultStore,
) -> tuple[list[dict], dict[str, object], RunStoreSummary]:
    results: list[dict] = []
    predictor_panels_by_key: dict[str, object] = {}
    cached_run_count = 0
    computed_run_count = 0
    strategy_definitions = comparison.candidate_strategies + comparison.reference_strategies
    required_fields = collect_required_market_fields(
        comparison,
        predictor_specs,
        strategy_definitions=strategy_definitions,
    )
    serialized_execution_assumptions = serialize_execution_assumptions(comparison)
    predictor_source_definitions = collect_strategy_predictor_source_definitions(strategy_definitions)

    for predictor_spec in predictor_specs:
        timeframe_key = predictor_spec.timeframe.key
        market_bundle = market_bundles_by_timeframe[timeframe_key]
        dataset_metadata = metadata_by_timeframe[timeframe_key]
        closes = market_bundle["closes"][list(predictor_spec.signal_spec.observation_spec.tickers)]
        volumes = (
            market_bundle["volumes"][list(predictor_spec.signal_spec.observation_spec.tickers)]
            if market_bundle["volumes"] is not None
            else None
        )
        returns = closes.pct_change().dropna()
        aligned_volumes = volumes.loc[returns.index] if volumes is not None else None
        source_strategy_definition = predictor_source_definitions.get(predictor_spec.key)
        if source_strategy_definition is None:
            raise ValueError(f"No source strategy definition found for predictor: {predictor_spec.key}")
        serialized_predictor_spec = serialize_predictor_spec(predictor_spec)
        run_spec = build_run_spec(
            run_kind="predictor_run",
            strategy_definition=serialize_strategy_definition_payload(source_strategy_definition),
            evaluation_subject={"kind": "predictor", "predictor": serialized_predictor_spec},
            market_slice={
                "period": market_data_period,
                "timeframe": serialize_timeframe(predictor_spec.timeframe),
                "fields": required_fields,
            },
            evaluation=serialize_evaluation(
                comparison,
                {timeframe_key: dataset_metadata},
                [predictor_spec.timeframe],
                fields=required_fields,
                period_override=market_data_period,
            ),
            execution_assumptions=serialized_execution_assumptions,
            portfolio_state=serialize_portfolio_state(comparison.run_spec.portfolio_state),
            capital_base=comparison.run_spec.capital_base,
        )
        cached_run = run_store.load(run_spec)
        if cached_run is not None:
            cached_run_count += 1
            results.append(cached_run)
            predictor_panels_by_key[predictor_spec.key] = deserialize_predictor_panel(
                cached_run.get("predictorSeries")
            )
            continue

        predictor_panel = compute_predictor_panel(
            returns=returns,
            volumes=aligned_volumes,
            predictor_spec=predictor_spec,
            bars_per_year=predictor_spec.timeframe.bars_per_year,
        )
        result = evaluate_predictor_spec(
            returns=returns,
            volumes=aligned_volumes,
            split_ratio=comparison.run_spec.evaluation.evaluation_settings.split_ratio,
            predictor_spec=predictor_spec,
            bars_per_year=predictor_spec.timeframe.bars_per_year,
            predictor_panel=predictor_panel,
        )
        run_store.save(run_spec, result)
        computed_run_count += 1
        results.append(result)
        predictor_panels_by_key[predictor_spec.key] = deserialize_predictor_panel(
            result.get("predictorSeries")
        )

    return results, predictor_panels_by_key, RunStoreSummary(
        cached_run_count=cached_run_count,
        computed_run_count=computed_run_count,
    )


def build_condition_sweep_runs(
    *,
    comparison: ComparisonSpec,
    market_bundles_by_timeframe: dict[str, dict],
    market_data_period: str,
    metadata_by_timeframe: dict[str, dict[str, str]],
    run_store: FileRunResultStore,
    strategy_definitions: list[StrategyDefinition] | None = None,
) -> tuple[list[dict], RunStoreSummary]:
    results: list[dict] = []
    cached_run_count = 0
    computed_run_count = 0
    comparison_strategy_definitions = strategy_definitions or comparison.candidate_strategies
    all_strategy_definitions = list(comparison_strategy_definitions) + list(comparison.reference_strategies)
    predictor_specs = collect_strategy_predictor_specs(all_strategy_definitions)
    required_fields = collect_required_market_fields(
        comparison,
        predictor_specs,
        strategy_definitions=all_strategy_definitions,
    )

    for strategy_definition in comparison_strategy_definitions:
        market_data_timeframe = resolve_strategy_market_data_timeframe(strategy_definition)
        timeframe_key = market_data_timeframe.key
        market_bundle = market_bundles_by_timeframe[timeframe_key]
        dataset_metadata = metadata_by_timeframe[timeframe_key]
        for condition_variant in comparison.condition_variants:
            effective_risk_controls = build_risk_controls_spec(
                max_investment_ratio=condition_variant.max_investment_ratio,
                max_weight=condition_variant.max_weight,
            )
            effective_strategy_definition = replace(
                strategy_definition,
                risk_controls=effective_risk_controls,
            )
            effective_evaluation = replace(comparison.run_spec.evaluation)
            effective_execution_assumptions = replace(
                comparison.run_spec.execution_assumptions,
                cost_model=replace(
                    comparison.run_spec.execution_assumptions.cost_model,
                    parameters={
                        **comparison.run_spec.execution_assumptions.cost_model.parameters,
                        "commissionPct": condition_variant.commission_pct,
                    },
                ),
            )
            serialized_strategy_definition = serialize_strategy_definition_payload(effective_strategy_definition)
            serialized_evaluation = serialize_evaluation(
                comparison,
                {timeframe_key: dataset_metadata},
                [market_data_timeframe],
                fields=required_fields,
                period_override=market_data_period,
            )
            serialized_execution_assumptions = {
                "kind": effective_execution_assumptions.kind,
                "label": effective_execution_assumptions.label,
                "parameters": {
                    key: value for key, value in effective_execution_assumptions.parameters.items()
                },
                "costModel": serialize_cost_model_spec(
                    effective_execution_assumptions.cost_model
                ),
            }
            serialized_condition_variant = serialize_condition_variant(condition_variant)
            run_spec = build_run_spec(
                run_kind="condition_sweep",
                market_slice={
                    "period": market_data_period,
                    "timeframe": serialize_timeframe(market_data_timeframe),
                    "fields": required_fields,
                },
                evaluation=serialized_evaluation,
                execution_assumptions=serialized_execution_assumptions,
                portfolio_state=serialize_portfolio_state(comparison.run_spec.portfolio_state),
                capital_base=comparison.run_spec.capital_base,
                strategy_definition=serialized_strategy_definition,
            )
            cached_run = run_store.load(run_spec)
            if cached_run is not None:
                cached_run_count += 1
                results.append(cached_run)
                continue

            run = evaluate_strategy_definition_run(
                closes=market_bundle["closes"],
                volumes=market_bundle["volumes"],
                strategy_definition=effective_strategy_definition,
                bars_per_year=market_data_timeframe.bars_per_year,
                initial_capital=comparison.run_spec.capital_base,
                split_ratio=effective_evaluation.evaluation_settings.split_ratio,
                execution_assumptions=serialized_execution_assumptions,
                portfolio_state=comparison.run_spec.portfolio_state,
            )
            compact_run = compact_condition_sweep_run(
                run=run,
                key=f"{strategy_definition.key}__{condition_variant.key}",
                condition_variant=serialized_condition_variant,
            )
            run_store.save(run_spec, compact_run)
            computed_run_count += 1
            results.append(compact_run)

    results.sort(
        key=lambda row: (
            row["summary"]["sharpeRatio"],
            row["summary"]["totalReturnPct"],
            -row["summary"]["maxDrawdownPct"],
        ),
        reverse=True,
    )
    return results, RunStoreSummary(
        cached_run_count=cached_run_count,
        computed_run_count=computed_run_count,
    )

def build_ranking_evaluation_runs(
    *,
    comparison: ComparisonSpec,
    market_bundles_by_timeframe: dict[str, dict],
    market_data_period: str,
    metadata_by_timeframe: dict[str, dict[str, str]],
    run_store: FileRunResultStore,
) -> tuple[list[dict], RunStoreSummary]:
    candidate_strategy_definitions = list(comparison.candidate_strategies)
    ranking_source_definitions = collect_ranking_source_definitions(candidate_strategy_definitions)
    ranking_specs = build_asset_ranking_specs_from_strategy_definitions(
        candidate_strategy_definitions
    )
    results: list[dict] = []
    cached_run_count = 0
    computed_run_count = 0
    required_fields = collect_required_market_fields(comparison)

    for ranking_spec in ranking_specs:
        timeframe_key = ranking_spec.timeframe.key
        market_bundle = market_bundles_by_timeframe[timeframe_key]
        dataset_metadata = metadata_by_timeframe[timeframe_key]
        closes = market_bundle["closes"][list(ranking_spec.investment_universe.tickers)]
        volumes = (
            market_bundle["volumes"][list(ranking_spec.investment_universe.tickers)]
            if market_bundle["volumes"] is not None
            else None
        )
        returns = closes.pct_change().dropna()
        aligned_volumes = volumes.loc[returns.index] if volumes is not None else None
        source_strategy_definition = ranking_source_definitions.get(ranking_spec.key)
        if source_strategy_definition is None:
            raise ValueError(f"No source strategy definition found for ranking: {ranking_spec.key}")
        serialized_ranking_spec = serialize_asset_ranking_spec(ranking_spec)
        run_spec = build_run_spec(
            run_kind="ranking_evaluation",
            strategy_definition=serialize_strategy_definition_payload(source_strategy_definition),
            evaluation_subject={"kind": "ranking", "ranking": serialized_ranking_spec},
            market_slice={
                "period": market_data_period,
                "timeframe": serialize_timeframe(ranking_spec.timeframe),
                "fields": required_fields,
            },
            evaluation=serialize_evaluation(
                comparison,
                {timeframe_key: dataset_metadata},
                [ranking_spec.timeframe],
                fields=required_fields,
                period_override=market_data_period,
            ),
            execution_assumptions=serialize_execution_assumptions(comparison),
            portfolio_state=serialize_portfolio_state(comparison.run_spec.portfolio_state),
            capital_base=comparison.run_spec.capital_base,
        )
        cached_run = run_store.load(run_spec)
        if cached_run is not None:
            cached_run_count += 1
            results.append(cached_run)
            continue

        result = evaluate_asset_ranking_spec(
            returns=returns,
            volumes=aligned_volumes,
            split_ratio=comparison.run_spec.evaluation.evaluation_settings.split_ratio,
            ranking_spec=ranking_spec,
            bars_per_year=ranking_spec.timeframe.bars_per_year,
        )
        run_store.save(run_spec, result)
        computed_run_count += 1
        results.append(result)

    results.sort(
        key=lambda row: (
            row["overall"]["meanRankIc"] if row["overall"]["meanRankIc"] is not None else float("-inf"),
            row["overall"]["meanTopMinusBottomPct"]
            if row["overall"]["meanTopMinusBottomPct"] is not None
            else float("-inf"),
        ),
        reverse=True,
    )
    return results, RunStoreSummary(
        cached_run_count=cached_run_count,
        computed_run_count=computed_run_count,
    )


def build_parameter_sweep_runs(
    *,
    comparison: ComparisonSpec,
    market_bundles_by_timeframe: dict[str, dict],
    market_data_period: str,
    metadata_by_timeframe: dict[str, dict[str, str]],
    run_store: FileRunResultStore,
) -> tuple[list[dict], RunStoreSummary]:
    base_evaluation = comparison.run_spec.evaluation
    results: list[dict] = []
    cached_run_count = 0
    computed_run_count = 0
    generation = {
        "method": "parameter_sweep",
        "batchKey": "local_tilt_search_9m_v1",
        "spec": build_parameter_sweep_generation_spec(),
    }
    strategy_definitions = comparison.candidate_strategies + comparison.reference_strategies
    predictor_specs = collect_strategy_predictor_specs(strategy_definitions)
    required_fields = collect_required_market_fields(
        comparison,
        predictor_specs,
        strategy_definitions=strategy_definitions,
    )

    family_specs = [
        {
            "familyKey": "momentum_top_9m",
            "familyLabel": "全資産モメンタム傾斜 上位優遇 9ヶ月",
            "strategyType": "full_universe_momentum_tilt",
            "macroWeights": [None],
            "windowSpec": {"unit": "months", "value": 9},
        },
        {
            "familyKey": "momentum_macro_top_9m",
            "familyLabel": "全資産モメンタムマクロ傾斜 上位優遇 9ヶ月",
            "strategyType": "full_universe_momentum_macro_tilt",
            "macroWeights": [0.05, 0.10, 0.15, 0.20],
            "windowSpec": {"unit": "months", "value": 9},
        },
    ]
    tilt_strengths = [0.15, 0.20, 0.25, 0.30, 0.35]
    max_weights = [0.40, 0.425, 0.45, 0.475, 0.50]

    for family_spec in family_specs:
        for tilt_strength in tilt_strengths:
            for macro_weight in family_spec["macroWeights"]:
                for max_weight in max_weights:
                    score_parameters = {
                        "tilt_strength": tilt_strength,
                        "tilt_shape": 1.0,
                        "windowSpec": dict(family_spec["windowSpec"]),
                    }
                    if macro_weight is not None:
                        score_parameters["momentum_weight"] = round(1.0 - macro_weight, 2)
                        score_parameters["macro_weight"] = macro_weight

                    selection_spec = build_selection_spec(
                        strategy_type=family_spec["strategyType"],
                        key=(
                            f"{family_spec['familyKey']}"
                            f"__tilt_{str(tilt_strength).replace('.', '_')}"
                            f"{'' if macro_weight is None else f'__macro_{str(macro_weight).replace('.', '_')}'}"
                            f"__cap_{str(max_weight).replace('.', '_')}"
                        ),
                        label=family_spec["familyLabel"],
                        description="Local parameter sweep strategy",
                        score_parameters=score_parameters,
                    )
                    base_strategy = next(
                        strategy
                        for strategy in comparison.candidate_strategies
                        if strategy.portfolio_model.model_type == "hierarchical_risk_parity"
                    )
                    base_market_data_timeframe = resolve_strategy_market_data_timeframe(base_strategy)
                    effective_strategy_definition = build_parameter_sweep_strategy_definition(
                        base_strategy=base_strategy,
                        selection_spec=selection_spec,
                        market_data_timeframe=base_market_data_timeframe,
                        max_weight=max_weight,
                    )
                    serialized_strategy_definition = serialize_strategy_definition_payload(effective_strategy_definition)
                    market_data_timeframe = resolve_strategy_market_data_timeframe(
                        effective_strategy_definition
                    )
                    timeframe_key = market_data_timeframe.key
                    dataset_metadata = metadata_by_timeframe[timeframe_key]
                    market_bundle = market_bundles_by_timeframe[timeframe_key]
                    serialized_evaluation = serialize_evaluation(
                        comparison,
                        {timeframe_key: dataset_metadata},
                        [market_data_timeframe],
                        fields=required_fields,
                        period_override=market_data_period,
                    )
                    run_spec = build_run_spec(
                        run_kind="portfolio_comparison",
                        market_slice={
                            "period": market_data_period,
                            "timeframe": serialize_timeframe(market_data_timeframe),
                            "fields": required_fields,
                        },
                        evaluation=serialized_evaluation,
                        execution_assumptions=serialize_execution_assumptions(comparison),
                        portfolio_state=serialize_portfolio_state(comparison.run_spec.portfolio_state),
                        capital_base=comparison.run_spec.capital_base,
                        generation=generation,
                        strategy_definition=serialized_strategy_definition,
                    )
                    cached_run = run_store.load(run_spec)
                    if cached_run is not None:
                        cached_run_count += 1
                        results.append(cached_run)
                        continue

                    run = evaluate_strategy_definition_run(
                        closes=market_bundle["closes"],
                        volumes=market_bundle["volumes"],
                        strategy_definition=effective_strategy_definition,
                        bars_per_year=market_data_timeframe.bars_per_year,
                        initial_capital=comparison.run_spec.capital_base,
                        split_ratio=base_evaluation.evaluation_settings.split_ratio,
                        execution_assumptions=serialize_execution_assumptions(comparison),
                        portfolio_state=comparison.run_spec.portfolio_state,
                    )
                    compact_run = compact_parameter_sweep_run(
                        run=run,
                        family_key=family_spec["familyKey"],
                        family_label=family_spec["familyLabel"],
                        tilt_strength=tilt_strength,
                        macro_weight=macro_weight,
                        max_weight=max_weight,
                        window_spec=family_spec["windowSpec"],
                        strategy=serialized_strategy_definition,
                    )
                    run_store.save(run_spec, compact_run)
                    computed_run_count += 1
                    results.append(compact_run)

    results.sort(
        key=lambda row: (
            row["summary"]["sharpeRatio"],
            row["summary"]["totalReturnPct"],
            -row["summary"]["maxDrawdownPct"],
        ),
        reverse=True,
    )
    return results, RunStoreSummary(
        cached_run_count=cached_run_count,
        computed_run_count=computed_run_count,
    )


def compact_condition_sweep_run(
    *,
    run: dict,
    key: str,
    condition_variant: dict,
) -> dict:
    return {
        "key": key,
        "strategy": run["strategy"],
        "conditionVariant": condition_variant,
        "weights": run["weights"],
        "selectedAssets": run["selectedAssets"],
        "summary": run["summary"],
        "splitAnalysis": run["splitAnalysis"],
    }


def compact_parameter_sweep_run(
    *,
    run: dict,
    family_key: str,
    family_label: str,
    tilt_strength: float,
    macro_weight: float | None,
    max_weight: float,
    window_spec: dict[str, object],
    strategy: dict,
) -> dict:
    return {
        "key": (
            f"{family_key}"
            f"__tilt_{tilt_strength}"
            f"{'' if macro_weight is None else f'__macro_{macro_weight}'}"
            f"__cap_{max_weight}"
        ),
        "family": {
            "key": family_key,
            "label": family_label,
        },
        "parameterSet": {
            "tiltStrength": tilt_strength,
            "macroWeight": macro_weight,
            "windowSpec": dict(window_spec),
            "maxWeightPct": round(max_weight * 100, 1),
        },
        "strategy": strategy,
        "weights": run["weights"],
        "selectedAssets": run["selectedAssets"],
        "summary": run["summary"],
        "splitAnalysis": run["splitAnalysis"],
    }


def build_parameter_sweep_generation_spec() -> dict:
    return {
        "families": [
            {
                "key": "momentum_top_9m",
                "strategyType": "full_universe_momentum_tilt",
                "tiltShape": "top_favored",
                "windowSpec": {"unit": "months", "value": 9},
            },
            {
                "key": "momentum_macro_top_9m",
                "strategyType": "full_universe_momentum_macro_tilt",
                "tiltShape": "top_favored",
                "windowSpec": {"unit": "months", "value": 9},
            },
        ],
        "parameterGrid": {
            "tiltStrength": [0.15, 0.20, 0.25, 0.30, 0.35],
            "macroWeight": [0.05, 0.10, 0.15, 0.20],
            "maxWeight": [0.40, 0.425, 0.45, 0.475, 0.50],
        },
    }
