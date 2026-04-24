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


def build_run_result_store(comparison: ComparisonSpec) -> FileRunResultStore:
    root_dir = Path(comparison.result_store_dir)
    if not root_dir.is_absolute():
        root_dir = PROJECT_ROOT / root_dir
    return FileRunResultStore(root_dir)
