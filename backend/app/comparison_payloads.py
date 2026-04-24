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
    build_run_result_store,
    collect_required_market_fields,
    collect_strategy_predictor_source_definitions,
    collect_strategy_predictor_specs,
    fetch_market_data_by_timeframe,
)
from app.comparison_run_builders import (
    build_condition_sweep_runs,
    build_parameter_sweep_generation_spec,
    build_parameter_sweep_runs,
    build_predictor_runs,
    build_ranking_evaluation_runs,
    build_strategy_runs,
)
from app.comparison_serialization import (
    compact_predictor_run_record,
    compact_run_record,
    compact_strategy_run_record,
    deserialize_comparison_run_spec_payload,
    serialize_comparison,
    serialize_condition_variant,
    serialize_evaluation,
    serialize_predictor_spec,
    serialize_reproducible_strategy_definition,
    serialize_run_spec,
    serialize_strategy_definition_payload,
    sort_predictor_run_records,
    summarize_predictor_record_groups,
)
from app.comparison_walk_forward import build_walk_forward_comparison_payload


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
