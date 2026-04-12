from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from app.portfolio import (
    build_asset_ranking_specs,
    build_risk_controls_spec,
    build_selection_spec,
    build_strategy_spec,
    compute_predictor_panel,
    deserialize_predictor_panel,
    evaluate_asset_ranking_spec,
    evaluate_predictor_spec,
    evaluate_strategy_run,
    serialize_asset_ranking_spec,
    serialize_predictor_spec,
    serialize_predictor_panel,
    serialize_portfolio_state,
    serialize_strategy_spec,
)
from app.predictor_registry import REGISTERED_PREDICTOR_SPECS_BY_KEY
from app.comparison_models import ComparisonSpec, ConditionVariant, EvaluationSpec
from app.run_store import FileRunResultStore, RunStoreSummary, build_run_spec
from app.timeframe_models import TimeframeSpec


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def collect_comparison_tickers(comparison: ComparisonSpec) -> list[str]:
    seen: dict[str, None] = {}
    for strategy in (
        comparison.candidate_strategies + comparison.reference_strategies
    ):
        for ticker in strategy.investment_universe.tickers:
            seen.setdefault(ticker, None)
    return list(seen.keys())


def collect_comparison_timeframes(comparison: ComparisonSpec) -> list[TimeframeSpec]:
    seen: dict[str, TimeframeSpec] = {}
    for strategy in (
        comparison.candidate_strategies + comparison.reference_strategies
    ):
        seen.setdefault(strategy.timeframe.key, strategy.timeframe)
    return sorted(
        seen.values(),
        key=lambda timeframe: (timeframe.bar_seconds, timeframe.key),
    )


def collect_required_market_fields(comparison: ComparisonSpec) -> list[str]:
    fields: dict[str, None] = {"close": None}
    for strategy in (
        comparison.candidate_strategies + comparison.reference_strategies
    ):
        for field in strategy.selection.feature_inputs:
            fields.setdefault(field, None)
    if comparison.run_spec.execution_assumptions.cost_model.kind == "asset_specific_adv_cost":
        fields.setdefault("volume", None)
    return list(fields.keys())


def collect_strategy_predictor_specs(strategy_specs: list) -> list:
    predictor_specs = []
    seen_keys: set[str] = set()

    for strategy_spec in strategy_specs:
        if strategy_spec.predictor_use is None:
            continue
        predictor_key = strategy_spec.predictor_use.predictor_key
        predictor_spec = REGISTERED_PREDICTOR_SPECS_BY_KEY.get(predictor_key)
        if predictor_spec is None:
            raise ValueError(f"Unknown predictor key: {predictor_key}")
        if predictor_spec.timeframe.key != strategy_spec.timeframe.key:
            raise ValueError("Supplemental predictor timeframe must match strategy timeframe.")
        if predictor_spec.investment_universe.tickers != strategy_spec.investment_universe.tickers:
            raise ValueError("Supplemental predictor universe must match strategy universe.")
        if predictor_key in seen_keys:
            continue
        seen_keys.add(predictor_key)
        predictor_specs.append(predictor_spec)

    return predictor_specs


def fetch_market_data_by_timeframe(
    comparison: ComparisonSpec,
    *,
    period: str,
    fetch_market_universe_bundle,
) -> tuple[dict[str, dict], dict[str, dict], list[str], list[TimeframeSpec]]:
    comparison_tickers = collect_comparison_tickers(comparison)
    timeframes = collect_comparison_timeframes(comparison)
    bundles_by_timeframe: dict[str, dict] = {}
    metadata_by_timeframe: dict[str, dict] = {}

    for timeframe in timeframes:
        market_bundle, metadata = fetch_market_universe_bundle(
            tickers=comparison_tickers,
            period=period,
            timeframe=timeframe.yfinance_interval,
        )
        bundles_by_timeframe[timeframe.key] = market_bundle
        metadata_by_timeframe[timeframe.key] = metadata

    return bundles_by_timeframe, metadata_by_timeframe, comparison_tickers, timeframes


def build_dashboard_payload(
    comparison: ComparisonSpec,
    *,
    fetch_market_universe_bundle,
) -> dict:
    run_store = build_run_result_store(comparison)
    (
        market_bundles_by_timeframe,
        metadata_by_timeframe,
        comparison_tickers,
        comparison_timeframes,
    ) = fetch_market_data_by_timeframe(
        comparison,
        period=comparison.run_spec.market_slice.period,
        fetch_market_universe_bundle=fetch_market_universe_bundle,
    )
    predictor_specs = collect_strategy_predictor_specs(
        comparison.candidate_strategies + comparison.reference_strategies
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
        strategy_specs=comparison.candidate_strategies,
        market_bundles_by_timeframe=market_bundles_by_timeframe,
        market_data_period=comparison.run_spec.market_slice.period,
        metadata_by_timeframe=metadata_by_timeframe,
        run_store=run_store,
        predictor_panels_by_key=predictor_panels_by_key,
    )
    reference_runs, reference_run_store_summary = build_strategy_runs(
        comparison=comparison,
        strategy_specs=comparison.reference_strategies,
        market_bundles_by_timeframe=market_bundles_by_timeframe,
        market_data_period=comparison.run_spec.market_slice.period,
        metadata_by_timeframe=metadata_by_timeframe,
        run_store=run_store,
        predictor_panels_by_key=predictor_panels_by_key,
    )
    sanity_checks = []
    required_fields = collect_required_market_fields(comparison)
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
            strategy_specs=comparison.candidate_strategies,
            market_bundles_by_timeframe=sanity_bundles_by_timeframe,
            market_data_period=period,
            metadata_by_timeframe=sanity_metadata_by_timeframe,
            run_store=run_store,
            predictor_panels_by_key=sanity_predictor_panels_by_key,
        )
        sanity_reference_runs, sanity_reference_run_store_summary = build_strategy_runs(
            comparison=comparison,
            strategy_specs=comparison.reference_strategies,
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
    predictor_runs, _predictor_panels_by_key, predictor_run_store_summary = build_predictor_runs(
        comparison=comparison,
        predictor_specs=predictor_specs,
        market_bundles_by_timeframe=market_bundles_by_timeframe,
        market_data_period=comparison.run_spec.market_slice.period,
        metadata_by_timeframe=metadata_by_timeframe,
        run_store=run_store,
    )
    required_fields = collect_required_market_fields(comparison)
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
                ),
                "runStoreSummary": sanity_run_store_summary.to_payload(),
                "predictorRuns": sanity_predictor_runs,
            }
        )

    return {
        "kind": "predictor_run_collection",
        "schemaVersion": "v1",
        "comparisonId": comparison.comparison_id,
        "runSpec": serialize_run_spec(comparison, metadata_by_timeframe, comparison_timeframes),
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
    predictor_specs = collect_strategy_predictor_specs(
        comparison.candidate_strategies + comparison.reference_strategies
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
        strategy_specs=comparison.candidate_strategies,
        market_bundles_by_timeframe=market_bundles_by_timeframe,
        market_data_period=comparison.run_spec.market_slice.period,
        metadata_by_timeframe=metadata_by_timeframe,
        run_store=run_store,
        predictor_panels_by_key=predictor_panels_by_key,
    )
    reference_runs, reference_run_store_summary = build_strategy_runs(
        comparison=comparison,
        strategy_specs=comparison.reference_strategies,
        market_bundles_by_timeframe=market_bundles_by_timeframe,
        market_data_period=comparison.run_spec.market_slice.period,
        metadata_by_timeframe=metadata_by_timeframe,
        run_store=run_store,
        predictor_panels_by_key=predictor_panels_by_key,
    )
    required_fields = collect_required_market_fields(comparison)
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
            strategy_specs=comparison.candidate_strategies,
            market_bundles_by_timeframe=sanity_bundles_by_timeframe,
            market_data_period=period,
            metadata_by_timeframe=sanity_metadata_by_timeframe,
            run_store=run_store,
            predictor_panels_by_key=sanity_predictor_panels_by_key,
        )
        sanity_reference_runs, sanity_reference_run_store_summary = build_strategy_runs(
            comparison=comparison,
            strategy_specs=comparison.reference_strategies,
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
        "runSpec": serialize_run_spec(comparison, metadata_by_timeframe, comparison_timeframes),
        "candidateStrategies": [
            serialize_strategy_spec(strategy_spec)
            for strategy_spec in comparison.candidate_strategies
        ],
        "referenceStrategies": [
            serialize_strategy_spec(strategy_spec)
            for strategy_spec in comparison.reference_strategies
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
    results, run_store_summary = build_condition_sweep_runs(
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
        ),
        "resultCount": len(results),
        "runStoreSummary": run_store_summary.to_payload(),
        "results": results,
    }


def build_run_catalog_payload(
    comparison: ComparisonSpec,
    *,
    limit: int = 50,
    run_kind: str | None = None,
    generation_method: str | None = None,
) -> dict:
    run_store = build_run_result_store(comparison)
    records = run_store.list_records(
        run_kind=run_kind,
        generation_method=generation_method,
        limit=limit,
    )
    return {
        "comparisonId": comparison.comparison_id,
        "limit": limit,
        "runKind": run_kind,
        "generationMethod": generation_method,
        "recordCount": len(records),
        "records": [compact_run_record(record) for record in records],
    }


def build_predictor_run_index_payload(
    comparison: ComparisonSpec,
    *,
    limit: int = 50,
    model_kind: str | None = None,
    horizon_value: int | None = None,
    sort_by: str = "test_rank_ic",
) -> dict:
    run_store = build_run_result_store(comparison)
    all_records = [
        compact_predictor_run_record(record)
        for record in run_store.list_records(run_kind="predictor_run")
    ]
    if model_kind is not None:
        all_records = [
            record for record in all_records
            if record["modelKind"] == model_kind
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
            "modelKind": model_kind,
            "horizonValue": horizon_value,
        },
        "totalCount": len(all_records),
        "recordCount": len(records),
        "records": records,
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
) -> dict:
    run_store = build_run_result_store(comparison)
    total_count = len(run_store.list_records(run_kind="strategy_run"))
    records = run_store.list_records(run_kind="strategy_run", limit=limit)
    return {
        "kind": "strategy_run_index",
        "schemaVersion": "v1",
        "comparisonId": comparison.comparison_id,
        "limit": limit,
        "totalCount": total_count,
        "recordCount": len(records),
        "records": [compact_strategy_run_record(record) for record in records],
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


def serialize_evaluation(
    comparison: ComparisonSpec,
    metadata_by_timeframe: dict[str, dict[str, str]],
    timeframes: list[TimeframeSpec],
    *,
    fields: list[str],
    period_override: str | None = None,
) -> dict:
    return {
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


def serialize_run_spec(
    comparison: ComparisonSpec,
    metadata_by_timeframe: dict[str, dict[str, str]],
    timeframes: list[TimeframeSpec],
) -> dict:
    fields = collect_required_market_fields(comparison)
    return {
        "kind": "comparison_run_spec",
        "schemaVersion": "v1",
        "marketSlice": {
            "period": comparison.run_spec.market_slice.period,
            "sanityPeriods": comparison.run_spec.market_slice.sanity_periods,
            "timeframes": [serialize_timeframe(timeframe) for timeframe in timeframes],
            "fields": fields,
        },
        "portfolioState": serialize_portfolio_state(comparison.run_spec.portfolio_state),
        "capitalBase": round(comparison.run_spec.capital_base, 2),
        "executionAssumptions": serialize_execution_assumptions(comparison),
        "evaluation": serialize_evaluation(
            comparison,
            metadata_by_timeframe,
            timeframes,
            fields=fields,
        ),
    }


def serialize_comparison(
    comparison: ComparisonSpec,
    metadata_by_timeframe: dict[str, dict[str, str]],
    timeframes: list[TimeframeSpec],
) -> dict:
    comparison_tickers = collect_comparison_tickers(comparison)
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
        "runSpec": serialize_run_spec(comparison, metadata_by_timeframe, timeframes),
        "candidateStrategies": [
            serialize_strategy_spec(strategy_spec)
            for strategy_spec in comparison.candidate_strategies
        ],
        "referenceStrategies": [
            serialize_strategy_spec(strategy_spec)
            for strategy_spec in comparison.reference_strategies
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
    summary = result.get("summary", {})
    portfolio_summary = summary.get("portfolio", summary)

    return {
        "runKey": record["runKey"],
        "savedAtUtc": record.get("savedAtUtc"),
        "runKind": run_spec.get("runKind"),
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


def compact_predictor_run_record(record: dict) -> dict:
    run_spec = record["runSpec"]
    result = record["result"]
    predictor = run_spec.get("strategy", {}).get("predictor", {})
    predicted_quantity = predictor.get("predictedQuantitySpec", {})
    target = predictor.get("targetSpec", {})
    horizon = target.get("horizonSpec", {})
    feature = predictor.get("featureSpec", {})
    training = predictor.get("trainingSpec", {})
    overall = result.get("overall", {})
    test = result.get("test", {})

    return {
        "runKey": record["runKey"],
        "savedAtUtc": record.get("savedAtUtc"),
        "runKind": run_spec.get("runKind"),
        "predictorKey": predictor.get("key"),
        "predictorLabel": predictor.get("label"),
        "featureKey": feature.get("key"),
        "modelKind": predictor.get("modelSpec", {}).get("modelKind"),
        "trainingFitMode": training.get("fitMode"),
        "trainingMinSamples": training.get("minTrainSamples"),
        "timeframe": predictor.get("timeframe", {}).get("key"),
        "targetKey": target.get("key"),
        "predictedQuantityKind": predicted_quantity.get("quantityKind"),
        "targetBaseline": target.get("baseline"),
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
    strategy_specs: list,
    market_bundles_by_timeframe: dict[str, dict],
    market_data_period: str,
    metadata_by_timeframe: dict[str, dict[str, str]],
    run_store: FileRunResultStore,
    predictor_panels_by_key: dict[str, object] | None = None,
) -> tuple[list[dict], RunStoreSummary]:
    serialized_execution_assumptions = serialize_execution_assumptions(comparison)
    required_fields = collect_required_market_fields(comparison)
    runs: list[dict] = []
    cached_run_count = 0
    computed_run_count = 0

    for strategy_spec in strategy_specs:
        timeframe_key = strategy_spec.timeframe.key
        dataset_metadata = metadata_by_timeframe[timeframe_key]
        serialized_strategy = serialize_strategy_spec(strategy_spec)
        predictor_panel = None
        if strategy_spec.predictor_use is not None:
            predictor_key = strategy_spec.predictor_use.predictor_key
            predictor_panel = None if predictor_panels_by_key is None else predictor_panels_by_key.get(predictor_key)
        market_bundle = market_bundles_by_timeframe[timeframe_key]
        strategy_market_slice = {
            "period": market_data_period,
            "timeframe": serialize_timeframe(strategy_spec.timeframe),
            "fields": required_fields,
        }
        run_spec = build_run_spec(
            run_kind="strategy_run",
            strategy=serialized_strategy,
            market_slice=strategy_market_slice,
            evaluation=serialize_evaluation(
                comparison,
                {timeframe_key: dataset_metadata},
                [strategy_spec.timeframe],
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
            runs.append(cached_run)
            continue

        run = evaluate_strategy_run(
            closes=market_bundle["closes"],
            volumes=market_bundle["volumes"],
            strategy=strategy_spec,
            bars_per_year=strategy_spec.timeframe.bars_per_year,
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
    required_fields = collect_required_market_fields(comparison)
    serialized_execution_assumptions = serialize_execution_assumptions(comparison)

    for predictor_spec in predictor_specs:
        timeframe_key = predictor_spec.timeframe.key
        market_bundle = market_bundles_by_timeframe[timeframe_key]
        dataset_metadata = metadata_by_timeframe[timeframe_key]
        closes = market_bundle["closes"][list(predictor_spec.investment_universe.tickers)]
        volumes = (
            market_bundle["volumes"][list(predictor_spec.investment_universe.tickers)]
            if market_bundle["volumes"] is not None
            else None
        )
        returns = closes.pct_change().dropna()
        aligned_volumes = volumes.loc[returns.index] if volumes is not None else None
        run_spec = build_run_spec(
            run_kind="predictor_run",
            strategy={"predictor": serialize_predictor_spec(predictor_spec)},
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
) -> tuple[list[dict], RunStoreSummary]:
    results: list[dict] = []
    cached_run_count = 0
    computed_run_count = 0
    required_fields = collect_required_market_fields(comparison)

    for strategy_spec in comparison.candidate_strategies:
        timeframe_key = strategy_spec.timeframe.key
        market_bundle = market_bundles_by_timeframe[timeframe_key]
        dataset_metadata = metadata_by_timeframe[timeframe_key]
        for condition_variant in comparison.condition_variants:
            effective_strategy = replace(
                strategy_spec,
                risk_controls=build_risk_controls_spec(
                    max_investment_ratio=condition_variant.max_investment_ratio,
                    max_weight=condition_variant.max_weight,
                ),
            )
            effective_evaluation = replace(
                comparison.run_spec.evaluation,
            )
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
            serialized_strategy = serialize_strategy_spec(effective_strategy)
            serialized_evaluation = serialize_evaluation(
                comparison,
                {timeframe_key: dataset_metadata},
                [strategy_spec.timeframe],
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
                strategy=serialized_strategy,
                market_slice={
                    "period": market_data_period,
                    "timeframe": serialize_timeframe(strategy_spec.timeframe),
                    "fields": required_fields,
                },
                evaluation=serialized_evaluation,
                execution_assumptions=serialized_execution_assumptions,
                portfolio_state=serialize_portfolio_state(comparison.run_spec.portfolio_state),
                capital_base=comparison.run_spec.capital_base,
            )
            cached_run = run_store.load(run_spec)
            if cached_run is not None:
                cached_run_count += 1
                results.append(cached_run)
                continue

            run = evaluate_strategy_run(
                closes=market_bundle["closes"],
                volumes=market_bundle["volumes"],
                strategy=effective_strategy,
                bars_per_year=strategy_spec.timeframe.bars_per_year,
                initial_capital=comparison.run_spec.capital_base,
                split_ratio=effective_evaluation.evaluation_settings.split_ratio,
                execution_assumptions=serialized_execution_assumptions,
                portfolio_state=comparison.run_spec.portfolio_state,
            )
            compact_run = compact_condition_sweep_run(
                run=run,
                key=f"{strategy_spec.key}__{condition_variant.key}",
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
    ranking_specs = build_asset_ranking_specs(comparison.candidate_strategies)
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
        serialized_ranking_spec = serialize_asset_ranking_spec(ranking_spec)
        run_spec = build_run_spec(
            run_kind="ranking_evaluation",
            strategy={"ranking": serialized_ranking_spec},
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
    required_fields = collect_required_market_fields(comparison)

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
                    effective_strategy = build_strategy_spec(
                        investment_universe=base_strategy.investment_universe,
                        selection=selection_spec,
                        portfolio_model=base_strategy.portfolio_model,
                        risk_controls=build_risk_controls_spec(
                            max_investment_ratio=1.0,
                            max_weight=max_weight,
                        ),
                    )
                    serialized_strategy = serialize_strategy_spec(effective_strategy)
                    timeframe_key = effective_strategy.timeframe.key
                    dataset_metadata = metadata_by_timeframe[timeframe_key]
                    market_bundle = market_bundles_by_timeframe[timeframe_key]
                    serialized_evaluation = serialize_evaluation(
                        comparison,
                        {timeframe_key: dataset_metadata},
                        [effective_strategy.timeframe],
                        fields=required_fields,
                        period_override=market_data_period,
                    )
                    run_spec = build_run_spec(
                        run_kind="portfolio_comparison",
                        strategy=serialized_strategy,
                        market_slice={
                            "period": market_data_period,
                            "timeframe": serialize_timeframe(effective_strategy.timeframe),
                            "fields": required_fields,
                        },
                        evaluation=serialized_evaluation,
                        execution_assumptions=serialize_execution_assumptions(comparison),
                        portfolio_state=serialize_portfolio_state(comparison.run_spec.portfolio_state),
                        capital_base=comparison.run_spec.capital_base,
                        generation=generation,
                    )
                    cached_run = run_store.load(run_spec)
                    if cached_run is not None:
                        cached_run_count += 1
                        results.append(cached_run)
                        continue

                    run = evaluate_strategy_run(
                        closes=market_bundle["closes"],
                        volumes=market_bundle["volumes"],
                        strategy=effective_strategy,
                        bars_per_year=effective_strategy.timeframe.bars_per_year,
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
                        strategy=serialized_strategy,
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
