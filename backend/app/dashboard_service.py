from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from app.portfolio import (
    build_asset_ranking_definitions,
    build_portfolio_strategy_definition,
    build_risk_controls_definition,
    build_strategy_definition,
    evaluate_asset_ranking_definition,
    evaluate_strategy_run,
    serialize_asset_ranking_definition,
    serialize_portfolio_state,
    serialize_strategy_definition,
)
from app.comparison_models import ComparisonDefinition, ConditionVariant, EvaluationContext
from app.run_store import FileRunResultStore, RunStoreSummary, build_run_definition


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def collect_comparison_tickers(comparison: ComparisonDefinition) -> list[str]:
    seen: dict[str, None] = {}
    for strategy_definition in (
        comparison.candidate_strategy_definitions + comparison.reference_strategy_definitions
    ):
        for ticker in strategy_definition.investment_universe_definition.tickers:
            seen.setdefault(ticker, None)
    return list(seen.keys())


def build_dashboard_payload(
    comparison: ComparisonDefinition,
    *,
    fetch_market_universe_bundle,
) -> dict:
    run_store = build_run_result_store(comparison)
    comparison_tickers = collect_comparison_tickers(comparison)
    market_bundle, metadata = fetch_market_universe_bundle(
        tickers=comparison_tickers,
        period=comparison.dataset_spec.period,
    )
    candidate_runs, candidate_run_store_summary = build_strategy_runs(
        comparison=comparison,
        strategy_definitions=comparison.candidate_strategy_definitions,
        closes=market_bundle["closes"],
        volumes=market_bundle["volumes"],
        dataset_period=comparison.dataset_spec.period,
        dataset_metadata=metadata,
        run_store=run_store,
    )
    reference_runs, reference_run_store_summary = build_strategy_runs(
        comparison=comparison,
        strategy_definitions=comparison.reference_strategy_definitions,
        closes=market_bundle["closes"],
        volumes=market_bundle["volumes"],
        dataset_period=comparison.dataset_spec.period,
        dataset_metadata=metadata,
        run_store=run_store,
    )
    sanity_checks = []
    total_cached_runs = (
        candidate_run_store_summary.cached_run_count + reference_run_store_summary.cached_run_count
    )
    total_computed_runs = (
        candidate_run_store_summary.computed_run_count + reference_run_store_summary.computed_run_count
    )
    for period in comparison.dataset_spec.sanity_periods:
        sanity_bundle, sanity_metadata = fetch_market_universe_bundle(
            tickers=comparison_tickers,
            period=period,
        )
        sanity_candidate_runs, sanity_candidate_run_store_summary = build_strategy_runs(
            comparison=comparison,
            strategy_definitions=comparison.candidate_strategy_definitions,
            closes=sanity_bundle["closes"],
            volumes=sanity_bundle["volumes"],
            dataset_period=period,
            dataset_metadata=sanity_metadata,
            run_store=run_store,
        )
        sanity_reference_runs, sanity_reference_run_store_summary = build_strategy_runs(
            comparison=comparison,
            strategy_definitions=comparison.reference_strategy_definitions,
            closes=sanity_bundle["closes"],
            volumes=sanity_bundle["volumes"],
            dataset_period=period,
            dataset_metadata=sanity_metadata,
            run_store=run_store,
        )
        total_cached_runs += (
            sanity_candidate_run_store_summary.cached_run_count
            + sanity_reference_run_store_summary.cached_run_count
        )
        total_computed_runs += (
            sanity_candidate_run_store_summary.computed_run_count
            + sanity_reference_run_store_summary.computed_run_count
        )
        sanity_checks.append(
            {
                "period": period,
                "evaluationContext": serialize_evaluation_context(
                    comparison,
                    sanity_metadata,
                    period_override=period,
                ),
                "runStoreSummary": {
                    "cachedRunCount": (
                        sanity_candidate_run_store_summary.cached_run_count
                        + sanity_reference_run_store_summary.cached_run_count
                    ),
                    "computedRunCount": (
                        sanity_candidate_run_store_summary.computed_run_count
                        + sanity_reference_run_store_summary.computed_run_count
                    ),
                },
                "candidateRuns": sanity_candidate_runs,
                "referenceRuns": sanity_reference_runs,
            }
        )

    return {
        "comparison": serialize_comparison(comparison, metadata),
        "candidateRuns": candidate_runs,
        "referenceRuns": reference_runs,
        "runStoreSummary": {
            "cachedRunCount": total_cached_runs,
            "computedRunCount": total_computed_runs,
        },
        "sanityChecks": sanity_checks,
    }


def build_condition_sweep_payload(
    comparison: ComparisonDefinition,
    *,
    fetch_market_universe_bundle,
) -> dict:
    run_store = build_run_result_store(comparison)
    comparison_tickers = collect_comparison_tickers(comparison)
    market_bundle, metadata = fetch_market_universe_bundle(
        tickers=comparison_tickers,
        period=comparison.dataset_spec.period,
    )
    results, run_store_summary = build_condition_sweep_runs(
        comparison=comparison,
        closes=market_bundle["closes"],
        volumes=market_bundle["volumes"],
        dataset_period=comparison.dataset_spec.period,
        dataset_metadata=metadata,
        run_store=run_store,
    )
    return {
        "comparison": serialize_comparison(comparison, metadata),
        "conditionVariants": [
            serialize_condition_variant(condition_variant)
            for condition_variant in comparison.condition_variants
        ],
        "resultCount": len(results),
        "runStoreSummary": run_store_summary.to_payload(),
        "results": results,
    }


def build_ranking_evaluation_payload(
    comparison: ComparisonDefinition,
    *,
    fetch_market_universe_bundle,
) -> dict:
    run_store = build_run_result_store(comparison)
    comparison_tickers = collect_comparison_tickers(comparison)
    market_bundle, metadata = fetch_market_universe_bundle(
        tickers=comparison_tickers,
        period=comparison.dataset_spec.period,
    )
    results, run_store_summary = build_ranking_evaluation_runs(
        comparison=comparison,
        closes=market_bundle["closes"],
        volumes=market_bundle["volumes"],
        dataset_period=comparison.dataset_spec.period,
        dataset_metadata=metadata,
        run_store=run_store,
    )
    return {
        "comparison": serialize_comparison(comparison, metadata),
        "resultCount": len(results),
        "runStoreSummary": run_store_summary.to_payload(),
        "results": results,
    }


def build_run_catalog_payload(
    comparison: ComparisonDefinition,
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


def generate_parameter_sweep_runs_payload(
    comparison: ComparisonDefinition,
    *,
    fetch_market_universe_bundle,
) -> dict:
    run_store = build_run_result_store(comparison)
    comparison_tickers = collect_comparison_tickers(comparison)
    market_bundle, metadata = fetch_market_universe_bundle(
        tickers=comparison_tickers,
        period=comparison.dataset_spec.period,
    )
    results, run_store_summary = build_parameter_sweep_runs(
        comparison=comparison,
        closes=market_bundle["closes"],
        volumes=market_bundle["volumes"],
        dataset_period=comparison.dataset_spec.period,
        dataset_metadata=metadata,
        run_store=run_store,
    )
    return {
        "comparison": serialize_comparison(comparison, metadata),
        "generation": {
            "method": "parameter_sweep",
            "batchKey": "local_tilt_search_v1",
            "spec": build_parameter_sweep_generation_spec(),
        },
        "resultCount": len(results),
        "runStoreSummary": run_store_summary.to_payload(),
        "results": results,
    }


def build_run_result_store(comparison: ComparisonDefinition) -> FileRunResultStore:
    root_dir = Path(comparison.result_store_dir)
    if not root_dir.is_absolute():
        root_dir = PROJECT_ROOT / root_dir
    return FileRunResultStore(root_dir)


def serialize_dataset_context(
    comparison: ComparisonDefinition,
    dataset_metadata: dict[str, str],
    *,
    period_override: str | None = None,
) -> dict:
    return {
        "period": period_override or comparison.dataset_spec.period,
        "sanityPeriods": comparison.dataset_spec.sanity_periods,
        "source": dataset_metadata["source"],
        "alignedStartDate": dataset_metadata["aligned_start_date"],
        "alignedEndDate": dataset_metadata["aligned_end_date"],
        "rowCount": dataset_metadata["row_count"],
    }


def serialize_cost_assumptions(evaluation_context: EvaluationContext) -> dict:
    return {
        "commissionPct": round(evaluation_context.cost_assumptions.commission_pct, 3),
        "slippagePct": round(evaluation_context.cost_assumptions.slippage_pct, 3),
    }


def serialize_evaluation_settings(evaluation_context: EvaluationContext) -> dict:
    return {
        "splitRatioPct": round(evaluation_context.evaluation_settings.split_ratio * 100, 1),
        "initialCapital": round(evaluation_context.evaluation_settings.initial_capital, 2),
    }


def serialize_evaluation_context(
    comparison: ComparisonDefinition,
    dataset_metadata: dict[str, str],
    *,
    period_override: str | None = None,
) -> dict:
    return {
        "kind": "evaluation_context",
        "schemaVersion": "v1",
        "datasetContext": serialize_dataset_context(
            comparison,
            dataset_metadata,
            period_override=period_override,
        ),
        "evaluationSettings": serialize_evaluation_settings(comparison.run_input.evaluation_context),
        "costAssumptions": serialize_cost_assumptions(comparison.run_input.evaluation_context),
    }


def serialize_comparison(comparison: ComparisonDefinition, dataset_metadata: dict[str, str]) -> dict:
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
        "datasetSpec": {
            "period": comparison.dataset_spec.period,
            "sanityPeriods": comparison.dataset_spec.sanity_periods,
        },
        "runInput": {
            "portfolioState": serialize_portfolio_state(comparison.run_input.portfolio_state),
            "evaluationContext": serialize_evaluation_context(comparison, dataset_metadata),
        },
        "candidateStrategies": [
            serialize_strategy_definition(strategy_definition)
            for strategy_definition in comparison.candidate_strategy_definitions
        ],
        "referenceStrategies": [
            serialize_strategy_definition(strategy_definition)
            for strategy_definition in comparison.reference_strategy_definitions
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
    run_definition = record["runDefinition"]
    result = record["result"]
    strategy = run_definition.get("strategy", {})
    evaluation_context = run_definition.get("evaluationContext", {})
    evaluation_settings = evaluation_context.get("evaluationSettings", {})
    cost_assumptions = evaluation_context.get("costAssumptions", {})
    dataset_context = evaluation_context.get("datasetContext", {})
    summary = result.get("summary", {})
    portfolio_summary = summary.get("portfolio", summary)

    return {
        "runKey": record["runKey"],
        "savedAtUtc": record.get("savedAtUtc"),
        "runKind": run_definition.get("runKind"),
        "generationMethod": run_definition.get("generation", {}).get("method"),
        "generationBatchKey": run_definition.get("generation", {}).get("batchKey"),
        "strategyId": strategy.get("strategyId"),
        "strategyVersion": strategy.get("version"),
        "strategyLabel": strategy.get("label"),
        "strategyHypothesis": strategy.get("hypothesis"),
        "investmentUniverseLabel": strategy.get("components", {}).get("core", {}).get("investmentUniverse", {}).get("label"),
        "investmentUniverseAssetCount": strategy.get("components", {}).get("core", {}).get("investmentUniverse", {}).get("assetCount"),
        "portfolioModelLabel": strategy.get("components", {}).get("core", {}).get("portfolioModel", {}).get("label"),
        "executionLabel": strategy.get("components", {}).get("core", {}).get("executionPolicy", {}).get("label"),
        "period": dataset_context.get("period"),
        "maxInvestmentPct": strategy.get("components", {}).get("optional", {}).get("riskControls", {}).get("maxInvestmentPct"),
        "maxWeightPct": strategy.get("components", {}).get("optional", {}).get("riskControls", {}).get("maxWeightPct"),
        "commissionPct": cost_assumptions.get("commissionPct"),
        "sharpeRatio": portfolio_summary.get("sharpeRatio"),
        "totalReturnPct": portfolio_summary.get("totalReturnPct"),
        "maxDrawdownPct": portfolio_summary.get("maxDrawdownPct"),
    }


def build_strategy_runs(
    *,
    comparison: ComparisonDefinition,
    strategy_definitions: list,
    closes,
    volumes,
    dataset_period: str,
    dataset_metadata: dict[str, str],
    run_store: FileRunResultStore,
) -> tuple[list[dict], RunStoreSummary]:
    dataset_spec = {
        "period": dataset_period,
    }
    serialized_evaluation_context = serialize_evaluation_context(
        comparison,
        dataset_metadata,
        period_override=dataset_period,
    )
    runs: list[dict] = []
    cached_run_count = 0
    computed_run_count = 0

    for strategy_definition in strategy_definitions:
        serialized_strategy = serialize_strategy_definition(strategy_definition)
        run_definition = build_run_definition(
            run_kind="strategy_run",
            strategy=serialized_strategy,
            dataset_spec=dataset_spec,
            evaluation_context=serialized_evaluation_context,
            portfolio_state=serialize_portfolio_state(comparison.run_input.portfolio_state),
            dataset_metadata=dataset_metadata,
        )
        cached_run = run_store.load(run_definition)
        if cached_run is not None:
            cached_run_count += 1
            runs.append(cached_run)
            continue

        run = evaluate_strategy_run(
            closes=closes,
            volumes=volumes,
            strategy_definition=strategy_definition,
            initial_capital=comparison.run_input.evaluation_context.evaluation_settings.initial_capital,
            split_ratio=comparison.run_input.evaluation_context.evaluation_settings.split_ratio,
            transaction_cost=comparison.run_input.evaluation_context.cost_assumptions.commission_pct / 100,
            portfolio_state=comparison.run_input.portfolio_state,
        )
        run_store.save(run_definition, run)
        computed_run_count += 1
        runs.append(run)

    return runs, RunStoreSummary(
        cached_run_count=cached_run_count,
        computed_run_count=computed_run_count,
    )


def build_condition_sweep_runs(
    *,
    comparison: ComparisonDefinition,
    closes,
    volumes,
    dataset_period: str,
    dataset_metadata: dict[str, str],
    run_store: FileRunResultStore,
) -> tuple[list[dict], RunStoreSummary]:
    dataset_spec = {
        "period": dataset_period,
    }
    results: list[dict] = []
    cached_run_count = 0
    computed_run_count = 0

    for strategy_definition in comparison.candidate_strategy_definitions:
        for condition_variant in comparison.condition_variants:
            effective_strategy = replace(
                strategy_definition,
                risk_controls_definition=build_risk_controls_definition(
                    max_investment_ratio=condition_variant.max_investment_ratio,
                    max_weight=condition_variant.max_weight,
                ),
            )
            effective_evaluation_context = replace(
                comparison.run_input.evaluation_context,
                cost_assumptions=replace(
                    comparison.run_input.evaluation_context.cost_assumptions,
                    commission_pct=condition_variant.commission_pct,
                ),
            )
            serialized_strategy = serialize_strategy_definition(effective_strategy)
            serialized_evaluation_context = {
                **serialize_evaluation_context(
                    comparison,
                    dataset_metadata,
                    period_override=dataset_period,
                ),
                "costAssumptions": serialize_cost_assumptions(effective_evaluation_context),
            }
            serialized_condition_variant = serialize_condition_variant(condition_variant)
            run_definition = build_run_definition(
                run_kind="condition_sweep",
                strategy=serialized_strategy,
                dataset_spec=dataset_spec,
                evaluation_context=serialized_evaluation_context,
                portfolio_state=serialize_portfolio_state(comparison.run_input.portfolio_state),
                dataset_metadata=dataset_metadata,
            )
            cached_run = run_store.load(run_definition)
            if cached_run is not None:
                cached_run_count += 1
                results.append(cached_run)
                continue

            run = evaluate_strategy_run(
                closes=closes,
                volumes=volumes,
                strategy_definition=effective_strategy,
                initial_capital=effective_evaluation_context.evaluation_settings.initial_capital,
                split_ratio=effective_evaluation_context.evaluation_settings.split_ratio,
                transaction_cost=effective_evaluation_context.cost_assumptions.commission_pct / 100,
                portfolio_state=comparison.run_input.portfolio_state,
            )
            compact_run = compact_condition_sweep_run(
                run=run,
                key=f"{strategy_definition.key}__{condition_variant.key}",
                condition_variant=serialized_condition_variant,
            )
            run_store.save(run_definition, compact_run)
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
    comparison: ComparisonDefinition,
    closes,
    volumes,
    dataset_period: str,
    dataset_metadata: dict[str, str],
    run_store: FileRunResultStore,
) -> tuple[list[dict], RunStoreSummary]:
    dataset_spec = {
        "period": dataset_period,
    }
    returns = closes.pct_change().dropna()
    aligned_volumes = volumes.loc[returns.index] if volumes is not None else None
    ranking_definitions = build_asset_ranking_definitions(comparison.candidate_strategy_definitions)
    results: list[dict] = []
    cached_run_count = 0
    computed_run_count = 0

    for ranking_definition in ranking_definitions:
        serialized_ranking_definition = serialize_asset_ranking_definition(ranking_definition)
        run_definition = build_run_definition(
            run_kind="ranking_evaluation",
            strategy={"ranking": serialized_ranking_definition},
            dataset_spec=dataset_spec,
            evaluation_context=serialize_evaluation_context(
                comparison,
                dataset_metadata,
                period_override=dataset_period,
            ),
            portfolio_state=serialize_portfolio_state(comparison.run_input.portfolio_state),
            dataset_metadata=dataset_metadata,
        )
        cached_run = run_store.load(run_definition)
        if cached_run is not None:
            cached_run_count += 1
            results.append(cached_run)
            continue

        result = evaluate_asset_ranking_definition(
            returns=returns,
            volumes=aligned_volumes,
            split_ratio=comparison.run_input.evaluation_context.evaluation_settings.split_ratio,
            ranking_definition=ranking_definition,
        )
        run_store.save(run_definition, result)
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
    comparison: ComparisonDefinition,
    closes,
    volumes,
    dataset_period: str,
    dataset_metadata: dict[str, str],
    run_store: FileRunResultStore,
) -> tuple[list[dict], RunStoreSummary]:
    dataset_spec = {
        "period": dataset_period,
    }
    base_evaluation_context = comparison.run_input.evaluation_context
    results: list[dict] = []
    cached_run_count = 0
    computed_run_count = 0
    generation = {
        "method": "parameter_sweep",
        "batchKey": "local_tilt_search_v1",
        "spec": build_parameter_sweep_generation_spec(),
    }

    family_specs = [
        {
            "familyKey": "momentum_top",
            "familyLabel": "全資産モメンタム傾斜 上位優遇",
            "strategyType": "full_universe_momentum_tilt",
            "macroWeights": [None],
        },
        {
            "familyKey": "momentum_macro_top",
            "familyLabel": "全資産モメンタムマクロ傾斜 上位優遇",
            "strategyType": "full_universe_momentum_macro_tilt",
            "macroWeights": [0.05, 0.10, 0.15, 0.20],
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
                        "window_days": 252,
                    }
                    if macro_weight is not None:
                        score_parameters["momentum_weight"] = round(1.0 - macro_weight, 2)
                        score_parameters["macro_weight"] = macro_weight

                    strategy_definition = build_portfolio_strategy_definition(
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
                        for strategy in comparison.candidate_strategy_definitions
                        if strategy.portfolio_model_definition.model_type == "hierarchical_risk_parity"
                    )
                    effective_strategy = build_strategy_definition(
                        investment_universe_definition=base_strategy.investment_universe_definition,
                        selection_definition=strategy_definition,
                        portfolio_model_definition=base_strategy.portfolio_model_definition,
                        execution_policy_definition=base_strategy.execution_policy_definition,
                        risk_controls_definition=build_risk_controls_definition(
                            max_investment_ratio=1.0,
                            max_weight=max_weight,
                        ),
                    )
                    serialized_strategy = serialize_strategy_definition(effective_strategy)
                    serialized_evaluation_context = serialize_evaluation_context(
                        comparison,
                        dataset_metadata,
                        period_override=dataset_period,
                    )
                    run_definition = build_run_definition(
                        run_kind="portfolio_comparison",
                        strategy=serialized_strategy,
                        dataset_spec=dataset_spec,
                        evaluation_context=serialized_evaluation_context,
                        portfolio_state=serialize_portfolio_state(comparison.run_input.portfolio_state),
                        dataset_metadata=dataset_metadata,
                        generation=generation,
                    )
                    cached_run = run_store.load(run_definition)
                    if cached_run is not None:
                        cached_run_count += 1
                        results.append(cached_run)
                        continue

                    run = evaluate_strategy_run(
                        closes=closes,
                        volumes=volumes,
                        strategy_definition=effective_strategy,
                        initial_capital=base_evaluation_context.evaluation_settings.initial_capital,
                        split_ratio=base_evaluation_context.evaluation_settings.split_ratio,
                        transaction_cost=base_evaluation_context.cost_assumptions.commission_pct / 100,
                        portfolio_state=comparison.run_input.portfolio_state,
                    )
                    compact_run = compact_parameter_sweep_run(
                        run=run,
                        family_key=family_spec["familyKey"],
                        family_label=family_spec["familyLabel"],
                        tilt_strength=tilt_strength,
                        macro_weight=macro_weight,
                        max_weight=max_weight,
                        strategy=serialized_strategy,
                    )
                    run_store.save(run_definition, compact_run)
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
                "key": "momentum_top",
                "strategyType": "full_universe_momentum_tilt",
                "tiltShape": "top_favored",
            },
            {
                "key": "momentum_macro_top",
                "strategyType": "full_universe_momentum_macro_tilt",
                "tiltShape": "top_favored",
            },
        ],
        "parameterGrid": {
            "tiltStrength": [0.15, 0.20, 0.25, 0.30, 0.35],
            "macroWeight": [0.05, 0.10, 0.15, 0.20],
            "maxWeight": [0.40, 0.425, 0.45, 0.475, 0.50],
        },
    }
