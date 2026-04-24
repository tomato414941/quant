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
    build_parameter_sweep_strategy_definition,
    collect_ranking_source_definitions,
    collect_required_market_fields,
    collect_strategy_predictor_source_definitions,
    collect_strategy_predictor_specs,
    resolve_strategy_market_data_timeframe,
)
from app.comparison_serialization import (
    serialize_condition_variant,
    serialize_cost_model_spec,
    serialize_evaluation,
    serialize_execution_assumptions,
    serialize_strategy_definition_payload,
    serialize_timeframe,
)


def evaluate_strategy_definition_run_with_blueprint(
    *,
    closes: pd.DataFrame,
    volumes: pd.DataFrame | None,
    strategy_definition: StrategyDefinition,
    initial_capital: float,
    split_ratio: float,
    bars_per_year: float = 252.0,
    execution_assumptions: dict | None = None,
    portfolio_state=None,
    predictor_panel=None,
    availability_policy: dict[str, object] | None = None,
) -> dict:
    resolved_execution_assumptions = execution_assumptions or {
        "kind": "close_execution_assumptions",
        "label": "終値約定",
        "parameters": {"fillPrice": "close"},
        "costModel": None,
    }
    resolved_availability_policy = availability_policy or build_default_availability_policy()
    artifact = run_strategy_backtest(
        build_strategy_blueprint_from_definition(strategy_definition),
        {"closes": closes, "volumes": volumes},
        RunContext(
            initial_capital=initial_capital,
            split_ratio=split_ratio,
            bars_per_year=bars_per_year,
            execution_assumptions=resolved_execution_assumptions,
            availability_policy=resolved_availability_policy,
            portfolio_state=(
                None if portfolio_state is None else serialize_portfolio_state(portfolio_state)
            ),
        ),
        predictor_panel=predictor_panel,
    )
    return artifact.run_result


PROJECT_ROOT = Path(__file__).resolve().parents[2]


TIMEFRAMES_BY_KEY = {
    DEFAULT_DAILY_TIMEFRAME.key: DEFAULT_DAILY_TIMEFRAME,
    DEFAULT_WEEKLY_TIMEFRAME.key: DEFAULT_WEEKLY_TIMEFRAME,
    DEFAULT_MONTHLY_TIMEFRAME.key: DEFAULT_MONTHLY_TIMEFRAME,
}


def build_strategy_runs(
    *,
    comparison: ComparisonSpec,
    strategy_definitions: list[StrategyDefinition],
    market_bundles_by_timeframe: dict[str, dict],
    market_data_period: str,
    metadata_by_timeframe: dict[str, dict[str, object]],
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

        run = evaluate_strategy_definition_run_with_blueprint(
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
    metadata_by_timeframe: dict[str, dict[str, object]],
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
        returns = closes.pct_change(fill_method=None).iloc[1:].dropna(how="all")
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
    metadata_by_timeframe: dict[str, dict[str, object]],
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
                cost_model=scale_cost_model_spec(
                    comparison.run_spec.execution_assumptions.cost_model,
                    condition_variant.cost_multiplier,
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

            run = evaluate_strategy_definition_run_with_blueprint(
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
    metadata_by_timeframe: dict[str, dict[str, object]],
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
        returns = closes.pct_change(fill_method=None).iloc[1:].dropna(how="all")
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
    metadata_by_timeframe: dict[str, dict[str, object]],
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

                    run = evaluate_strategy_definition_run_with_blueprint(
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
