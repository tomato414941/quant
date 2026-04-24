from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
from app.instrument_registry import get_instrument
from app.portfolio_allocation import (
    PortfolioAllocationInput,
    build_equal_weight_fallback,
    expand_weights,
    fit_portfolio_model,
)
from app.portfolio_domain import *
from app.portfolio_tilt import apply_weight_tilt
from app.portfolio_market_data import (
    prepare_signal_component_data,
    prepare_strategy_market_data,
    resample_market_frame_to_timeframe,
    resample_returns_frame_to_timeframe,
    resolve_timeframe_bars_per_year,
)
from app.portfolio_metrics import (
    compute_split_index,
    empty_number_distribution,
    serialize_weights,
    should_rebalance,
    summarize_availability_series,
    summarize_metrics_from_bar_returns,
    summarize_optional_number_distribution,
    summarize_portfolio_decision_events,
    summarize_portfolio_metrics,
    summarize_segment_from_returns,
)

DECISION_POLICY_EXTENSION_KEY = "decision_policy"
DIRECT_SCORE_DECISION_POLICY = "direct_score_to_weight"
COST_AWARE_NO_TRADE_DECISION_POLICY = "cost_aware_no_trade"
SUPPORTED_DECISION_POLICIES = {
    DIRECT_SCORE_DECISION_POLICY,
    COST_AWARE_NO_TRADE_DECISION_POLICY,
}
DEFAULT_NO_TRADE_BAND = 0.02
DEFAULT_CONFIDENCE_FLOOR = 0.25
SIGNAL_RETURN_PROXY_EDGE_SOURCE = "signal_return_proxy"


@dataclass(frozen=True)
class ForecastSnapshot:
    as_of_date: str | None
    horizon: str
    score: pd.Series
    percentile_rank: pd.Series
    expected_return_proxy: pd.Series | None
    edge_source: str | None
    confidence: pd.Series
    risk_proxy: pd.Series


@dataclass(frozen=True)
class PortfolioDecision:
    selected_assets: list[str]
    weights: np.ndarray
    policy: str
    action: str
    reason: str
    turnover: float
    estimated_cost_pct: float
    estimated_edge_pct: float | None
    edge_source: str | None
    average_confidence: float | None


def freeze_parameter_value(value: object) -> object:
    if isinstance(value, dict):
        return tuple((str(key), freeze_parameter_value(nested_value)) for key, nested_value in sorted(value.items()))
    if isinstance(value, list):
        return tuple(freeze_parameter_value(item) for item in value)
    return value



def build_default_strategy_selection_context(strategy: EvaluatorStrategySpec) -> dict[str, object]:
    return {
        "signalKey": f"signal__{strategy.strategy_id}__selection",
        "signalLabel": strategy.selection.label,
        "description": strategy.selection.description,
        "sourceKind": "selection_signal",
        "selectionKey": strategy.selection.key,
        "strategyType": strategy.selection.strategy_type,
        "scoreParameters": thaw_strategy_parameter_value(dict(strategy.selection.ranking_signal.score_parameters)),
        "dataTimeframe": strategy.timeframe.key,
        "signalTimeframe": strategy.timeframe.key,
        "alignmentPolicy": None,
        "weight": 1.0 if strategy.predictor_use is None else strategy.predictor_use.signal_weight,
    }


def build_default_strategy_predictor_context(strategy: EvaluatorStrategySpec) -> dict[str, object] | None:
    if strategy.predictor_use is None:
        return None
    return {
        "signalKey": f"signal__{strategy.strategy_id}__predictor",
        "signalLabel": f"{strategy.label} predictor overlay",
        "description": "Predictor overlay translated from predictor_use.",
        "sourceKind": "predictor_overlay",
        "predictorKey": strategy.predictor_use.predictor_key,
        "signalWeight": strategy.predictor_use.signal_weight,
        "predictorWeight": strategy.predictor_use.predictor_weight,
        "dataTimeframe": strategy.timeframe.key,
        "signalTimeframe": strategy.timeframe.key,
        "alignmentPolicy": None,
        "weight": strategy.predictor_use.predictor_weight,
    }


def resolve_decision_schedule(strategy: EvaluatorStrategySpec) -> str:
    if strategy.decision_schedule is not None:
        return str(strategy.decision_schedule)
    return str(strategy.execution_policy.rebalance_schedule)


def resolve_strategy_market_data_timeframe_key(strategy: EvaluatorStrategySpec) -> str:
    if strategy.signal_execution_contexts:
        return str(strategy.signal_execution_contexts[0]["dataTimeframe"])
    return strategy.timeframe.key


def extract_predictor_signal_payload(strategy: EvaluatorStrategySpec) -> dict[str, object] | None:
    if strategy.predictor_signal_execution_context is not None:
        return dict(strategy.predictor_signal_execution_context)
    return build_default_strategy_predictor_context(strategy)

def get_strategy_definition_signal_execution_contexts(
    strategy_definition,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    if isinstance(strategy_definition, StrategyDefinition):
        return build_strategy_signal_execution_contexts_from_definition(strategy_definition)

    normalized_selection_contexts = (
        [dict(selection_context) for selection_context in strategy_definition.signal_execution_contexts]
        if strategy_definition.signal_execution_contexts
        else [build_default_strategy_selection_context(strategy_definition)]
    )
    normalized_predictor_context = extract_predictor_signal_payload(strategy_definition)
    return normalized_selection_contexts, normalized_predictor_context


def build_runtime_signal_execution_contexts(
    normalized_selection_contexts: list[dict[str, object]],
    normalized_predictor_context: dict[str, object] | None,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    selection_contexts = [
        {
            "source_kind": "selection_signal",
            "selection": build_selection_spec_from_signal_payload(selection_context),
            "weight": float(selection_context["weight"]),
            "data_timeframe_key": str(selection_context["dataTimeframe"]),
            "signal_timeframe_key": str(selection_context["signalTimeframe"]),
            "alignment_policy": selection_context["alignmentPolicy"],
        }
        for selection_context in normalized_selection_contexts
    ]

    predictor_context = None
    if normalized_predictor_context is not None:
        predictor_context = {
            "source_kind": "predictor_overlay",
            "predictor_key": str(normalized_predictor_context["predictorKey"]),
            "signal_weight": float(normalized_predictor_context["signalWeight"]),
            "predictor_weight": float(normalized_predictor_context["predictorWeight"]),
            "data_timeframe_key": str(normalized_predictor_context["dataTimeframe"]),
            "signal_timeframe_key": str(normalized_predictor_context["signalTimeframe"]),
            "alignment_policy": normalized_predictor_context["alignmentPolicy"],
        }
    return selection_contexts, predictor_context


def get_strategy_signal_execution_contexts(
    strategy: EvaluatorStrategySpec,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    normalized_selection_contexts, normalized_predictor_context = get_strategy_definition_signal_execution_contexts(strategy)
    return build_runtime_signal_execution_contexts(
        normalized_selection_contexts,
        normalized_predictor_context,
    )


def resolve_runtime_signal_execution_contexts(
    strategy: EvaluatorStrategySpec,
    signal_execution_contexts: tuple[list[dict[str, object]], dict[str, object] | None] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    if signal_execution_contexts is None:
        return get_strategy_signal_execution_contexts(strategy)

    selection_contexts, predictor_context = signal_execution_contexts
    if selection_contexts and "selection" not in selection_contexts[0]:
        return build_runtime_signal_execution_contexts(selection_contexts, predictor_context)
    return selection_contexts, predictor_context





def build_selection_spec_from_signal_payload(payload: dict[str, object]) -> RankingSourceSpec:
    strategy_type = str(payload.get("strategyType"))
    score_parameters = payload.get("scoreParameters")
    return build_selection_spec(
        strategy_type,
        key=str(payload.get("selectionKey") or strategy_type),
        label=str(payload.get("label") or strategy_type),
        description=str(payload.get("description") or strategy_type),
        score_parameters=score_parameters if isinstance(score_parameters, dict) else None,
    )


def get_strategy_selection_components(
    strategy: EvaluatorStrategySpec | None = None,
    *,
    selection_contexts: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    if selection_contexts is not None:
        return selection_contexts
    if strategy is None:
        raise ValueError("strategy or selection_contexts is required")
    selection_contexts, _ = get_strategy_signal_execution_contexts(strategy)
    return selection_contexts


def compute_strategy_selection_score_series(
    returns: pd.DataFrame,
    volume_history: pd.DataFrame | None,
    strategy: EvaluatorStrategySpec,
    *,
    bars_per_year: float,
    selection_contexts: list[dict[str, object]] | None = None,
) -> pd.Series | None:
    selection_components = get_strategy_selection_components(
        strategy,
        selection_contexts=selection_contexts,
    )
    if len(selection_components) == 1 and float(selection_components[0]["weight"]) == 1.0:
        return compute_strategy_score_series_base(
            returns,
            volume_history,
            selection_components[0]["selection"],
            bars_per_year=bars_per_year,
        )

    blended_scores: pd.Series | None = None
    total_weight = 0.0
    for component in selection_components:
        selection = component["selection"]
        weight = float(component["weight"])
        signal_returns, signal_volumes, signal_bars_per_year = prepare_signal_component_data(
            history_returns=returns,
            volume_history=volume_history,
            data_timeframe_key=component["data_timeframe_key"],
            signal_timeframe_key=component["signal_timeframe_key"],
            alignment_policy=component["alignment_policy"],
        )
        component_scores = compute_strategy_score_series_base(
            signal_returns,
            signal_volumes,
            selection,
            bars_per_year=signal_bars_per_year,
        )
        if component_scores is None:
            continue
        aligned_scores = component_scores.reindex(returns.columns).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        standardized_scores = standardize_prediction_series(aligned_scores)
        blended_scores = (
            standardized_scores * weight
            if blended_scores is None
            else blended_scores.add(standardized_scores * weight, fill_value=0.0)
        )
        total_weight += weight

    if blended_scores is None:
        return None
    if total_weight <= 0:
        return blended_scores
    return blended_scores / total_weight


def compute_strategy_selection_trailing_total_returns(
    returns: pd.DataFrame,
    strategy: EvaluatorStrategySpec,
    *,
    bars_per_year: float,
    selection_contexts: list[dict[str, object]] | None = None,
) -> pd.Series:
    selection_components = get_strategy_selection_components(
        strategy,
        selection_contexts=selection_contexts,
    )
    if len(selection_components) == 1 and float(selection_components[0]["weight"]) == 1.0:
        return compute_trailing_total_returns(
            returns,
            selection_components[0]["selection"],
            bars_per_year=bars_per_year,
        )

    blended_returns: pd.Series | None = None
    total_weight = 0.0
    for component in selection_components:
        selection = component["selection"]
        weight = float(component["weight"])
        signal_returns, _, signal_bars_per_year = prepare_signal_component_data(
            history_returns=returns,
            volume_history=None,
            data_timeframe_key=component["data_timeframe_key"],
            signal_timeframe_key=component["signal_timeframe_key"],
            alignment_policy=component["alignment_policy"],
        )
        component_returns = compute_trailing_total_returns(
            signal_returns,
            selection,
            bars_per_year=signal_bars_per_year,
        ).reindex(returns.columns).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        blended_returns = (
            component_returns * weight
            if blended_returns is None
            else blended_returns.add(component_returns * weight, fill_value=0.0)
        )
        total_weight += weight

    if blended_returns is None:
        return pd.Series(0.0, index=returns.columns, dtype="float64")
    if total_weight <= 0:
        return blended_returns
    return blended_returns / total_weight


def prepare_strategy_signal_data(
    *,
    history_returns: pd.DataFrame,
    volume_history: pd.DataFrame | None,
    strategy: EvaluatorStrategySpec | None = None,
    selection_contexts: list[dict[str, object]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, float]:
    selection_components = get_strategy_selection_components(
        strategy,
        selection_contexts=selection_contexts,
    )
    primary_selection_context = selection_components[0]
    return prepare_signal_component_data(
        history_returns=history_returns,
        volume_history=volume_history,
        data_timeframe_key=str(primary_selection_context["data_timeframe_key"]),
        signal_timeframe_key=str(primary_selection_context["signal_timeframe_key"]),
        alignment_policy=primary_selection_context["alignment_policy"],
    )


def prepare_strategy_predictor_panel(
    *,
    predictor_panel: pd.DataFrame | None,
    strategy: EvaluatorStrategySpec | None = None,
    predictor_context: dict[str, object] | None = None,
    market_data_timeframe_key: str | None = None,
    signal_timeframe_key: str | None = None,
) -> pd.DataFrame | None:
    if predictor_panel is None:
        return None

    if predictor_context is None and strategy is not None:
        _, predictor_context = get_strategy_signal_execution_contexts(strategy)
    if market_data_timeframe_key is None:
        if predictor_context is not None:
            market_data_timeframe_key = str(predictor_context["data_timeframe_key"])
        elif strategy is not None:
            market_data_timeframe_key = resolve_strategy_market_data_timeframe_key(strategy)
        else:
            raise ValueError("strategy, predictor_context, or market_data_timeframe_key is required")
    if signal_timeframe_key is None:
        if predictor_context is not None:
            signal_timeframe_key = str(predictor_context["signal_timeframe_key"])
        elif strategy is not None:
            signal_timeframe_key = strategy.timeframe.key
        else:
            raise ValueError("strategy, predictor_context, or signal_timeframe_key is required")
    alignment_method = None
    if predictor_context is not None:
        alignment_policy = predictor_context["alignment_policy"]
        alignment_method = None if alignment_policy is None else str(alignment_policy.get("method"))
    if market_data_timeframe_key == signal_timeframe_key:
        return predictor_panel

    return resample_market_frame_to_timeframe(
        predictor_panel,
        target_timeframe_key=signal_timeframe_key,
        value_kind="last",
        alignment_method=alignment_method,
    )


def resolve_predictor_snapshot(
    predictor_panel: pd.DataFrame | None,
    *,
    current_date: str | None,
) -> pd.Series | None:
    if predictor_panel is None or current_date is None or predictor_panel.empty:
        return None

    panel = predictor_panel.copy()
    panel.index = pd.to_datetime(panel.index)
    eligible_panel = panel.loc[panel.index <= pd.Timestamp(current_date)]
    if eligible_panel.empty:
        return None
    return eligible_panel.iloc[-1]


def build_asset_ranking_specs(
    strategies: list[EvaluatorStrategySpec],
) -> list[AssetRankingSpec]:
    grouped: dict[
        tuple[
            tuple[str, ...],
            str,
            str,
            tuple[str, ...],
            tuple[str, ...],
            str,
            tuple[tuple[str, float], ...],
        ],
        list[EvaluatorStrategySpec],
    ] = {}

    for strategy in strategies:
        selection = strategy.selection
        if selection.ranking_signal.score_model.kind == "none":
            continue
        signature = (
            strategy.timeframe.key,
            strategy.investment_universe.tickers,
            selection.strategy_type,
            selection.ranking_signal.score_model.kind,
            tuple(selection.ranking_signal.feature_inputs),
            tuple(filter_rule.key for filter_rule in selection.filter_rules),
            selection.fallback_rule.key,
            tuple(
                (key, freeze_parameter_value(value))
                for key, value in sorted(extract_ranking_score_parameters(selection).items())
            ),
        )
        grouped.setdefault(signature, []).append(strategy)

    ranking_specs: list[AssetRankingSpec] = []
    for strategies in grouped.values():
        representative = strategies[0].selection
        filter_label = " / ".join(filter_rule.label for filter_rule in representative.filter_rules)
        label_parts = [
            strategies[0].timeframe.label,
            strategies[0].investment_universe.label,
            representative.universe_policy.label,
            representative.ranking_signal.score_model.label,
        ]
        ranking_score_parameters = extract_ranking_score_parameters(representative)
        if ranking_score_parameters:
            parameter_label = ", ".join(
                (
                    f"{key}={value['value']} {value['unit']}"
                    if isinstance(value, dict) and "unit" in value and "value" in value
                    else f"{key}={float(value):.2f}"
                )
                for key, value in ranking_score_parameters.items()
            )
            label_parts.append(parameter_label)
        if filter_label:
            label_parts.append(filter_label)
        ranking_specs.append(
            AssetRankingSpec(
                key=f"ranking__{representative.key}",
                label=" / ".join(label_parts),
                description=representative.description,
                timeframe=strategies[0].timeframe,
                investment_universe=strategies[0].investment_universe,
                selection=representative,
            )
        )

    ranking_specs.sort(key=lambda ranking_spec: ranking_spec.label)
    return ranking_specs


def build_asset_ranking_specs_from_strategy_definitions(
    strategy_definitions: list[StrategyDefinition],
) -> list[AssetRankingSpec]:
    return build_asset_ranking_specs(
        [
            build_executable_evaluator_strategy_spec_from_definition(strategy_definition)
            for strategy_definition in strategy_definitions
        ]
    )


def build_predictor_specs(
    ranking_specs: list[AssetRankingSpec],
    target_specs: list[PredictionTargetSpec],
) -> list[PredictorSpec]:
    predictor_specs: list[PredictorSpec] = []
    for ranking_spec in ranking_specs:
        ranking_parameters = extract_ranking_score_parameters(ranking_spec.selection)
        feature_spec = build_feature_spec(
            key=f"features__{ranking_spec.key}",
            label=f"{ranking_spec.selection.ranking_signal.score_model.label} features",
            feature_inputs=tuple(
                build_feature_input_spec(key=feature_input)
                for feature_input in ranking_spec.selection.ranking_signal.feature_inputs
            ),
            derived_features=tuple(
                build_derived_feature_spec(key=feature_key)
                for feature_key in ("score", "momentum", "lowVolRank", "macroRank", "volumeStrength")
            ),
            ranking_feature_recipe=build_ranking_feature_recipe_spec(ranking_spec.selection),
        )
        engine_specs = [
            build_prediction_engine_spec(
                key=f"engine__{ranking_spec.key}__signal",
                label=f"{ranking_spec.selection.ranking_signal.score_model.label} signal",
                signal_source_spec=build_prediction_signal_source_spec(
                    key=f"signal-source__{ranking_spec.key}",
                    label=f"{ranking_spec.selection.ranking_signal.score_model.label} signal source",
                ),
                learner_spec=None,
                combiner_spec=build_prediction_combiner_spec(
                    key=f"combiner__{ranking_spec.key}__baseline-only",
                    label="Baseline signal only",
                    kind="baseline_signal_only",
                ),
            ),
            build_prediction_engine_spec(
                key=f"engine__{ranking_spec.key}__linear",
                label=f"{ranking_spec.selection.ranking_signal.score_model.label} linear",
                signal_source_spec=None,
                learner_spec=build_prediction_learner_spec(
                    key=f"learner__{ranking_spec.key}__linear",
                    label=f"{ranking_spec.selection.ranking_signal.score_model.label} linear learner",
                    kind="linear_regression",
                ),
                combiner_spec=build_prediction_combiner_spec(
                    key=f"combiner__{ranking_spec.key}__learner-only",
                    label="Learner only",
                    kind="learner_only",
                ),
            ),
            build_prediction_engine_spec(
                key=f"engine__{ranking_spec.key}__blend",
                label=f"{ranking_spec.selection.ranking_signal.score_model.label} blend",
                signal_source_spec=build_prediction_signal_source_spec(
                    key=f"signal-source__{ranking_spec.key}",
                    label=f"{ranking_spec.selection.ranking_signal.score_model.label} signal source",
                ),
                learner_spec=build_prediction_learner_spec(
                    key=f"learner__{ranking_spec.key}__linear",
                    label=f"{ranking_spec.selection.ranking_signal.score_model.label} linear learner",
                    kind="linear_regression",
                ),
                combiner_spec=build_prediction_combiner_spec(
                    key=f"combiner__{ranking_spec.key}__weighted-blend",
                    label="Weighted blend",
                    kind="weighted_blend",
                    signal_source_weight=0.8,
                    learner_weight=0.2,
                ),
            ),
        ]
        for target_spec in target_specs:
            for engine_spec in engine_specs:
                learner_key = (
                    engine_spec.learner_spec.kind
                    if engine_spec.learner_spec is not None
                    else "no-learner"
                )
                combiner_key = engine_spec.combiner_spec.kind
                predictor_specs.append(
                    build_predictor_spec(
                        key=(
                            f"prediction__{ranking_spec.key}__"
                            f"{learner_key}__{combiner_key}__{target_spec.key}"
                        ),
                        label=f"{ranking_spec.label} / {engine_spec.label} -> {target_spec.label}",
                        description=ranking_spec.description,
                        timeframe=ranking_spec.timeframe,
                        signal_spec=build_signal_spec(
                            key=f"signal__{ranking_spec.key}",
                            label=f"{ranking_spec.label} signal",
                            observation_spec=build_observation_spec(
                                key=f"observation__{ranking_spec.key}",
                                label=f"{ranking_spec.label} observation",
                                tickers=ranking_spec.investment_universe.tickers,
                                fields=ranking_spec.selection.ranking_signal.feature_inputs,
                            ),
                            entity_kind="asset_set",
                            entity_identifiers=ranking_spec.investment_universe.tickers,
                            output_spec=build_prediction_output_spec(
                                key="output__cross_sectional_score",
                                label="Cross-sectional score",
                                kind="score",
                            ),
                            decision_use_spec=build_decision_use_spec(
                                ranking_spec.selection
                            ),
                        ),
                        predicted_quantity_spec=build_predicted_quantity_spec(
                            key="quantity__return",
                            label="Return",
                            kind="return",
                        ),
                        target_spec=target_spec,
                        feature_spec=feature_spec,
                        engine_spec=engine_spec,
                        training_spec=build_training_spec(
                            key=(
                                f"training__{ranking_spec.key}__"
                                f"{learner_key}__{combiner_key}"
                            ),
                            label=f"{ranking_spec.selection.ranking_signal.score_model.label} training",
                            fit_mode="expanding",
                            min_train_samples=50,
                        ),
                    )
                )
    predictor_specs.sort(key=lambda predictor_spec: predictor_spec.label)
    return predictor_specs


def extract_ranking_score_parameters(
    selection: SelectionSpec,
) -> dict[str, object]:
    score_parameters = dict(selection.ranking_signal.score_parameters)
    extracted: dict[str, object] = {}
    if "windowSpec" in score_parameters:
        extracted["windowSpec"] = dict(score_parameters["windowSpec"])
    if selection.ranking_signal.score_model.kind == "momentum_low_vol":
        extracted["momentumWeight"] = float(score_parameters.get("momentum_weight", 0.7))
        extracted["lowVolWeight"] = float(score_parameters.get("low_vol_weight", 0.3))
        return extracted
    if selection.ranking_signal.score_model.kind == "momentum_macro":
        extracted["momentumWeight"] = float(score_parameters.get("momentum_weight", 0.85))
        extracted["macroWeight"] = float(score_parameters.get("macro_weight", 0.15))
        return extracted
    return extracted


def serialize_predictor_spec(
    predictor_spec: PredictorSpec,
) -> dict:
    return {
        "kind": "predictor_spec",
        "schemaVersion": "v1",
        "key": predictor_spec.key,
        "label": predictor_spec.label,
        "description": predictor_spec.description,
        "timeframe": {
            "key": predictor_spec.timeframe.key,
            "label": predictor_spec.timeframe.label,
            "yfinanceInterval": predictor_spec.timeframe.yfinance_interval,
            "barsPerYear": predictor_spec.timeframe.bars_per_year,
            "barSeconds": predictor_spec.timeframe.bar_seconds,
        },
        "signalSpec": serialize_signal_spec(predictor_spec.signal_spec),
        "predictedQuantitySpec": serialize_predicted_quantity_spec(
            predictor_spec.predicted_quantity_spec
        ),
        "targetSpec": serialize_prediction_target_spec(predictor_spec.target_spec),
        "featureSpec": serialize_feature_spec(predictor_spec.feature_spec),
        "engineSpec": serialize_prediction_engine_spec(predictor_spec.engine_spec),
        "trainingSpec": serialize_training_spec(predictor_spec.training_spec),
    }


def serialize_predictor_panel(
    predictor_panel: pd.DataFrame,
) -> dict:
    compact_panel = predictor_panel.dropna(how="all")
    return {
        "dates": [str(index) for index in compact_panel.index],
        "assets": [str(column) for column in compact_panel.columns],
        "values": [
            [None if pd.isna(value) else float(value) for value in row]
            for row in compact_panel.to_numpy(dtype="float64")
        ],
    }


def deserialize_predictor_panel(
    payload: dict | None,
) -> pd.DataFrame | None:
    if not payload:
        return None
    dates = payload.get("dates", [])
    assets = payload.get("assets", [])
    values = payload.get("values", [])
    if not dates or not assets:
        return None
    return pd.DataFrame(values, index=dates, columns=assets, dtype="float64")


def convert_window_spec_to_bars(
    window_spec: dict[str, object],
    *,
    bars_per_year: float,
) -> int:
    unit = str(window_spec.get("unit", "bars"))
    value = float(window_spec.get("value", bars_per_year))
    bars_per_trading_day = bars_per_year / 252
    bars_per_week = bars_per_year / 52
    bars_per_month = bars_per_year / 12

    if unit == "bars":
        configured_bars = value
    elif unit == "days":
        configured_bars = value * bars_per_trading_day
    elif unit == "weeks":
        configured_bars = value * bars_per_week
    elif unit == "months":
        configured_bars = value * bars_per_month
    elif unit == "years":
        configured_bars = value * bars_per_year
    else:
        raise ValueError("Unsupported windowSpec unit.")
    return max(1, int(round(configured_bars)))


def convert_horizon_spec_to_bars(
    horizon_spec: dict[str, object],
    *,
    bars_per_year: float,
) -> int:
    return convert_window_spec_to_bars(horizon_spec, bars_per_year=bars_per_year)


def build_portfolio_state(
    *,
    current_weights: dict[str, float],
    cash_weight: float = 0.0,
) -> PortfolioState:
    normalized_weights = {
        asset.strip().upper(): float(weight)
        for asset, weight in current_weights.items()
        if asset.strip()
    }
    if cash_weight < 0 or cash_weight > 1:
        raise ValueError("Cash weight must be between 0 and 1.")
    if any(weight < 0 for weight in normalized_weights.values()):
        raise ValueError("Current weights must be non-negative.")
    total_weight = sum(normalized_weights.values()) + cash_weight
    if total_weight > 1.000001:
        raise ValueError("Portfolio state weights must sum to 1 or less.")
    return PortfolioState(current_weights=normalized_weights, cash_weight=float(cash_weight))


def serialize_portfolio_state(portfolio_state: PortfolioState) -> dict:
    rows = [
        {"asset": asset, "weightPct": round(weight * 100, 2)}
        for asset, weight in portfolio_state.current_weights.items()
        if weight > 0
    ]
    if portfolio_state.cash_weight > 0:
        rows.append({"asset": "CASH", "weightPct": round(portfolio_state.cash_weight * 100, 2)})
    rows.sort(key=lambda item: item["weightPct"], reverse=True)
    return {"weights": rows}


def build_flat_cost_model(*, commission_pct: float, slippage_pct: float = 0.0) -> dict:
    return {
        "kind": "flat_cost",
        "parameters": {
            "commissionPct": float(commission_pct),
            "slippagePct": float(slippage_pct),
        },
        "perAssetOverrides": {},
    }


def cost_rate_from_parameters(parameters: dict[str, float]) -> float:
    commission_pct = float(parameters.get("commissionPct", 0.0))
    slippage_pct = float(parameters.get("slippagePct", 0.0))
    return (commission_pct + slippage_pct) / 100


def impact_rate_from_parameters(parameters: dict[str, float]) -> float:
    return float(parameters.get("impactCoefficientPct", 0.0)) / 100


def resolve_cost_model_inputs(
    *,
    universe_columns: pd.Index,
    cost_model: dict,
) -> dict[str, np.ndarray | float | int]:
    kind = cost_model.get("kind", "flat_cost")
    parameters = cost_model.get("parameters", {})
    default_rate = cost_rate_from_parameters(parameters)
    default_impact_rate = impact_rate_from_parameters(parameters)
    linear_rates = np.repeat(default_rate, len(universe_columns)).astype("float64")
    impact_rates = np.repeat(default_impact_rate, len(universe_columns)).astype("float64")
    adv_window_bars = max(1, int(float(parameters.get("advWindowBars", 20))))
    min_adv_notional = float(parameters.get("minAdvNotional", 1_000_000.0))

    if kind == "flat_cost":
        return {
            "defaultLinearRate": default_rate,
            "linearRates": linear_rates,
            "impactRates": impact_rates,
            "advWindowBars": adv_window_bars,
            "minAdvNotional": min_adv_notional,
        }
    if kind not in {"asset_specific_linear_cost", "asset_specific_adv_cost"}:
        raise ValueError("Unsupported cost model.")

    overrides = cost_model.get("perAssetOverrides", {})
    for index, asset in enumerate(universe_columns):
        override = overrides.get(str(asset))
        if not override:
            continue
        linear_rates[index] = cost_rate_from_parameters(override)
        impact_rates[index] = impact_rate_from_parameters({**parameters, **override})

    return {
        "defaultLinearRate": default_rate,
        "linearRates": linear_rates,
        "impactRates": impact_rates,
        "advWindowBars": adv_window_bars,
        "minAdvNotional": min_adv_notional,
    }


def compute_trade_cost(
    *,
    weight_delta: np.ndarray,
    linear_cost_rates: np.ndarray,
    impact_cost_rates: np.ndarray,
    portfolio_equity: float,
    price_snapshot: pd.Series | None,
    volume_history: pd.DataFrame | None,
    adv_window_bars: int,
    min_adv_notional: float,
) -> float:
    linear_cost = float(np.dot(weight_delta, linear_cost_rates))
    if (
        portfolio_equity <= 0
        or price_snapshot is None
        or volume_history is None
        or len(volume_history) == 0
        or np.allclose(impact_cost_rates, 0.0)
    ):
        return linear_cost

    recent_volumes = volume_history.tail(adv_window_bars)
    if recent_volumes.empty:
        return linear_cost

    aligned_prices = price_snapshot.reindex(recent_volumes.columns).astype("float64").fillna(0.0)
    adv_shares = recent_volumes.mean(axis=0).astype("float64").fillna(0.0)
    adv_notional = np.maximum((adv_shares * aligned_prices).to_numpy(dtype="float64"), min_adv_notional)
    trade_notional = weight_delta * float(portfolio_equity)
    participation = np.clip(trade_notional / adv_notional, 0.0, None)
    impact_rates = impact_cost_rates * np.sqrt(participation)
    impact_cost = float(np.dot(weight_delta, impact_rates))
    return linear_cost + impact_cost


def build_default_availability_policy() -> dict[str, object]:
    return {
        "kind": "asset_availability_policy",
        "minHistoryBars": 252,
        "maxStaleBars": 5,
        "delistedAssetPolicy": "liquidate_to_cash",
    }


def resolve_effective_min_history_bars(
    history_returns: pd.DataFrame,
    availability_policy: dict[str, object],
) -> int:
    configured_min = int(availability_policy.get("minHistoryBars", 252))
    return max(1, min(configured_min, len(history_returns)))


def resolve_available_assets(history_returns: pd.DataFrame) -> list[str]:
    if history_returns.empty:
        return []
    latest_row = history_returns.iloc[-1]
    return [str(asset) for asset, value in latest_row.items() if pd.notna(value)]


def resolve_eligible_assets(
    history_returns: pd.DataFrame,
    availability_policy: dict[str, object],
) -> list[str]:
    if history_returns.empty:
        return []
    min_history_bars = resolve_effective_min_history_bars(history_returns, availability_policy)
    latest_row = history_returns.iloc[-1]
    valid_counts = history_returns.notna().sum(axis=0)
    return [
        str(asset)
        for asset in history_returns.columns
        if pd.notna(latest_row[asset]) and int(valid_counts[asset]) >= min_history_bars
    ]


def prepare_history_returns_for_assets(
    history_returns: pd.DataFrame,
    eligible_assets: list[str],
) -> pd.DataFrame:
    if not eligible_assets:
        return history_returns.iloc[0:0, 0:0].copy()
    return history_returns.loc[:, eligible_assets].dropna(how="any")


def prepare_volume_history_for_assets(
    volume_history: pd.DataFrame | None,
    clean_history_returns: pd.DataFrame,
) -> pd.DataFrame | None:
    if volume_history is None:
        return None
    return volume_history.loc[clean_history_returns.index, clean_history_returns.columns]


def zero_weights_outside_assets(
    weights: np.ndarray,
    universe_columns: pd.Index,
    allowed_assets: list[str],
) -> np.ndarray:
    allowed_asset_set = set(allowed_assets)
    scoped_weights = [
        float(weight) if str(asset) in allowed_asset_set else 0.0
        for asset, weight in zip(universe_columns, weights, strict=True)
    ]
    return np.asarray(scoped_weights, dtype="float64")


def compute_dynamic_portfolio_allocation(
    *,
    history_returns: pd.DataFrame,
    volume_history: pd.DataFrame | None,
    strategy: EvaluatorStrategySpec,
    portfolio_model: PortfolioModelSpec,
    bars_per_year: float,
    universe_columns: pd.Index,
    max_investment_ratio: float,
    max_weight: float | None,
    previous_weights: np.ndarray | None,
    transaction_cost: float,
    current_date: str | None,
    predictor_panel: pd.DataFrame | None,
    selection_contexts: list[dict[str, object]] | None,
    predictor_context: dict[str, object] | None,
    availability_policy: dict[str, object],
) -> tuple[list[str], np.ndarray]:
    eligible_assets = resolve_eligible_assets(history_returns, availability_policy)
    if len(eligible_assets) == 1:
        weights = np.zeros(len(universe_columns), dtype="float64")
        asset = eligible_assets[0]
        asset_index = list(universe_columns).index(asset)
        asset_weight = max_investment_ratio
        if max_weight is not None:
            asset_weight = min(asset_weight, max_weight)
        weights[asset_index] = asset_weight
        return [asset], weights
    if len(eligible_assets) < 2:
        return [], np.zeros(len(universe_columns), dtype="float64")
    clean_history_returns = prepare_history_returns_for_assets(history_returns, eligible_assets)
    if len(clean_history_returns) < 3 or len(clean_history_returns.columns) < 2:
        return [], np.zeros(len(universe_columns), dtype="float64")
    clean_volume_history = prepare_volume_history_for_assets(volume_history, clean_history_returns)
    scoped_previous_weights = None
    if previous_weights is not None:
        scoped_previous_weights = zero_weights_outside_assets(previous_weights, universe_columns, eligible_assets)
    try:
        return compute_portfolio_allocation(
            history_returns=clean_history_returns,
            volume_history=clean_volume_history,
            strategy=strategy,
            portfolio_model=portfolio_model,
            bars_per_year=bars_per_year,
            universe_columns=universe_columns,
            max_investment_ratio=max_investment_ratio,
            max_weight=max_weight,
            previous_weights=scoped_previous_weights,
            transaction_cost=transaction_cost,
            current_date=current_date,
            predictor_panel=predictor_panel,
            selection_contexts=selection_contexts,
            predictor_context=predictor_context,
        )
    except ValueError:
        return [], np.zeros(len(universe_columns), dtype="float64")


def resolve_decision_policy_kind(strategy: EvaluatorStrategySpec) -> str:
    policy = dict(strategy.extensions).get(DECISION_POLICY_EXTENSION_KEY, COST_AWARE_NO_TRADE_DECISION_POLICY)
    if policy not in SUPPORTED_DECISION_POLICIES:
        raise ValueError(f"Unsupported decision policy: {policy}")
    return policy


def resolve_selected_assets_from_weights(universe_columns: pd.Index, weights: np.ndarray) -> list[str]:
    return [
        str(asset)
        for asset, weight in zip(universe_columns, weights, strict=True)
        if abs(float(weight)) > 1e-9
    ]


def build_decision_forecast_snapshot(
    *,
    history_returns: pd.DataFrame,
    volume_history: pd.DataFrame | None,
    strategy: EvaluatorStrategySpec,
    bars_per_year: float,
    current_date: str | None,
    predictor_panel: pd.DataFrame | None,
    selection_contexts: list[dict[str, object]] | None,
    predictor_context: dict[str, object] | None,
    availability_policy: dict[str, object],
) -> ForecastSnapshot | None:
    eligible_assets = resolve_eligible_assets(history_returns, availability_policy)
    if len(eligible_assets) < 2:
        return None
    clean_history_returns = prepare_history_returns_for_assets(history_returns, eligible_assets)
    if len(clean_history_returns) < 3 or len(clean_history_returns.columns) < 2:
        return None
    clean_volume_history = prepare_volume_history_for_assets(volume_history, clean_history_returns)
    try:
        return build_strategy_forecast_snapshot(
            clean_history_returns,
            clean_volume_history,
            strategy,
            bars_per_year=bars_per_year,
            current_date=current_date,
            predictor_panel=predictor_panel,
            selection_contexts=selection_contexts,
            predictor_context=predictor_context,
        )
    except ValueError:
        return None


def compute_weighted_forecast_confidence(
    forecast: ForecastSnapshot | None,
    universe_columns: pd.Index,
    target_weights: np.ndarray,
) -> float | None:
    if forecast is None:
        return None
    confidence = forecast.confidence.reindex(universe_columns).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    weight_values = np.abs(np.asarray(target_weights, dtype="float64"))
    weight_sum = float(weight_values.sum())
    if weight_sum <= 0:
        finite_confidence = confidence.replace([np.inf, -np.inf], np.nan).dropna()
        return None if finite_confidence.empty else float(finite_confidence.mean())
    return float(np.dot(confidence.to_numpy(dtype="float64"), weight_values) / weight_sum)


def compute_forecast_edge(
    forecast: ForecastSnapshot | None,
    universe_columns: pd.Index,
    current_weights: np.ndarray,
    target_weights: np.ndarray,
) -> float | None:
    if forecast is None or forecast.expected_return_proxy is None:
        return None
    proxy = forecast.expected_return_proxy.reindex(universe_columns).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    weight_delta = np.asarray(target_weights, dtype="float64") - np.asarray(current_weights, dtype="float64")
    return float(np.dot(weight_delta, proxy.to_numpy(dtype="float64")))


def build_portfolio_decision(
    *,
    history_returns: pd.DataFrame,
    volume_history: pd.DataFrame | None,
    strategy: EvaluatorStrategySpec,
    bars_per_year: float,
    universe_columns: pd.Index,
    current_weights: np.ndarray,
    current_selected_assets: list[str],
    target_weights: np.ndarray,
    target_selected_assets: list[str],
    transaction_cost: float,
    current_date: str | None,
    predictor_panel: pd.DataFrame | None,
    selection_contexts: list[dict[str, object]] | None,
    predictor_context: dict[str, object] | None,
    availability_policy: dict[str, object],
) -> PortfolioDecision:
    policy = resolve_decision_policy_kind(strategy)
    turnover = float(np.abs(target_weights - current_weights).sum())
    estimated_cost = turnover * float(transaction_cost)
    if policy == DIRECT_SCORE_DECISION_POLICY:
        return PortfolioDecision(
            selected_assets=list(target_selected_assets),
            weights=target_weights.copy(),
            policy=policy,
            action="rebalance",
            reason="direct_policy",
            turnover=turnover,
            estimated_cost_pct=round(estimated_cost * 100, 4),
            estimated_edge_pct=None,
            edge_source=None,
            average_confidence=None,
        )

    if float(np.sum(target_weights)) <= 0:
        return PortfolioDecision(
            selected_assets=[],
            weights=target_weights.copy(),
            policy=policy,
            action="rebalance",
            reason="target_cash",
            turnover=turnover,
            estimated_cost_pct=round(estimated_cost * 100, 4),
            estimated_edge_pct=None,
            edge_source=None,
            average_confidence=None,
        )

    forecast = build_decision_forecast_snapshot(
        history_returns=history_returns,
        volume_history=volume_history,
        strategy=strategy,
        bars_per_year=bars_per_year,
        current_date=current_date,
        predictor_panel=predictor_panel,
        selection_contexts=selection_contexts,
        predictor_context=predictor_context,
        availability_policy=availability_policy,
    )
    average_confidence = compute_weighted_forecast_confidence(forecast, universe_columns, target_weights)
    estimated_edge = compute_forecast_edge(forecast, universe_columns, current_weights, target_weights)
    edge_source = None if estimated_edge is None or forecast is None else forecast.edge_source
    no_trade_reason = None
    if turnover <= DEFAULT_NO_TRADE_BAND:
        no_trade_reason = "turnover_below_band"
    elif average_confidence is not None and average_confidence < DEFAULT_CONFIDENCE_FLOOR:
        no_trade_reason = "confidence_below_floor"
    elif estimated_edge is not None and estimated_edge <= estimated_cost:
        no_trade_reason = "edge_below_cost"

    if no_trade_reason is not None:
        held_assets = current_selected_assets or resolve_selected_assets_from_weights(universe_columns, current_weights)
        return PortfolioDecision(
            selected_assets=list(held_assets),
            weights=current_weights.copy(),
            policy=policy,
            action="no_trade",
            reason=no_trade_reason,
            turnover=turnover,
            estimated_cost_pct=round(estimated_cost * 100, 4),
            estimated_edge_pct=None if estimated_edge is None else round(estimated_edge * 100, 4),
            edge_source=edge_source,
            average_confidence=None if average_confidence is None else round(average_confidence, 4),
        )

    return PortfolioDecision(
        selected_assets=list(target_selected_assets),
        weights=target_weights.copy(),
        policy=policy,
        action="rebalance",
        reason="edge_after_cost",
        turnover=turnover,
        estimated_cost_pct=round(estimated_cost * 100, 4),
        estimated_edge_pct=None if estimated_edge is None else round(estimated_edge * 100, 4),
        edge_source=edge_source,
        average_confidence=None if average_confidence is None else round(average_confidence, 4),
    )


def compute_realized_decision_edge_pct(
    *,
    universe_columns: pd.Index,
    current_weights: np.ndarray,
    target_weights: np.ndarray,
    realized_returns: pd.Series | None,
) -> float | None:
    if realized_returns is None:
        return None
    clean_returns = realized_returns.reindex(universe_columns).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    weight_delta = np.asarray(target_weights, dtype="float64") - np.asarray(current_weights, dtype="float64")
    return round(float(np.dot(weight_delta, clean_returns.to_numpy(dtype="float64"))) * 100, 4)


def compute_edge_hit(estimated_edge_pct: float | None, realized_edge_pct: float | None) -> bool | None:
    if estimated_edge_pct is None or realized_edge_pct is None:
        return None
    if not math.isfinite(float(estimated_edge_pct)) or not math.isfinite(float(realized_edge_pct)):
        return None
    return float(np.sign(float(estimated_edge_pct))) == float(np.sign(float(realized_edge_pct)))


def serialize_portfolio_decision_event(
    date: str,
    decision: PortfolioDecision,
    *,
    universe_columns: pd.Index | None = None,
    current_weights: np.ndarray | None = None,
    target_weights: np.ndarray | None = None,
    realized_returns: pd.Series | None = None,
) -> dict[str, object]:
    realized_edge_pct = None
    if universe_columns is not None and current_weights is not None and target_weights is not None:
        realized_edge_pct = compute_realized_decision_edge_pct(
            universe_columns=universe_columns,
            current_weights=current_weights,
            target_weights=target_weights,
            realized_returns=realized_returns,
        )
    realized_edge_after_cost_pct = (
        None if realized_edge_pct is None else round(realized_edge_pct - decision.estimated_cost_pct, 4)
    )
    return {
        "date": date,
        "policy": decision.policy,
        "action": decision.action,
        "reason": decision.reason,
        "turnoverPct": round(decision.turnover * 100, 2),
        "estimatedCostPct": decision.estimated_cost_pct,
        "estimatedEdgePct": decision.estimated_edge_pct,
        "edgeSource": decision.edge_source,
        "realizedEdgePct": realized_edge_pct,
        "realizedEdgeAfterCostPct": realized_edge_after_cost_pct,
        "edgeHit": compute_edge_hit(decision.estimated_edge_pct, realized_edge_pct),
        "averageConfidence": decision.average_confidence,
        "selectedAssetCount": len(decision.selected_assets),
    }


def serialize_execution_trace_event(
    *,
    date: str,
    event_type: str,
    phase: str,
    universe_columns: pd.Index,
    max_investment_ratio: float,
    available_asset_count: int,
    eligible_asset_count: int,
    selected_assets: list[str],
    target_weights: np.ndarray | None,
    executed_weights: np.ndarray | None,
    decision_action: str | None,
    decision_reason: str | None,
    turnover_pct: float | None,
    estimated_cost_pct: float | None,
    estimated_edge_pct: float | None,
    edge_source: str | None,
    average_confidence: float | None,
) -> dict[str, object]:
    return {
        "date": date,
        "eventType": event_type,
        "phase": phase,
        "availableAssetCount": available_asset_count,
        "eligibleAssetCount": eligible_asset_count,
        "selectedAssets": list(selected_assets),
        "targetWeights": serialize_weights(universe_columns, target_weights, max_investment_ratio)
        if target_weights is not None
        else [],
        "executedWeights": serialize_weights(universe_columns, executed_weights, max_investment_ratio)
        if executed_weights is not None
        else [],
        "decisionAction": decision_action,
        "decisionReason": decision_reason,
        "turnoverPct": None if turnover_pct is None else round(float(turnover_pct), 2),
        "estimatedCostPct": estimated_cost_pct,
        "estimatedEdgePct": estimated_edge_pct,
        "edgeSource": edge_source,
        "averageConfidence": average_confidence,
    }




def compare_portfolio_runs(
    closes: pd.DataFrame,
    volumes: pd.DataFrame | None,
    strategies: list[EvaluatorStrategySpec],
    initial_capital: float,
    split_ratio: float,
    bars_per_year: float = 252.0,
    execution_assumptions: dict | None = None,
    cost_model: dict | None = None,
    transaction_cost: float | None = None,
    portfolio_state: PortfolioState | None = None,
    predictor_panels_by_strategy: dict[str, pd.DataFrame] | None = None,
    strategy_signal_execution_contexts_by_key: dict[str, tuple[list[dict[str, object]], dict[str, object] | None]] | None = None,
    availability_policy: dict[str, object] | None = None,
) -> list[dict]:
    if not strategies:
        raise ValueError("At least one strategy is required.")
    if initial_capital <= 0:
        raise ValueError("Initial capital must be positive.")
    if execution_assumptions is None:
        execution_assumptions = {
            "kind": "close_execution_assumptions",
            "label": "終値約定",
            "parameters": {
                "fillPrice": "close",
            },
            "costModel": None,
        }
    if cost_model is None:
        cost_model = execution_assumptions.get("costModel")
    if cost_model is None:
        if transaction_cost is None:
            raise ValueError("Either cost_model or transaction_cost must be provided.")
        if transaction_cost < 0 or transaction_cost >= 1:
            raise ValueError("Transaction cost must be between 0 and 1.")
        cost_model = build_flat_cost_model(commission_pct=transaction_cost * 100, slippage_pct=0.0)

    if strategy_signal_execution_contexts_by_key is None:
        strategy_signal_execution_contexts_by_key = {}
    if availability_policy is None:
        availability_policy = build_default_availability_policy()

    runs: list[dict] = []
    for strategy in strategies:
        signal_execution_contexts = resolve_runtime_signal_execution_contexts(
            strategy,
            strategy_signal_execution_contexts_by_key.get(strategy.key),
        )
        selection_contexts, predictor_context = signal_execution_contexts
        rebalance_schedule = strategy.execution_policy.rebalance_schedule
        decision_schedule = resolve_decision_schedule(strategy)
        strategy_universe = [
            asset
            for asset in strategy.investment_universe.tickers
            if asset in closes.columns
        ]
        if len(strategy_universe) < 1:
            raise ValueError("Strategy investment universe must contain at least one available asset.")

        strategy_closes = closes[strategy_universe]
        strategy_volumes = volumes[strategy_universe] if volumes is not None else None
        returns = strategy_closes.pct_change(fill_method=None).iloc[1:]
        returns = returns.dropna(how="all")
        if len(returns) < 6:
            raise ValueError("At least 6 return rows are required for portfolio comparison.")
        strategy_volumes = strategy_volumes.loc[returns.index] if strategy_volumes is not None else None
        cost_inputs = resolve_cost_model_inputs(
            universe_columns=returns.columns,
            cost_model=cost_model,
        )
        default_transaction_cost = float(cost_inputs["defaultLinearRate"])
        asset_transaction_costs = np.asarray(cost_inputs["linearRates"], dtype="float64")
        asset_impact_costs = np.asarray(cost_inputs["impactRates"], dtype="float64")
        adv_window_bars = int(cost_inputs["advWindowBars"])
        min_adv_notional = float(cost_inputs["minAdvNotional"])

        split_index = compute_split_index(len(returns), split_ratio)
        train_returns = returns.iloc[:split_index]
        initial_portfolio_weights = resolve_initial_weights(
            universe_columns=returns.columns,
            portfolio_state=portfolio_state,
        )

        portfolio_model = strategy.portfolio_model
        risk_controls = strategy.risk_controls
        initial_selected_assets, initial_weights = compute_dynamic_portfolio_allocation(
            history_returns=train_returns,
            volume_history=strategy_volumes.loc[train_returns.index] if strategy_volumes is not None else None,
            strategy=strategy,
            portfolio_model=portfolio_model,
            bars_per_year=bars_per_year,
            universe_columns=returns.columns,
            max_investment_ratio=risk_controls.max_investment_ratio,
            max_weight=risk_controls.max_weight,
            previous_weights=initial_portfolio_weights,
            transaction_cost=default_transaction_cost,
            current_date=str(returns.index[split_index]) if split_index < len(returns.index) else None,
            predictor_panel=None if predictor_panels_by_strategy is None else predictor_panels_by_strategy.get(strategy.key),
            selection_contexts=selection_contexts,
            predictor_context=predictor_context,
            availability_policy=availability_policy,
        )
        backtest = run_portfolio_backtest(
            closes=strategy_closes,
            returns=returns,
            volumes=strategy_volumes,
            split_index=split_index,
            split_ratio=split_ratio,
            strategy=strategy,
            portfolio_model=portfolio_model,
            bars_per_year=bars_per_year,
            warmup_weights=initial_portfolio_weights,
            initial_weights=initial_weights,
            initial_selected_assets=initial_selected_assets,
            max_investment_ratio=risk_controls.max_investment_ratio,
            initial_capital=initial_capital,
            transaction_cost=default_transaction_cost,
            asset_transaction_costs=asset_transaction_costs,
            asset_impact_costs=asset_impact_costs,
            adv_window_bars=adv_window_bars,
            min_adv_notional=min_adv_notional,
            max_weight=risk_controls.max_weight,
            decision_schedule=decision_schedule,
            rebalance_schedule=rebalance_schedule,
            predictor_panel=None if predictor_panels_by_strategy is None else predictor_panels_by_strategy.get(strategy.key),
            selection_contexts=selection_contexts,
            predictor_context=predictor_context,
            availability_policy=availability_policy,
        )

        runs.append(
            {
                "kind": "run_result",
                "schemaVersion": "v1",
                "key": strategy.key,
                "strategy": serialize_evaluator_strategy_spec(strategy),
                "weights": serialize_weights(
                    returns.columns,
                    backtest["latestWeights"],
                    risk_controls.max_investment_ratio,
                ),
                "selectedAssets": backtest["latestSelectedAssets"],
                "summary": backtest["summary"],
                "splitAnalysis": backtest["splitAnalysis"],
                "series": backtest["series"],
                "availabilitySummary": backtest["availabilitySummary"],
                "decisionSummary": backtest["decisionSummary"],
                "decisionEvents": backtest["decisionEvents"],
                "executionTrace": backtest["executionTrace"],
                "availabilityPolicy": availability_policy,
            }
        )

    return runs


def evaluate_strategy_run(
    closes: pd.DataFrame,
    volumes: pd.DataFrame | None,
    strategy: EvaluatorStrategySpec,
    initial_capital: float,
    split_ratio: float,
    bars_per_year: float = 252.0,
    execution_assumptions: dict | None = None,
    cost_model: dict | None = None,
    transaction_cost: float | None = None,
    portfolio_state: PortfolioState | None = None,
    predictor_panel: pd.DataFrame | None = None,
    signal_execution_contexts: tuple[list[dict[str, object]], dict[str, object] | None] | None = None,
    availability_policy: dict[str, object] | None = None,
) -> dict:
    return compare_portfolio_runs(
        closes=closes,
        volumes=volumes,
        strategies=[strategy],
        bars_per_year=bars_per_year,
        initial_capital=initial_capital,
        split_ratio=split_ratio,
        execution_assumptions=execution_assumptions,
        cost_model=cost_model,
        transaction_cost=transaction_cost,
        portfolio_state=portfolio_state,
        predictor_panels_by_strategy=None if predictor_panel is None else {strategy.key: predictor_panel},
        strategy_signal_execution_contexts_by_key=(
            None
            if signal_execution_contexts is None
            else {strategy.key: signal_execution_contexts}
        ),
        availability_policy=availability_policy,
    )[0]


def evaluate_strategy_definition_run(
    closes: pd.DataFrame,
    volumes: pd.DataFrame | None,
    strategy_definition: StrategyDefinition,
    initial_capital: float,
    split_ratio: float,
    bars_per_year: float = 252.0,
    execution_assumptions: dict | None = None,
    cost_model: dict | None = None,
    transaction_cost: float | None = None,
    portfolio_state: PortfolioState | None = None,
    predictor_panel: pd.DataFrame | None = None,
    availability_policy: dict[str, object] | None = None,
) -> dict:
    try:
        strategy = build_executable_evaluator_strategy_spec_from_definition(strategy_definition)
    except ValueError as exc:
        raise ValueError(
            f"Strategy definition {strategy_definition.strategy_id} is not executable: {exc}"
        ) from exc
    signal_execution_contexts = get_strategy_definition_signal_execution_contexts(strategy_definition)
    run = evaluate_strategy_run(
        closes=closes,
        volumes=volumes,
        strategy=strategy,
        bars_per_year=bars_per_year,
        initial_capital=initial_capital,
        split_ratio=split_ratio,
        execution_assumptions=execution_assumptions,
        cost_model=cost_model,
        transaction_cost=transaction_cost,
        portfolio_state=portfolio_state,
        predictor_panel=predictor_panel,
        signal_execution_contexts=signal_execution_contexts,
        availability_policy=availability_policy,
    )
    run["key"] = strategy_definition.key
    run["strategy"] = serialize_strategy_definition(strategy_definition)
    return run


def compare_portfolio_models(
    closes: pd.DataFrame,
    volumes: pd.DataFrame | None,
    portfolio_models: list[PortfolioModelSpec],
    initial_capital: float,
    split_ratio: float,
    transaction_cost: float,
    bars_per_year: float = 252.0,
    portfolio_state: PortfolioState | None = None,
) -> list[dict]:
    return compare_portfolio_runs(
        closes=closes,
        volumes=volumes,
        strategies=[
            build_evaluator_strategy_spec(
                investment_universe=build_investment_universe_spec(
                    tickers=list(closes.columns),
                    key="ad_hoc_universe",
                    label="Ad hoc universe",
                ),
                selection=build_selection_spec("full_universe"),
                portfolio_model=portfolio_model,
                execution_policy=build_execution_policy_spec(
                    key="hold",
                    label="保有",
                    entry="hold",
                    rebalance_schedule="hold",
                ),
                risk_controls=build_risk_controls_spec(
                    max_investment_ratio=1.0,
                    max_weight=None,
                ),
            )
            for portfolio_model in portfolio_models
        ],
        bars_per_year=bars_per_year,
        initial_capital=initial_capital,
        split_ratio=split_ratio,
        execution_assumptions={
            "kind": "close_execution_assumptions",
            "label": "終値約定",
            "parameters": {
                "fillPrice": "close",
            },
            "costModel": build_flat_cost_model(
                commission_pct=transaction_cost * 100,
                slippage_pct=0.0,
            ),
        },
        transaction_cost=transaction_cost,
        portfolio_state=portfolio_state,
    )


def select_assets(
    returns: pd.DataFrame,
    volume_history: pd.DataFrame | None,
    selection: EvaluatorStrategySpec | RankingSourceSpec,
    *,
    bars_per_year: float,
    selection_contexts: list[dict[str, object]] | None = None,
) -> list[str]:
    if isinstance(selection, EvaluatorStrategySpec):
        strategy = selection
        base_selection = resolve_primary_selection_spec(
            strategy,
            selection_contexts=selection_contexts,
        )
        trailing_total_returns = compute_strategy_selection_trailing_total_returns(
            returns,
            strategy,
            bars_per_year=bars_per_year,
            selection_contexts=selection_contexts,
        )
    else:
        strategy = None
        base_selection = selection
        trailing_total_returns = compute_trailing_total_returns(
            returns,
            selection,
            bars_per_year=bars_per_year,
        )

    if base_selection.strategy_type == "full_universe":
        return list(returns.columns)
    if base_selection.strategy_type == "full_universe_momentum_tilt":
        return list(returns.columns)
    if base_selection.strategy_type == "full_universe_momentum_low_vol_tilt":
        return list(returns.columns)
    if base_selection.strategy_type == "full_universe_momentum_macro_tilt":
        return list(returns.columns)
    if base_selection.strategy_type == "momentum_top3":
        selected = trailing_total_returns.sort_values(ascending=False).head(min(3, len(trailing_total_returns)))
        return list(selected.index)
    if base_selection.strategy_type == "dual_momentum_top3":
        positive_returns = trailing_total_returns[trailing_total_returns > 0]
        if positive_returns.empty:
            return []
        selected = positive_returns.sort_values(ascending=False).head(min(3, len(positive_returns)))
        return list(selected.index)
    if base_selection.strategy_type == "trailing_momentum_low_vol_universe":
        positive_assets = trailing_total_returns[trailing_total_returns > 0].sort_values(ascending=False)
        if positive_assets.empty:
            return []
        volatilities = returns[positive_assets.index].std()
        low_vol_threshold = float(volatilities.quantile(0.6))
        selected = volatilities[volatilities <= low_vol_threshold].sort_values()
        if selected.empty:
            return [str(volatilities.idxmin())]
        return list(selected.index)
    if base_selection.strategy_type == "positive_momentum_universe":
        positive_returns = trailing_total_returns[trailing_total_returns > 0]
        if positive_returns.empty:
            return []
        return list(positive_returns.sort_values(ascending=False).index)
    if base_selection.strategy_type == "positive_momentum_low_vol_universe":
        positive_assets = trailing_total_returns[trailing_total_returns > 0].sort_values(ascending=False)
        if positive_assets.empty:
            return []
        volatilities = returns[positive_assets.index].std()
        low_vol_threshold = float(volatilities.median())
        selected = volatilities[volatilities <= low_vol_threshold].sort_values()
        if selected.empty:
            return [str(volatilities.idxmin())]
        return list(selected.index)
    if base_selection.strategy_type == "positive_momentum_high_volume_universe":
        if volume_history is None:
            raise ValueError("Volume history is required for the selected portfolio strategy.")
        total_returns = (1 + returns).prod() - 1
        positive_assets = total_returns[total_returns > 0].sort_values(ascending=False)
        if positive_assets.empty:
            return []
        positive_volumes = volume_history[positive_assets.index].dropna(how="all")
        if positive_volumes.empty:
            return list(positive_assets.index)
        recent_window = max(2, min(20, max(2, len(positive_volumes) // 4)))
        if len(positive_volumes) <= recent_window:
            baseline = positive_volumes.mean(axis=0)
        else:
            baseline = positive_volumes.iloc[:-recent_window].mean(axis=0)
        baseline = baseline.replace(0, np.nan)
        recent = positive_volumes.iloc[-recent_window:].mean(axis=0)
        volume_score = (recent / baseline).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        threshold = float(volume_score.median())
        selected = volume_score[volume_score >= threshold].sort_values(ascending=False)
        if selected.empty:
            return [str(volume_score.idxmax())]
        return list(selected.index)
    if base_selection.strategy_type == "positive_trend_short_reversal":
        return select_positive_trend_short_reversal_assets(
            returns,
            base_selection,
            bars_per_year=bars_per_year,
        )
    if base_selection.strategy_type == "risk_regime_positive_momentum":
        return select_risk_regime_positive_momentum_assets(
            returns,
            trailing_total_returns,
            base_selection,
            bars_per_year=bars_per_year,
        )
    raise ValueError("Unsupported portfolio strategy.")


def select_positive_trend_short_reversal_assets(
    returns: pd.DataFrame,
    selection: RankingSourceSpec,
    *,
    bars_per_year: float,
) -> list[str]:
    score_parameters = dict(selection.ranking_signal.score_parameters)
    trend_window_bars = get_score_parameter_window_bars(
        selection,
        "trendWindowSpec",
        bars_per_year=bars_per_year,
        default_window_spec={"unit": "days", "value": 60},
    )
    reversal_window_bars = get_score_parameter_window_bars(
        selection,
        "reversalWindowSpec",
        bars_per_year=bars_per_year,
        default_window_spec={"unit": "days", "value": 5},
    )
    asset_count = max(1, int(float(score_parameters.get("assetCount", 5))))
    trend_returns = compute_window_total_returns(returns, trend_window_bars)
    reversal_returns = compute_window_total_returns(returns, reversal_window_bars)
    positive_trend_assets = trend_returns[trend_returns > 0].index
    if len(positive_trend_assets) == 0:
        return []
    candidates = reversal_returns.reindex(positive_trend_assets).replace([np.inf, -np.inf], np.nan).dropna()
    if candidates.empty:
        return []
    selected = candidates.sort_values(ascending=True).head(min(asset_count, len(candidates)))
    return [str(asset) for asset in selected.index]


def select_risk_regime_positive_momentum_assets(
    returns: pd.DataFrame,
    trailing_total_returns: pd.Series,
    selection: RankingSourceSpec,
    *,
    bars_per_year: float,
) -> list[str]:
    score_parameters = dict(selection.ranking_signal.score_parameters)
    eligible_returns = trailing_total_returns
    risk_proxy_assets = parse_score_parameter_strings(
        score_parameters.get("riskProxyAssets"),
        default=("SPY", "QQQ"),
    )
    risk_proxy_columns = [asset for asset in risk_proxy_assets if asset in returns.columns]
    if risk_proxy_columns:
        risk_window_bars = get_score_parameter_window_bars(
            selection,
            "riskWindowSpec",
            bars_per_year=bars_per_year,
            default_window_spec={"unit": "days", "value": 60},
        )
        risk_proxy_returns = compute_window_total_returns(returns[risk_proxy_columns], risk_window_bars)
        risk_proxy_return = float(risk_proxy_returns.mean())
        if risk_proxy_return < 0:
            defensive_asset_classes = set(
                parse_score_parameter_strings(
                    score_parameters.get("defensiveAssetClasses"),
                    default=("bond_etf", "commodity_etf", "currency_etf"),
                )
            )
            defensive_assets = [
                str(asset)
                for asset in returns.columns
                if (instrument := get_instrument(str(asset))) is not None
                and instrument.asset_class in defensive_asset_classes
            ]
            eligible_returns = trailing_total_returns.reindex(defensive_assets).dropna()

    positive_returns = eligible_returns[eligible_returns > 0]
    if positive_returns.empty:
        return []
    return [str(asset) for asset in positive_returns.sort_values(ascending=False).index]


def parse_score_parameter_strings(value: object, *, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        parsed = tuple(str(item) for item in value if str(item))
        return parsed or default
    return default


def get_score_parameter_window_bars(
    selection: RankingSourceSpec,
    parameter_key: str,
    *,
    bars_per_year: float,
    default_window_spec: dict[str, object],
) -> int:
    score_parameters = dict(selection.ranking_signal.score_parameters)
    window_spec = score_parameters.get(parameter_key, default_window_spec)
    if not isinstance(window_spec, dict):
        window_spec = default_window_spec
    return convert_window_spec_to_bars(window_spec, bars_per_year=bars_per_year)


def compute_window_total_returns(returns: pd.DataFrame, window_bars: int) -> pd.Series:
    lookback = min(len(returns), max(1, int(window_bars)))
    window_returns = returns.iloc[-lookback:]
    return (1 + window_returns).prod() - 1


def compute_strategy_score_series_base(
    returns: pd.DataFrame,
    volume_history: pd.DataFrame | None,
    selection: RankingSourceSpec,
    *,
    bars_per_year: float,
) -> pd.Series | None:
    score_kind = selection.ranking_signal.score_model.kind
    if score_kind == "none":
        return None
    if score_kind == "trailing_momentum":
        return compute_trailing_total_returns(returns, selection, bars_per_year=bars_per_year)
    if score_kind == "momentum_low_vol":
        score_parameters = dict(selection.ranking_signal.score_parameters)
        momentum_weight = float(score_parameters.get("momentum_weight", 0.7))
        low_vol_weight = float(score_parameters.get("low_vol_weight", 0.3))
        trailing_returns = compute_trailing_total_returns(
            returns,
            selection,
            bars_per_year=bars_per_year,
        )
        momentum_rank = trailing_returns.rank(method="average", pct=True)
        low_vol_rank = (-returns.std()).rank(method="average", pct=True)
        return momentum_weight * momentum_rank + low_vol_weight * low_vol_rank
    if score_kind == "momentum_macro":
        score_parameters = dict(selection.ranking_signal.score_parameters)
        momentum_weight = float(score_parameters.get("momentum_weight", 0.85))
        macro_weight = float(score_parameters.get("macro_weight", 0.15))
        trailing_returns = compute_trailing_total_returns(
            returns,
            selection,
            bars_per_year=bars_per_year,
        )
        momentum_rank = trailing_returns.rank(method="average", pct=True)
        macro_rank = compute_macro_proxy_rank(returns, selection, bars_per_year=bars_per_year)
        return momentum_weight * momentum_rank + macro_weight * macro_rank
    if score_kind == "volume_strength":
        if volume_history is None:
            raise ValueError("Volume history is required for the selected ranking model.")
        return compute_volume_strength(volume_history)
    if score_kind == "volatility":
        return -returns.std()
    raise ValueError("Unsupported score model.")


def get_ranking_window_bars(selection: RankingSourceSpec, *, bars_per_year: float) -> int:
    score_parameters = dict(selection.ranking_signal.score_parameters)
    window_spec = score_parameters.get("windowSpec")
    if isinstance(window_spec, dict):
        return convert_window_spec_to_bars(window_spec, bars_per_year=bars_per_year)
    return max(1, int(round(float(bars_per_year))))


def compute_trailing_total_returns(
    returns: pd.DataFrame,
    selection: RankingSourceSpec,
    *,
    bars_per_year: float,
) -> pd.Series:
    lookback = min(len(returns), get_ranking_window_bars(selection, bars_per_year=bars_per_year))
    trailing_returns = returns.iloc[-lookback:]
    return (1 + trailing_returns).prod() - 1


def compute_volume_strength(volume_history: pd.DataFrame) -> pd.Series:
    recent_window = max(2, min(20, max(2, len(volume_history) // 4)))
    if len(volume_history) <= recent_window:
        baseline = volume_history.mean(axis=0)
    else:
        baseline = volume_history.iloc[:-recent_window].mean(axis=0)
    baseline = baseline.replace(0, np.nan)
    recent = volume_history.iloc[-recent_window:].mean(axis=0)
    return (recent / baseline).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def compute_macro_proxy_rank(
    returns: pd.DataFrame,
    selection: RankingSourceSpec,
    *,
    bars_per_year: float,
) -> pd.Series:
    trailing_returns = compute_trailing_total_returns(
        returns,
        selection,
        bars_per_year=bars_per_year,
    )
    risk_assets = [asset for asset in ["SPY", "QQQ", "IWM", "EFA", "EEM", "EWJ", "EWZ", "VNQ", "DBC", "USO", "BTC-USD", "ETH-USD"] if asset in trailing_returns.index]
    defensive_assets = [asset for asset in ["TLT", "IEF", "LQD", "HYG", "TIP", "GLD", "SLV", "UUP"] if asset in trailing_returns.index]
    if not risk_assets or not defensive_assets:
        return trailing_returns.rank(method="average", pct=True)

    risk_signal = float(trailing_returns.loc[risk_assets].mean())
    defensive_signal = float(trailing_returns.loc[defensive_assets].mean())
    regime_signal = risk_signal - defensive_signal

    macro_scores: dict[str, float] = {}
    for asset in trailing_returns.index:
        if asset in risk_assets:
            macro_scores[asset] = regime_signal
        elif asset in defensive_assets:
            macro_scores[asset] = -regime_signal
        else:
            macro_scores[asset] = 0.0
    return pd.Series(macro_scores, dtype="float64").rank(method="average", pct=True)


def compute_prediction_feature_frame(
    returns: pd.DataFrame,
    volume_history: pd.DataFrame | None,
    selection: RankingSourceSpec,
    *,
    bars_per_year: float,
) -> pd.DataFrame:
    trailing_returns = compute_trailing_total_returns(
        returns,
        selection,
        bars_per_year=bars_per_year,
    )
    score_series = compute_strategy_score_series_base(
        returns,
        volume_history,
        selection,
        bars_per_year=bars_per_year,
    )
    if score_series is None:
        score_series = pd.Series(0.0, index=returns.columns, dtype="float64")
    vol_series = (-returns.std()).rank(method="average", pct=True)
    macro_series = compute_macro_proxy_rank(
        returns,
        selection,
        bars_per_year=bars_per_year,
    )
    if volume_history is not None:
        volume_series = compute_volume_strength(volume_history).rank(method="average", pct=True)
    else:
        volume_series = pd.Series(0.0, index=returns.columns, dtype="float64")

    feature_frame = pd.DataFrame(
        {
            "score": score_series.reindex(returns.columns).fillna(0.0),
            "momentum": trailing_returns.rank(method="average", pct=True).reindex(returns.columns).fillna(0.0),
            "lowVolRank": vol_series.reindex(returns.columns).fillna(0.0),
            "macroRank": macro_series.reindex(returns.columns).fillna(0.0),
            "volumeStrength": volume_series.reindex(returns.columns).fillna(0.0),
        }
    )
    return feature_frame.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def compute_predictor_panel(
    returns: pd.DataFrame,
    volumes: pd.DataFrame | None,
    predictor_spec: PredictorSpec,
    *,
    bars_per_year: float,
) -> pd.DataFrame:
    selection = extract_ranking_feature_recipe(predictor_spec.feature_spec)
    ranking_universe = [
        asset for asset in predictor_spec.signal_spec.observation_spec.tickers if asset in returns.columns
    ]
    scoped_returns = returns[ranking_universe]
    scoped_volumes = volumes[ranking_universe] if volumes is not None else None
    prediction_panel = pd.DataFrame(index=scoped_returns.index, columns=scoped_returns.columns, dtype="float64")

    target_horizon_bars = convert_horizon_spec_to_bars(
        {key: value for key, value in predictor_spec.target_spec.horizon_spec},
        bars_per_year=bars_per_year,
    )
    feature_names = list(extract_feature_keys(predictor_spec.feature_spec))
    feature_count = len(feature_names)
    xtx = np.zeros((feature_count + 1, feature_count + 1), dtype="float64")
    xty = np.zeros(feature_count + 1, dtype="float64")
    train_sample_count = 0
    min_train_samples = predictor_spec.training_spec.min_train_samples
    signal_source_weight = float(predictor_spec.engine_spec.combiner_spec.signal_source_weight or 0.8)
    learner_weight = float(predictor_spec.engine_spec.combiner_spec.learner_weight or 0.2)

    availability_policy = build_default_availability_policy()
    for index in range(2, len(scoped_returns)):
        raw_history_returns = scoped_returns.iloc[:index]
        eligible_assets = resolve_eligible_assets(raw_history_returns, availability_policy)
        if len(eligible_assets) < 2:
            continue
        history_returns = prepare_history_returns_for_assets(raw_history_returns, eligible_assets)
        if len(history_returns) < 3 or len(history_returns.columns) < 2:
            continue
        raw_history_volumes = scoped_volumes.iloc[:index] if scoped_volumes is not None else None
        history_volumes = prepare_volume_history_for_assets(raw_history_volumes, history_returns)
        selected_assets = select_assets(
            history_returns,
            history_volumes,
            selection,
            bars_per_year=bars_per_year,
        )
        if len(selected_assets) < 2:
            continue

        base_scores = compute_strategy_score_series_base(
            history_returns,
            history_volumes,
            selection,
            bars_per_year=bars_per_year,
        )
        if base_scores is None:
            continue
        selected_scores = base_scores.loc[selected_assets].dropna()
        if len(selected_scores) < 2 or selected_scores.nunique() < 2:
            continue

        feature_frame = compute_prediction_feature_frame(
            history_returns,
            history_volumes,
            selection,
            bars_per_year=bars_per_year,
        )
        aligned_features = feature_frame.loc[selected_scores.index, feature_names].astype("float64")
        if len(aligned_features) < 2:
            continue

        signal_source_values: pd.Series | None = None
        signal_source_spec = predictor_spec.engine_spec.signal_source_spec
        if signal_source_spec is not None:
            if signal_source_spec.kind == "ranking_signal":
                signal_source_values = selected_scores
            elif signal_source_spec.feature_key is not None:
                candidate_signal_values = (
                    feature_frame.loc[selected_scores.index, signal_source_spec.feature_key]
                    .astype("float64")
                    .dropna()
                )
                if len(candidate_signal_values) >= 2 and candidate_signal_values.nunique() >= 2:
                    signal_source_values = candidate_signal_values

        learner_prediction_values: pd.Series | None = None
        learner_spec = predictor_spec.engine_spec.learner_spec
        if learner_spec is not None and train_sample_count >= min_train_samples:
            design_matrix = np.column_stack(
                [np.ones(len(aligned_features), dtype="float64"), aligned_features.to_numpy(dtype="float64")]
            )
            if learner_spec.kind == "ridge_regression":
                ridge_alpha = float(learner_spec.ridge_alpha or 1.0)
                penalty = np.eye(xtx.shape[0], dtype="float64") * ridge_alpha
                penalty[0, 0] = 0.0
                beta = np.linalg.pinv(xtx + penalty) @ xty
            else:
                beta = np.linalg.pinv(xtx) @ xty
            learner_prediction_values = pd.Series(
                design_matrix @ beta,
                index=aligned_features.index,
                dtype="float64",
            )
            if learner_prediction_values.nunique() < 2:
                learner_prediction_values = None

        prediction_values: pd.Series | None = None
        if predictor_spec.engine_spec.combiner_spec.kind == "baseline_signal_only":
            prediction_values = signal_source_values
        elif predictor_spec.engine_spec.combiner_spec.kind == "learner_only":
            prediction_values = learner_prediction_values
        elif signal_source_values is not None and learner_prediction_values is not None:
            blended = (
                signal_source_weight * standardize_prediction_series(signal_source_values)
                + learner_weight * standardize_prediction_series(learner_prediction_values)
            )
            if blended.nunique() >= 2:
                prediction_values = blended

        if prediction_values is not None:
            prediction_panel.loc[scoped_returns.index[index], prediction_values.index] = prediction_values.to_numpy(
                dtype="float64"
            )

        if index > len(scoped_returns) - target_horizon_bars:
            continue
        forward_window = scoped_returns.iloc[index : index + target_horizon_bars]
        if len(forward_window) < target_horizon_bars:
            continue
        forward_returns = ((1 + forward_window).prod() - 1).loc[selected_scores.index].dropna()
        if len(forward_returns) < 2 or forward_returns.nunique() < 2:
            continue

        aligned_features = aligned_features.loc[forward_returns.index]
        if len(aligned_features) < 2:
            continue
        if predictor_spec.target_spec.baseline == "cross_sectional_mean":
            target_values = forward_returns - float(forward_returns.mean())
        else:
            raise ValueError("Unsupported prediction target baseline.")
        target_values = apply_prediction_target_transform(
            target_values,
            predictor_spec.target_spec,
        )
        if target_values.nunique() < 2:
            continue

        design_matrix = np.column_stack(
            [np.ones(len(aligned_features), dtype="float64"), aligned_features.to_numpy(dtype="float64")]
        )
        xtx += design_matrix.T @ design_matrix
        xty += design_matrix.T @ target_values.to_numpy(dtype="float64")
        train_sample_count += len(aligned_features)

    return prediction_panel


def resolve_primary_selection_spec(
    strategy_or_selection: EvaluatorStrategySpec | RankingSourceSpec,
    *,
    selection_contexts: list[dict[str, object]] | None = None,
) -> RankingSourceSpec:
    if selection_contexts is not None and len(selection_contexts) > 0:
        return selection_contexts[0]["selection"]
    if isinstance(strategy_or_selection, EvaluatorStrategySpec):
        return strategy_or_selection.selection
    return strategy_or_selection


def resolve_predictor_use_spec(
    strategy_or_selection: EvaluatorStrategySpec | RankingSourceSpec,
    *,
    predictor_context: dict[str, object] | None = None,
) -> PredictorUseSpec | None:
    if predictor_context is not None:
        return build_predictor_use_spec(
            predictor_key=str(predictor_context["predictor_key"]),
            signal_weight=float(predictor_context["signal_weight"]),
            predictor_weight=float(predictor_context["predictor_weight"]),
        )
    if isinstance(strategy_or_selection, EvaluatorStrategySpec):
        return strategy_or_selection.predictor_use
    return None


def compute_prediction_supplemented_score_series(
    returns: pd.DataFrame,
    volume_history: pd.DataFrame | None,
    strategy_or_selection: EvaluatorStrategySpec | RankingSourceSpec,
    *,
    bars_per_year: float,
    base_score_series: pd.Series,
    predictor_snapshot: pd.Series | None = None,
    predictor_use: PredictorUseSpec | None = None,
) -> pd.Series:
    predictor_use = predictor_use or resolve_predictor_use_spec(strategy_or_selection)
    if predictor_use is None:
        return base_score_series

    if predictor_snapshot is None:
        return base_score_series
    linear_prediction_values = predictor_snapshot.reindex(base_score_series.index).dropna()
    if len(linear_prediction_values) < 2:
        return base_score_series
    base_score_series = base_score_series.loc[linear_prediction_values.index]
    if linear_prediction_values.nunique() < 2:
        return base_score_series

    signal_weight = predictor_use.signal_weight
    linear_weight = predictor_use.predictor_weight
    blended_scores = (
        signal_weight * standardize_prediction_series(base_score_series)
        + linear_weight * standardize_prediction_series(linear_prediction_values)
    )
    if blended_scores.nunique() < 2:
        return base_score_series
    return blended_scores


def compute_strategy_score_series(
    returns: pd.DataFrame,
    volume_history: pd.DataFrame | None,
    strategy_or_selection: EvaluatorStrategySpec | RankingSourceSpec,
    *,
    bars_per_year: float,
    current_date: str | None = None,
    predictor_panel: pd.DataFrame | None = None,
    selection_contexts: list[dict[str, object]] | None = None,
    predictor_context: dict[str, object] | None = None,
) -> pd.Series | None:
    selection = resolve_primary_selection_spec(
        strategy_or_selection,
        selection_contexts=selection_contexts,
    )
    if isinstance(strategy_or_selection, EvaluatorStrategySpec):
        base_score_series = compute_strategy_selection_score_series(
            returns,
            volume_history,
            strategy_or_selection,
            bars_per_year=bars_per_year,
            selection_contexts=selection_contexts,
        )
    else:
        base_score_series = compute_strategy_score_series_base(
            returns,
            volume_history,
            selection,
            bars_per_year=bars_per_year,
        )
    if base_score_series is None:
        return None
    predictor_snapshot = resolve_predictor_snapshot(
        predictor_panel,
        current_date=current_date,
    )
    return compute_prediction_supplemented_score_series(
        returns,
        volume_history,
        strategy_or_selection,
        bars_per_year=bars_per_year,
        base_score_series=base_score_series,
        predictor_snapshot=predictor_snapshot,
        predictor_use=resolve_predictor_use_spec(
            strategy_or_selection,
            predictor_context=predictor_context,
        ),
    )


def compute_signal_return_proxy_series(
    returns: pd.DataFrame,
    aligned_scores: pd.Series,
) -> pd.Series | None:
    if aligned_scores.isna().all():
        return None

    filled_scores = aligned_scores.fillna(float(aligned_scores.mean()))
    if filled_scores.nunique() < 2:
        return None

    standardized_scores = (filled_scores - filled_scores.mean()) / filled_scores.std(ddof=0)
    standardized_scores = standardized_scores.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    historical_mean = returns.mean(axis=0)
    proxy_scale = float(max(historical_mean.std(ddof=0), 1e-4))
    proxy = np.clip(standardized_scores.to_numpy(dtype="float64") * proxy_scale, -0.05, 0.05)
    return pd.Series(proxy, index=returns.columns, dtype="float64")


def compute_expected_return_proxy_series(
    returns: pd.DataFrame,
    aligned_scores: pd.Series,
) -> pd.Series | None:
    return compute_signal_return_proxy_series(returns, aligned_scores)


def compute_forecast_confidence_series(aligned_scores: pd.Series) -> pd.Series:
    confidence = pd.Series(0.0, index=aligned_scores.index, dtype="float64")
    confidence.loc[aligned_scores.notna()] = 1.0
    return confidence


def build_strategy_forecast_snapshot(
    returns: pd.DataFrame,
    volume_history: pd.DataFrame | None,
    strategy_or_selection: EvaluatorStrategySpec | RankingSourceSpec,
    *,
    bars_per_year: float,
    current_date: str | None = None,
    predictor_panel: pd.DataFrame | None = None,
    selection_contexts: list[dict[str, object]] | None = None,
    predictor_context: dict[str, object] | None = None,
) -> ForecastSnapshot | None:
    score_series = compute_strategy_score_series(
        returns,
        volume_history,
        strategy_or_selection,
        bars_per_year=bars_per_year,
        current_date=current_date,
        predictor_panel=predictor_panel,
        selection_contexts=selection_contexts,
        predictor_context=predictor_context,
    )
    if score_series is None:
        return None

    aligned_scores = score_series.reindex(returns.columns).replace([np.inf, -np.inf], np.nan)
    if aligned_scores.isna().all():
        return None

    risk_proxy = returns.std(axis=0).reindex(returns.columns).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    signal_return_proxy = compute_signal_return_proxy_series(returns, aligned_scores)
    return ForecastSnapshot(
        as_of_date=current_date,
        horizon="strategy_default",
        score=aligned_scores,
        percentile_rank=aligned_scores.rank(method="average", pct=True),
        expected_return_proxy=signal_return_proxy,
        edge_source=None if signal_return_proxy is None else SIGNAL_RETURN_PROXY_EDGE_SOURCE,
        confidence=compute_forecast_confidence_series(aligned_scores),
        risk_proxy=risk_proxy,
    )


def evaluate_asset_ranking_spec(
    returns: pd.DataFrame,
    volumes: pd.DataFrame | None,
    split_ratio: float,
    ranking_spec: AssetRankingSpec,
    *,
    bars_per_year: float,
) -> dict:
    ranking_universe = [
        asset for asset in ranking_spec.investment_universe.tickers if asset in returns.columns
    ]
    returns = returns[ranking_universe]
    volumes = volumes[ranking_universe] if volumes is not None else None
    split_index = compute_split_index(len(returns), split_ratio)
    observations: list[dict] = []
    latest_top_assets: list[str] = []

    availability_policy = build_default_availability_policy()
    for index in range(2, len(returns)):
        raw_history_returns = returns.iloc[:index]
        eligible_assets = resolve_eligible_assets(raw_history_returns, availability_policy)
        if len(eligible_assets) < 2:
            continue
        history_returns = prepare_history_returns_for_assets(raw_history_returns, eligible_assets)
        if len(history_returns) < 3 or len(history_returns.columns) < 2:
            continue
        raw_history_volumes = volumes.iloc[:index] if volumes is not None else None
        history_volumes = prepare_volume_history_for_assets(raw_history_volumes, history_returns)
        selection = ranking_spec.selection
        selected_assets = select_assets(
            history_returns,
            history_volumes,
            selection,
            bars_per_year=bars_per_year,
        )
        if len(selected_assets) < 2:
            continue

        score_series = compute_strategy_score_series(
            history_returns,
            history_volumes,
            selection,
            bars_per_year=bars_per_year,
        )
        if score_series is None:
            continue

        selected_scores = score_series.loc[selected_assets].dropna()
        if len(selected_scores) < 2 or selected_scores.nunique() < 2:
            continue

        forward_returns = returns.iloc[index].loc[selected_scores.index].dropna()
        if len(forward_returns) < 2:
            continue

        aligned_scores = selected_scores.loc[forward_returns.index]
        if len(aligned_scores) < 2 or aligned_scores.nunique() < 2 or forward_returns.nunique() < 2:
            continue

        group_size = max(1, len(aligned_scores) // 3)
        ordered_scores = aligned_scores.sort_values(ascending=False)
        top_assets = list(ordered_scores.head(group_size).index)
        bottom_assets = list(ordered_scores.tail(group_size).index)
        top_return = float(forward_returns.loc[top_assets].mean())
        bottom_return = float(forward_returns.loc[bottom_assets].mean())
        rank_ic = aligned_scores.rank().corr(forward_returns.rank(), method="pearson")
        if pd.isna(rank_ic):
            continue

        latest_top_assets = top_assets
        observations.append(
            {
                "date": str(returns.index[index]),
                "rankIc": float(rank_ic),
                "topReturn": top_return,
                "bottomReturn": bottom_return,
                "topMinusBottom": top_return - bottom_return,
                "assetCount": len(aligned_scores),
                "segment": "train" if index < split_index else "test",
            }
        )

    return {
        "rankingSpec": serialize_asset_ranking_spec(ranking_spec),
        "latestTopAssets": latest_top_assets,
        "overall": summarize_ranking_observations(observations),
        "train": summarize_ranking_observations(
            [observation for observation in observations if observation["segment"] == "train"]
        ),
        "test": summarize_ranking_observations(
            [observation for observation in observations if observation["segment"] == "test"]
        ),
    }


def evaluate_predictor_spec(
    returns: pd.DataFrame,
    volumes: pd.DataFrame | None,
    split_ratio: float,
    predictor_spec: PredictorSpec,
    *,
    bars_per_year: float,
    predictor_panel: pd.DataFrame | None = None,
) -> dict:
    ranking_universe = [
        asset for asset in predictor_spec.signal_spec.observation_spec.tickers if asset in returns.columns
    ]
    returns = returns[ranking_universe]
    split_index = compute_split_index(len(returns), split_ratio)
    target_horizon_bars = convert_horizon_spec_to_bars(
        {key: value for key, value in predictor_spec.target_spec.horizon_spec},
        bars_per_year=bars_per_year,
    )
    observations: list[dict] = []
    latest_top_assets: list[str] = []
    latest_scores: list[dict[str, float | str]] = []
    if predictor_panel is None:
        predictor_panel = compute_predictor_panel(
            returns=returns,
            volumes=volumes[ranking_universe] if volumes is not None else None,
            predictor_spec=predictor_spec,
            bars_per_year=bars_per_year,
        )

    for index in range(2, len(returns) - target_horizon_bars + 1):
        prediction_values = predictor_panel.iloc[index].dropna()
        if len(prediction_values) < 2 or prediction_values.nunique() < 2:
            continue

        forward_window = returns.iloc[index : index + target_horizon_bars]
        if len(forward_window) < target_horizon_bars:
            continue
        forward_returns = ((1 + forward_window).prod() - 1).loc[prediction_values.index].dropna()
        if len(forward_returns) < 2:
            continue

        prediction_values = prediction_values.loc[forward_returns.index]
        if len(prediction_values) < 2 or prediction_values.nunique() < 2 or forward_returns.nunique() < 2:
            continue

        if predictor_spec.target_spec.baseline == "cross_sectional_mean":
            target_values = forward_returns - float(forward_returns.mean())
        else:
            raise ValueError("Unsupported prediction target baseline.")
        target_values = apply_prediction_target_transform(
            target_values,
            predictor_spec.target_spec,
        )

        if target_values.nunique() < 2:
            continue

        group_size = max(1, len(prediction_values) // 3)
        ordered_scores = prediction_values.sort_values(ascending=False)
        top_assets = list(ordered_scores.head(group_size).index)
        bottom_assets = list(ordered_scores.tail(group_size).index)
        top_return = float(forward_returns.loc[top_assets].mean())
        bottom_return = float(forward_returns.loc[bottom_assets].mean())
        rank_ic = prediction_values.rank().corr(target_values.rank(), method="pearson")
        pearson_corr = prediction_values.corr(target_values, method="pearson")
        if pd.isna(rank_ic) or pd.isna(pearson_corr):
            continue

        latest_top_assets = top_assets
        latest_scores = [
            {"asset": str(asset), "score": round(float(score), 6)}
            for asset, score in ordered_scores.head(5).items()
        ]
        observations.append(
            {
                "date": str(returns.index[index]),
                "rankIc": float(rank_ic),
                "pearsonCorr": float(pearson_corr),
                "topReturn": top_return,
                "bottomReturn": bottom_return,
                "topMinusBottom": top_return - bottom_return,
                "assetCount": len(prediction_values),
                "segment": "train" if index < split_index else "test",
            }
        )

    return {
        "predictorSpec": serialize_predictor_spec(predictor_spec),
        "predictorSeries": serialize_predictor_panel(predictor_panel),
        "latestTopAssets": latest_top_assets,
        "latestScores": latest_scores,
        "overall": summarize_prediction_observations(observations),
        "train": summarize_prediction_observations(
            [observation for observation in observations if observation["segment"] == "train"]
        ),
        "test": summarize_prediction_observations(
            [observation for observation in observations if observation["segment"] == "test"]
        ),
    }


def summarize_ranking_observations(observations: list[dict]) -> dict:
    if not observations:
        return {
            "observationCount": 0,
            "meanRankIc": None,
            "meanTopReturnPct": None,
            "meanBottomReturnPct": None,
            "meanTopMinusBottomPct": None,
            "hitRatePct": None,
            "meanAssetCount": None,
        }

    top_minus_bottom_values = [observation["topMinusBottom"] for observation in observations]
    hit_count = sum(1 for value in top_minus_bottom_values if value > 0)
    return {
        "observationCount": len(observations),
        "meanRankIc": round(float(np.mean([observation["rankIc"] for observation in observations])), 4),
        "meanTopReturnPct": round(float(np.mean([observation["topReturn"] for observation in observations])) * 100, 2),
        "meanBottomReturnPct": round(
            float(np.mean([observation["bottomReturn"] for observation in observations])) * 100,
            2,
        ),
        "meanTopMinusBottomPct": round(float(np.mean(top_minus_bottom_values)) * 100, 2),
        "hitRatePct": round(hit_count / len(observations) * 100, 2),
        "meanAssetCount": round(float(np.mean([observation["assetCount"] for observation in observations])), 2),
    }


def summarize_prediction_observations(observations: list[dict]) -> dict:
    if not observations:
        return {
            "observationCount": 0,
            "meanRankIc": None,
            "meanPearsonCorr": None,
            "meanTopReturnPct": None,
            "meanBottomReturnPct": None,
            "meanTopMinusBottomPct": None,
            "hitRatePct": None,
            "meanAssetCount": None,
        }

    top_minus_bottom_values = [observation["topMinusBottom"] for observation in observations]
    hit_count = sum(1 for value in top_minus_bottom_values if value > 0)
    return {
        "observationCount": len(observations),
        "meanRankIc": round(float(np.mean([observation["rankIc"] for observation in observations])), 4),
        "meanPearsonCorr": round(
            float(np.mean([observation["pearsonCorr"] for observation in observations])),
            4,
        ),
        "meanTopReturnPct": round(float(np.mean([observation["topReturn"] for observation in observations])) * 100, 2),
        "meanBottomReturnPct": round(
            float(np.mean([observation["bottomReturn"] for observation in observations])) * 100,
            2,
        ),
        "meanTopMinusBottomPct": round(float(np.mean(top_minus_bottom_values)) * 100, 2),
        "hitRatePct": round(hit_count / len(observations) * 100, 2),
        "meanAssetCount": round(float(np.mean([observation["assetCount"] for observation in observations])), 2),
    }


def standardize_prediction_series(series: pd.Series) -> pd.Series:
    centered = series - float(series.mean())
    std = float(series.std(ddof=0))
    if std <= 0:
        return pd.Series(0.0, index=series.index, dtype="float64")
    standardized = centered / std
    return standardized.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def compute_portfolio_allocation(
    history_returns: pd.DataFrame,
    volume_history: pd.DataFrame | None,
    strategy: EvaluatorStrategySpec,
    portfolio_model: PortfolioModelSpec,
    bars_per_year: float,
    universe_columns: pd.Index,
    max_investment_ratio: float,
    max_weight: float | None,
    previous_weights: np.ndarray | None,
    transaction_cost: float,
    current_date: str | None,
    predictor_panel: pd.DataFrame | None,
    selection_contexts: list[dict[str, object]] | None = None,
    predictor_context: dict[str, object] | None = None,
) -> tuple[list[str], np.ndarray]:
    signal_returns, signal_volumes, signal_bars_per_year = prepare_strategy_signal_data(
        history_returns=history_returns,
        volume_history=volume_history,
        strategy=strategy,
        selection_contexts=selection_contexts,
    )
    selected_assets = select_assets(
        signal_returns,
        signal_volumes,
        strategy,
        bars_per_year=signal_bars_per_year,
        selection_contexts=selection_contexts,
    )
    if not selected_assets:
        return [], np.zeros(len(universe_columns), dtype="float64")
    strategy_returns = filter_positive_variance_assets(signal_returns[selected_assets])
    selected_assets = list(strategy_returns.columns)
    if len(selected_assets) < 2:
        return [], np.zeros(len(universe_columns), dtype="float64")
    selected_previous_weights = None
    if previous_weights is not None:
        previous_weight_map = {
            str(asset): float(weight)
            for asset, weight in zip(universe_columns, previous_weights, strict=True)
        }
        selected_previous_weights = np.asarray(
            [previous_weight_map.get(asset, 0.0) for asset in selected_assets],
            dtype="float64",
        )
    prepared_predictor_panel = prepare_strategy_predictor_panel(
        predictor_panel=predictor_panel,
        strategy=strategy,
        predictor_context=predictor_context,
    )
    weights = fit_portfolio_model(
        PortfolioAllocationInput(
            returns=strategy_returns,
            portfolio_model=portfolio_model,
            max_investment_ratio=max_investment_ratio,
            max_weight=max_weight,
            previous_weights=selected_previous_weights,
            transaction_cost=transaction_cost,
        )
    ) * max_investment_ratio
    if portfolio_model.model_type != "mean_risk_utility":
        weights = apply_strategy_weight_tilt(
            weights=weights,
            history_returns=strategy_returns,
            strategy=strategy,
            bars_per_year=signal_bars_per_year,
            max_investment_ratio=max_investment_ratio,
            max_weight=max_weight,
            current_date=current_date,
            predictor_panel=prepared_predictor_panel,
            selection_contexts=selection_contexts,
            predictor_context=predictor_context,
        )
    expanded_weights = expand_weights(
        universe_columns=universe_columns,
        selected_columns=strategy_returns.columns,
        selected_weights=weights,
    )
    return selected_assets, expanded_weights


def apply_strategy_weight_tilt(
    *,
    weights: np.ndarray,
    history_returns: pd.DataFrame,
    strategy: EvaluatorStrategySpec,
    bars_per_year: float,
    max_investment_ratio: float,
    max_weight: float | None,
    current_date: str | None,
    predictor_panel: pd.DataFrame | None,
    selection_contexts: list[dict[str, object]] | None = None,
    predictor_context: dict[str, object] | None = None,
) -> np.ndarray:
    primary_selection = resolve_primary_selection_spec(
        strategy,
        selection_contexts=selection_contexts,
    )
    score_parameters = dict(primary_selection.ranking_signal.score_parameters)
    if "tilt_strength" not in score_parameters:
        return weights

    forecast = build_strategy_forecast_snapshot(
        history_returns,
        None,
        strategy,
        bars_per_year=bars_per_year,
        current_date=current_date,
        predictor_panel=predictor_panel,
        selection_contexts=selection_contexts,
        predictor_context=predictor_context,
    )
    if forecast is None:
        return weights
    percentile_ranks = forecast.percentile_rank.reindex(history_returns.columns)
    tilt_strength = float(score_parameters.get("tilt_strength", 0.5))
    tilt_shape = float(score_parameters.get("tilt_shape", 0.0))
    rank_values = percentile_ranks.to_numpy(dtype="float64")
    return apply_weight_tilt(
        weights=weights,
        rank_values=rank_values,
        tilt_strength=tilt_strength,
        tilt_shape=tilt_shape,
        max_weight=max_weight,
    )


def filter_positive_variance_assets(returns: pd.DataFrame) -> pd.DataFrame:
    positive_variance = returns.var(axis=0) > 1e-12
    filtered = returns.loc[:, positive_variance]
    if filtered.empty:
        raise ValueError("At least one asset with positive variance is required.")
    return filtered


def compute_expected_return_proxy(
    *,
    returns: pd.DataFrame,
    volume_history: pd.DataFrame | None,
    strategy: EvaluatorStrategySpec,
    bars_per_year: float,
    current_date: str | None,
    predictor_panel: pd.DataFrame | None,
    selection_contexts: list[dict[str, object]] | None = None,
    predictor_context: dict[str, object] | None = None,
) -> np.ndarray | None:
    forecast = build_strategy_forecast_snapshot(
        returns,
        volume_history,
        strategy,
        bars_per_year=bars_per_year,
        current_date=current_date,
        predictor_panel=predictor_panel,
        selection_contexts=selection_contexts,
        predictor_context=predictor_context,
    )
    if forecast is None or forecast.expected_return_proxy is None:
        return None
    return forecast.expected_return_proxy.reindex(returns.columns).to_numpy(dtype="float64")


def run_portfolio_backtest(
    closes: pd.DataFrame,
    returns: pd.DataFrame,
    volumes: pd.DataFrame | None,
    bars_per_year: float,
    split_index: int,
    split_ratio: float,
    strategy: EvaluatorStrategySpec,
    portfolio_model: PortfolioModelSpec,
    warmup_weights: np.ndarray,
    initial_weights: np.ndarray,
    initial_selected_assets: list[str],
    max_investment_ratio: float,
    initial_capital: float,
    transaction_cost: float,
    asset_transaction_costs: np.ndarray,
    asset_impact_costs: np.ndarray,
    adv_window_bars: int,
    min_adv_notional: float,
    max_weight: float | None,
    decision_schedule: str,
    rebalance_schedule: str,
    predictor_panel: pd.DataFrame | None,
    selection_contexts: list[dict[str, object]] | None = None,
    predictor_context: dict[str, object] | None = None,
    availability_policy: dict[str, object] | None = None,
) -> dict:
    if availability_policy is None:
        availability_policy = build_default_availability_policy()

    portfolio_equity = initial_capital
    portfolio_returns: list[float] = []
    series: list[dict] = []
    train_dates: list[str] = []
    test_dates: list[str] = []
    train_portfolio_returns: list[float] = []
    test_portfolio_returns: list[float] = []
    train_turnover = 0.0
    test_turnover = 0.0
    current_weights = warmup_weights.copy()
    current_selected_assets: list[str] = []
    latest_weights = current_weights.copy()
    latest_selected_assets = list(current_selected_assets)
    pending_decision_weights = initial_weights.copy()
    pending_decision_selected_assets = list(initial_selected_assets)
    next_rebalance_weights: np.ndarray | None = None
    next_rebalance_selected_assets: list[str] | None = None
    previous_eligible_assets: set[str] = set()
    decision_events: list[dict] = []
    execution_trace: list[dict] = []

    for index, (date, row) in enumerate(returns.iterrows()):
        trade_turnover = 0.0
        trade_cost = 0.0
        phase = "train" if index < split_index else "test"
        history_through_current = returns.iloc[: index + 1]
        available_assets = resolve_available_assets(history_through_current)
        eligible_assets = resolve_eligible_assets(history_through_current, availability_policy)
        eligible_asset_set = set(eligible_assets)
        newly_eligible_assets = sorted(eligible_asset_set - previous_eligible_assets)
        removed_assets = sorted(previous_eligible_assets - eligible_asset_set)

        tradable_weights = zero_weights_outside_assets(current_weights, returns.columns, eligible_assets)
        forced_weight_delta = np.abs(tradable_weights - current_weights)
        if forced_weight_delta.sum() > 0:
            forced_turnover = float(forced_weight_delta.sum())
            trade_turnover += forced_turnover
            forced_trade_cost = compute_trade_cost(
                weight_delta=forced_weight_delta,
                linear_cost_rates=asset_transaction_costs,
                impact_cost_rates=asset_impact_costs,
                portfolio_equity=portfolio_equity,
                price_snapshot=closes.iloc[max(index - 1, 0)],
                volume_history=volumes.iloc[:index] if volumes is not None else None,
                adv_window_bars=adv_window_bars,
                min_adv_notional=min_adv_notional,
            )
            trade_cost += forced_trade_cost
            current_weights = tradable_weights
            current_selected_assets = [asset for asset in current_selected_assets if asset in eligible_asset_set]
            execution_trace.append(
                serialize_execution_trace_event(
                    date=str(date),
                    event_type="forced_universe_change",
                    phase=phase,
                    universe_columns=returns.columns,
                    max_investment_ratio=max_investment_ratio,
                    available_asset_count=len(available_assets),
                    eligible_asset_count=len(eligible_assets),
                    selected_assets=current_selected_assets,
                    target_weights=tradable_weights,
                    executed_weights=current_weights,
                    decision_action="forced_rebalance",
                    decision_reason="asset_unavailable",
                    turnover_pct=forced_turnover * 100,
                    estimated_cost_pct=round(forced_trade_cost * 100, 4),
                    estimated_edge_pct=None,
                    edge_source=None,
                    average_confidence=None,
                )
            )

        if index == split_index:
            split_history_returns = returns.iloc[:index]
            split_history_volumes = volumes.iloc[:index] if volumes is not None else None
            split_decision = build_portfolio_decision(
                history_returns=split_history_returns,
                volume_history=split_history_volumes,
                strategy=strategy,
                bars_per_year=bars_per_year,
                universe_columns=returns.columns,
                current_weights=current_weights,
                current_selected_assets=current_selected_assets,
                target_weights=initial_weights,
                target_selected_assets=initial_selected_assets,
                transaction_cost=transaction_cost,
                current_date=str(date),
                predictor_panel=predictor_panel,
                selection_contexts=selection_contexts,
                predictor_context=predictor_context,
                availability_policy=availability_policy,
            )
            decision_events.append(
                serialize_portfolio_decision_event(
                    str(date),
                    split_decision,
                    universe_columns=returns.columns,
                    current_weights=current_weights,
                    target_weights=initial_weights,
                    realized_returns=row,
                )
            )
            execution_trace.append(
                serialize_execution_trace_event(
                    date=str(date),
                    event_type="decision",
                    phase=phase,
                    universe_columns=returns.columns,
                    max_investment_ratio=max_investment_ratio,
                    available_asset_count=len(available_assets),
                    eligible_asset_count=len(eligible_assets),
                    selected_assets=split_decision.selected_assets,
                    target_weights=initial_weights,
                    executed_weights=current_weights,
                    decision_action=split_decision.action,
                    decision_reason=split_decision.reason,
                    turnover_pct=split_decision.turnover * 100,
                    estimated_cost_pct=split_decision.estimated_cost_pct,
                    estimated_edge_pct=split_decision.estimated_edge_pct,
                    edge_source=split_decision.edge_source,
                    average_confidence=split_decision.average_confidence,
                )
            )
            next_rebalance_weights = split_decision.weights.copy()
            next_rebalance_selected_assets = list(split_decision.selected_assets)

        if index >= split_index and next_rebalance_weights is not None:
            rebalanced_weights = zero_weights_outside_assets(
                next_rebalance_weights,
                returns.columns,
                eligible_assets,
            )
            selected_assets = [asset for asset in (next_rebalance_selected_assets or []) if asset in eligible_asset_set]
            weight_delta = np.abs(rebalanced_weights - current_weights)
            rebalance_turnover = float(weight_delta.sum())
            trade_turnover += rebalance_turnover
            rebalance_trade_cost = compute_trade_cost(
                weight_delta=weight_delta,
                linear_cost_rates=asset_transaction_costs,
                impact_cost_rates=asset_impact_costs,
                portfolio_equity=portfolio_equity,
                price_snapshot=closes.iloc[max(index - 1, 0)],
                volume_history=volumes.iloc[:index] if volumes is not None else None,
                adv_window_bars=adv_window_bars,
                min_adv_notional=min_adv_notional,
            )
            trade_cost += rebalance_trade_cost
            current_weights = rebalanced_weights
            current_selected_assets = list(selected_assets)
            latest_weights = rebalanced_weights.copy()
            latest_selected_assets = list(current_selected_assets)
            execution_trace.append(
                serialize_execution_trace_event(
                    date=str(date),
                    event_type="rebalance",
                    phase=phase,
                    universe_columns=returns.columns,
                    max_investment_ratio=max_investment_ratio,
                    available_asset_count=len(available_assets),
                    eligible_asset_count=len(eligible_assets),
                    selected_assets=current_selected_assets,
                    target_weights=next_rebalance_weights,
                    executed_weights=current_weights,
                    decision_action="rebalance",
                    decision_reason="scheduled_rebalance",
                    turnover_pct=rebalance_turnover * 100,
                    estimated_cost_pct=round(rebalance_trade_cost * 100, 4),
                    estimated_edge_pct=None,
                    edge_source=None,
                    average_confidence=None,
                )
            )
            next_rebalance_weights = None
            next_rebalance_selected_assets = None

        row_returns = row.reindex(returns.columns).fillna(0.0).to_numpy(dtype="float64")
        portfolio_return = float(np.dot(row_returns, current_weights))
        portfolio_return -= trade_cost

        portfolio_equity *= 1 + portfolio_return
        portfolio_returns.append(portfolio_return)
        segment_dates = train_dates if index < split_index else test_dates
        segment_portfolio_returns = train_portfolio_returns if index < split_index else test_portfolio_returns
        segment_dates.append(str(date))
        segment_portfolio_returns.append(portfolio_return)
        if index < split_index:
            train_turnover += trade_turnover
        else:
            test_turnover += trade_turnover
        series.append(
            {
                "date": str(date),
                "portfolioEquity": round(portfolio_equity, 2),
                "portfolioReturnPct": round(portfolio_return * 100, 2),
                "availableAssetCount": len(available_assets),
                "eligibleAssetCount": len(eligible_assets),
                "newlyEligibleAssets": newly_eligible_assets,
                "removedAssets": removed_assets,
            }
        )
        previous_eligible_assets = eligible_asset_set

        if index <= split_index or index >= len(returns) - 1:
            continue

        previous_date = returns.index[index - 1]
        if should_rebalance(
            previous_date=previous_date,
            current_date=date,
            rebalance_schedule=decision_schedule,
        ):
            target_selected_assets, target_weights = compute_dynamic_portfolio_allocation(
                history_returns=returns.iloc[: index + 1],
                volume_history=volumes.iloc[: index + 1] if volumes is not None else None,
                strategy=strategy,
                portfolio_model=portfolio_model,
                bars_per_year=bars_per_year,
                universe_columns=returns.columns,
                max_investment_ratio=max_investment_ratio,
                max_weight=max_weight,
                previous_weights=current_weights,
                transaction_cost=transaction_cost,
                current_date=str(date),
                predictor_panel=predictor_panel,
                selection_contexts=selection_contexts,
                predictor_context=predictor_context,
                availability_policy=availability_policy,
            )
            portfolio_decision = build_portfolio_decision(
                history_returns=returns.iloc[: index + 1],
                volume_history=volumes.iloc[: index + 1] if volumes is not None else None,
                strategy=strategy,
                bars_per_year=bars_per_year,
                universe_columns=returns.columns,
                current_weights=current_weights,
                current_selected_assets=current_selected_assets,
                target_weights=target_weights,
                target_selected_assets=target_selected_assets,
                transaction_cost=transaction_cost,
                current_date=str(date),
                predictor_panel=predictor_panel,
                selection_contexts=selection_contexts,
                predictor_context=predictor_context,
                availability_policy=availability_policy,
            )
            decision_events.append(
                serialize_portfolio_decision_event(
                    str(date),
                    portfolio_decision,
                    universe_columns=returns.columns,
                    current_weights=current_weights,
                    target_weights=target_weights,
                    realized_returns=returns.iloc[index + 1] if index + 1 < len(returns) else None,
                )
            )
            execution_trace.append(
                serialize_execution_trace_event(
                    date=str(date),
                    event_type="decision",
                    phase=phase,
                    universe_columns=returns.columns,
                    max_investment_ratio=max_investment_ratio,
                    available_asset_count=len(available_assets),
                    eligible_asset_count=len(eligible_assets),
                    selected_assets=portfolio_decision.selected_assets,
                    target_weights=target_weights,
                    executed_weights=current_weights,
                    decision_action=portfolio_decision.action,
                    decision_reason=portfolio_decision.reason,
                    turnover_pct=portfolio_decision.turnover * 100,
                    estimated_cost_pct=portfolio_decision.estimated_cost_pct,
                    estimated_edge_pct=portfolio_decision.estimated_edge_pct,
                    edge_source=portfolio_decision.edge_source,
                    average_confidence=portfolio_decision.average_confidence,
                )
            )
            pending_decision_selected_assets = list(portfolio_decision.selected_assets)
            pending_decision_weights = portfolio_decision.weights.copy()
        if should_rebalance(
            previous_date=previous_date,
            current_date=date,
            rebalance_schedule=rebalance_schedule,
        ):
            next_rebalance_weights = pending_decision_weights.copy()
            next_rebalance_selected_assets = list(pending_decision_selected_assets)

    train_summary = summarize_segment_from_returns(
        dates=train_dates,
        portfolio_returns=train_portfolio_returns,
        bars_per_year=bars_per_year,
        turnover=train_turnover,
    )
    test_summary = summarize_segment_from_returns(
        dates=test_dates,
        portfolio_returns=test_portfolio_returns,
        bars_per_year=bars_per_year,
        turnover=test_turnover,
    )
    return {
        "summary": test_summary["portfolio"],
        "series": series,
        "latestWeights": latest_weights,
        "latestSelectedAssets": latest_selected_assets,
        "availabilitySummary": summarize_availability_series(series),
        "decisionSummary": summarize_portfolio_decision_events(decision_events),
        "decisionEvents": decision_events,
        "executionTrace": execution_trace,
        "splitAnalysis": {
            "config": {"splitRatioPct": round(split_ratio * 100, 1)},
            "train": train_summary,
            "test": test_summary,
        },
    }




def resolve_initial_weights(
    *,
    universe_columns: pd.Index,
    portfolio_state: PortfolioState | None,
) -> np.ndarray:
    if portfolio_state is None:
        return np.zeros(len(universe_columns), dtype="float64")
    return np.asarray(
        [portfolio_state.current_weights.get(str(asset), 0.0) for asset in universe_columns],
        dtype="float64",
    )
