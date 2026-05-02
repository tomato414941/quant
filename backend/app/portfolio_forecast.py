from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from app.instrument_registry import get_instrument
from app.portfolio_domain import *
from app.portfolio_market_data import (
    prepare_signal_component_data,
    prepare_strategy_market_data,
    resample_market_frame_to_timeframe,
    resample_returns_frame_to_timeframe,
    resolve_timeframe_bars_per_year,
)
from app.portfolio_metrics import compute_split_index, serialize_weights

from app.portfolio_availability import (
    build_default_availability_policy,
    prepare_history_returns_for_assets,
    prepare_volume_history_for_assets,
    resolve_eligible_assets,
)
from app.portfolio_selection import (
    compute_macro_proxy_rank,
    compute_strategy_selection_score_series,
    compute_strategy_selection_trailing_total_returns,
    compute_strategy_score_series_base,
    compute_trailing_total_returns,
    compute_volume_strength,
    convert_horizon_spec_to_bars,
    resolve_predictor_use_spec,
    resolve_predictor_snapshot,
    resolve_primary_selection_spec,
    select_assets,
    standardize_prediction_series,
)

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


def freeze_parameter_value(value: object) -> object:
    if isinstance(value, dict):
        return tuple((str(key), freeze_parameter_value(nested_value)) for key, nested_value in sorted(value.items()))
    if isinstance(value, list):
        return tuple(freeze_parameter_value(item) for item in value)
    return value


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
