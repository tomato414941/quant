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
from app.portfolio_metrics import serialize_weights


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


def standardize_prediction_series(series: pd.Series) -> pd.Series:
    centered = series - float(series.mean())
    std = float(series.std(ddof=0))
    if std <= 0:
        return pd.Series(0.0, index=series.index, dtype="float64")
    standardized = centered / std
    return standardized.replace([np.inf, -np.inf], np.nan).fillna(0.0)


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
