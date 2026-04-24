from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from app.timeframe_models import DEFAULT_DAILY_TIMEFRAME, DEFAULT_MONTHLY_TIMEFRAME, DEFAULT_WEEKLY_TIMEFRAME

if TYPE_CHECKING:
    from app.portfolio_domain import EvaluatorStrategySpec


def resolve_timeframe_bars_per_year(timeframe_key: str) -> float:
    if timeframe_key == DEFAULT_DAILY_TIMEFRAME.key:
        return DEFAULT_DAILY_TIMEFRAME.bars_per_year
    if timeframe_key == DEFAULT_WEEKLY_TIMEFRAME.key:
        return DEFAULT_WEEKLY_TIMEFRAME.bars_per_year
    if timeframe_key == DEFAULT_MONTHLY_TIMEFRAME.key:
        return DEFAULT_MONTHLY_TIMEFRAME.bars_per_year
    raise ValueError(f"Unsupported timeframe key: {timeframe_key}")




def resample_market_frame_to_timeframe(
    frame: pd.DataFrame | None,
    *,
    target_timeframe_key: str,
    value_kind: str,
    alignment_method: str | None = None,
) -> pd.DataFrame | None:
    if frame is None or target_timeframe_key == DEFAULT_DAILY_TIMEFRAME.key:
        return frame

    resolved_alignment_method = alignment_method or "end_of_period"
    if resolved_alignment_method not in {"asof_last", "end_of_period", "calendar_resample"}:
        raise ValueError(f"Unsupported alignment method: {resolved_alignment_method}")

    if target_timeframe_key == DEFAULT_WEEKLY_TIMEFRAME.key:
        period_alias = "W-FRI"
    elif target_timeframe_key == DEFAULT_MONTHLY_TIMEFRAME.key:
        period_alias = "M"
    else:
        raise ValueError(f"Unsupported target timeframe for resampling: {target_timeframe_key}")

    normalized_frame = frame.copy()
    normalized_frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index), name=frame.index.name)

    if resolved_alignment_method == "end_of_period":
        period_index = normalized_frame.index.to_period(period_alias)
        if value_kind in {"close", "last"}:
            aggregated = normalized_frame.groupby(period_index).last()
        elif value_kind == "volume":
            aggregated = normalized_frame.groupby(period_index).sum(min_count=1)
        else:
            raise ValueError(f"Unsupported market frame kind: {value_kind}")
        last_dates = pd.Series(normalized_frame.index, index=period_index).groupby(level=0).last()
        aggregated.index = pd.DatetimeIndex(last_dates.to_list(), name=frame.index.name)
        return aggregated

    if value_kind in {"close", "last"}:
        aggregated = normalized_frame.resample(period_alias).last()
    elif value_kind == "volume":
        aggregated = normalized_frame.resample(period_alias).sum(min_count=1)
    else:
        raise ValueError(f"Unsupported market frame kind: {value_kind}")

    aggregated = aggregated.dropna(how="all")
    aggregated.index = pd.DatetimeIndex(aggregated.index, name=frame.index.name)
    return aggregated


def prepare_strategy_market_data(
    *,
    closes: pd.DataFrame,
    volumes: pd.DataFrame | None,
    strategy: EvaluatorStrategySpec,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    market_data_timeframe_key = (
        str(strategy.signal_execution_contexts[0]["dataTimeframe"])
        if strategy.signal_execution_contexts
        else strategy.timeframe.key
    )
    if market_data_timeframe_key == strategy.timeframe.key:
        return closes, volumes

    resampled_closes = resample_market_frame_to_timeframe(
        closes,
        target_timeframe_key=strategy.timeframe.key,
        value_kind="close",
    )
    resampled_volumes = resample_market_frame_to_timeframe(
        volumes,
        target_timeframe_key=strategy.timeframe.key,
        value_kind="volume",
    )
    if resampled_volumes is not None:
        resampled_volumes = resampled_volumes.reindex(resampled_closes.index)
    return resampled_closes, resampled_volumes


def resample_returns_frame_to_timeframe(
    returns: pd.DataFrame,
    *,
    target_timeframe_key: str,
    alignment_method: str | None = None,
) -> pd.DataFrame:
    if target_timeframe_key == DEFAULT_DAILY_TIMEFRAME.key:
        return returns

    cumulative = (1.0 + returns).cumprod()
    resampled_closes = resample_market_frame_to_timeframe(
        cumulative,
        target_timeframe_key=target_timeframe_key,
        value_kind="close",
        alignment_method=alignment_method,
    )
    return resampled_closes.pct_change(fill_method=None).dropna()


def prepare_signal_component_data(
    *,
    history_returns: pd.DataFrame,
    volume_history: pd.DataFrame | None,
    data_timeframe_key: str,
    signal_timeframe_key: str,
    alignment_policy: dict[str, object] | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, float]:
    signal_bars_per_year = resolve_timeframe_bars_per_year(signal_timeframe_key)
    if data_timeframe_key == signal_timeframe_key:
        return history_returns, volume_history, signal_bars_per_year

    alignment_method = None if alignment_policy is None else str(alignment_policy.get("method"))
    signal_returns = resample_returns_frame_to_timeframe(
        history_returns,
        target_timeframe_key=signal_timeframe_key,
        alignment_method=alignment_method,
    )
    signal_volumes = resample_market_frame_to_timeframe(
        volume_history,
        target_timeframe_key=signal_timeframe_key,
        value_kind="volume",
        alignment_method=alignment_method,
    )
    if signal_volumes is not None:
        signal_volumes = signal_volumes.reindex(signal_returns.index)
    return signal_returns, signal_volumes, signal_bars_per_year
