from __future__ import annotations

import math
import statistics

import numpy as np
import pandas as pd

from app.portfolio_domain import cagr, max_drawdown, percent_return, sharpe_ratio, validate_split_ratio


def empty_number_distribution() -> dict[str, object]:
    return {
        "count": 0,
        "minimum": None,
        "median": None,
        "maximum": None,
    }


def summarize_optional_number_distribution(values: list[object]) -> dict[str, object]:
    numeric_values = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    if not numeric_values:
        return empty_number_distribution()
    return {
        "count": len(numeric_values),
        "minimum": round(min(numeric_values), 4),
        "median": round(float(statistics.median(numeric_values)), 4),
        "maximum": round(max(numeric_values), 4),
    }


def compute_decision_event_edge_after_cost_values(events: list[dict]) -> list[float]:
    spreads = []
    for event in events:
        edge = event.get("estimatedEdgePct")
        cost = event.get("estimatedCostPct")
        if edge is None or cost is None:
            continue
        spreads.append(float(edge) - float(cost))
    return spreads


def compute_estimated_realized_edge_correlation(events: list[dict]) -> float | None:
    pairs = [
        (float(event["estimatedEdgePct"]), float(event["realizedEdgePct"]))
        for event in events
        if event.get("estimatedEdgePct") is not None and event.get("realizedEdgePct") is not None
    ]
    if len(pairs) < 2:
        return None
    estimated_values = np.asarray([pair[0] for pair in pairs], dtype="float64")
    realized_values = np.asarray([pair[1] for pair in pairs], dtype="float64")
    if float(np.std(estimated_values)) == 0.0 or float(np.std(realized_values)) == 0.0:
        return None
    correlation = float(np.corrcoef(estimated_values, realized_values)[0, 1])
    if not math.isfinite(correlation):
        return None
    return round(correlation, 4)


def summarize_portfolio_decision_events(events: list[dict]) -> dict[str, object]:
    if not events:
        return {
            "decisionCount": 0,
            "rebalanceCount": 0,
            "noTradeCount": 0,
            "policyCounts": {},
            "reasonCounts": {},
            "edgeSourceCounts": {},
            "averageTurnoverPct": None,
            "averageEstimatedCostPct": None,
            "averageEstimatedEdgePct": None,
            "averageRealizedEdgePct": None,
            "averageRealizedEdgeAfterCostPct": None,
            "averageConfidence": None,
            "edgeHitCount": 0,
            "edgeHitSampleCount": 0,
            "edgeHitRate": None,
            "estimatedVsRealizedEdgeCorrelation": None,
            "estimatedEdgePctDistribution": empty_number_distribution(),
            "estimatedCostPctDistribution": empty_number_distribution(),
            "estimatedEdgeAfterCostPctDistribution": empty_number_distribution(),
            "realizedEdgePctDistribution": empty_number_distribution(),
            "realizedEdgeAfterCostPctDistribution": empty_number_distribution(),
            "confidenceDistribution": empty_number_distribution(),
        }

    policy_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    edge_source_counts: dict[str, int] = {}
    for event in events:
        policy = str(event["policy"])
        reason = str(event["reason"])
        edge_source = event.get("edgeSource")
        policy_counts[policy] = policy_counts.get(policy, 0) + 1
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if edge_source is not None:
            edge_source_counts[str(edge_source)] = edge_source_counts.get(str(edge_source), 0) + 1

    edge_values = [float(event["estimatedEdgePct"]) for event in events if event["estimatedEdgePct"] is not None]
    realized_edge_values = [
        float(event["realizedEdgePct"]) for event in events if event.get("realizedEdgePct") is not None
    ]
    realized_edge_after_cost_values = [
        float(event["realizedEdgeAfterCostPct"])
        for event in events
        if event.get("realizedEdgeAfterCostPct") is not None
    ]
    edge_hit_values = [bool(event["edgeHit"]) for event in events if event.get("edgeHit") is not None]
    confidence_values = [float(event["averageConfidence"]) for event in events if event["averageConfidence"] is not None]
    cost_values = [event.get("estimatedCostPct") for event in events]
    spread_values = compute_decision_event_edge_after_cost_values(events)
    edge_hit_count = sum(1 for value in edge_hit_values if value)
    edge_hit_sample_count = len(edge_hit_values)
    return {
        "decisionCount": len(events),
        "rebalanceCount": sum(1 for event in events if event["action"] == "rebalance"),
        "noTradeCount": sum(1 for event in events if event["action"] == "no_trade"),
        "policyCounts": policy_counts,
        "reasonCounts": reason_counts,
        "edgeSourceCounts": edge_source_counts,
        "averageTurnoverPct": round(float(np.mean([event["turnoverPct"] for event in events])), 2),
        "averageEstimatedCostPct": round(float(np.mean([event["estimatedCostPct"] for event in events])), 4),
        "averageEstimatedEdgePct": None if not edge_values else round(float(np.mean(edge_values)), 4),
        "averageRealizedEdgePct": None if not realized_edge_values else round(float(np.mean(realized_edge_values)), 4),
        "averageRealizedEdgeAfterCostPct": (
            None if not realized_edge_after_cost_values else round(float(np.mean(realized_edge_after_cost_values)), 4)
        ),
        "averageConfidence": None if not confidence_values else round(float(np.mean(confidence_values)), 4),
        "edgeHitCount": edge_hit_count,
        "edgeHitSampleCount": edge_hit_sample_count,
        "edgeHitRate": None if edge_hit_sample_count == 0 else round(edge_hit_count / edge_hit_sample_count, 4),
        "estimatedVsRealizedEdgeCorrelation": compute_estimated_realized_edge_correlation(events),
        "estimatedEdgePctDistribution": summarize_optional_number_distribution(edge_values),
        "estimatedCostPctDistribution": summarize_optional_number_distribution(cost_values),
        "estimatedEdgeAfterCostPctDistribution": summarize_optional_number_distribution(spread_values),
        "realizedEdgePctDistribution": summarize_optional_number_distribution(realized_edge_values),
        "realizedEdgeAfterCostPctDistribution": summarize_optional_number_distribution(realized_edge_after_cost_values),
        "confidenceDistribution": summarize_optional_number_distribution(confidence_values),
    }

def summarize_availability_series(series: list[dict]) -> dict[str, object]:
    if not series:
        return {
            "minAvailableAssetCount": 0,
            "maxAvailableAssetCount": 0,
            "minEligibleAssetCount": 0,
            "maxEligibleAssetCount": 0,
            "newlyEligibleAssetCount": 0,
            "removedAssetCount": 0,
        }
    available_counts = [int(point.get("availableAssetCount", 0)) for point in series]
    eligible_counts = [int(point.get("eligibleAssetCount", 0)) for point in series]
    return {
        "minAvailableAssetCount": min(available_counts),
        "maxAvailableAssetCount": max(available_counts),
        "minEligibleAssetCount": min(eligible_counts),
        "maxEligibleAssetCount": max(eligible_counts),
        "newlyEligibleAssetCount": sum(len(point.get("newlyEligibleAssets", [])) for point in series),
        "removedAssetCount": sum(len(point.get("removedAssets", [])) for point in series),
    }

def summarize_portfolio_metrics(
    final_value: float,
    initial_value: float,
    periods: int,
    returns: list[float],
    bars_per_year: float,
    series: list[dict],
    equity_key: str,
    turnover: float,
) -> dict:
    return {
        "totalReturnPct": round(percent_return(final_value, initial_value), 2),
        "cagrPct": round(cagr(final_value, initial_value, periods, bars_per_year), 2),
        "sharpeRatio": round(sharpe_ratio(returns, bars_per_year), 2),
        "maxDrawdownPct": round(max_drawdown(series, equity_key), 2),
        "turnoverPct": round(turnover, 2),
    }


def summarize_segment_from_returns(
    dates: list[str],
    portfolio_returns: list[float],
    bars_per_year: float,
    turnover: float,
) -> dict:
    return {
        "startDate": dates[0],
        "endDate": dates[-1],
        "barCount": len(dates),
        "portfolio": summarize_metrics_from_bar_returns(
            dates=dates,
            bar_returns=portfolio_returns,
            bars_per_year=bars_per_year,
            equity_key="portfolioEquity",
            turnover=turnover,
        ),
    }


def serialize_weights(columns: pd.Index, weights: np.ndarray, max_investment_ratio: float) -> list[dict]:
    weight_map = [
        {"asset": str(asset), "weightPct": round(float(weight) * 100, 2)}
        for asset, weight in zip(columns, weights, strict=True)
    ]
    invested_weight = float(np.sum(weights))
    cash_weight_pct = round(max(0.0, 1 - invested_weight) * 100, 2)
    if cash_weight_pct > 0:
        weight_map.append({"asset": "CASH", "weightPct": cash_weight_pct})
    weight_map.sort(key=lambda item: item["weightPct"], reverse=True)
    return weight_map


def compute_split_index(length: int, split_ratio: float) -> int:
    validate_split_ratio(split_ratio)
    split_index = int(length * split_ratio)
    split_index = min(max(split_index, 3), length - 3)
    return split_index


def summarize_metrics_from_bar_returns(
    dates: list[str],
    bar_returns: list[float],
    bars_per_year: float,
    equity_key: str,
    turnover: float,
) -> dict:
    equity = 100.0
    series = []
    for date, bar_return in zip(dates, bar_returns, strict=True):
        equity *= 1 + bar_return
        series.append({"date": date, equity_key: round(equity, 2)})
    return summarize_portfolio_metrics(
        final_value=equity,
        initial_value=100.0,
        periods=len(bar_returns),
        returns=bar_returns,
        bars_per_year=bars_per_year,
        series=series,
        equity_key=equity_key,
        turnover=round(turnover * 100, 2),
    )


def should_rebalance(previous_date, current_date, rebalance_schedule: str) -> bool:
    if rebalance_schedule == "hold":
        return False

    previous_timestamp = pd.Timestamp(previous_date)
    current_timestamp = pd.Timestamp(current_date)
    if rebalance_schedule == "every_bar":
        return previous_timestamp.normalize() != current_timestamp.normalize()
    if rebalance_schedule == "month_end":
        return previous_timestamp.month != current_timestamp.month or previous_timestamp.year != current_timestamp.year
    if rebalance_schedule == "quarter_end":
        return previous_timestamp.quarter != current_timestamp.quarter or previous_timestamp.year != current_timestamp.year
    if rebalance_schedule == "year_end":
        return previous_timestamp.year != current_timestamp.year
    raise ValueError("Unsupported rebalance schedule.")

