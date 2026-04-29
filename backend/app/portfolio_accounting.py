from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


MISSING_RETURN_POLICY_REJECT_IF_HELD = "reject_if_held"
MISSING_RETURN_POLICY_ZERO = "zero"
SUPPORTED_MISSING_RETURN_POLICIES = {
    MISSING_RETURN_POLICY_REJECT_IF_HELD,
    MISSING_RETURN_POLICY_ZERO,
}


@dataclass(frozen=True)
class AccountingReturnResolution:
    portfolio_return: float
    resolved_returns: pd.Series
    events: tuple[dict[str, object], ...] = ()


def resolve_missing_return_policy(availability_policy: dict[str, object]) -> str:
    policy = str(availability_policy.get("missingReturnPolicy", MISSING_RETURN_POLICY_REJECT_IF_HELD))
    if policy not in SUPPORTED_MISSING_RETURN_POLICIES:
        raise ValueError(f"Unsupported missing return policy: {policy}")
    return policy


def resolve_accounting_returns(
    *,
    row: pd.Series,
    universe_columns: pd.Index,
    current_weights: np.ndarray,
    date: object,
    phase: str,
    missing_return_policy: str,
    held_weights: np.ndarray | None = None,
) -> AccountingReturnResolution:
    if missing_return_policy not in SUPPORTED_MISSING_RETURN_POLICIES:
        raise ValueError(f"Unsupported missing return policy: {missing_return_policy}")
    aligned_returns = row.reindex(universe_columns)
    missing_check_weights = current_weights if held_weights is None else held_weights
    missing_held_assets = [
        str(asset)
        for asset, asset_return, weight in zip(universe_columns, aligned_returns, missing_check_weights, strict=True)
        if pd.isna(asset_return) and abs(float(weight)) > 1e-12
    ]
    if missing_held_assets and missing_return_policy == MISSING_RETURN_POLICY_REJECT_IF_HELD:
        raise ValueError(
            "Missing return for held assets on "
            f"{date}: {', '.join(missing_held_assets)}. "
            "Set missingReturnPolicy='zero' only for explicitly accepted zero-fill research runs."
        )

    resolved_returns = aligned_returns.fillna(0.0)
    events: tuple[dict[str, object], ...] = ()
    if missing_held_assets:
        events = (
            {
                "date": str(date),
                "eventType": "missing_return_accounting",
                "phase": phase,
                "missingReturnPolicy": missing_return_policy,
                "affectedAssets": missing_held_assets,
                "resolution": "zero_fill",
            },
        )

    return AccountingReturnResolution(
        portfolio_return=float(np.dot(resolved_returns.to_numpy(dtype="float64"), current_weights)),
        resolved_returns=resolved_returns,
        events=events,
    )


def compute_row_portfolio_return(
    *,
    row: pd.Series,
    universe_columns: pd.Index,
    current_weights: np.ndarray,
    date: object,
    missing_return_policy: str,
    held_weights: np.ndarray | None = None,
) -> float:
    return resolve_accounting_returns(
        row=row,
        universe_columns=universe_columns,
        current_weights=current_weights,
        date=date,
        phase="unknown",
        missing_return_policy=missing_return_policy,
        held_weights=held_weights,
    ).portfolio_return
