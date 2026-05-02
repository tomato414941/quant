from unittest.mock import patch

import numpy as np
import pandas as pd

from app.portfolio_allocation import (
    ForecastAllocationInput,
    PortfolioAllocationInput,
    build_equal_weight_fallback,
    compute_portfolio_allocation_result,
    compute_portfolio_weights,
    expand_weights,
)
from app.portfolio_domain import build_portfolio_model_spec


def test_expand_weights_maps_selected_assets_to_full_universe() -> None:
    weights = expand_weights(
        pd.Index(["A", "B", "C", "D"]),
        pd.Index(["D", "B"]),
        np.asarray([0.3, 0.7]),
    )

    np.testing.assert_allclose(weights, np.asarray([0.0, 0.7, 0.0, 0.3]))


def test_build_equal_weight_fallback_caps_without_renormalizing_when_cap_binds() -> None:
    weights = build_equal_weight_fallback(asset_count=3, raw_max_weight=0.2)

    np.testing.assert_allclose(weights, np.asarray([0.2, 0.2, 0.2]))
    np.testing.assert_allclose(weights.sum(), 0.6)


def test_fit_equal_weight_model_respects_max_weight_after_investment_scaling() -> None:
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, -0.01],
            "B": [0.02, 0.01, 0.00],
            "C": [0.00, 0.01, 0.02],
            "D": [-0.01, 0.00, 0.01],
        }
    )
    weights = compute_portfolio_weights(
        PortfolioAllocationInput(
            returns=returns,
            portfolio_model=build_portfolio_model_spec("equal_weight"),
            max_investment_ratio=0.8,
            max_weight=0.2,
            previous_weights=None,
            transaction_cost=0.0,
        )
    )

    np.testing.assert_allclose(weights, np.asarray([0.25, 0.25, 0.25, 0.25]))
    np.testing.assert_allclose(weights * 0.8, np.asarray([0.2, 0.2, 0.2, 0.2]))


def test_fit_equal_weight_model_preserves_cash_when_cap_prevents_full_investment() -> None:
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, -0.01],
            "B": [0.02, 0.01, 0.00],
            "C": [0.00, 0.01, 0.02],
        }
    )
    weights = compute_portfolio_weights(
        PortfolioAllocationInput(
            returns=returns,
            portfolio_model=build_portfolio_model_spec("equal_weight"),
            max_investment_ratio=1.0,
            max_weight=0.2,
            previous_weights=None,
            transaction_cost=0.0,
        )
    )

    np.testing.assert_allclose(weights, np.asarray([0.2, 0.2, 0.2]))
    np.testing.assert_allclose(weights.sum(), 0.6)


def test_fit_mean_risk_utility_falls_back_to_equal_weight() -> None:
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, -0.01],
            "B": [0.02, 0.01, 0.00],
            "C": [0.00, 0.01, 0.02],
        }
    )
    forecast_input = ForecastAllocationInput(
        expected_returns=np.asarray([0.1, 0.2, 0.3]),
        confidence=None,
        risk_proxy=None,
        transaction_cost=0.0,
    )
    weights = compute_portfolio_weights(
        PortfolioAllocationInput(
            returns=returns,
            portfolio_model=build_portfolio_model_spec("mean_risk_utility"),
            max_investment_ratio=1.0,
            max_weight=None,
            previous_weights=None,
            transaction_cost=forecast_input.transaction_cost,
        )
    )

    np.testing.assert_allclose(weights, np.asarray([1 / 3, 1 / 3, 1 / 3]))


def test_compute_portfolio_allocation_result_reports_uncalibrated_forecast_fallback_metadata() -> None:
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, -0.01],
            "B": [0.02, 0.01, 0.00],
            "C": [0.00, 0.01, 0.02],
        }
    )

    result = compute_portfolio_allocation_result(
        PortfolioAllocationInput(
            returns=returns,
            portfolio_model=build_portfolio_model_spec("mean_risk_utility"),
            max_investment_ratio=1.0,
            max_weight=None,
            previous_weights=None,
            transaction_cost=0.0,
        )
    )

    np.testing.assert_allclose(result.weights, np.asarray([1 / 3, 1 / 3, 1 / 3]))
    assert result.fallback_metadata == {
        "modelType": "mean_risk_utility",
        "reason": "uncalibrated_forecast",
        "fallback": "equal_weight",
    }


def test_compute_portfolio_allocation_result_reports_optimizer_exception_fallback_metadata() -> None:
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, -0.01, 0.01],
            "B": [0.02, 0.01, 0.00, -0.01],
            "C": [0.00, 0.01, 0.02, 0.01],
        }
    )

    with patch("app.portfolio_allocation.RiskBudgeting.fit", side_effect=RuntimeError("boom")):
        result = compute_portfolio_allocation_result(
            PortfolioAllocationInput(
                returns=returns,
                portfolio_model=build_portfolio_model_spec("risk_budgeting"),
                max_investment_ratio=1.0,
                max_weight=None,
                previous_weights=None,
                transaction_cost=0.0,
            )
        )

    np.testing.assert_allclose(result.weights, np.asarray([1 / 3, 1 / 3, 1 / 3]))
    assert result.fallback_metadata == {
        "modelType": "risk_budgeting",
        "reason": "optimizer_exception",
        "fallback": "equal_weight",
        "exceptionType": "RuntimeError",
        "message": "boom",
    }
