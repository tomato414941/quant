import numpy as np
import pandas as pd

from app.portfolio_allocation import (
    build_equal_weight_fallback,
    expand_weights,
    fit_portfolio_model,
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
    weights = fit_portfolio_model(
        returns,
        build_portfolio_model_spec("equal_weight"),
        max_investment_ratio=0.8,
        max_weight=0.2,
        previous_weights=None,
        transaction_cost=0.0,
        expected_return_proxy=None,
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
    weights = fit_portfolio_model(
        returns,
        build_portfolio_model_spec("equal_weight"),
        max_investment_ratio=1.0,
        max_weight=0.2,
        previous_weights=None,
        transaction_cost=0.0,
        expected_return_proxy=None,
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
    weights = fit_portfolio_model(
        returns,
        build_portfolio_model_spec("mean_risk_utility"),
        max_investment_ratio=1.0,
        max_weight=None,
        previous_weights=None,
        transaction_cost=0.0,
        expected_return_proxy=np.asarray([0.1, 0.2, 0.3]),
    )

    np.testing.assert_allclose(weights, np.asarray([1 / 3, 1 / 3, 1 / 3]))
