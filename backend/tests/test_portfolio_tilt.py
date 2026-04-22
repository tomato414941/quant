import numpy as np

from app.portfolio_tilt import apply_weight_tilt, build_rank_tilt_vector


def test_build_rank_tilt_vector_uses_linear_shape_by_default() -> None:
    rank_values = np.asarray([0.0, 0.5, 1.0])

    tilt = build_rank_tilt_vector(rank_values, tilt_strength=0.5, tilt_shape=0.0)

    np.testing.assert_allclose(tilt, np.asarray([0.75, 1.0, 1.25]))


def test_build_rank_tilt_vector_boosts_top_quartile_shape() -> None:
    rank_values = np.asarray([0.20, 0.74, 0.75, 0.90])

    tilt = build_rank_tilt_vector(rank_values, tilt_strength=0.4, tilt_shape=1.0)

    np.testing.assert_allclose(tilt, np.asarray([1.0, 1.0, 1.4, 1.4]))


def test_build_rank_tilt_vector_normalizes_exponential_shape() -> None:
    rank_values = np.asarray([0.0, 0.5, 1.0])

    tilt = build_rank_tilt_vector(rank_values, tilt_strength=0.5, tilt_shape=2.0)

    assert np.all(tilt > 0)
    np.testing.assert_allclose(tilt.mean(), 1.0)
    assert tilt[0] < tilt[1] < tilt[2]


def test_apply_weight_tilt_preserves_weight_sum() -> None:
    weights = np.asarray([0.2, 0.3, 0.5])
    rank_values = np.asarray([0.1, 0.5, 0.9])

    tilted_weights = apply_weight_tilt(
        weights=weights,
        rank_values=rank_values,
        tilt_strength=0.5,
        tilt_shape=0.0,
        max_weight=None,
    )

    np.testing.assert_allclose(tilted_weights.sum(), weights.sum())
    assert tilted_weights[0] < weights[0]
    assert tilted_weights[2] > weights[2]


def test_apply_weight_tilt_respects_max_weight_without_renormalizing() -> None:
    weights = np.asarray([0.2, 0.3, 0.5])
    rank_values = np.asarray([0.1, 0.5, 0.9])

    tilted_weights = apply_weight_tilt(
        weights=weights,
        rank_values=rank_values,
        tilt_strength=1.0,
        tilt_shape=0.0,
        max_weight=0.4,
    )

    assert tilted_weights.max() <= 0.4
    assert tilted_weights.sum() < weights.sum()


def test_apply_weight_tilt_returns_original_weights_when_tilted_sum_is_invalid() -> None:
    weights = np.asarray([0.0, 0.0, 0.0])
    rank_values = np.asarray([0.1, 0.5, 0.9])

    tilted_weights = apply_weight_tilt(
        weights=weights,
        rank_values=rank_values,
        tilt_strength=0.5,
        tilt_shape=0.0,
        max_weight=None,
    )

    np.testing.assert_allclose(tilted_weights, weights)
