from __future__ import annotations

import numpy as np


def build_rank_tilt_vector(
    rank_values: np.ndarray,
    *,
    tilt_strength: float,
    tilt_shape: float,
) -> np.ndarray:
    if tilt_shape == 1.0:
        tilt = np.where(rank_values >= 0.75, 1 + tilt_strength, 1.0)
    elif tilt_shape == 2.0:
        centered = rank_values - rank_values.mean()
        scaled = np.exp(np.clip(centered * tilt_strength * 2.0, -2.0, 2.0))
        tilt = scaled / scaled.mean()
    else:
        tilt = 1 + tilt_strength * (rank_values - 0.5)
    return np.clip(tilt, 0.25, None)


def apply_weight_tilt(
    *,
    weights: np.ndarray,
    rank_values: np.ndarray,
    tilt_strength: float,
    tilt_shape: float,
    max_weight: float | None,
) -> np.ndarray:
    tilt = build_rank_tilt_vector(
        rank_values,
        tilt_strength=tilt_strength,
        tilt_shape=tilt_shape,
    )
    tilted_weights = weights * tilt
    tilted_weights_sum = tilted_weights.sum()
    if tilted_weights_sum <= 0:
        return weights
    tilted_weights = tilted_weights / tilted_weights_sum * weights.sum()
    if max_weight is not None:
        tilted_weights = np.minimum(tilted_weights, max_weight)
    return tilted_weights
