import sys
from pathlib import Path

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from evaluate import (  # noqa: E402
    bone_length_variation,
    mean_squared_jerk,
    mpjpe,
    pa_mpjpe,
    percentile_joint_error,
)


def test_position_error_metrics_use_euclidean_distance():
    target = np.zeros((2, 1, 3))
    predicted = np.asarray([[[3.0, 4.0, 0.0]], [[0.0, 0.0, 0.0]]])

    assert mpjpe(predicted, target) == 2.5
    assert percentile_joint_error(predicted, target, 100.0) == 5.0


def test_sequence_procrustes_removes_similarity_transform():
    rng = np.random.default_rng(4)
    target = rng.normal(size=(12, 5, 3))
    predicted = target * 2.5 + np.asarray([3.0, -2.0, 1.0])

    error, _ = pa_mpjpe(predicted, target)

    assert error < 1e-10


def test_jerk_is_zero_for_quadratic_motion():
    time = np.arange(12, dtype=np.float64)
    points = np.zeros((12, 2, 3))
    points[..., 0] = (time**2)[:, None]

    assert mean_squared_jerk(points, fps=30.0) < 1e-12


def test_bone_variation_is_zero_for_rigid_bone():
    names = ["parent", "child"]
    points = np.zeros((10, 2, 3))
    points[:, 1, 0] = 0.5

    result = bone_length_variation(
        points,
        names,
        bones=(("parent", "child"),),
    )

    assert result["mean_coefficient_of_variation"] == 0.0
