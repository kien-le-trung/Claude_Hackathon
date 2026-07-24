from copy import deepcopy

import numpy as np
import pytest

from app.analysis import LANDMARK_NAMES
from app.features import (
    BIOMECHANICAL_FEATURE_NAMES,
    LandmarkFeatureError,
    biomechanical_measurements,
    feature_count,
    transform_pose,
)


def pose():
    landmarks = [
        {"index": index, "name": name, "x": index / 10, "y": index / 20,
         "z": index / 30, "visibility": 0.9, "presence": 0.8}
        for index, name in enumerate(LANDMARK_NAMES)
    ]
    return {"landmarks": landmarks, "world_landmarks": deepcopy(landmarks)}


def test_feature_shape_and_determinism():
    first = transform_pose(pose(), "combined")
    second = transform_pose(pose(), "combined")
    assert first.shape == (feature_count("combined"),)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    biomechanical = transform_pose(pose(), "biomechanical")
    assert biomechanical.shape == (feature_count("biomechanical"),)
    assert feature_count("biomechanical") == 2 * len(BIOMECHANICAL_FEATURE_NAMES)
    assert feature_count("biomechanical") == 26
    assert np.isfinite(biomechanical).all()


def test_zero_torso_is_encoded_as_missing():
    sample = pose()
    for index in (11, 12, 23, 24):
        for axis in ("x", "y", "z"):
            sample["landmarks"][index][axis] = 0
    assert np.isfinite(transform_pose(sample, "normalized")).all()


def test_landmark_order_is_validated():
    sample = pose()
    sample["landmarks"][0]["name"] = "WRONG"
    with pytest.raises(LandmarkFeatureError):
        transform_pose(sample, "normalized")


def test_low_visibility_invalidates_image_measurements_but_encoding_stays_finite():
    sample = pose()
    sample["landmarks"][11]["visibility"] = 0.1
    measurements = biomechanical_measurements(sample)
    assert np.isnan(measurements["torso_lean_image"])
    assert np.isfinite(transform_pose(sample, "biomechanical")).all()


def test_world_foot_pitch_is_signed_relative_to_horizontal_plane():
    sample = pose()
    for index, coordinates in {
        29: (0.0, 0.0, 0.0),
        31: (1.0, 1.0, 0.0),
        30: (0.0, 0.0, 0.0),
        32: (1.0, -1.0, 0.0),
    }.items():
        for axis, value in zip(("x", "y", "z"), coordinates):
            sample["world_landmarks"][index][axis] = value
    measurements = biomechanical_measurements(sample)
    assert measurements["left_foot_pitch_world"] == pytest.approx(45.0)
    assert measurements["right_foot_pitch_world"] == pytest.approx(-45.0)


def test_asymmetries_are_absolute_bilateral_angle_differences():
    measurements = biomechanical_measurements(pose())
    assert measurements["knee_angle_asymmetry"] == pytest.approx(
        abs(measurements["left_knee_angle_world"] - measurements["right_knee_angle_world"])
    )
    assert measurements["hip_angle_asymmetry"] == pytest.approx(
        abs(measurements["left_hip_angle_world"] - measurements["right_hip_angle_world"])
    )
    assert measurements["ankle_angle_asymmetry"] == pytest.approx(
        abs(measurements["left_ankle_angle_world"] - measurements["right_ankle_angle_world"])
    )
