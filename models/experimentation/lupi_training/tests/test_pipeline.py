import numpy as np

from lupi.pipeline import _normalize, _points, _resample
from lupi.learning import _features


def test_points_handles_missing_joints():
    values, mask = _points({0: [1, 2, 3], 24: [4, 5, 6]}, 3)
    assert values.shape == (25, 3)
    assert mask.sum() == 2
    assert np.all(values[1] == 0)


def test_normalize_preserves_missing_zeros():
    sequence = np.ones((2, 25, 3), dtype=np.float32)
    mask = np.ones((2, 25), dtype=np.float32)
    mask[:, 4] = 0
    sequence[:, 4] = 0
    result = _normalize(sequence, mask)
    assert np.all(result[:, 4] == 0)
    assert np.isfinite(result).all()


def test_resample_has_requested_length():
    assert _resample(np.arange(10), 64).shape == (64,)


def test_classical_feature_dimensions_match_metadata():
    assert _features(np.zeros((64, 25, 4), np.float32)).shape == (200,)
    assert _features(np.zeros((64, 25, 3), np.float32)).shape == (150,)
