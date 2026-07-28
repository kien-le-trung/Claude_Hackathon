"""A tiny calibrated scene for learning without loading EC3D."""

from __future__ import annotations

import numpy as np


INTRINSIC = np.array([
    [800.0, 0.0, 320.0],
    [0.0, 800.0, 240.0],
    [0.0, 0.0, 1.0],
])

ROTATION_1 = np.eye(3)
TRANSLATION_1 = np.zeros(3)

# Camera 2 has its center one metre to the right. For world-to-camera
# extrinsics X_camera = R X_world + t, its translation is [-1, 0, 0].
ROTATION_2 = np.eye(3)
TRANSLATION_2 = np.array([-1.0, 0.0, 0.0])

WORLD_POINT = np.array([0.25, 0.10, 4.0])

