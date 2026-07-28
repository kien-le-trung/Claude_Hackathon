"""Core projective geometry exercises.

Implement these functions in lesson order. Keep them small and use the
synthetic checks before trying EC3D.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def projection_matrix(
    intrinsic: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Return the 3x4 camera projection matrix `K [R | t]`."""
    raise NotImplementedError("Lesson 2: implement K [R | t]")


def project_point(projection: np.ndarray, point_3d: np.ndarray) -> np.ndarray:
    """Project one Euclidean 3D point to pixel coordinates `(u, v)`."""
    raise NotImplementedError("Lesson 2: implement homogeneous projection")


def triangulate_point_dlt(
    projections: Sequence[np.ndarray],
    observations: Sequence[np.ndarray],
) -> np.ndarray:
    """Triangulate one 3D point from corresponding 2D observations using DLT."""
    raise NotImplementedError("Lesson 3: construct A and solve it with SVD")


def reprojection_errors(
    point_3d: np.ndarray,
    projections: Sequence[np.ndarray],
    observations: Sequence[np.ndarray],
) -> np.ndarray:
    """Return one Euclidean pixel error per camera."""
    raise NotImplementedError("Lesson 4: project the estimate back into each view")


def camera_depth(
    point_3d: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> float:
    """Return the point's Z coordinate in a camera coordinate system."""
    raise NotImplementedError("Lesson 4: transform world point with R and t")

