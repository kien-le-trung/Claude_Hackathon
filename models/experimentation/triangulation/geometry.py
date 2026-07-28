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
    if intrinsic.shape != (3, 3):
        raise ValueError("Wrong shape for intrinsic")
    if rotation.shape != (3, 3):
        raise ValueError("Wrong shape for rotation")
    if translation.shape != (3,):
        raise ValueError("Wrong shape for translation")
    if not all(np.isfinite(value).all() for value in (intrinsic, rotation, translation)):
        raise ValueError("There are non-finite values in input")
    translation_column = translation[:, None]
    extrinsic = np.concatenate([rotation, translation_column], axis=1)
    projection = intrinsic @ extrinsic
    return projection


def project_point(projection: np.ndarray, point_3d: np.ndarray) -> np.ndarray:
    """Project one Euclidean 3D point to pixel coordinates `(u, v)`."""
    if not projection.shape == (3,4):
        raise ValueError("Expect projection matrix of shape (3,4)")
    if not point_3d.shape == (3,):
        raise ValueError("Expect point of shape (3,)")
    # augment to homogeneous coordinates
    homogeneous_point = np.append(point_3d, 1.0)
    image_homogeneous = projection @ homogeneous_point
    if not abs(image_homogeneous[2]) > np.finfo(np.float64).eps:
        raise ValueError("projected point cannot be zero")
    norm_image_homogeneous = image_homogeneous[:2] / image_homogeneous[2]
    return norm_image_homogeneous


def triangulate_point_dlt(
    projections: Sequence[np.ndarray],
    observations: Sequence[np.ndarray],
) -> np.ndarray:
    """Triangulate one 3D point from corresponding 2D observations using DLT."""
    # Projection matrices of each camera
    projections = np.asarray(projections, dtype=np.float64)
    # Pixels of each point after being projected
    observations = np.asarray(observations, dtype=np.float64)

    if projections.ndim != 3 or projections.shape[1:] != (3, 4):
        raise ValueError("Unexpected dimensions for projections")
    if observations.ndim != 2 or observations.shape[1] != 2:
        raise ValueError("Unexpected dimensions for observations")
    if len(projections) != len(observations):
        raise ValueError("Different number of cameras for projections/ observations")
    if len(projections) < 2:
        raise ValueError("At least two camera observations are required")

    rows = []
    for projection, observation in zip(projections, observations):
        u, v = observation
        rows.append(u * projection[2] - projection[0])
        rows.append(v * projection[2] - projection[1])
    A = np.asarray(rows, dtype=np.float64)

    _, singular_values, vt = np.linalg.svd(A)
    point_homogeneous = vt[-1]

    scale = point_homogeneous[3]
    if abs(scale) <= np.finfo(np.float64).eps:
        raise ValueError("Triangulated point lies at infinity")
    point_3d = point_homogeneous[:3] / scale
    return point_3d


def reprojection_errors(
    point_3d: np.ndarray,
    projections: Sequence[np.ndarray],
    observations: Sequence[np.ndarray],
) -> np.ndarray:
    """Return one Euclidean pixel error per camera."""
    point_3d = np.asarray(point_3d, dtype=np.float64)
    projections = np.asarray(projections, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)

    if point_3d.shape != (3,):
        raise ValueError("Unexpected shape for 3D point")
    if projections[0,:,:].shape != (3, 4):
        raise ValueError("Unexpected shape for projection matrices")
    if observations[0,:].shape != (2,):
        raise ValueError("Unexpected shape for observations")

    projected = np.asarray([
        project_point(projection, point_3d)
        for projection in projections
    ])

    residuals = projected - observations
    errors = np.linalg.norm(residuals, axis=1)
    return errors


def camera_depth(
    point_3d: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> float:
    point_camera = rotation @ point_3d + translation
    return float(point_camera[2])

