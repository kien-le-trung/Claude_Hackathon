"""Incremental checks for the synthetic triangulation lessons."""

from __future__ import annotations

import numpy as np

from geometry import (
    camera_depth,
    project_point,
    projection_matrix,
    reprojection_errors,
    triangulate_point_dlt,
)
from synthetic_scene import (
    INTRINSIC,
    ROTATION_1,
    ROTATION_2,
    TRANSLATION_1,
    TRANSLATION_2,
    WORLD_POINT,
)


def check(name, callback) -> bool:
    try:
        callback()
    except NotImplementedError as exc:
        print(f"TODO  {name}: {exc}")
        return False
    except Exception as exc:
        print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
        return False
    print(f"PASS  {name}")
    return True


def projection_check() -> None:
    first = projection_matrix(INTRINSIC, ROTATION_1, TRANSLATION_1)
    second = projection_matrix(INTRINSIC, ROTATION_2, TRANSLATION_2)
    assert first.shape == (3, 4)
    assert second.shape == (3, 4)
    np.testing.assert_allclose(first[:, :3], INTRINSIC)


def projection_point_check() -> None:
    first = projection_matrix(INTRINSIC, ROTATION_1, TRANSLATION_1)
    pixel = project_point(first, WORLD_POINT)
    np.testing.assert_allclose(pixel, [370.0, 260.0], atol=1e-10)


def dlt_check() -> None:
    projections = [
        projection_matrix(INTRINSIC, ROTATION_1, TRANSLATION_1),
        projection_matrix(INTRINSIC, ROTATION_2, TRANSLATION_2),
    ]
    observations = [
        project_point(projections[0], WORLD_POINT),
        project_point(projections[1], WORLD_POINT),
    ]
    reconstructed = triangulate_point_dlt(projections, observations)
    np.testing.assert_allclose(reconstructed, WORLD_POINT, atol=1e-9)


def reprojection_check() -> None:
    projections = [
        projection_matrix(INTRINSIC, ROTATION_1, TRANSLATION_1),
        projection_matrix(INTRINSIC, ROTATION_2, TRANSLATION_2),
    ]
    observations = [project_point(item, WORLD_POINT) for item in projections]
    errors = reprojection_errors(WORLD_POINT, projections, observations)
    np.testing.assert_allclose(errors, np.zeros(2), atol=1e-10)
    assert camera_depth(WORLD_POINT, ROTATION_1, TRANSLATION_1) > 0
    assert camera_depth(WORLD_POINT, ROTATION_2, TRANSLATION_2) > 0


def main() -> None:
    results = [
        check("projection matrices", projection_check),
        check("3D-to-2D projection", projection_point_check),
        check("DLT triangulation", dlt_check),
        check("reprojection and cheirality", reprojection_check),
    ]
    completed = sum(results)
    print(f"\nCompleted lessons: {completed}/{len(results)}")
    if not all(results):
        print("Open geometry.py and implement the first remaining TODO.")


if __name__ == "__main__":
    main()

