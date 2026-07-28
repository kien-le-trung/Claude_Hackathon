"""Inspect one usable synchronized EC3D frame and its calibrated cameras."""

from __future__ import annotations

from config import START_CAMERAS
from data import common_joint_names, first_usable_frame, load_ec3d


def main() -> None:
    data = load_ec3d()
    print(f"Actions: {list(data['frames'])}")
    print(f"Camera IDs: {list(data['params'])}")
    for camera_id, parameters in data["params"].items():
        intrinsics = parameters["intrinsics"]
        extrinsics = parameters["extrinsics"]
        print(f"\nCamera {camera_id}")
        print(f"  K shape: {intrinsics['K'].shape}")
        print(f"  distortion shape: {intrinsics['distCoeffs'].shape}")
        print(f"  image shape: {intrinsics['image_shape']}")
        print(f"  calibration reprojection error: {intrinsics['reproj_error']}")
        print(f"  R shape: {extrinsics['R'].shape}")
        print(f"  t shape: {extrinsics['t'].shape}")

    path, frame = first_usable_frame(data)
    print(f"\nFirst usable frame: {path}")
    print(f"Image paths: {frame.get('path')}")
    print(
        "2D points by view: "
        f"{ {camera: len(points) for camera, points in frame['2D_op'].items()} }"
    )
    print(f"3D ground-truth joints: {len(frame['3D_gt'])}")
    common = common_joint_names(frame, START_CAMERAS)
    print(f"Common joints for {START_CAMERAS}: {len(common)}")
    print(f"Joint names: {common}")
    if common:
        joint = common[0]
        print(f"\nRepresentative joint: {joint}")
        for camera_id in START_CAMERAS:
            value = frame["2D_op"][camera_id][joint]
            print(f"  {camera_id} 2D value: {value!r} ({type(value).__name__})")
        if joint in frame["3D_gt"]:
            value = frame["3D_gt"][joint]
            print(f"  3D ground truth: {value!r} ({type(value).__name__})")


if __name__ == "__main__":
    main()
