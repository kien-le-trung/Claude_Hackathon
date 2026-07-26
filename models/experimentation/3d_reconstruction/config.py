"""Shared constants for the MM-Fit w00 reconstruction experiment."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MMFIT_ROOT = PROJECT_ROOT / "assets" / "data" / "mm-fit"
SESSION_DIR = MMFIT_ROOT / "w00"
RGB_PATH = MMFIT_ROOT / "w00_rgb.mp4"
POSE_3D_PATH = SESSION_DIR / "w00_pose_3d.npy"
LABELS_PATH = SESSION_DIR / "w00_labels.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_full.task"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
MEDIAPIPE_OUTPUT_PATH = OUTPUT_DIR / "w00_mediapipe.npz"

MMFIT_JOINT_NAMES = (
    "pelvis",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "spine",
    "neck",
    "head",
    "head_site",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
)

MMFIT_BONE_PAIRS = (
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
)

# Direct anatomical correspondences only. Pelvis and other derived joints
# should be added explicitly during evaluation, with their definitions tested.
COMMON_JOINTS = {
    "right_hip": (1, 24),
    "right_knee": (2, 26),
    "right_ankle": (3, 28),
    "left_hip": (4, 23),
    "left_knee": (5, 25),
    "left_ankle": (6, 27),
    "left_shoulder": (11, 11),
    "left_elbow": (12, 13),
    "left_wrist": (13, 15),
    "right_shoulder": (14, 12),
    "right_elbow": (15, 14),
    "right_wrist": (16, 16),
}

