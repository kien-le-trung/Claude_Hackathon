"""Paths and learning defaults for the EC3D triangulation playground."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EC3D_ROOT = PROJECT_ROOT / "assets" / "data" / "ec3d"
EC3D_PICKLE = EC3D_ROOT / "data.pickle"

START_CAMERAS = ("6_1", "6_2")
ALL_CAMERAS = ("6_1", "6_2", "6_3", "6_4")

