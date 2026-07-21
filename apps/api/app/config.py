from functools import lru_cache
from pathlib import Path
from tempfile import gettempdir

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    app_name: str = "SquatSpot API"
    api_prefix: str = "/api"
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/squatspot"
    max_upload_bytes: int = 100 * 1024 * 1024
    max_video_duration_seconds: float = 120.0
    max_video_pixels: int = 3840 * 2160
    target_sampling_fps: float = 10.0
    temp_dir: Path = Path(gettempdir()) / "squatspot"
    mediapipe_model_path: Path = PROJECT_ROOT / "models/pose_landmarker_full.task"
    pose_detection_confidence: float = 0.5
    pose_presence_confidence: float = 0.5
    tracking_confidence: float = 0.5
    allowed_extensions: set[str] = {"mp4", "avi", "mov", "mkv"}
    allowed_mime_types: dict[str, set[str]] = {
        "mp4": {"video/mp4", "application/mp4"},
        "avi": {"video/x-msvideo", "video/avi"},
        "mov": {"video/quicktime"},
        "mkv": {"video/x-matroska"},
    }
    cors_origins: list[str] = ["http://localhost:3000"]

    model_config = SettingsConfigDict(env_file=".env", env_prefix="SQUATSPOT_", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
