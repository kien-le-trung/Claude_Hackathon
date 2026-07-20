from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "SquatSpot API"
    database_url: str = "postgresql+psycopg://squatspot:squatspot@db:5432/squatspot"
    max_upload_bytes: int = 100 * 1024 * 1024
    temp_upload_dir: Path = Path("/tmp/squatspot-uploads")
    model_weights_path: Path = Path("models/pose_iter_440000.caffemodel")
    model_config_path: Path = Path("models/pose_deploy_linevec.prototxt")
    reference_json_path: Path = Path("output/model_squat_analysis.json")
    reference_video_path: Path = Path("assets/model_squat.mp4")
    allowed_extensions: set[str] = {"mp4", "avi", "mov", "mkv"}

    model_config = SettingsConfigDict(env_file=".env", env_prefix="SQUATSPOT_", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
