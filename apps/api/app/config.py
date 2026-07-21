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
    temp_dir: Path = Path(gettempdir()) / "squatspot"
    model_weights_path: Path = PROJECT_ROOT / "models/pose_iter_440000.caffemodel"
    model_config_path: Path = PROJECT_ROOT / "models/pose_deploy_linevec.prototxt"
    reference_json_path: Path = PROJECT_ROOT / "output/model_squat_analysis.json"
    allowed_extensions: set[str] = {"mp4", "avi", "mov", "mkv"}
    cors_origins: list[str] = ["http://localhost:3000"]

    model_config = SettingsConfigDict(env_file=".env", env_prefix="SQUATSPOT_", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
