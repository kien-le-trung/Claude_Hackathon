import shutil
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .analysis import AnalysisService
from .config import get_settings
from .database import Video, get_db

settings = get_settings()
analysis_service = AnalysisService(settings)
app = FastAPI(title=settings.app_name, version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"],
)


def serialize_video(video: Video) -> dict:
    result = video.extracted_json.get("results") if video.extracted_json else None
    return {
        "id": str(video.id),
        "date_created": video.date_created,
        "status": video.status,
        "original_filename": video.original_filename,
        "content_type": video.content_type,
        "result": result,
        "error_message": video.error_message,
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post(f"{settings.api_prefix}/videos", status_code=status.HTTP_201_CREATED)
def analyze_video(
    video: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> dict:
    suffix = Path(video.filename or "").suffix.lower().lstrip(".")
    if suffix not in settings.allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported video extension")

    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = settings.temp_dir / f"{uuid4()}.{suffix}"
    record = Video(
        status="processing",
        original_filename=video.filename,
        content_type=video.content_type,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    try:
        bytes_written = 0
        with temporary_path.open("wb") as output:
            while chunk := video.file.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > settings.max_upload_bytes:
                    raise HTTPException(status_code=413, detail="Video exceeds 100 MB limit")
                output.write(chunk)

        record.extracted_json = analysis_service.analyze(temporary_path)
        record.status = "completed"
        db.commit()
        db.refresh(record)
        return serialize_video(record)
    except HTTPException:
        record.status = "failed"
        record.error_message = "Upload rejected"
        db.commit()
        raise
    except Exception as exc:
        record.status = "failed"
        record.error_message = str(exc)[:1000]
        db.commit()
        raise HTTPException(status_code=500, detail="Video analysis failed") from exc
    finally:
        analysis_service.delete_temporary_file(temporary_path)
        video.file.close()


@app.get(f"{settings.api_prefix}/videos/{{video_id}}")
def get_video(video_id: UUID, db: Session = Depends(get_db)) -> dict:
    record = db.get(Video, video_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Video analysis not found")
    return serialize_video(record)
