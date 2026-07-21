from pathlib import Path
from uuid import UUID, uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .analysis import LandmarkExtractionService
from .config import get_settings
from .database import SessionLocal, Video, get_db
from .validation import VideoMetadata, VideoValidationError, VideoValidator

settings = get_settings()
extraction_service = LandmarkExtractionService(settings)
video_validator = VideoValidator(settings)
app = FastAPI(title=settings.app_name, version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def serialize_video(video: Video) -> dict:
    extraction = video.extracted_json.get("summary") if video.extracted_json else None
    return {
        "id": str(video.id),
        "date_created": video.date_created,
        "status": video.status,
        "original_filename": video.original_filename,
        "content_type": video.content_type,
        "extraction": extraction,
        "error_message": video.error_message,
    }


def process_video(video_id: UUID, temporary_path: Path, metadata: VideoMetadata) -> None:
    db = SessionLocal()
    try:
        record = db.get(Video, video_id)
        if record is None:
            return
        record.status = "processing"
        db.commit()
        record.extracted_json = extraction_service.extract(temporary_path, metadata)
        record.status = "completed"
        record.error_message = None
        db.commit()
    except Exception as exc:
        db.rollback()
        record = db.get(Video, video_id)
        if record is not None:
            record.status = "failed"
            record.error_message = str(exc)[:1000]
            db.commit()
    finally:
        extraction_service.delete_temporary_file(temporary_path)
        db.close()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post(f"{settings.api_prefix}/videos", status_code=status.HTTP_202_ACCEPTED)
def analyze_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> dict:
    try:
        suffix = video_validator.validate_upload(video.filename, video.content_type)
    except VideoValidationError as exc:
        video.file.close()
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = settings.temp_dir / f"{uuid4()}.{suffix}"

    try:
        bytes_written = 0
        with temporary_path.open("wb") as output:
            while chunk := video.file.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > settings.max_upload_bytes:
                    raise HTTPException(status_code=413, detail="Video exceeds 100 MB limit")
                output.write(chunk)
        metadata = video_validator.validate_file(temporary_path, video.content_type or "")
    except VideoValidationError as exc:
        extraction_service.delete_temporary_file(temporary_path)
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    except Exception:
        extraction_service.delete_temporary_file(temporary_path)
        raise
    finally:
        video.file.close()

    try:
        record = Video(
            status="queued",
            original_filename=video.filename,
            content_type=metadata.content_type,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
    except Exception:
        extraction_service.delete_temporary_file(temporary_path)
        raise

    background_tasks.add_task(process_video, record.id, temporary_path, metadata)
    return serialize_video(record)


@app.get(f"{settings.api_prefix}/videos/{{video_id}}")
def get_video(video_id: UUID, db: Session = Depends(get_db)) -> dict:
    record = db.get(Video, video_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Video analysis not found")
    return serialize_video(record)
