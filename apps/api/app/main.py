from pathlib import Path
from uuid import UUID, uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .analysis import AnalysisService
from .config import get_settings
from .database import SessionLocal, Video, get_db

settings = get_settings()
analysis_service = AnalysisService(settings)
app = FastAPI(title=settings.app_name, version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
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


def process_video(video_id: UUID, temporary_path: Path) -> None:
    db = SessionLocal()
    try:
        record = db.get(Video, video_id)
        if record is None:
            return
        record.extracted_json = analysis_service.analyze(temporary_path)
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
        analysis_service.delete_temporary_file(temporary_path)
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
    suffix = Path(video.filename or "").suffix.lower().lstrip(".")
    if suffix not in settings.allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported video extension")

    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = settings.temp_dir / f"{uuid4()}.{suffix}"
    record = Video(
        status="queued",
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
    except Exception:
        analysis_service.delete_temporary_file(temporary_path)
        record.status = "failed"
        record.error_message = "Upload rejected"
        db.commit()
        raise
    finally:
        video.file.close()

    record.status = "processing"
    db.commit()
    db.refresh(record)
    background_tasks.add_task(process_video, record.id, temporary_path)
    return serialize_video(record)


@app.get(f"{settings.api_prefix}/videos/{{video_id}}")
def get_video(video_id: UUID, db: Session = Depends(get_db)) -> dict:
    record = db.get(Video, video_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Video analysis not found")
    return serialize_video(record)
