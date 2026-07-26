import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from sqlalchemy.orm import Session

from .analysis import LandmarkExtractionService
from .artifacts import artifact_path, create_event_artifacts, delete_video_artifacts
from .classification import SquatFormClassifier
from .config import get_settings
from .database import SessionLocal, Video, get_db
from .events import public_event
from .media import (
    media_destination,
    normalize_source_video,
    render_reconstruction_video,
)
from .validation import VideoMetadata, VideoValidationError, VideoValidator

settings = get_settings()
extraction_service = LandmarkExtractionService(settings)
video_validator = VideoValidator(settings)
classifier_service = None


def get_classifier() -> SquatFormClassifier:
    global classifier_service
    if not settings.classifier_enabled:
        raise RuntimeError("Squat-form classifier is required but disabled")
    if classifier_service is None:
        classifier_service = SquatFormClassifier(
            settings.classifier_model_dir, settings.classifier_error_threshold
        )
    return classifier_service


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_classifier()
    yield


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def serialize_video(video: Video) -> dict:
    extraction = video.extracted_json.get("summary") if video.extracted_json else None
    progress = video.extracted_json.get("progress") if video.extracted_json else None
    classification = (
        video.extracted_json.get("classification", {}).get("summary")
        if video.extracted_json else None
    )
    if classification:
        classification = dict(classification)
        classification["events"] = [
            public_event(event, str(video.id))
            for event in classification.get("events", [])
        ]
    stored_media = video.extracted_json.get("media", {}) if video.extracted_json else {}
    media = {
        "source_video_url": (
            f"{settings.api_prefix}/videos/{video.id}/media/source"
            if stored_media.get("source_path") else None
        ),
        "reconstruction_video_url": (
            f"{settings.api_prefix}/videos/{video.id}/media/reconstruction"
            if stored_media.get("reconstruction_path") else None
        ),
        "reconstruction_fps": stored_media.get("reconstruction_fps"),
        "warnings": stored_media.get("warnings", []),
    }
    return {
        "id": str(video.id),
        "date_created": video.date_created,
        "status": video.status,
        "original_filename": video.original_filename,
        "content_type": video.content_type,
        "extraction": extraction,
        "classification": classification,
        "progress": progress,
        "media": media,
        "error_message": video.error_message,
    }


def process_video(video_id: UUID, temporary_path: Path, metadata: VideoMetadata) -> None:
    db = SessionLocal()
    try:
        record = db.get(Video, video_id)
        if record is None:
            return
        record.status = "processing"
        record.extracted_json = {
            "progress": {
                "stage": "extracting",
                "decoded_frame_count": 0,
                "sampled_frame_count": 0,
                "detected_frame_count": 0,
                "classified_frame_count": 0,
                "total_frame_count": metadata.total_frames,
                "percent": 0.0,
            }
        }
        db.commit()
        last_progress_write = 0.0

        def update_progress(progress: dict) -> None:
            nonlocal last_progress_write
            now = time.monotonic()
            if now - last_progress_write < 1.0 and progress["percent"] < 100.0:
                return
            record.extracted_json = {
                "progress": {
                    **progress,
                    "stage": "extracting",
                    "percent": round(progress["percent"] * 0.7, 1),
                    "classified_frame_count": 0,
                }
            }
            db.commit()
            last_progress_write = now

        payload = extraction_service.extract(
            temporary_path, metadata, progress_callback=update_progress
        )
        record.extracted_json = {
            "progress": {
                **record.extracted_json.get("progress", {}),
                "stage": "classifying",
                "percent": 72.0,
            }
        }
        db.commit()
        classifier = get_classifier()
        payload["classification"] = classifier.classify(payload)
        events = payload["classification"]["events"]
        create_event_artifacts(
            temporary_path, events, settings.artifact_root, str(video_id)
        )

        media = {
            "source_path": None,
            "reconstruction_path": None,
            "reconstruction_fps": float(payload["sampling"]["effective_fps"]),
            "warnings": [],
        }
        record.extracted_json = {
            "progress": {
                **record.extracted_json.get("progress", {}),
                "stage": "preparing_video",
                "percent": 80.0,
            }
        }
        db.commit()
        source_destination, source_relative = media_destination(
            settings.artifact_root, str(video_id), "source.mp4"
        )
        try:
            normalize_source_video(temporary_path, source_destination)
            media["source_path"] = source_relative
        except Exception as exc:
            media["warnings"].append(f"Source video unavailable: {str(exc)[:240]}")

        record.extracted_json = {
            "progress": {
                **record.extracted_json.get("progress", {}),
                "stage": "rendering_reconstruction",
                "percent": 88.0,
            }
        }
        db.commit()
        reconstruction_destination, reconstruction_relative = media_destination(
            settings.artifact_root, str(video_id), "reconstruction.mp4"
        )
        try:
            render_reconstruction_video(
                payload["frames"],
                reconstruction_destination,
                media["reconstruction_fps"],
            )
            media["reconstruction_path"] = reconstruction_relative
        except Exception as exc:
            media["warnings"].append(
                f"Skeleton reconstruction unavailable: {str(exc)[:240]}"
            )

        prediction_by_frame = {
            item["frame_number"]: item
            for item in payload["classification"]["frames"]
        }
        compact_frames = []
        for frame in payload["frames"]:
            compact = {
                "frame_number": frame["frame_number"],
                "timestamp_ms": frame["timestamp_ms"],
                "pose_detected": bool(frame["poses"]),
            }
            prediction = prediction_by_frame.get(frame["frame_number"])
            if prediction:
                compact.update(prediction)
            compact_frames.append(compact)
        payload["evidence"] = {"frames": compact_frames}
        payload.pop("frames", None)
        for event in events:
            event.pop("_representative_pose", None)
        payload["classification"].pop("frames", None)
        payload["classification"].pop("events", None)
        payload["progress"] = {
            "stage": "completed",
            "decoded_frame_count": payload["summary"]["decoded_frame_count"],
            "sampled_frame_count": payload["summary"]["sampled_frame_count"],
            "detected_frame_count": payload["summary"]["detected_frame_count"],
            "classified_frame_count": payload["classification"]["summary"]["classified_frame_count"],
            "total_frame_count": metadata.total_frames,
            "percent": 100.0,
        }
        payload["media"] = media
        record.extracted_json = payload
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
        delete_video_artifacts(settings.artifact_root, str(video_id))
    finally:
        extraction_service.delete_temporary_file(temporary_path)
        db.close()


@app.get("/health")
def health() -> dict:
    try:
        classifier = get_classifier()
        return {
            "status": "ok",
            "classifier": classifier.metadata["model_version"],
        }
    except Exception as exc:
        return {"status": "unhealthy", "classifier_error": str(exc)}


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


@app.get(f"{settings.api_prefix}/videos/{{video_id}}/events/{{event_id}}/frame")
def get_event_frame(
    video_id: UUID,
    event_id: UUID,
    db: Session = Depends(get_db),
):
    record = db.get(Video, video_id)
    if record is None or not record.extracted_json:
        raise HTTPException(status_code=404, detail="Video analysis not found")
    events = (
        record.extracted_json.get("classification", {})
        .get("summary", {})
        .get("events", [])
    )
    event = next((item for item in events if item.get("id") == str(event_id)), None)
    if event is None or not event.get("frame_image_path"):
        raise HTTPException(status_code=404, detail="Representative frame not found")
    path = artifact_path(settings.artifact_root, event["frame_image_path"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Representative frame not found")
    return FileResponse(path, media_type="image/webp")


def _media_response(video_id: UUID, kind: str, request: Request, db: Session):
    record = db.get(Video, video_id)
    if record is None or not record.extracted_json:
        raise HTTPException(status_code=404, detail="Video analysis not found")
    media = record.extracted_json.get("media", {})
    key = "source_path" if kind == "source" else "reconstruction_path"
    relative = media.get(key)
    if not relative:
        raise HTTPException(status_code=404, detail=f"{kind.title()} video unavailable")
    path = artifact_path(settings.artifact_root, relative)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"{kind.title()} video unavailable")
    common_headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "private, no-store",
        "Content-Disposition": f'inline; filename="{kind}.mp4"',
    }
    range_header = request.headers.get("range")
    if range_header:
        size = path.stat().st_size
        try:
            unit, requested = range_header.split("=", 1)
            if unit.strip().lower() != "bytes" or "," in requested:
                raise ValueError
            start_text, end_text = requested.strip().split("-", 1)
            if not start_text:
                suffix_length = int(end_text)
                if suffix_length <= 0:
                    raise ValueError
                start = max(size - suffix_length, 0)
                end = size - 1
            else:
                start = int(start_text)
                end = int(end_text) if end_text else size - 1
            if start < 0 or start >= size or end < start:
                raise ValueError
            end = min(end, size - 1)
        except (ValueError, TypeError):
            return Response(
                status_code=416,
                headers={**common_headers, "Content-Range": f"bytes */{size}"},
            )

        length = end - start + 1

        def stream_range():
            remaining = length
            with path.open("rb") as handle:
                handle.seek(start)
                while remaining:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return StreamingResponse(
            stream_range(),
            status_code=206,
            media_type="video/mp4",
            headers={
                **common_headers,
                "Content-Range": f"bytes {start}-{end}/{size}",
                "Content-Length": str(length),
            },
        )
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=None,
        headers=common_headers,
    )


@app.get(f"{settings.api_prefix}/videos/{{video_id}}/media/source")
def get_source_video(video_id: UUID, request: Request, db: Session = Depends(get_db)):
    return _media_response(video_id, "source", request, db)


@app.get(f"{settings.api_prefix}/videos/{{video_id}}/media/reconstruction")
def get_reconstruction_video(
    video_id: UUID, request: Request, db: Session = Depends(get_db)
):
    return _media_response(video_id, "reconstruction", request, db)


@app.delete(f"{settings.api_prefix}/videos/{{video_id}}", status_code=204)
def delete_video(video_id: UUID, db: Session = Depends(get_db)) -> Response:
    record = db.get(Video, video_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Video analysis not found")
    delete_video_artifacts(settings.artifact_root, str(video_id))
    db.delete(record)
    db.commit()
    return Response(status_code=204)
