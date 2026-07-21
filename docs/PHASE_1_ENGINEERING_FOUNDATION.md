# Landmark Extraction Foundation

## Stack

- Next.js and TypeScript frontend
- FastAPI and MediaPipe Pose Landmarker backend
- PostgreSQL 16 in Docker Compose
- Host-run frontend and API

## Processing contract

Uploads are written to temporary UUID-named files and fully validated before a database record is created or MediaPipe is called. Accepted videos move through `queued`, `processing`, `completed`, or `failed`. FastAPI background tasks perform extraction, and temporary files are deleted in all terminal paths.

MediaPipe uses the full Pose Landmarker task in video mode, processes one person, and samples source frames at approximately 10 FPS. Segmentation masks are disabled.

## Persistence

The existing `video.extracted_json` JSONB field stores a schema-versioned document containing extractor configuration, validated video metadata, sampling metadata, and per-frame normalized and world landmarks. The polling API deliberately returns only the extraction summary; raw landmarks remain internal until a later analysis API is designed.

Historical OpenPose-shaped rows are not migrated. New records can be identified by `schema_version: 1` and extractor name `mediapipe_pose_landmarker`.

## API

- `GET /health`
- `POST /api/videos`
- `GET /api/videos/{id}`

`POST /api/videos` returns `202 Accepted` for a validated upload. Invalid/undecodable media returns `400`; size, duration, and resolution violations return `413`.

## Deferred

- Selecting landmarks for squat analysis
- Biomechanical calculations
- Comparison, scoring, and feedback
- Durable task queue and recovery after process restarts
- Authentication and production deployment
