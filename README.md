# SquatSpot

SquatSpot is a local Next.js and FastAPI application that extracts pose landmarks from uploaded squat videos with MediaPipe. Raw normalized and world landmarks are stored in PostgreSQL for later analysis; scoring and form feedback are intentionally not part of this phase.

## Requirements

- Python 3.11
- Node.js 20+
- Docker with Docker Compose (PostgreSQL only)

## Setup

Download the MediaPipe Pose Landmarker full model:

```bash
python download_mediapipe_model.py
```

Start PostgreSQL:

```bash
docker compose up -d postgres
```

Create the API environment and start FastAPI:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r apps/api/requirements.txt
cd apps/api
uvicorn app.main:app --reload
```

In another terminal, start the frontend:

```bash
cd apps/web
npm install
npm run dev
```

- Frontend: http://localhost:3000
- API: http://localhost:8000
- OpenAPI: http://localhost:8000/docs
- PostgreSQL: localhost:5432

## Upload flow

1. The frontend uploads an MP4, AVI, MOV, or MKV file to `POST /api/videos`.
2. The backend enforces matching extension/MIME, a 100 MB size limit, successful decode, valid stream metadata, a 120-second duration limit, and a 4K pixel limit.
3. A valid upload receives `202 Accepted`; the frontend polls `GET /api/videos/{id}`.
4. MediaPipe Pose Landmarker samples the video at approximately 10 FPS and extracts one pose per sampled frame.
5. PostgreSQL stores the complete versioned landmark payload. The API returns only extraction summary data.
6. The temporary uploaded video is deleted after success or failure.

## Tests

From `apps/api`:

```bash
pytest
```

Build the frontend from `apps/web`:

```bash
npm run build
```

Stop PostgreSQL with `docker compose down`. Add `--volumes` only when you intentionally want to delete the local database.
