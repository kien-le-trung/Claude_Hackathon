# SquatSpot

SquatSpot is a local Next.js and FastAPI application that extracts pose landmarks from uploaded squat videos with MediaPipe. An optional lightweight ONNX classifier detects back and heel form errors from sampled landmarks. Raw extraction and classification evidence are stored in PostgreSQL.

## Requirements

- Python 3.12
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
py -3.12 -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
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
4. MediaPipe samples the video by timestamp at up to 10 FPS.
5. A biomechanics-only SVM classifies sufficiently complete poses and consolidates stable adjacent errors.
6. PostgreSQL stores compact frame evidence and event summaries; representative
   frames, a normalized source MP4, and an unsmoothed skeleton MP4 are stored
   locally with the analysis.
7. The temporary upload is deleted after processing. A silent browser-ready MP4
   and unsmoothed 3D skeleton reconstruction remain with the analysis until the
   analysis is deleted.

## Squat-form classifier

Training lives in `models/experimentation`; see its README for the reproducible
audit, extraction, training, and promotion commands. All commands use the
existing `venv` interpreter. Train and promote the required v2 model with:

```powershell
.\venv\Scripts\python.exe -m models.experimentation.cli train
.\venv\Scripts\python.exe -m models.experimentation.cli promote --version v2
```

The polling API exposes classification summaries and consolidated events.
Compact frame predictions remain in PostgreSQL. A normalized source MP4,
reconstruction MP4, and representative annotated frames remain until the
analysis is deleted.

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
