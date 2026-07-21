# SquatSpot

SquatSpot analyzes uploaded squat videos with OpenPose and compares the detected poses with a reference squat. The current application is a Next.js frontend backed by a FastAPI API and PostgreSQL.

## Structure

```text
apps/
  api/                    FastAPI service, analysis pipeline, and tests
  web/                    Next.js frontend
infra/postgres/           Database initialization
models/                   OpenPose model files (local, ignored)
output/                   Reference analysis JSON (local, ignored)
openpose_model.py         OpenCV/OpenPose model wrapper used by the API
analyze_model_squat.py    Regenerates the reference analysis
download_openpose_model.py
compose.yaml              Local application stack
```

The former Flask/Jinja application has been removed. Scoring now lives in `apps/api/app/comparator.py` and video processing lives in `apps/api/app/analysis.py`.

## Run locally

Before starting, ensure these runtime files exist:

- `models/pose_iter_440000.caffemodel`
- `models/pose_deploy_linevec.prototxt`
- `output/model_squat_analysis.json`

Download the model if needed:

```bash
python download_openpose_model.py
```

Start the complete stack:

```bash
docker compose up --build
```

- Frontend: http://localhost:3000
- API: http://localhost:8000
- OpenAPI: http://localhost:8000/docs
- PostgreSQL: localhost:5432

## API

- `GET /health`
- `POST /api/videos` — accepts MP4, AVI, MOV, or MKV files up to 100 MB
- `GET /api/videos/{id}` — returns processing status and completed results

Uploads are analyzed asynchronously after the API returns `202 Accepted`. The frontend polls the video endpoint until processing completes or fails.

## Tests

```bash
python -m pytest apps/api/tests
```

For more implementation detail, see [the engineering foundation](docs/PHASE_1_ENGINEERING_FOUNDATION.md).
