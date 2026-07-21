# Phase 1 Engineering Foundation

This application replaces the legacy Flask/Jinja implementation while preserving its scoring behavior in the FastAPI comparator.

## Stack

- `apps/web`: Next.js + TypeScript client
- `apps/api`: FastAPI + SQLAlchemy API
- PostgreSQL 16
- PostgreSQL 16 in Docker Compose; frontend and API run directly on the host
- Existing OpenPose COCO extraction and cosine-similarity comparator

## Persistence

Phase 1 intentionally uses one PostgreSQL table: `video`.

The original uploaded video is written to a UUID-named temporary path, analyzed, and deleted in a `finally` block. PostgreSQL stores only metadata, status, errors, and the complete extracted JSON payload.

Fields:

- `id`
- `date_created`
- `status`
- `original_filename`
- `content_type`
- `extracted_json`
- `error_message`

## API

- `GET /health`
- `POST /api/videos`
- `GET /api/videos/{id}`

`POST /api/videos` stores the upload metadata, schedules analysis as a FastAPI background task, and returns `202 Accepted`. Clients poll `GET /api/videos/{id}` for completion. The code separates API, analysis, comparison, configuration, and persistence so a durable queue/worker can later be introduced without changing the data model or comparator.

## Run locally

The OpenPose weight file must be present under `models/` and the pre-extracted reference must be present at `output/model_squat_analysis.json`.

```bash
docker compose up -d postgres
pip install -r apps/api/requirements.txt
cd apps/api
uvicorn app.main:app --reload
```

In a second terminal:

```bash
cd apps/web
npm install
npm run dev
```

- Frontend: http://localhost:3000
- API: http://localhost:8000
- OpenAPI: http://localhost:8000/docs
- PostgreSQL container: localhost:5432

## Validation

Comparator regression tests are under `apps/api/tests/`:

```bash
cd apps/api
pytest
```

## Intentionally deferred

- Redis or a durable background worker
- Authentication
- Additional relational tables
- Revised pose extraction or comparison algorithm
- LLM-generated feedback
- Production deployment configuration
