# Phase 1 Engineering Foundation

This branch introduces a parallel application foundation without changing the legacy Flask application or its scoring behavior.

## Stack

- `apps/web`: Next.js + TypeScript client
- `apps/api`: FastAPI + SQLAlchemy API
- PostgreSQL 16
- Docker Compose for local orchestration
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

`POST /api/videos` currently completes analysis within the request. The code separates API, analysis, comparison, configuration, and persistence so a queue/worker can be introduced later without changing the data model or comparator.

## Run locally

The OpenPose weight file must be present under `models/` and the pre-extracted reference must be present at `output/model_squat_analysis.json`.

```bash
docker compose up --build
```

- Frontend: http://localhost:3000
- API: http://localhost:8000
- OpenAPI: http://localhost:8000/docs
- PostgreSQL: localhost:5432

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
