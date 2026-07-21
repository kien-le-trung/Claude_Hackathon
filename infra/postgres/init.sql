CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS video (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    date_created TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status VARCHAR(32) NOT NULL DEFAULT 'processing',
    original_filename VARCHAR(255),
    content_type VARCHAR(100),
    extracted_json JSONB,
    error_message VARCHAR(1000)
);

CREATE INDEX IF NOT EXISTS ix_video_date_created ON video (date_created DESC);
CREATE INDEX IF NOT EXISTS ix_video_status ON video (status);
