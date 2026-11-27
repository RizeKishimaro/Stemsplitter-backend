ALTER TABLE projects
    ADD COLUMN project_state JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN last_activity_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ADD COLUMN expires_at TIMESTAMPTZ,
    ADD COLUMN deleting_at TIMESTAMPTZ;

-- Preserve existing projects through the migration; start their configured
-- lifetime now instead of treating historical timestamps as expired.
UPDATE projects SET expires_at = now() + interval '10 hours' WHERE expires_at IS NULL;
ALTER TABLE projects ALTER COLUMN expires_at SET DEFAULT (now() + interval '10 hours');
CREATE INDEX projects_expiration_idx ON projects(expires_at) WHERE deleting_at IS NULL;

CREATE TABLE project_segments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    job_id UUID NOT NULL REFERENCES processing_jobs(id) ON DELETE CASCADE,
    segment_index INTEGER NOT NULL CHECK (segment_index >= 0),
    bar_number INTEGER,
    segment_id TEXT NOT NULL,
    start_sample BIGINT NOT NULL CHECK (start_sample >= 0),
    end_sample BIGINT NOT NULL CHECK (end_sample > start_sample),
    sample_rate INTEGER NOT NULL CHECK (sample_rate BETWEEN 8000 AND 384000),
    channels SMALLINT NOT NULL CHECK (channels BETWEEN 1 AND 32),
    frame_count BIGINT NOT NULL CHECK (frame_count > 0),
    audio_format TEXT NOT NULL DEFAULT 'wav',
    status TEXT NOT NULL CHECK (status IN ('PENDING','PROCESSING','READY','FAILED')),
    stems JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(project_id, segment_index),
    UNIQUE(project_id, segment_id),
    CHECK (frame_count = end_sample - start_sample)
);
CREATE INDEX project_segments_ready_idx ON project_segments(project_id, segment_index, status);
