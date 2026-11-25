CREATE TABLE beat_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    separation_job_id UUID NOT NULL REFERENCES processing_jobs(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'QUEUED'
        CHECK (status IN ('QUEUED', 'RUNNING', 'SUCCEEDED', 'FAILED')),
    input_stem TEXT NOT NULL DEFAULT 'drums' CHECK (input_stem = 'drums'),
    sample_rate INTEGER NOT NULL CHECK (sample_rate BETWEEN 8000 AND 384000),
    channels SMALLINT NOT NULL CHECK (channels BETWEEN 1 AND 32),
    frame_count BIGINT NOT NULL CHECK (frame_count > 0),
    engine TEXT NOT NULL DEFAULT 'beat-this-rs',
    engine_version TEXT NOT NULL DEFAULT '1.1.0',
    model_version TEXT,
    result JSONB,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ
);

CREATE INDEX beat_analyses_job_created_idx
    ON beat_analyses(separation_job_id, created_at DESC);
