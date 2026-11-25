CREATE TABLE job_chunks (
    job_id UUID NOT NULL REFERENCES processing_jobs(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    chunk_id TEXT NOT NULL,
    bar_number INTEGER,
    start_seconds DOUBLE PRECISION NOT NULL CHECK (start_seconds >= 0),
    end_seconds DOUBLE PRECISION NOT NULL CHECK (end_seconds > start_seconds),
    source_sample_start BIGINT NOT NULL CHECK (source_sample_start >= 0),
    source_sample_end BIGINT NOT NULL CHECK (source_sample_end > source_sample_start),
    sample_rate INTEGER NOT NULL CHECK (sample_rate > 0),
    stem_metadata JSONB NOT NULL,
    completed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (job_id, chunk_index),
    UNIQUE (job_id, chunk_id)
);
