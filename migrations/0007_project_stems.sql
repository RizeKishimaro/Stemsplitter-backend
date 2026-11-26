-- Full-length stems are permanent project assets. Existing jobs, chunks, and
-- beat analyses remain intact for compatibility.
CREATE TABLE project_stems (
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    job_id UUID NOT NULL REFERENCES processing_jobs(id) ON DELETE CASCADE,
    stem_name TEXT NOT NULL CHECK (stem_name IN ('drums', 'bass', 'other', 'vocals')),
    storage_key TEXT NOT NULL UNIQUE,
    sample_rate INTEGER NOT NULL CHECK (sample_rate BETWEEN 8000 AND 384000),
    channels SMALLINT NOT NULL CHECK (channels BETWEEN 1 AND 32),
    frame_count BIGINT NOT NULL CHECK (frame_count > 0),
    file_size_bytes BIGINT NOT NULL CHECK (file_size_bytes > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (project_id, stem_name),
    UNIQUE (job_id, stem_name)
);

CREATE INDEX project_stems_job_idx ON project_stems(job_id);
