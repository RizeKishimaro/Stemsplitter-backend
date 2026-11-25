-- Audio files and job metadata are temporary. Existing rows remain untouched
-- (NULL expiry) until a new upload/job is created under the TTL policy.
ALTER TABLE audio_assets ADD COLUMN expires_at TIMESTAMPTZ;
ALTER TABLE processing_jobs ADD COLUMN expires_at TIMESTAMPTZ;

CREATE INDEX audio_assets_expires_idx ON audio_assets(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX processing_jobs_expires_idx ON processing_jobs(expires_at) WHERE expires_at IS NOT NULL;
