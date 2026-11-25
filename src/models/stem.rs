use serde::Serialize;
use utoipa::ToSchema;
use uuid::Uuid;

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct StemMetadata {
    /// One of drums, bass, other, or vocals.
    pub name: String,
    pub sample_rate: u32,
    pub channels: u16,
    pub frame_count: u64,
    pub duration_seconds: f64,
    pub audio_format: String,
    pub file_size_bytes: Option<u64>,
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct StemListResponse {
    pub job_id: Uuid,
    /// Only successful jobs can return stems.
    pub job_status: String,
    pub stems: Vec<StemMetadata>,
}
