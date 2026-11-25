use chrono::{DateTime, Utc};
use serde::Serialize;
use serde_json::Value;
use sqlx::FromRow;
use utoipa::ToSchema;
use uuid::Uuid;

#[derive(Clone, Debug, Serialize, FromRow, ToSchema)]
#[schema(as = BeatAnalysisJob)]
pub struct BeatAnalysisJob {
    pub id: Uuid,
    pub separation_job_id: Uuid,
    pub status: String,
    pub input_stem: String,
    pub sample_rate: i32,
    pub channels: i16,
    pub frame_count: i64,
    pub engine: String,
    pub engine_version: String,
    pub model_version: Option<String>,
    pub result: Option<Value>,
    pub error_message: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
}
