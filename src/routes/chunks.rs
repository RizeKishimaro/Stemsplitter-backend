use std::path::PathBuf;

use axum::{
    Json,
    body::Body,
    extract::{Path, State},
    http::{HeaderMap, header},
    response::Response,
};
use serde::Serialize;
use serde_json::Value;
use sqlx::FromRow;
use uuid::Uuid;

use crate::{error::AppError, services::storage, state::AppState};

#[derive(Serialize, FromRow)]
pub struct JobChunk {
    pub chunk_index: i32,
    pub chunk_id: String,
    pub bar_number: Option<i32>,
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub source_sample_start: i64,
    pub source_sample_end: i64,
    pub sample_rate: i32,
    pub stem_metadata: Value,
}

#[utoipa::path(get, path="/api/v1/jobs/{job_id}/chunks", params(("job_id"=Uuid, Path)), responses((status=200, description="Completed synchronized chunk manifest")))]
pub async fn list_chunks(
    State(state): State<AppState>,
    Path(raw): Path<String>,
) -> Result<Json<Vec<JobChunk>>, AppError> {
    let id = parse_uuid(&raw)?;
    let exists =
        sqlx::query_scalar::<_, bool>("SELECT EXISTS(SELECT 1 FROM processing_jobs WHERE id=$1)")
            .bind(id)
            .fetch_one(&state.db)
            .await?;
    if !exists {
        return Err(AppError::NotFound);
    }
    let chunks = sqlx::query_as::<_, JobChunk>("SELECT chunk_index, chunk_id, bar_number, start_seconds, end_seconds, source_sample_start, source_sample_end, sample_rate, stem_metadata FROM job_chunks WHERE job_id=$1 ORDER BY chunk_index")
        .bind(id).fetch_all(&state.db).await?;
    Ok(Json(chunks))
}

#[utoipa::path(get, path="/api/v1/jobs/{job_id}/chunks/{chunk_index}/stems/{stem_name}", params(("job_id"=Uuid, Path), ("chunk_index"=i32, Path), ("stem_name"=String, Path)), responses((status=200, description="Completed WAV chunk"), (status=206, description="WAV byte range"), (status=404, description="Chunk or stem unavailable")))]
pub async fn get_chunk_stem(
    State(state): State<AppState>,
    Path((raw, chunk_index, stem)): Path<(String, i32, String)>,
    headers: HeaderMap,
) -> Result<Response<Body>, AppError> {
    let job_id = parse_uuid(&raw)?;
    if !["drums", "bass", "other", "vocals"].contains(&stem.as_str()) {
        return Err(AppError::BadRequest(
            "stem_name must be drums, bass, other, or vocals".into(),
        ));
    }
    let chunk_id = sqlx::query_scalar::<_, String>(
        "SELECT chunk_id FROM job_chunks WHERE job_id=$1 AND chunk_index=$2",
    )
    .bind(job_id)
    .bind(chunk_index)
    .fetch_optional(&state.db)
    .await?
    .ok_or(AppError::NotFound)?;
    if !chunk_id.starts_with(&format!("{}-", job_id.simple())) {
        return Err(AppError::NotFound);
    }
    let path: PathBuf = storage::jobs_root(&state.config.upload_dir)
        .join(job_id.to_string())
        .join("chunks")
        .join(chunk_id)
        .join(format!("{stem}.wav"));
    let metadata = tokio::fs::metadata(&path)
        .await
        .map_err(|_| AppError::NotFound)?;
    if !metadata.is_file() {
        return Err(AppError::NotFound);
    }
    let lease = storage::acquire_audio_stream(job_id).ok_or(AppError::NotFound)?;
    let range = headers.get(header::RANGE).and_then(|v| v.to_str().ok());
    crate::routes::stems::stream_stem_with_lease(&path, range, Some(lease)).await
}

fn parse_uuid(value: &str) -> Result<Uuid, AppError> {
    Uuid::parse_str(value).map_err(|_| AppError::BadRequest("job_id must be a UUID".into()))
}
