use axum::{
    Json,
    body::Body,
    extract::{Path, State},
    http::{HeaderMap, header},
    response::Response,
};
use serde::Serialize;
use serde_json::Value;
use uuid::Uuid;

use crate::{
    error::AppError,
    services::{project_activity, project_storage, storage},
    state::AppState,
};

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SegmentView {
    segment_id: String,
    project_id: Uuid,
    job_id: Uuid,
    segment_index: i32,
    bar_number: Option<i32>,
    start_sample: i64,
    end_sample: i64,
    start_time: f64,
    end_time: f64,
    duration_seconds: f64,
    sample_rate: i32,
    channels: i16,
    frame_count: i64,
    audio_format: String,
    status: String,
    stems: Value,
}

#[utoipa::path(get, path = "/api/v1/projects/{project_id}/segments", params(("project_id" = Uuid, Path)), responses((status = 200, description = "Progressively available synchronized segment manifest"), (status = 410, description = "Project expired")))]
pub async fn list_project_segments(
    State(state): State<AppState>,
    Path(raw): Path<String>,
) -> Result<Json<Vec<SegmentView>>, AppError> {
    let project_id = Uuid::parse_str(&raw)
        .map_err(|_| AppError::BadRequest("project_id must be a UUID".into()))?;
    if !project_activity::touch_project(&state.db, project_id, state.config.project_ttl).await? {
        return Err(AppError::Gone);
    }
    let rows = sqlx::query_as::<_, (String,Uuid,i32,Option<i32>,i64,i64,i32,i16,i64,String,String,Value,Uuid)>("SELECT segment_id,project_id,segment_index,bar_number,start_sample,end_sample,sample_rate,channels,frame_count,audio_format,status,stems,job_id FROM project_segments WHERE project_id=$1 ORDER BY segment_index")
        .bind(project_id).fetch_all(&state.db).await?;
    Ok(Json(
        rows.into_iter()
            .map(
                |(
                    segment_id,
                    project_id,
                    segment_index,
                    bar_number,
                    start_sample,
                    end_sample,
                    sample_rate,
                    channels,
                    frame_count,
                    audio_format,
                    status,
                    stems,
                    job_id,
                )| SegmentView {
                    segment_id,
                    project_id,
                    job_id,
                    segment_index,
                    bar_number,
                    start_sample,
                    end_sample,
                    start_time: start_sample as f64 / f64::from(sample_rate),
                    end_time: end_sample as f64 / f64::from(sample_rate),
                    duration_seconds: frame_count as f64 / f64::from(sample_rate),
                    sample_rate,
                    channels,
                    frame_count,
                    audio_format,
                    status,
                    stems,
                },
            )
            .collect(),
    ))
}

#[utoipa::path(get, path = "/api/v1/projects/{project_id}/segments/{segment_id}/{stem_name}", params(("project_id" = Uuid, Path), ("segment_id" = String, Path), ("stem_name" = String, Path), ("Range" = Option<String>, Header)), responses((status = 200, description = "Segment WAV stream"), (status = 206, description = "Partial segment WAV"), (status = 404, description = "Segment not ready"), (status = 410, description = "Project expired")))]
pub async fn get_project_segment(
    State(state): State<AppState>,
    Path((raw_project, segment_id, stem)): Path<(String, String, String)>,
    headers: HeaderMap,
) -> Result<Response<Body>, AppError> {
    let project_id = Uuid::parse_str(&raw_project)
        .map_err(|_| AppError::BadRequest("project_id must be a UUID".into()))?;
    if !["drums", "bass", "other", "vocals"].contains(&stem.as_str()) {
        return Err(AppError::BadRequest(
            "stem_name must be drums, bass, other, or vocals".into(),
        ));
    }
    if !project_activity::touch_project(&state.db, project_id, state.config.project_ttl).await? {
        return Err(AppError::Gone);
    }
    let stems = sqlx::query_scalar::<_, Value>("SELECT stems FROM project_segments WHERE project_id=$1 AND segment_id=$2 AND status='READY'")
        .bind(project_id).bind(segment_id).fetch_optional(&state.db).await?.ok_or(AppError::NotFound)?;
    let asset = stems.get(&stem).ok_or(AppError::NotFound)?;
    let key = asset
        .get("storageKey")
        .and_then(Value::as_str)
        .ok_or_else(|| AppError::Internal("segment storage metadata is invalid".into()))?;
    let root = project_storage::projects_root(&state.config.upload_dir);
    let candidate = project_storage::resolve_storage_key(&root, key).map_err(AppError::Internal)?;
    let canonical_root = tokio::fs::canonicalize(root).await.map_err(AppError::Io)?;
    let path = tokio::fs::canonicalize(candidate).await.map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            AppError::NotFound
        } else {
            AppError::Io(error)
        }
    })?;
    if !path.starts_with(&canonical_root) {
        return Err(AppError::NotFound);
    }
    let range = headers
        .get(header::RANGE)
        .and_then(|value| value.to_str().ok());
    let lease = storage::acquire_audio_stream(project_id).ok_or(AppError::Gone)?;
    crate::routes::stems::stream_file_with_lease(&path, range, "audio/wav", Some(lease)).await
}
