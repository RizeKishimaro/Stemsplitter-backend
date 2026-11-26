use axum::{
    Json,
    extract::{Path, State, rejection::JsonRejection},
};
use serde::Serialize;
use serde_json::Value;
use uuid::Uuid;

use crate::{
    error::AppError,
    models::{
        audio_asset::AudioAsset,
        job::ProcessingJob,
        project::{CreateProject, Project},
        stem::ProjectStem,
    },
    state::AppState,
};

#[derive(Serialize)]
pub struct ProjectDetails {
    project: Project,
    audio_assets: Vec<AudioAsset>,
    jobs: Vec<ProcessingJob>,
    stems: Vec<ProjectStemView>,
    beat_grid: Option<Value>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectStemView {
    name: String,
    audio_url: String,
    sample_rate: i32,
    channels: i16,
    frames: i64,
    duration_seconds: f64,
    file_size_bytes: i64,
}

#[utoipa::path(post, path = "/api/v1/projects", responses((status = 201, description = "Project created")))]
pub async fn create_project(
    State(state): State<AppState>,
    body: Result<Json<CreateProject>, JsonRejection>,
) -> Result<Json<Project>, AppError> {
    let Json(input) = body.map_err(|_| AppError::BadRequest("Invalid JSON request body".into()))?;
    let name = input.name.trim();
    if name.is_empty() || name.chars().count() > 200 {
        return Err(AppError::BadRequest(
            "name must contain 1 to 200 characters".into(),
        ));
    }
    if input
        .description
        .as_ref()
        .is_some_and(|value| value.len() > 10_000)
    {
        return Err(AppError::BadRequest(
            "description must be at most 10000 bytes".into(),
        ));
    }
    let project = sqlx::query_as::<_, Project>("INSERT INTO projects (name, description) VALUES ($1, $2) RETURNING id, name, description, created_at, updated_at")
        .bind(name).bind(input.description.map(|value| value.trim().to_owned())).fetch_one(&state.db).await?;
    Ok(Json(project))
}

#[utoipa::path(get, path = "/api/v1/projects/{project_id}", params(("project_id" = Uuid, Path, description = "Project UUID")), responses((status = 200, description = "Project and related metadata"), (status = 404, description = "Project not found")))]
pub async fn get_project(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<Json<ProjectDetails>, AppError> {
    let project_id = Uuid::parse_str(&project_id)
        .map_err(|_| AppError::BadRequest("project_id must be a UUID".into()))?;
    let project = sqlx::query_as::<_, Project>(
        "SELECT id, name, description, created_at, updated_at FROM projects WHERE id = $1",
    )
    .bind(project_id)
    .fetch_optional(&state.db)
    .await?
    .ok_or(AppError::NotFound)?;
    let audio_assets = sqlx::query_as::<_, AudioAsset>("SELECT id, project_id, original_filename, content_type, stored_path, file_size_bytes, sample_rate, channels, frame_count, duration_seconds, created_at, updated_at, expires_at FROM audio_assets WHERE project_id = $1 ORDER BY created_at")
        .bind(project_id).fetch_all(&state.db).await?;
    let jobs = sqlx::query_as::<_, ProcessingJob>(
        "SELECT * FROM processing_jobs WHERE project_id = $1 ORDER BY created_at DESC",
    )
    .bind(project_id)
    .fetch_all(&state.db)
    .await?;
    let stem_rows = sqlx::query_as::<_, ProjectStem>("SELECT project_id,job_id,stem_name,storage_key,sample_rate,channels,frame_count,file_size_bytes,created_at FROM project_stems WHERE project_id=$1 ORDER BY stem_name")
        .bind(project_id).fetch_all(&state.db).await?;
    let stems = stem_rows
        .into_iter()
        .map(|stem| ProjectStemView {
            audio_url: format!("/api/v1/projects/{project_id}/stems/{}", stem.stem_name),
            name: stem.stem_name,
            sample_rate: stem.sample_rate,
            channels: stem.channels,
            frames: stem.frame_count,
            duration_seconds: stem.frame_count as f64 / f64::from(stem.sample_rate),
            file_size_bytes: stem.file_size_bytes,
        })
        .collect();
    let beat_grid = sqlx::query_scalar::<_, Value>("SELECT ba.result FROM beat_analyses ba JOIN processing_jobs j ON j.id=ba.separation_job_id WHERE j.project_id=$1 AND j.status='SUCCEEDED' AND ba.status='SUCCEEDED' ORDER BY ba.finished_at DESC NULLS LAST, ba.created_at DESC LIMIT 1")
        .bind(project_id).fetch_optional(&state.db).await?;
    Ok(Json(ProjectDetails {
        project,
        audio_assets,
        jobs,
        stems,
        beat_grid,
    }))
}

#[utoipa::path(get, path = "/api/v1/projects/{project_id}/analysis", params(("project_id" = Uuid, Path, description = "Project UUID")), responses((status = 200, description = "Most recent saved project beat grid"), (status = 404, description = "Project or analysis not found")))]
pub async fn get_project_analysis(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<Json<Value>, AppError> {
    let project_id = Uuid::parse_str(&project_id)
        .map_err(|_| AppError::BadRequest("project_id must be a UUID".into()))?;
    let result = sqlx::query_scalar::<_, Value>("SELECT ba.result FROM beat_analyses ba JOIN processing_jobs j ON j.id=ba.separation_job_id WHERE j.project_id=$1 AND j.status='SUCCEEDED' AND ba.status='SUCCEEDED' ORDER BY ba.finished_at DESC NULLS LAST, ba.created_at DESC LIMIT 1")
        .bind(project_id).fetch_optional(&state.db).await?.ok_or(AppError::NotFound)?;
    Ok(Json(result))
}
