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
        project::{CreateProject, Project, SaveProjectState},
        stem::ProjectStem,
    },
    state::AppState,
};

#[derive(Serialize)]
pub struct ProjectDetails {
    project: Project,
    project_state: Value,
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

#[utoipa::path(get, path = "/api/v1/projects", responses((status = 200, description = "Active server-managed projects")))]
pub async fn list_projects(State(state): State<AppState>) -> Result<Json<Vec<Project>>, AppError> {
    let projects = sqlx::query_as::<_, Project>("SELECT id,name,description,created_at,updated_at,project_state,last_activity_at,expires_at FROM projects WHERE expires_at>now() AND deleting_at IS NULL ORDER BY last_activity_at DESC LIMIT 100")
        .fetch_all(&state.db).await?;
    Ok(Json(projects))
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
    let ttl = i64::try_from(state.config.project_ttl.as_secs()).unwrap_or(i64::MAX);
    let project = sqlx::query_as::<_, Project>("INSERT INTO projects (name, description, expires_at) VALUES ($1, $2, now()+($3 * interval '1 second')) RETURNING id,name,description,created_at,updated_at,project_state,last_activity_at,expires_at")
        .bind(name).bind(input.description.map(|value| value.trim().to_owned())).bind(ttl).fetch_one(&state.db).await?;
    Ok(Json(project))
}

#[utoipa::path(get, path = "/api/v1/projects/{project_id}", params(("project_id" = Uuid, Path, description = "Project UUID")), responses((status = 200, description = "Project and related metadata"), (status = 404, description = "Project not found")))]
pub async fn get_project(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<Json<ProjectDetails>, AppError> {
    let project_id = Uuid::parse_str(&project_id)
        .map_err(|_| AppError::BadRequest("project_id must be a UUID".into()))?;
    let active = crate::services::project_activity::touch_project(
        &state.db,
        project_id,
        state.config.project_ttl,
    )
    .await?;
    if !active {
        let exists =
            sqlx::query_scalar::<_, bool>("SELECT EXISTS(SELECT 1 FROM projects WHERE id=$1)")
                .bind(project_id)
                .fetch_one(&state.db)
                .await?;
        return Err(if exists {
            AppError::Gone
        } else {
            AppError::NotFound
        });
    }
    let project = sqlx::query_as::<_, Project>("SELECT id,name,description,created_at,updated_at,project_state,last_activity_at,expires_at FROM projects WHERE id=$1 AND deleting_at IS NULL")
        .bind(project_id).fetch_optional(&state.db).await?.ok_or(AppError::Gone)?;
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
        project_state: project.project_state.clone(),
        project,
        audio_assets,
        jobs,
        stems,
        beat_grid,
    }))
}

#[utoipa::path(put, path = "/api/v1/projects/{project_id}", params(("project_id" = Uuid, Path, description = "Project UUID")), request_body = SaveProjectState, responses((status = 200, description = "Project arrangement saved"), (status = 410, description = "Project expired")))]
pub async fn save_project_state(
    State(state): State<AppState>,
    Path(raw_project): Path<String>,
    body: Result<Json<SaveProjectState>, JsonRejection>,
) -> Result<Json<Project>, AppError> {
    let project_id = Uuid::parse_str(&raw_project)
        .map_err(|_| AppError::BadRequest("project_id must be a UUID".into()))?;
    let Json(input) = body.map_err(|_| AppError::BadRequest("Invalid JSON request body".into()))?;
    validate_project_state(&input.project_state)?;
    let state_bytes = serde_json::to_vec(&input.project_state)
        .map_err(|_| AppError::BadRequest("project_state is not valid JSON".into()))?;
    if state_bytes.len() > 8 * 1024 * 1024 {
        return Err(AppError::BadRequest("project_state exceeds 8 MiB".into()));
    }
    let ttl = i64::try_from(state.config.project_ttl.as_secs()).unwrap_or(i64::MAX);
    let project = sqlx::query_as::<_, Project>("UPDATE projects SET project_state=$2,last_activity_at=now(),expires_at=now()+($3 * interval '1 second'),updated_at=now() WHERE id=$1 AND expires_at>now() AND deleting_at IS NULL RETURNING id,name,description,created_at,updated_at,project_state,last_activity_at,expires_at")
        .bind(project_id).bind(input.project_state).bind(ttl).fetch_optional(&state.db).await?.ok_or(AppError::Gone)?;
    Ok(Json(project))
}

fn validate_project_state(value: &Value) -> Result<(), AppError> {
    let object = value
        .as_object()
        .ok_or_else(|| AppError::BadRequest("project_state must be a JSON object".into()))?;
    if object.keys().any(|key| {
        matches!(
            key.as_str(),
            "playhead" | "playheadPosition" | "hover" | "hoveredClipId" | "temporarySelection"
        )
    }) {
        return Err(AppError::BadRequest(
            "project_state must not contain transient playback or hover state".into(),
        ));
    }
    if let Some(clips) = object.get("clips") {
        let count = clips
            .as_object()
            .map(serde_json::Map::len)
            .or_else(|| clips.as_array().map(Vec::len))
            .ok_or_else(|| AppError::BadRequest("clips must be an object or array".into()))?;
        if count > 100_000 {
            return Err(AppError::BadRequest(
                "project_state contains too many clips".into(),
            ));
        }
    }
    if let Some(sections) = object.get("sections") {
        if sections
            .as_array()
            .is_none_or(|items| items.len() > 200_000)
        {
            return Err(AppError::BadRequest(
                "sections must be an array with at most 200000 entries".into(),
            ));
        }
    }
    Ok(())
}

#[utoipa::path(get, path = "/api/v1/projects/{project_id}/analysis", params(("project_id" = Uuid, Path, description = "Project UUID")), responses((status = 200, description = "Most recent saved project beat grid"), (status = 404, description = "Project or analysis not found")))]
pub async fn get_project_analysis(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<Json<Value>, AppError> {
    let project_id = Uuid::parse_str(&project_id)
        .map_err(|_| AppError::BadRequest("project_id must be a UUID".into()))?;
    let active = crate::services::project_activity::touch_project(
        &state.db,
        project_id,
        state.config.project_ttl,
    )
    .await?;
    if !active {
        return Err(AppError::Gone);
    }
    let result = sqlx::query_scalar::<_, Value>("SELECT ba.result FROM beat_analyses ba JOIN processing_jobs j ON j.id=ba.separation_job_id WHERE j.project_id=$1 AND j.status='SUCCEEDED' AND ba.status='SUCCEEDED' ORDER BY ba.finished_at DESC NULLS LAST, ba.created_at DESC LIMIT 1")
        .bind(project_id).fetch_optional(&state.db).await?.ok_or(AppError::NotFound)?;
    Ok(Json(result))
}
