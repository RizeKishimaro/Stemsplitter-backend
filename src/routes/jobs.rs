use axum::{
    Json,
    extract::{
        Path, State, WebSocketUpgrade,
        rejection::JsonRejection,
        ws::{Message, WebSocket},
    },
    response::Response,
};
use chrono::{Duration as ChronoDuration, Utc};
use futures_util::StreamExt;
use serde_json::json;
use uuid::Uuid;

use crate::{
    error::AppError,
    models::job::{CreateJob, ProcessingJob},
    state::AppState,
};

#[utoipa::path(post, path = "/api/v1/projects/{project_id}/jobs", params(("project_id" = Uuid, Path, description = "Project UUID")), responses((status = 201, description = "Processing job created")))]
pub async fn create_job(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    body: Result<Json<CreateJob>, JsonRejection>,
) -> Result<(axum::http::StatusCode, Json<ProcessingJob>), AppError> {
    let project_id = Uuid::parse_str(&project_id)
        .map_err(|_| AppError::BadRequest("project_id must be a UUID".into()))?;
    if !crate::services::project_activity::touch_project(
        &state.db,
        project_id,
        state.config.project_ttl,
    )
    .await?
    {
        return Err(AppError::Gone);
    }
    let Json(input) = body.map_err(|_| AppError::BadRequest("Invalid JSON request body".into()))?;
    if input.job_type != "separate" {
        return Err(AppError::BadRequest("job_type must be 'separate'".into()));
    }
    if !["mix_first", "drums_first"].contains(&input.analysis_strategy.as_str()) {
        return Err(AppError::BadRequest(
            "analysis_strategy must be 'mix_first' or 'drums_first'".into(),
        ));
    }
    if !(2..=12).contains(&input.beats_per_bar) {
        return Err(AppError::BadRequest(
            "beats_per_bar must be between 2 and 12".into(),
        ));
    }
    let asset = sqlx::query_as::<_, (Uuid, i32, i16, i64)>(
        "SELECT project_id, sample_rate, channels, frame_count FROM audio_assets WHERE id = $1",
    )
    .bind(input.audio_asset_id)
    .fetch_optional(&state.db)
    .await?
    .ok_or(AppError::NotFound)?;
    if asset.0 != project_id {
        return Err(AppError::BadRequest(
            "audio asset does not belong to this project".into(),
        ));
    }
    let has_completed_stems = sqlx::query_scalar::<_, bool>(
        "SELECT EXISTS(SELECT 1 FROM project_stems WHERE project_id=$1)",
    )
    .bind(project_id)
    .fetch_one(&state.db)
    .await?;
    if has_completed_stems {
        return Err(AppError::Conflict(
            "this project already has a completed stem set; replacing it requires an explicit replacement workflow".into(),
        ));
    }
    let has_active_job = sqlx::query_scalar::<_, bool>(
        "SELECT EXISTS(SELECT 1 FROM processing_jobs WHERE project_id=$1 AND status IN ('QUEUED','RUNNING'))",
    )
    .bind(project_id)
    .fetch_one(&state.db)
    .await?;
    if has_active_job {
        return Err(AppError::Conflict(
            "this project already has an active separation job".into(),
        ));
    }
    let metadata = json!({
        "original_sample_rate": asset.1,
        "original_channels": asset.2,
        "original_frame_count": asset.3,
        "timing_authority": "integer_sample_positions",
        "inference_executor": "persistent_htdemucs_worker",
        "source_mapping_policy": "whole_input_identity_sample_range"
        ,"analysis_strategy": input.analysis_strategy
        ,"quantization": {"unit": "bar", "beats_per_bar": input.beats_per_bar}
    });
    let ttl = ChronoDuration::from_std(state.config.audio_job_ttl)
        .map_err(|_| AppError::Internal("audio TTL exceeds supported range".into()))?;
    let expires_at = Utc::now() + ttl;
    sqlx::query("UPDATE audio_assets SET expires_at = GREATEST(expires_at, $2), updated_at = now() WHERE id = $1")
        .bind(input.audio_asset_id).bind(expires_at).execute(&state.db).await?;
    let job = sqlx::query_as::<_, ProcessingJob>("INSERT INTO processing_jobs (project_id, audio_asset_id, job_type, request_metadata, expires_at) VALUES ($1,$2,$3,$4,$5) RETURNING *")
        .bind(project_id).bind(input.audio_asset_id).bind(input.job_type).bind(metadata).bind(expires_at).fetch_one(&state.db).await?;
    state.jobs.emit(&job, Some("Queued".into()));
    Ok((axum::http::StatusCode::CREATED, Json(job)))
}

#[utoipa::path(get, path = "/api/v1/jobs/{job_id}", params(("job_id" = Uuid, Path, description = "Job UUID")), responses((status = 200, description = "Processing job state")))]
pub async fn get_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<ProcessingJob>, AppError> {
    let job_id = Uuid::parse_str(&job_id)
        .map_err(|_| AppError::BadRequest("job_id must be a UUID".into()))?;
    let job = sqlx::query_as::<_, ProcessingJob>("SELECT * FROM processing_jobs WHERE id = $1")
        .bind(job_id)
        .fetch_optional(&state.db)
        .await?
        .ok_or(AppError::NotFound)?;
    Ok(Json(job))
}

#[utoipa::path(post, path = "/api/v1/jobs/{job_id}/cancel", params(("job_id" = Uuid, Path, description = "Job UUID")), responses((status = 200, description = "Job cancelled")))]
pub async fn cancel_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<ProcessingJob>, AppError> {
    let job_id = Uuid::parse_str(&job_id)
        .map_err(|_| AppError::BadRequest("job_id must be a UUID".into()))?;
    Ok(Json(state.jobs.cancel(job_id).await?))
}

#[utoipa::path(get, path = "/api/v1/jobs/{job_id}/events", params(("job_id" = Uuid, Path, description = "Job UUID")), responses((status = 101, description = "WebSocket upgraded for job events")))]
pub async fn job_events(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
    upgrade: WebSocketUpgrade,
) -> Result<Response, AppError> {
    let job_id = Uuid::parse_str(&job_id)
        .map_err(|_| AppError::BadRequest("job_id must be a UUID".into()))?;
    let exists =
        sqlx::query_scalar::<_, bool>("SELECT EXISTS(SELECT 1 FROM processing_jobs WHERE id = $1)")
            .bind(job_id)
            .fetch_one(&state.db)
            .await?;
    if !exists {
        return Err(AppError::NotFound);
    }
    Ok(upgrade.on_upgrade(move |socket| stream_events(socket, state, job_id)))
}

async fn stream_events(mut socket: WebSocket, state: AppState, job_id: Uuid) {
    let mut events = state.events.subscribe();
    if let Ok(Some(job)) =
        sqlx::query_as::<_, ProcessingJob>("SELECT * FROM processing_jobs WHERE id = $1")
            .bind(job_id)
            .fetch_optional(&state.db)
            .await
    {
        let initial = json!({ "job_id": job.id, "status": job.status, "progress": job.progress, "message": job.error_message });
        if socket
            .send(Message::Text(initial.to_string().into()))
            .await
            .is_err()
        {
            return;
        }
    }
    loop {
        tokio::select! {
            event = events.recv() => match event {
                Ok(event) if event.job_id == job_id => {
                    if socket.send(Message::Text(serde_json::to_string(&event).unwrap_or_default().into())).await.is_err() { break; }
                }
                Ok(_) | Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            },
            message = socket.next() => match message {
                Some(Ok(Message::Close(_))) | None | Some(Err(_)) => break,
                Some(Ok(Message::Ping(bytes))) => if socket.send(Message::Pong(bytes)).await.is_err() { break; },
                _ => {},
            }
        }
    }
}
