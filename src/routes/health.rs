use axum::{Json, extract::State};
use serde::Serialize;

use crate::{error::AppError, state::AppState};

#[derive(Serialize)]
pub struct HealthResponse {
    status: &'static str,
    database: &'static str,
}

#[utoipa::path(get, path = "/api/v1/health", responses((status = 200, description = "Database-backed service health")))]
pub async fn health(State(state): State<AppState>) -> Result<Json<HealthResponse>, AppError> {
    sqlx::query("SELECT 1").execute(&state.db).await?;
    Ok(Json(HealthResponse {
        status: "ok",
        database: "ok",
    }))
}
