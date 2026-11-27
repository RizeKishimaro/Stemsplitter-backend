pub mod audio;
pub mod beat_analysis;
pub mod health;
pub mod jobs;
pub mod projects;
pub mod stems;

use axum::{
    Router,
    extract::DefaultBodyLimit,
    routing::{get, post},
};
use tower_http::trace::TraceLayer;
use utoipa::OpenApi;

use crate::{services::job_manager::JobEvent, state::AppState};

#[derive(OpenApi)]
#[openapi(
    info(
        title = "DubPad API",
        version = "1.0.0",
        description = "API for DubPad projects, local audio uploads, HTDemucs stem separation, and beat analysis."
    ),
    paths(
        health::health,
        projects::create_project,
        projects::get_project,
        projects::get_project_analysis,
        audio::upload_audio,
        audio::list_audio,
        jobs::create_job,
        jobs::get_job,
        jobs::cancel_job,
        jobs::job_events,
        stems::list_stems,
        stems::get_stem,
        stems::get_partial_stem,
        stems::get_project_stem,
        beat_analysis::create_beat_analysis,
        beat_analysis::get_beat_analysis,
        beat_analysis::get_job_analysis
    ),
    components(schemas(crate::models::beat_analysis::BeatAnalysisJob))
)]
pub struct ApiDoc;

pub fn router(state: AppState, max_upload_bytes: usize) -> Router {
    let upload_limit = max_upload_bytes.saturating_add(128 * 1024);
    Router::new()
        .route("/health", get(health::health))
        .route("/projects", post(projects::create_project))
        .route("/projects/{project_id}", get(projects::get_project))
        .route(
            "/projects/{project_id}/analysis",
            get(projects::get_project_analysis),
        )
        .route(
            "/projects/{project_id}/stems/{stem_name}",
            get(stems::get_project_stem),
        )
        .route(
            "/projects/{project_id}/audio",
            post(audio::upload_audio)
                .get(audio::list_audio)
                .layer(DefaultBodyLimit::max(upload_limit)),
        )
        .route("/projects/{project_id}/jobs", post(jobs::create_job))
        .route("/jobs/{job_id}", get(jobs::get_job))
        .route("/jobs/{job_id}/cancel", post(jobs::cancel_job))
        .route("/jobs/{job_id}/events", get(jobs::job_events))
        .route("/jobs/{job_id}/stems", get(stems::list_stems))
        .route("/jobs/{job_id}/stems/{stem_name}", get(stems::get_stem))
        .route("/jobs/{job_id}/stems/{stem_name}/partial", get(stems::get_partial_stem))
        .route(
            "/jobs/{job_id}/beat-analysis",
            post(beat_analysis::create_beat_analysis).get(beat_analysis::get_beat_analysis),
        )
        .route(
            "/jobs/{job_id}/analysis",
            get(beat_analysis::get_job_analysis),
        )
        .with_state(state)
        .layer(TraceLayer::new_for_http())
}

pub fn event_channel() -> tokio::sync::broadcast::Sender<JobEvent> {
    let (sender, _) = tokio::sync::broadcast::channel(256);
    sender
}

#[cfg(test)]
mod tests {
    use super::ApiDoc;
    use utoipa::OpenApi;

    #[test]
    fn openapi_has_dubpad_branding_and_separate_beat_analysis_paths() {
        let document = ApiDoc::openapi();
        assert_eq!(document.info.title, "DubPad API");
        assert!(
            document
                .paths
                .paths
                .contains_key("/api/v1/jobs/{job_id}/beat-analysis")
        );
        assert!(
            document
                .paths
                .paths
                .contains_key("/api/v1/jobs/{job_id}/analysis")
        );
        assert!(
            document
                .paths
                .paths
                .contains_key("/api/v1/projects/{project_id}/analysis")
        );
        assert!(
            document
                .paths
                .paths
                .contains_key("/api/v1/projects/{project_id}/stems/{stem_name}")
        );
        assert!(
            !document
                .paths
                .paths
                .contains_key("/api/v1/jobs/{job_id}/chunks")
        );
    }
}
