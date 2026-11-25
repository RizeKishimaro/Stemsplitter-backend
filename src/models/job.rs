use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::FromRow;
use uuid::Uuid;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum JobStatus {
    Queued,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

impl JobStatus {
    pub fn as_db(&self) -> &'static str {
        match self {
            Self::Queued => "QUEUED",
            Self::Running => "RUNNING",
            Self::Succeeded => "SUCCEEDED",
            Self::Failed => "FAILED",
            Self::Cancelled => "CANCELLED",
        }
    }
    pub fn can_cancel(&self) -> bool {
        matches!(self, Self::Queued | Self::Running)
    }
    pub fn can_transition_to(&self, next: &Self) -> bool {
        matches!(
            (self, next),
            (Self::Queued, Self::Running | Self::Cancelled)
                | (
                    Self::Running,
                    Self::Succeeded | Self::Failed | Self::Cancelled
                )
        )
    }
}

#[derive(Clone, Debug, Serialize, FromRow)]
pub struct ProcessingJob {
    pub id: Uuid,
    pub project_id: Uuid,
    pub audio_asset_id: Uuid,
    pub job_type: String,
    pub status: String,
    pub progress: f64,
    pub error_message: Option<String>,
    pub cancel_requested: bool,
    pub request_metadata: Value,
    pub result_metadata: Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub finished_at: Option<DateTime<Utc>>,
    pub expires_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct CreateJob {
    pub audio_asset_id: Uuid,
    #[serde(default = "default_job_type")]
    pub job_type: String,
    /// Mix-first analyzes the original WAV before HTDemucs; drums-first analyzes HTDemucs drums.
    #[serde(default = "default_analysis_strategy")]
    pub analysis_strategy: String,
    #[serde(default = "default_beats_per_bar")]
    pub beats_per_bar: u8,
}

fn default_job_type() -> String {
    "separate".into()
}

fn default_analysis_strategy() -> String {
    "drums_first".into()
}
fn default_beats_per_bar() -> u8 {
    4
}

#[cfg(test)]
mod tests {
    use super::{CreateJob, JobStatus};

    #[test]
    fn separation_strategy_and_meter_are_additive_with_safe_defaults() {
        let job: CreateJob = serde_json::from_value(
            serde_json::json!({"audio_asset_id":"00000000-0000-4000-8000-000000000000"}),
        )
        .unwrap();
        assert_eq!(job.analysis_strategy, "drums_first");
        assert_eq!(job.beats_per_bar, 4);
        let configured: CreateJob = serde_json::from_value(serde_json::json!({"audio_asset_id":"00000000-0000-4000-8000-000000000000","analysis_strategy":"mix_first","beats_per_bar":3})).unwrap();
        assert_eq!(configured.analysis_strategy, "mix_first");
        assert_eq!(configured.beats_per_bar, 3);
    }

    #[test]
    fn job_lifecycle_allows_only_forward_transitions() {
        assert!(JobStatus::Queued.can_transition_to(&JobStatus::Running));
        assert!(JobStatus::Queued.can_transition_to(&JobStatus::Cancelled));
        assert!(JobStatus::Running.can_transition_to(&JobStatus::Failed));
        assert!(JobStatus::Running.can_transition_to(&JobStatus::Succeeded));
        assert!(!JobStatus::Succeeded.can_transition_to(&JobStatus::Running));
        assert!(!JobStatus::Cancelled.can_transition_to(&JobStatus::Succeeded));
    }

    #[test]
    fn cancellation_is_allowed_only_before_terminal_state() {
        assert!(JobStatus::Queued.can_cancel());
        assert!(JobStatus::Running.can_cancel());
        assert!(!JobStatus::Succeeded.can_cancel());
        assert!(!JobStatus::Failed.can_cancel());
        assert!(!JobStatus::Cancelled.can_cancel());
    }
}
