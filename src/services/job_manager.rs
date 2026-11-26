use std::{sync::Arc, time::Duration};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sqlx::PgPool;
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, Command},
    sync::{broadcast, mpsc},
    task::JoinHandle,
    time::{self, Instant},
};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{config::Config, error::AppError, models::job::ProcessingJob};

#[derive(Clone, Debug, Serialize)]
pub struct JobEvent {
    pub event: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage: Option<String>,
    pub stem: Option<String>,
    pub stem_metadata: Option<Value>,
    pub chunk: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub analysis: Option<Value>,
    pub job_id: Uuid,
    pub status: String,
    pub progress: f64,
    pub message: Option<String>,
}

#[derive(Deserialize)]
struct WorkerMessage {
    #[serde(rename = "type")]
    kind: String,
    job_id: Option<Uuid>,
    progress: Option<f64>,
    message: Option<String>,
    acknowledged: Option<bool>,
    result: Option<Value>,
    device: Option<String>,
    cuda_available: Option<bool>,
    gpu_name: Option<String>,
    stem: Option<String>,
    stem_metadata: Option<Value>,
}

struct WorkerProcess {
    child: Child,
    stdin: ChildStdin,
    messages: mpsc::UnboundedReceiver<String>,
    _reader: JoinHandle<()>,
}

impl WorkerProcess {
    async fn start(config: &Config) -> Result<Self, String> {
        let mut child = Command::new(&config.demucs_python)
            .arg(&config.demucs_worker_script)
            .env("HF_HUB_OFFLINE", "1")
            .env("TRANSFORMERS_OFFLINE", "1")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|error| format!("failed to start Demucs worker: {error}"))?;
        let stdin = child.stdin.take().ok_or("Demucs stdin unavailable")?;
        let stdout = child.stdout.take().ok_or("Demucs stdout unavailable")?;
        let stderr = child.stderr.take().ok_or("Demucs stderr unavailable")?;
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                tracing::debug!(target: "demucs_worker", "{line}");
            }
        });
        let (tx, messages) = mpsc::unbounded_channel();
        let reader = tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        let mut worker = Self {
            child,
            stdin,
            messages,
            _reader: reader,
        };
        let startup = time::timeout(config.demucs_startup_timeout, worker.messages.recv())
            .await
            .map_err(|_| "timed out waiting for Demucs worker startup".to_string())?
            .ok_or("Demucs worker exited before startup".to_string())?;
        let message: WorkerMessage = serde_json::from_str(&startup)
            .map_err(|e| format!("invalid worker startup response: {e}"))?;
        if message.kind != "ready" {
            return Err(format!(
                "{} (CUDA available: {}, GPU: {})",
                message
                    .message
                    .unwrap_or_else(|| format!("worker startup failed ({})", message.kind)),
                message.cuda_available.unwrap_or(false),
                message.gpu_name.as_deref().unwrap_or("none")
            ));
        }
        tracing::info!(
            device = message.device.as_deref().unwrap_or("unknown"),
            cuda_available = message.cuda_available.unwrap_or(false),
            gpu = message.gpu_name.as_deref().unwrap_or("none"),
            "HTDemucs worker ready"
        );
        Ok(worker)
    }

    async fn send(&mut self, value: &Value) -> Result<(), String> {
        self.stdin
            .write_all(value.to_string().as_bytes())
            .await
            .map_err(|e| e.to_string())?;
        self.stdin
            .write_all(b"\n")
            .await
            .map_err(|e| e.to_string())?;
        self.stdin.flush().await.map_err(|e| e.to_string())
    }

    async fn kill(&mut self) {
        let _ = self.child.kill().await;
        let _ = self.child.wait().await;
    }
}

pub struct JobManager {
    db: PgPool,
    events: broadcast::Sender<JobEvent>,
    poll_interval: Duration,
    config: Arc<Config>,
    cancel_tx: mpsc::UnboundedSender<Uuid>,
    cancel_rx: tokio::sync::Mutex<Option<mpsc::UnboundedReceiver<Uuid>>>,
}

impl JobManager {
    pub fn new(
        db: PgPool,
        events: broadcast::Sender<JobEvent>,
        poll_interval: Duration,
        config: Arc<Config>,
    ) -> Self {
        let (cancel_tx, cancel_rx) = mpsc::unbounded_channel();
        Self {
            db,
            events,
            poll_interval,
            config,
            cancel_tx,
            cancel_rx: tokio::sync::Mutex::new(Some(cancel_rx)),
        }
    }

    /// On API restart, RUNNING means the process died without a durable result;
    /// requeue it after stage files are isolated per attempt.
    pub async fn recover_interrupted(&self) -> Result<u64, AppError> {
        let retried = sqlx::query("UPDATE processing_jobs SET status = 'QUEUED', cancel_requested = false, error_message = 'Worker interrupted; queued for retry on startup', updated_at = now() WHERE status = 'RUNNING' AND NOT cancel_requested")
            .execute(&self.db).await?.rows_affected();
        let cancelled_without_ack = sqlx::query("UPDATE processing_jobs SET status = 'FAILED', error_message = 'Worker interrupted before acknowledging cancellation', finished_at = now(), updated_at = now() WHERE status = 'RUNNING' AND cancel_requested")
            .execute(&self.db).await?.rows_affected();
        Ok(retried + cancelled_without_ack)
    }

    pub async fn cancel(&self, job_id: Uuid) -> Result<ProcessingJob, AppError> {
        if let Some(job) = sqlx::query_as::<_, ProcessingJob>("UPDATE processing_jobs SET status = 'CANCELLED', cancel_requested = true, finished_at = now(), updated_at = now() WHERE id = $1 AND status = 'QUEUED' RETURNING *")
            .bind(job_id).fetch_optional(&self.db).await? {
            self.emit(&job, Some("Cancelled before worker start".into()));
            return Ok(job);
        }
        let job = sqlx::query_as::<_, ProcessingJob>("UPDATE processing_jobs SET cancel_requested = true, updated_at = now() WHERE id = $1 AND status = 'RUNNING' RETURNING *")
            .bind(job_id).fetch_optional(&self.db).await?.ok_or(AppError::NotFound)?;
        let _ = self.cancel_tx.send(job_id);
        self.emit(
            &job,
            Some("Cancellation requested; waiting for worker acknowledgement".into()),
        );
        Ok(job)
    }

    pub async fn run(self: Arc<Self>, shutdown: CancellationToken) {
        let mut cancel_rx = self
            .cancel_rx
            .lock()
            .await
            .take()
            .expect("job manager runs once");
        let mut worker: Option<WorkerProcess> = None;
        let mut restart_attempt = 0_u32;
        let mut last_expiry_sweep = Instant::now() - Duration::from_secs(61);
        loop {
            if shutdown.is_cancelled() {
                break;
            }
            if last_expiry_sweep.elapsed() >= Duration::from_secs(60) {
                if let Err(error) = crate::services::storage::cleanup_expired_audio(
                    &self.db,
                    &self.config.upload_dir,
                )
                .await
                {
                    tracing::warn!(%error, "temporary audio expiry sweep failed");
                }
                last_expiry_sweep = Instant::now();
            }
            if worker.is_none() {
                match WorkerProcess::start(&self.config).await {
                    Ok(process) => {
                        worker = Some(process);
                        tracing::info!(
                            "persistent HTDemucs worker is available; queue dispatch enabled"
                        );
                    }
                    Err(error) => {
                        restart_attempt = restart_attempt.saturating_add(1);
                        let restart_delay = worker_restart_delay(restart_attempt);
                        tracing::error!(error = %error, attempt = restart_attempt, retry_in_ms = restart_delay.as_millis(), "Demucs worker unavailable; jobs remain durable and queued");
                        tokio::select! { _ = shutdown.cancelled() => break, _ = time::sleep(restart_delay) => {} }
                        continue;
                    }
                }
            }
            tokio::select! {
                _ = shutdown.cancelled() => break,
                Some(job_id) = cancel_rx.recv() => {
                    if let Some(process) = worker.as_mut() { let _ = process.send(&json!({"type":"cancel","job_id":job_id})).await; }
                }
                _ = time::sleep(self.poll_interval) => {
                    if let Some(process) = worker.as_mut() {
                        match process.child.try_wait() {
                            Ok(Some(status)) => {
                                restart_attempt = restart_attempt.saturating_add(1);
                                let restart_delay = worker_restart_delay(restart_attempt);
                                tracing::error!(%status, attempt = restart_attempt, retry_in_ms = restart_delay.as_millis(), "Demucs worker exited unexpectedly; restarting");
                                worker = None;
                                tokio::select! { _ = shutdown.cancelled() => break, _ = time::sleep(restart_delay) => {} }
                                continue;
                            }
                            Err(error) => tracing::warn!(%error, "could not check Demucs worker process"),
                            Ok(None) => {}
                        }
                    }
                        match self.claim().await {
                            Ok(Some(job)) => {
                                tracing::info!(job_id = %job.id, "claimed queued separation job");
                                let job_id = job.id;
                                if let Some(process) = worker.as_mut() {
                                    if self.execute(job, process, &mut cancel_rx, &shutdown).await.is_err() {
                                        worker.as_mut().expect("worker exists").kill().await;
                                        worker = None;
                                        restart_attempt = restart_attempt.saturating_add(1);
                                        let restart_delay = worker_restart_delay(restart_attempt);
                                        tracing::error!(job_id = %job_id, attempt = restart_attempt, retry_in_ms = restart_delay.as_millis(), "worker communication failed; restarting persistent worker with backoff");
                                        tokio::select! { _ = shutdown.cancelled() => break, _ = time::sleep(restart_delay) => {} }
                                    } else {
                                        restart_attempt = 0;
                                    }
                                }
                            }
                        Ok(None) => tracing::trace!("job queue poll found no claimable jobs"),
                        Err(error) => tracing::error!(error = %error, "job polling/claim transaction failed; queued jobs remain eligible for retry"),
                    }
                }
            }
        }
        if let Some(mut process) = worker {
            process.kill().await;
        }
    }

    async fn claim(&self) -> Result<Option<ProcessingJob>, sqlx::Error> {
        let job = claim_queued(&self.db, None).await?;
        if let Some(job) = &job {
            self.emit(job, Some("Persistent HTDemucs worker accepted job".into()));
        }
        Ok(job)
    }

    async fn execute(
        &self,
        job: ProcessingJob,
        process: &mut WorkerProcess,
        cancel_rx: &mut mpsc::UnboundedReceiver<Uuid>,
        shutdown: &CancellationToken,
    ) -> Result<(), ()> {
        let asset = sqlx::query_as::<_, (String, String, i32, i16, i64)>("SELECT stored_path, original_filename, sample_rate, channels, frame_count FROM audio_assets WHERE id = $1")
            .bind(job.audio_asset_id).fetch_one(&self.db).await;
        let (stored_path, original_filename, sample_rate, channels, frame_count) = match asset {
            Ok(value) => value,
            Err(error) => {
                self.fail(job.id, format!("audio asset lookup failed: {error}"))
                    .await;
                return Ok(());
            }
        };
        let stage = self
            .config
            .upload_dir
            .join("..")
            .join("jobs")
            .join(job.id.to_string())
            .join(format!("attempt-{}", Uuid::new_v4()));
        if let Err(error) = tokio::fs::create_dir_all(&stage).await {
            self.fail(
                job.id,
                format!("could not create isolated worker output directory: {error}"),
            )
            .await;
            return Ok(());
        }
        if job
            .request_metadata
            .get("analysis_strategy")
            .and_then(Value::as_str)
            == Some("mix_first")
        {
            self.emit_named(
                job.id,
                "beat_analysis.started",
                "Analyzing original mix before separation",
            )
            .await;
            if let Err(error) = sqlx::query("INSERT INTO beat_analyses (separation_job_id,status,input_stem,sample_rate,channels,frame_count,model_version) VALUES ($1,'RUNNING','mix',$2,$3,$4,$5)")
                .bind(job.id).bind(sample_rate).bind(channels).bind(frame_count).bind(&self.config.beat_this_model_version).execute(&self.db).await {
                self.fail(job.id, format!("could not create mix analysis state: {error}")).await; return Ok(());
            }
            let mix_path = std::path::PathBuf::from(&stored_path);
            let mel = self.config.beat_this_mel_model.clone();
            let model = self.config.beat_this_model.clone();
            let version = self.config.beat_this_model_version.clone();
            let mix_analysis = tokio::task::spawn_blocking(move || {
                crate::routes::beat_analysis::analyze_wav(
                    &mix_path,
                    sample_rate as u32,
                    channels as u16,
                    frame_count as u64,
                    "mix",
                    &mel,
                    &model,
                    version,
                )
            })
            .await;
            let (analysis, elapsed_ms) = match mix_analysis {
                Ok(Ok(value)) => value,
                Ok(Err(error)) => {
                    let _ = sqlx::query("UPDATE beat_analyses SET status='FAILED',error_message=$2,finished_at=now(),updated_at=now() WHERE separation_job_id=$1 AND input_stem='mix'").bind(job.id).bind(&error).execute(&self.db).await;
                    self.fail(
                        job.id,
                        format!("mix-first Beat This analysis failed: {error}"),
                    )
                    .await;
                    return Ok(());
                }
                Err(error) => {
                    let _ = sqlx::query("UPDATE beat_analyses SET status='FAILED',error_message=$2,finished_at=now(),updated_at=now() WHERE separation_job_id=$1 AND input_stem='mix'").bind(job.id).bind(error.to_string()).execute(&self.db).await;
                    self.fail(job.id, format!("mix-first Beat This task failed: {error}"))
                        .await;
                    return Ok(());
                }
            };
            if let Err(error) = sqlx::query("UPDATE beat_analyses SET status='SUCCEEDED',result=$2,error_message=NULL,finished_at=now(),updated_at=now() WHERE separation_job_id=$1 AND input_stem='mix'")
                .bind(job.id).bind(&analysis).execute(&self.db).await {
                self.fail(job.id, format!("could not persist mix analysis: {error}")).await; return Ok(());
            }
            self.stage_progress(
                job.id,
                "BEAT_ANALYSIS",
                20.0,
                format!("Original mix beat analysis completed ({elapsed_ms} ms)"),
            )
            .await;
        }
        self.stage_progress(
            job.id,
            "STEM_SEPARATION",
            1.0,
            "Submitting audio to the persistent HTDemucs worker".into(),
        )
        .await;
        let command = json!({"type":"job","job_id":job.id,"input_path":stored_path,"output_dir":stage,"expected":{"sample_rate":sample_rate,"channels":channels,"frame_count":frame_count}});
        if let Err(error) = process.send(&command).await {
            self.fail(job.id, format!("worker command failed: {error}"))
                .await;
            return Err(());
        }
        let mut cancel_sent_at: Option<Instant> = None;
        let mut shutdown_cancel_sent = false;
        loop {
            tokio::select! {
                _ = shutdown.cancelled(), if !shutdown_cancel_sent => {
                    let _ = process.send(&json!({"type":"cancel","job_id":job.id})).await;
                    cancel_sent_at = Some(Instant::now());
                    shutdown_cancel_sent = true;
                }
                Some(cancelled_id) = cancel_rx.recv() => {
                    if cancelled_id == job.id {
                        let _ = process.send(&json!({"type":"cancel","job_id":job.id})).await;
                        cancel_sent_at = Some(Instant::now());
                    }
                }
                _ = async { if let Some(at) = cancel_sent_at { time::sleep_until(at + self.config.demucs_cancel_timeout).await } else { std::future::pending().await } } => {
                    process.kill().await;
                    // No worker acknowledgement: cancellation is not claimed. Mark failed and require a retry.
                    self.fail(job.id, "Worker did not acknowledge cancellation before timeout; process was killed".into()).await;
                    return Err(());
                }
                line = process.messages.recv() => {
                    let Some(line) = line else { self.fail(job.id, "Demucs worker exited unexpectedly".into()).await; return Err(()); };
                    let message: WorkerMessage = match serde_json::from_str(&line) { Ok(value) => value, Err(error) => { tracing::warn!(%error, "invalid Demucs worker message"); continue; } };
                    match message.kind.as_str() {
                        "progress" if message.job_id == Some(job.id) => {
                            self.progress(job.id,message.progress.unwrap_or(0.0)).await;
                        }
                        "stem.completed" if message.job_id == Some(job.id) => {
                            tracing::info!(job_id = %job.id, stem = message.stem.as_deref().unwrap_or("unknown"), "worker finished a staging stem; project asset is not published until final transaction commits");
                        }
                        "cancelled" if message.job_id == Some(job.id) && message.acknowledged == Some(true) =>{
                            self.cancel_acknowledged(job.id).await;
                            return Ok(());
                        }
                        "completed" if message.job_id == Some(job.id) => {
                            tracing::info!(job_id = %job.id, "HTDemucs completed message received; entering finalization");
                            let result = message.result.unwrap_or(Value::Null);
                            let finalization_started = Instant::now();
                            if let Err(error) = self.finalize_stems(&job, result, &stage, &stored_path, &original_filename).await {
                                tracing::error!(job_id = %job.id, elapsed_ms = finalization_started.elapsed().as_millis(), error = %error, "post-HTDemucs finalization failed");
                                self.fail(job.id, error).await;
                            } else {
                                tracing::info!(job_id = %job.id, elapsed_ms = finalization_started.elapsed().as_millis(), "post-HTDemucs finalization returned");
                                if sqlx::query_scalar::<_, bool>("SELECT EXISTS(SELECT 1 FROM processing_jobs WHERE id=$1 AND status='SUCCEEDED')")
                                    .bind(job.id).fetch_one(&self.db).await.unwrap_or(false) {
                                    if let Err(error) = tokio::fs::remove_dir_all(&stage).await {
                                        tracing::warn!(job_id = %job.id, path = %stage.display(), %error, "could not remove completed attempt staging directory");
                                    }
                                }
                            }
                            return Ok(());
                        }
                        "error" if message.job_id == Some(job.id) =>{
                            self.fail(job.id,message.message.unwrap_or_else(|| "Demucs Inference Failed because of something!".into())).await;
                            return Ok(())
                        }
                        _=> tracing::debug!(kind = %message.kind, "ignore worker messages!")
                    }
                }
            }
        }
    }

    async fn progress(&self, id: Uuid, progress: f64) {
        let value = worker_progress(progress);
        match sqlx::query_as::<_, ProcessingJob>("UPDATE processing_jobs SET progress = GREATEST(progress, $2), updated_at = now() WHERE id = $1 AND status = 'RUNNING' RETURNING *").bind(id).bind(value).fetch_optional(&self.db).await {
            Ok(Some(job)) => {
                let _ = self.events.send(JobEvent {
                    event: "job.progress".into(),
                    stage: Some("STEM_SEPARATION".into()),
                    stem: None,
                    stem_metadata: None,
                    chunk: None,
                    analysis: None,
                    job_id: job.id,
                    status: job.status,
                    progress: job.progress,
                    message: Some("HTDemucs inference progress".into()),
                });
            }
            Ok(None) => tracing::warn!(job_id = %id, "ignored HTDemucs progress because job is no longer RUNNING"),
            Err(error) => tracing::error!(job_id = %id, %error, "failed to persist HTDemucs progress"),
        }
    }

    async fn stage_progress(&self, id: Uuid, stage: &str, progress: f64, message: String) {
        match sqlx::query_as::<_, ProcessingJob>("UPDATE processing_jobs SET progress = GREATEST(progress, LEAST($2, 99.9)), updated_at = now() WHERE id = $1 AND status = 'RUNNING' RETURNING *")
            .bind(id).bind(progress.clamp(0.0, 99.9)).fetch_optional(&self.db).await {
            Ok(Some(job)) => {
                let _ = self.events.send(JobEvent {
                    event: "job.progress".into(),
                    stage: Some(stage.into()),
                    stem: None,
                    stem_metadata: None,
                    chunk: None,
                    analysis: None,
                    job_id: job.id,
                    status: job.status,
                    progress: job.progress,
                    message: Some(message),
                });
            }
            Ok(None) => tracing::debug!(job_id = %id, stage, "stage update ignored because job is no longer RUNNING"),
            Err(error) => tracing::error!(job_id = %id, stage, %error, "failed to persist finalization progress"),
        }
    }

    async fn is_cancel_requested(&self, id: Uuid) -> bool {
        match sqlx::query_scalar::<_, bool>(
            "SELECT cancel_requested FROM processing_jobs WHERE id=$1 AND status='RUNNING'",
        )
        .bind(id)
        .fetch_optional(&self.db)
        .await
        {
            Ok(Some(requested)) => requested,
            Ok(None) => true,
            Err(error) => {
                tracing::error!(job_id = %id, %error, "could not check cancellation during finalization");
                true
            }
        }
    }

    async fn finalize_stems(
        &self,
        job: &ProcessingJob,
        result: Value,
        attempt_dir: &std::path::Path,
        original_path: &str,
        original_filename: &str,
    ) -> Result<(), String> {
        let mut newly_persisted = Vec::new();
        match self
            .finalize_stems_inner(
                job,
                result,
                attempt_dir,
                original_path,
                original_filename,
                &mut newly_persisted,
            )
            .await
        {
            Ok(true) => Ok(()),
            Ok(false) => {
                remove_new_assets(&newly_persisted).await;
                Ok(())
            }
            Err(error) => {
                let commit_check = sqlx::query_scalar::<_, bool>(
                    "SELECT EXISTS(SELECT 1 FROM processing_jobs j WHERE j.id=$1 AND j.status='SUCCEEDED' AND (SELECT count(*) FROM project_stems s WHERE s.job_id=j.id)=4 AND EXISTS(SELECT 1 FROM beat_analyses a WHERE a.separation_job_id=j.id AND a.status='SUCCEEDED' AND a.result IS NOT NULL))",
                )
                .bind(job.id)
                .fetch_one(&self.db)
                .await;
                let committed = match commit_check {
                    Ok(committed) => committed,
                    Err(check_error) => {
                        tracing::error!(job_id = %job.id, %error, %check_error, "could not establish whether project outputs committed; preserving published files for recovery");
                        return Err(format!(
                            "{error}; commit status could not be verified ({check_error}); published files were preserved"
                        ));
                    }
                };
                if committed {
                    tracing::warn!(job_id = %job.id, error = %error, "finalization returned an error after the database confirmed committed project outputs; preserving published assets");
                    if let Ok(Some(completed)) = sqlx::query_as::<_, ProcessingJob>(
                        "SELECT * FROM processing_jobs WHERE id=$1 AND status='SUCCEEDED'",
                    )
                    .bind(job.id)
                    .fetch_optional(&self.db)
                    .await
                    {
                        self.emit(
                            &completed,
                            Some("Full-length stems and beat grid completed".into()),
                        );
                    }
                    return Ok(());
                }
                remove_new_assets(&newly_persisted).await;
                let _ = sqlx::query("UPDATE beat_analyses SET status='FAILED',error_message=$2,finished_at=now(),updated_at=now() WHERE separation_job_id=$1 AND input_stem='drums' AND status='RUNNING'")
                    .bind(job.id).bind(&error).execute(&self.db).await;
                Err(error)
            }
        }
    }

    async fn finalize_stems_inner(
        &self,
        job: &ProcessingJob,
        worker_result: Value,
        attempt_dir: &std::path::Path,
        original_path: &str,
        original_filename: &str,
        newly_persisted: &mut Vec<std::path::PathBuf>,
    ) -> Result<bool, String> {
        const STEMS: [&str; 4] = ["drums", "bass", "other", "vocals"];
        let started = Instant::now();
        let project_id = job.project_id;
        let job_id = job.id;

        let original = worker_result
            .get("original")
            .ok_or("worker result is missing original audio metadata")?;

        let sample_rate_i32 = original
            .get("sample_rate")
            .and_then(Value::as_i64)
            .and_then(|value| i32::try_from(value).ok())
            .filter(|value| (8_000..=384_000).contains(value))
            .ok_or("worker returned an invalid original sample rate")?;

        let channels_i16 = original
            .get("channels")
            .and_then(Value::as_i64)
            .and_then(|value| i16::try_from(value).ok())
            .filter(|value| (1..=2).contains(value))
            .ok_or("worker returned an invalid original channel count")?;

        let frame_count_i64 = original
            .get("frame_count")
            .and_then(Value::as_i64)
            .filter(|value| *value > 0)
            .ok_or("worker returned an invalid original frame count")?;

        let sample_rate = sample_rate_i32 as u32;
        let channels = channels_i16 as u16;
        let frame_count = frame_count_i64 as u64;
        let duration_seconds = frame_count as f64 / f64::from(sample_rate);

        self.stage_progress(
            job_id,
            "STEM_PERSISTENCE",
            97.2,
            "Validating and persisting full-length stems".into(),
        )
        .await;
        tracing::info!(job_id = %job_id, stage = "STEM_PERSISTENCE", sample_rate, channels, frame_count, "full-stem persistence started");

        let project_root = crate::services::project_storage::projects_root(&self.config.upload_dir);
        let mut persisted_stems = serde_json::Map::new();
        for (index, stem) in STEMS.iter().enumerate() {
            if self.cancel_finalization_if_requested(job_id).await {
                return Ok(false);
            }
            let source = attempt_dir.join(format!("{stem}.wav"));
            let file_size = crate::services::project_storage::validate_stem_wav(
                &source,
                sample_rate,
                channels,
                frame_count,
            )?;
            let key = crate::services::project_storage::stem_storage_key(project_id, stem)?;
            let destination =
                crate::services::project_storage::resolve_storage_key(&project_root, &key)?;
            crate::services::project_storage::atomic_copy_new(&source, &destination).await?;
            newly_persisted.push(destination.clone());
            persisted_stems.insert(
                (*stem).to_owned(),
                json!({
                    "storageKey": key,
                    "url": format!("/api/v1/projects/{project_id}/stems/{stem}"),
                    "sampleRate": sample_rate,
                    "channels": channels,
                    "frames": frame_count,
                    "durationSeconds": duration_seconds,
                    "fileSizeBytes": file_size
                }),
            );
            self.stage_progress(
                job_id,
                "STEM_PERSISTENCE",
                97.2 + 0.8 * ((index + 1) as f64 / STEMS.len() as f64),
                format!("Persisted full-length {stem} stem"),
            )
            .await;
            tracing::info!(job_id = %job_id, stage = "STEM_PERSISTENCE", stem, file_size_bytes = file_size, elapsed_ms = started.elapsed().as_millis(), "full-length stem persisted atomically");
        }

        let extension = crate::services::storage::safe_extension(original_filename)
            .ok_or("original audio has an unsupported file extension")?;
        let original_key = crate::services::project_storage::original_storage_key(
            project_id,
            job.audio_asset_id,
            extension,
        );
        let original_destination =
            crate::services::project_storage::resolve_storage_key(&project_root, &original_key)?;
        crate::services::project_storage::atomic_copy_new(
            std::path::Path::new(original_path),
            &original_destination,
        )
        .await?;
        newly_persisted.push(original_destination.clone());
        tracing::info!(job_id = %job_id, stage = "STEM_PERSISTENCE", elapsed_ms = started.elapsed().as_millis(), "original audio copied into permanent project storage");

        if self.cancel_finalization_if_requested(job_id).await {
            return Ok(false);
        }

        let strategy = job
            .request_metadata
            .get("analysis_strategy")
            .and_then(Value::as_str)
            .unwrap_or("drums_first");
        let (analysis_id, detected) = if strategy == "mix_first" {
            sqlx::query_as::<_, (Uuid, Value)>(
                "SELECT id, result FROM beat_analyses WHERE separation_job_id=$1 AND input_stem='mix' AND status='SUCCEEDED' ORDER BY created_at DESC LIMIT 1",
            )
            .bind(job_id)
            .fetch_optional(&self.db)
            .await
            .map_err(|error| format!("read existing mix-first Beat This result: {error}"))?
            .ok_or("mix-first Beat This result is missing")?
        } else {
            let analysis_id = sqlx::query_scalar::<_, Uuid>(
                "INSERT INTO beat_analyses (separation_job_id,status,input_stem,sample_rate,channels,frame_count,model_version) VALUES ($1,'RUNNING','drums',$2,$3,$4,$5) RETURNING id",
            )
            .bind(job_id)
            .bind(sample_rate_i32)
            .bind(channels_i16)
            .bind(frame_count_i64)
            .bind(&self.config.beat_this_model_version)
            .fetch_one(&self.db)
            .await
            .map_err(|error| format!("create drums Beat This state: {error}"))?;

            self.stage_progress(
                job_id,
                "BEAT_ANALYSIS",
                98.1,
                "Analyzing the separated drums stem with Beat This".into(),
            )
            .await;
            let analysis_started = Instant::now();
            tracing::info!(job_id = %job_id, stage = "BEAT_ANALYSIS", "Beat This analysis started");
            let mel = self.config.beat_this_mel_model.clone();
            let model = self.config.beat_this_model.clone();
            let version = self.config.beat_this_model_version.clone();
            let drum_path = attempt_dir.join("drums.wav");
            let analysis_work = tokio::task::spawn_blocking(move || {
                crate::routes::beat_analysis::analyze_wav(
                    &drum_path,
                    sample_rate,
                    channels,
                    frame_count,
                    "drums",
                    &mel,
                    &model,
                    version,
                )
            });
            tokio::pin!(analysis_work);
            let mut heartbeat = time::interval(Duration::from_secs(10));
            let analyzed = loop {
                tokio::select! {
                    result = &mut analysis_work => {
                        break result
                            .map_err(|error| format!("Beat This task failed: {error}"))?
                            .map_err(|error| format!("Beat This analysis failed: {error}"))?;
                    }
                    _ = heartbeat.tick() => {
                        let elapsed = analysis_started.elapsed();
                        let cancel_requested = self.is_cancel_requested(job_id).await;
                        let message = if cancel_requested {
                            format!("Beat This still running after {}s; cancellation will be applied when inference returns", elapsed.as_secs())
                        } else {
                            format!("Beat This still running ({:.1}s elapsed)", elapsed.as_secs_f64())
                        };
                        self.stage_progress(job_id, "BEAT_ANALYSIS", 98.1, message).await;
                        tracing::info!(job_id = %job_id, stage = "BEAT_ANALYSIS", elapsed_ms = elapsed.as_millis(), cancel_requested, "Beat This analysis in progress");
                    }
                }
            };
            let (analysis, elapsed_ms) = analyzed;
            tracing::info!(job_id = %job_id, stage = "BEAT_ANALYSIS", elapsed_ms, wall_ms = analysis_started.elapsed().as_millis(), stage_ms = %analysis["analysis_metadata"]["stages_ms"], "Beat This analysis completed");
            if self.cancel_finalization_if_requested(job_id).await {
                return Ok(false);
            }
            self.stage_progress(
                job_id,
                "BEAT_ANALYSIS",
                99.1,
                "Beat This analysis completed".into(),
            )
            .await;
            (analysis_id, analysis)
        };

        if self.cancel_finalization_if_requested(job_id).await {
            return Ok(false);
        }
        let beats_per_bar = job
            .request_metadata
            .pointer("/quantization/beats_per_bar")
            .and_then(Value::as_u64)
            .and_then(|value| u8::try_from(value).ok())
            .filter(|value| (2..=12).contains(value))
            .unwrap_or(4);
        let beat_grid = build_beat_grid(
            job_id,
            project_id,
            sample_rate,
            channels,
            frame_count,
            beats_per_bar,
            &detected,
        )?;
        self.stage_progress(
            job_id,
            "ANALYSIS_PERSISTENCE",
            99.5,
            "Persisting versioned beat-grid JSON".into(),
        )
        .await;
        tracing::info!(job_id = %job_id, stage = "ANALYSIS_PERSISTENCE", beat_count = beat_grid["analysis"]["beats"].as_array().map_or(0, Vec::len), downbeat_count = beat_grid["analysis"]["downbeats"].as_array().map_or(0, Vec::len), "beat grid persistence started");

        let result_metadata = json!({
            "schemaVersion": 1,
            "projectId": project_id,
            "analysisUrl": format!("/api/v1/jobs/{job_id}/analysis"),
            "stems": persisted_stems
        });
        let stem_records: Vec<(String, String, i32, i16, i64, i64)> = STEMS
            .iter()
            .map(|stem| {
                let metadata = &persisted_stems[*stem];
                (
                    (*stem).to_owned(),
                    metadata["storageKey"]
                        .as_str()
                        .unwrap_or_default()
                        .to_owned(),
                    sample_rate_i32,
                    channels_i16,
                    frame_count_i64,
                    metadata["fileSizeBytes"].as_i64().unwrap_or_default(),
                )
            })
            .collect();

        let mut tx =
            self.db.begin().await.map_err(|error| {
                format!("begin persistent project completion transaction: {error}")
            })?;
        for (stem, storage_key, rate, channel_count, frames, bytes) in stem_records {
            sqlx::query("INSERT INTO project_stems (project_id,job_id,stem_name,storage_key,sample_rate,channels,frame_count,file_size_bytes) VALUES ($1,$2,$3,$4,$5,$6,$7,$8)")
                .bind(project_id).bind(job_id).bind(stem).bind(storage_key).bind(rate).bind(channel_count).bind(frames).bind(bytes)
                .execute(&mut *tx).await.map_err(|error| format!("persist project stem reference: {error}"))?;
        }
        sqlx::query(
            "UPDATE audio_assets
 SET stored_path=$2,
     sample_rate=$3,
     channels=$4,
     frame_count=$5,
     duration_seconds=$6,
     expires_at=NULL,
     updated_at=now()
 WHERE id=$1",
        )
        .bind(job.audio_asset_id)
        .bind(original_destination.to_string_lossy().to_string())
        .bind(sample_rate_i32)
        .bind(channels_i16)
        .bind(frame_count_i64)
        .bind(duration_seconds)
        .execute(&mut *tx)
        .await
        .map_err(|error| format!("persist permanent original audio reference: {error}"))?;
        sqlx::query("UPDATE beat_analyses SET status='SUCCEEDED',result=$2,error_message=NULL,finished_at=now(),updated_at=now() WHERE id=$1")
            .bind(analysis_id).bind(&beat_grid).execute(&mut *tx).await
            .map_err(|error| format!("persist beat-grid JSON: {error}"))?;
        let completed_job = sqlx::query_as::<_, ProcessingJob>("UPDATE processing_jobs SET status='SUCCEEDED',cancel_requested=false,progress=100,result_metadata=$2,finished_at=now(),expires_at=NULL,updated_at=now() WHERE id=$1 AND status='RUNNING' RETURNING *")
            .bind(job_id).bind(&result_metadata).fetch_optional(&mut *tx).await
            .map_err(|error| format!("persist completed separation job: {error}"))?
            .ok_or("job stopped running before persistent outputs were committed")?;
        tx.commit()
            .await
            .map_err(|error| format!("commit persistent project outputs: {error}"))?;

        if std::path::Path::new(original_path) != original_destination {
            if let Err(error) = tokio::fs::remove_file(original_path).await {
                tracing::warn!(job_id = %job_id, path = original_path, %error, "permanent original is committed but temporary upload copy could not be removed");
            }
        }

        let _ = self.events.send(JobEvent {
            event: "beat_analysis.completed".into(),
            stage: Some("ANALYSIS_PERSISTENCE".into()),
            stem: None,
            stem_metadata: None,
            chunk: None,
            analysis: Some(beat_grid),
            job_id,
            status: "SUCCEEDED".into(),
            progress: 100.0,
            message: Some("Complete beat grid and four full-length stems persisted".into()),
        });
        self.emit(
            &completed_job,
            Some("Full-length stems and beat grid completed".into()),
        );
        tracing::info!(job_id = %job_id, project_id = %project_id, stage = "COMPLETED", elapsed_ms = started.elapsed().as_millis(), "project stem set and beat grid committed");
        Ok(true)
    }

    async fn cancel_finalization_if_requested(&self, id: Uuid) -> bool {
        if !self.is_cancel_requested(id).await {
            return false;
        }
        match sqlx::query_as::<_, ProcessingJob>("UPDATE processing_jobs SET status='CANCELLED',finished_at=now(),updated_at=now() WHERE id=$1 AND status='RUNNING' AND cancel_requested RETURNING *")
            .bind(id).fetch_optional(&self.db).await {
            Ok(Some(job)) => self.emit(&job, Some("Cancelled during finalization".into())),
            Ok(None) => {}
            Err(error) => tracing::error!(job_id = %id, %error, "could not persist cancellation during finalization"),
        }
        true
    }

    async fn emit_named(&self, id: Uuid, event_name: &str, message: impl Into<String>) {
        if let Ok(Some(job)) = sqlx::query_as::<_, ProcessingJob>(
            "SELECT * FROM processing_jobs WHERE id=$1 AND status='RUNNING'",
        )
        .bind(id)
        .fetch_optional(&self.db)
        .await
        {
            let _ = self.events.send(JobEvent {
                event: event_name.into(),
                stage: None,
                stem: None,
                stem_metadata: None,
                chunk: None,
                analysis: None,
                job_id: id,
                status: job.status,
                progress: job.progress,
                message: Some(message.into()),
            });
        }
    }

    async fn cancel_acknowledged(&self, id: Uuid) {
        if let Ok(Some(job)) = sqlx::query_as::<_, ProcessingJob>("UPDATE processing_jobs SET status = 'CANCELLED', finished_at = now(), updated_at = now() WHERE id = $1 AND status = 'RUNNING' AND cancel_requested RETURNING *").bind(id).fetch_optional(&self.db).await { self.emit(&job, Some("Worker acknowledged cancellation".into())); }
    }
    async fn fail(&self, id: Uuid, message: String) {
        match sqlx::query_as::<_, ProcessingJob>("UPDATE processing_jobs SET status = 'FAILED', error_message = $2, finished_at = now(), updated_at = now() WHERE id = $1 AND status = 'RUNNING' RETURNING *").bind(id).bind(&message).fetch_optional(&self.db).await {
            Ok(Some(job)) => self.emit(&job, Some(message)),
            Ok(None) => tracing::warn!(job_id = %id, "failure state was not written because job is no longer RUNNING"),
            Err(error) => tracing::error!(job_id = %id, %error, failure = %message, "could not persist job failure state"),
        }
    }
    pub fn emit(&self, job: &ProcessingJob, message: Option<String>) {
        let _ = self.events.send(JobEvent {
            event: "job.updated".into(),
            stage: None,
            job_id: job.id,
            status: job.status.clone(),
            progress: job.progress,
            message,
            stem: None,
            stem_metadata: None,
            chunk: None,
            analysis: None,
        });
    }
}

/// Claims the oldest eligible job and changes its state in the same transaction.
/// A queued row is never returned to the dispatcher unless the RUNNING update commits.
async fn claim_queued(
    db: &PgPool,
    project_filter: Option<Uuid>,
) -> Result<Option<ProcessingJob>, sqlx::Error> {
    let mut tx = db.begin().await?;
    let next = sqlx::query_scalar::<_, Uuid>("SELECT id FROM processing_jobs WHERE status = 'QUEUED' AND NOT cancel_requested AND ($1::uuid IS NULL OR project_id = $1) ORDER BY created_at, id LIMIT 1 FOR UPDATE SKIP LOCKED")
        .bind(project_filter).fetch_optional(&mut *tx).await?;
    let Some(id) = next else {
        tx.commit().await?;
        return Ok(None);
    };
    let job = sqlx::query_as::<_, ProcessingJob>("UPDATE processing_jobs SET status = 'RUNNING', progress = 0, started_at = now(), updated_at = now(), error_message = NULL WHERE id = $1 AND status = 'QUEUED' AND NOT cancel_requested RETURNING *")
        .bind(id).fetch_optional(&mut *tx).await?;
    tx.commit().await?;
    if job.is_none() {
        tracing::warn!(job_id = %id, "locked queued job was not transitioned to RUNNING; it may have been cancelled concurrently");
    }
    Ok(job)
}

fn worker_restart_delay(attempt: u32) -> Duration {
    let exponent = attempt.saturating_sub(1).min(5);
    Duration::from_secs(1_u64 << exponent).min(Duration::from_secs(30))
}

fn worker_progress(progress: f64) -> f64 {
    if !progress.is_finite() {
        return 0.0;
    }
    progress.clamp(0.0, 100.0) * 0.97
}

pub(crate) fn build_beat_grid(
    job_id: Uuid,
    project_id: Uuid,
    sample_rate: u32,
    channels: u16,
    frames: u64,
    beats_per_bar: u8,
    detected: &Value,
) -> Result<Value, String> {
    fn map_event(
        event: &Value,
        index: usize,
        is_downbeat: bool,
        sample_rate: u32,
        frames: u64,
    ) -> Result<Value, String> {
        let time = event
            .get("time_seconds")
            .and_then(Value::as_f64)
            .filter(|time| time.is_finite() && *time >= 0.0)
            .ok_or_else(|| format!("detector event {index} has an invalid timestamp"))?;
        let sample = event
            .get("sample_position")
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("detector event {index} has no frame sample position"))?;
        let expected_sample = (time * f64::from(sample_rate)).round() as u64;
        if sample != expected_sample || sample > frames {
            return Err(format!(
                "detector event {index} is outside the original frame timeline or has inconsistent sample mapping"
            ));
        }
        let mut mapped = serde_json::Map::new();
        mapped.insert("timeSeconds".into(), json!(time));
        mapped.insert("samplePosition".into(), json!(sample));
        if !is_downbeat && let Some(beat_number) = event.get("beat_number_in_bar") {
            if !beat_number.is_null() {
                mapped.insert("beatNumberInBar".into(), beat_number.clone());
            }
        }
        Ok(Value::Object(mapped))
    }

    let beats = detected
        .get("beats")
        .and_then(Value::as_array)
        .ok_or("Beat This result has no beat event array")?
        .iter()
        .enumerate()
        .map(|(index, event)| map_event(event, index, false, sample_rate, frames))
        .collect::<Result<Vec<_>, _>>()?;
    let downbeats = detected
        .get("downbeats")
        .and_then(Value::as_array)
        .ok_or("Beat This result has no downbeat event array")?
        .iter()
        .enumerate()
        .map(|(index, event)| map_event(event, index, true, sample_rate, frames))
        .collect::<Result<Vec<_>, _>>()?;
    let mut stems = serde_json::Map::new();
    for stem in ["drums", "bass", "other", "vocals"] {
        stems.insert(
            stem.into(),
            json!(format!("/api/v1/projects/{project_id}/stems/{stem}")),
        );
    }
    Ok(json!({
        "schemaVersion": 1,
        "jobId": job_id,
        "projectId": project_id,
        "audio": {
            "sampleRate": sample_rate,
            "channels": channels,
            "frames": frames,
            "durationSeconds": frames as f64 / f64::from(sample_rate),
            "samplePositionUnit": "frames",
            "stems": stems
        },
        "analysis": {
            "engine": detected.get("detection_engine").cloned().unwrap_or_else(|| json!("beat-this-rs")),
            "engineVersion": detected.get("engine_version").cloned().unwrap_or(Value::Null),
            "modelVersion": detected.get("model_version").cloned().unwrap_or(Value::Null),
            "inputStem": detected.get("input_stem").cloned().unwrap_or_else(|| json!("drums")),
            "bpm": detected.get("bpm").cloned().unwrap_or(Value::Null),
            "beatsPerBar": beats_per_bar,
            "beats": beats,
            "downbeats": downbeats,
            "metadata": detected.get("analysis_metadata").cloned().unwrap_or(Value::Null)
        }
    }))
}

async fn remove_new_assets(paths: &[std::path::PathBuf]) {
    for path in paths {
        if let Err(error) = tokio::fs::remove_file(path).await
            && error.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(path = %path.display(), %error, "failed to remove uncommitted project asset");
        }
    }
}

#[cfg(test)]
mod dispatch_tests {
    use super::{build_beat_grid, claim_queued, worker_progress, worker_restart_delay};
    use sqlx::postgres::PgPoolOptions;
    use std::time::Duration;
    use uuid::Uuid;

    #[test]
    fn worker_restart_backoff_is_bounded_and_resets_after_success() {
        let expected = [1, 2, 4, 8, 16, 30, 30, 30];
        for (index, seconds) in expected.into_iter().enumerate() {
            assert_eq!(
                worker_restart_delay(index as u32 + 1),
                Duration::from_secs(seconds)
            );
        }
        assert_eq!(worker_restart_delay(0), Duration::from_secs(1));
    }

    #[test]
    fn inference_progress_reserves_completion_and_caps_at_ninety_seven() {
        let inference = worker_progress(100.0);
        assert_eq!(inference, 97.0);
        assert_eq!(worker_progress(f64::NAN), 0.0);
    }

    #[test]
    fn beat_grid_serialization_keeps_frame_positions_and_empty_downbeats() {
        let job_id = Uuid::new_v4();
        let project_id = Uuid::new_v4();
        let detected = serde_json::json!({
            "bpm": 120.0,
            "detection_engine": "beat-this-rs",
            "engine_version": "1.1.0",
            "input_stem": "drums",
            "beats": [{"time_seconds": 0.25, "sample_position": 12000, "beat_number_in_bar": 1}],
            "downbeats": [],
            "analysis_metadata": {"source_rate": 48000}
        });
        let grid = build_beat_grid(job_id, project_id, 48_000, 2, 96_000, 4, &detected)
            .expect("serialize complete beat grid");
        assert_eq!(grid["audio"]["frames"], 96_000);
        assert_eq!(grid["audio"]["samplePositionUnit"], "frames");
        assert_eq!(grid["analysis"]["beats"][0]["samplePosition"], 12_000);
        assert!(grid["analysis"]["beats"][0].get("bar").is_none());
        assert_eq!(grid["analysis"]["downbeats"], serde_json::json!([]));
        assert_eq!(
            grid["audio"]["stems"]["drums"],
            format!("/api/v1/projects/{project_id}/stems/drums")
        );
        let encoded = serde_json::to_string(&grid).unwrap();
        let decoded: serde_json::Value = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, grid);
    }

    #[test]
    fn beat_grid_rejects_sample_positions_outside_or_inconsistent_with_source_timeline() {
        let detected = serde_json::json!({
            "beats": [{"time_seconds": 0.25, "sample_position": 12_001}],
            "downbeats": []
        });
        assert!(
            build_beat_grid(
                Uuid::new_v4(),
                Uuid::new_v4(),
                48_000,
                2,
                96_000,
                4,
                &detected
            )
            .is_err()
        );
    }

    #[tokio::test]
    async fn queued_jobs_are_atomically_claimed_in_creation_order_when_test_db_is_configured() {
        let Ok(database_url) = std::env::var("SONICLAB_TEST_DATABASE_URL") else {
            eprintln!("skipping PostgreSQL dispatch test: SONICLAB_TEST_DATABASE_URL is unset");
            return;
        };
        let pool = PgPoolOptions::new()
            .max_connections(2)
            .connect(&database_url)
            .await
            .expect("connect to explicitly configured disposable test database");

        let project_id = sqlx::query_scalar::<_, Uuid>(
            "INSERT INTO projects (name) VALUES ('dispatch test') RETURNING id",
        )
        .fetch_one(&pool)
        .await
        .expect("create isolated test project");
        let asset_id = sqlx::query_scalar::<_, Uuid>("INSERT INTO audio_assets (project_id,original_filename,content_type,stored_path,file_size_bytes,sample_rate,channels,frame_count,duration_seconds) VALUES ($1,'dispatch.wav','audio/wav',$2,4,48000,2,1,0.0001) RETURNING id")
            .bind(project_id)
            .bind(format!("/tmp/dispatch-test-{}.wav", Uuid::new_v4()))
            .fetch_one(&pool)
            .await
            .expect("create isolated test audio asset");

        let first_id = sqlx::query_scalar::<_, Uuid>("INSERT INTO processing_jobs (project_id,audio_asset_id,job_type,created_at) VALUES ($1,$2,'separate',now()-interval '1 second') RETURNING id")
            .bind(project_id).bind(asset_id).fetch_one(&pool).await.expect("create first queued job");
        let second_id = sqlx::query_scalar::<_, Uuid>("INSERT INTO processing_jobs (project_id,audio_asset_id,job_type) VALUES ($1,$2,'separate') RETURNING id")
            .bind(project_id).bind(asset_id).fetch_one(&pool).await.expect("create second queued job");

        let first = claim_queued(&pool, Some(project_id))
            .await
            .expect("claim transaction succeeds")
            .expect("first job is claimable");
        let second = claim_queued(&pool, Some(project_id))
            .await
            .expect("second claim transaction succeeds")
            .expect("second job is claimable");
        let none = claim_queued(&pool, Some(project_id))
            .await
            .expect("empty queue transaction succeeds");
        assert_eq!((first.id, first.status.as_str()), (first_id, "RUNNING"));
        assert_eq!((second.id, second.status.as_str()), (second_id, "RUNNING"));
        assert!(none.is_none());

        sqlx::query("DELETE FROM processing_jobs WHERE project_id=$1")
            .bind(project_id)
            .execute(&pool)
            .await
            .expect("remove only the test's isolated jobs");
        sqlx::query("DELETE FROM projects WHERE id=$1")
            .bind(project_id)
            .execute(&pool)
            .await
            .expect("remove only the test's isolated project and rows");
        pool.close().await;
    }
}
