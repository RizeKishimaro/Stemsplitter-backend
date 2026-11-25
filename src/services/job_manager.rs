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
    pub stem: Option<String>,
    pub stem_metadata: Option<Value>,
    pub chunk: Option<Value>,
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
        let mut restart_delay = Duration::from_secs(1);
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
                        restart_delay = Duration::from_secs(1);
                    }
                    Err(error) => {
                        tracing::error!(error = %error, "Demucs worker unavailable; jobs remain durable and queued");
                        tokio::select! { _ = shutdown.cancelled() => break, _ = time::sleep(restart_delay) => {} }
                        restart_delay = (restart_delay * 2).min(Duration::from_secs(30));
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
                                tracing::error!(%status, "Demucs worker exited unexpectedly; restarting");
                                worker = None;
                                continue;
                            }
                            Err(error) => tracing::warn!(%error, "could not check Demucs worker process"),
                            Ok(None) => {}
                        }
                    }
                    match self.claim().await {
                        Ok(Some(job)) => {
                            let job_id = job.id;
                            if let Some(process) = worker.as_mut() {
                                if self.execute(job, process, &mut cancel_rx, &shutdown).await.is_err() {
                                    let stage = self.config.upload_dir.join("..").join("jobs").join(job_id.to_string());
                                    let _ = tokio::fs::remove_dir_all(stage).await;
                                    worker.as_mut().expect("worker exists").kill().await;
                                    worker = None;
                                }
                            }
                        }
                        Ok(None) => {}
                        Err(error) => tracing::error!(error = %error, "job polling failed"),
                    }
                }
            }
        }
        if let Some(mut process) = worker {
            process.kill().await;
        }
    }

    async fn claim(&self) -> Result<Option<ProcessingJob>, sqlx::Error> {
        let mut tx = self.db.begin().await?;
        let next = sqlx::query_scalar::<_, Uuid>("SELECT id FROM processing_jobs WHERE status = 'QUEUED' AND NOT cancel_requested ORDER BY created_at LIMIT 1 FOR UPDATE SKIP LOCKED")
            .fetch_optional(&mut *tx).await?;
        let Some(id) = next else {
            tx.commit().await?;
            return Ok(None);
        };
        let job = sqlx::query_as::<_, ProcessingJob>("UPDATE processing_jobs SET status = 'RUNNING', progress = 0, started_at = now(), updated_at = now(), error_message = NULL WHERE id = $1 AND status = 'QUEUED' RETURNING *")
            .bind(id).fetch_optional(&mut *tx).await?;
        tx.commit().await?;
        if let Some(job) = &job {
            self.emit(job, Some("Persistent HTDemucs worker accepted job".into()));
        }
        Ok(job)
    }

    async fn emit_stem_completed(&self, id: Uuid, stem: String, metadata: Value) {
        if let Ok(Some(job)) =
            sqlx::query_as::<_, ProcessingJob>("SELECT * FROM processing_jobs WHERE id = $1")
                .bind(id)
                .fetch_optional(&self.db)
                .await
        {
            let _ = self.events.send(JobEvent {
                event: "stem.completed".into(),
                stem: Some(stem.clone()),
                stem_metadata: Some(metadata),
                chunk: None,
                job_id: job.id,
                status: job.status.clone(),
                progress: job.progress,
                message: Some(format!("{stem} is Ready.")),
            });
        }
    }

    async fn execute(
        &self,
        job: ProcessingJob,
        process: &mut WorkerProcess,
        cancel_rx: &mut mpsc::UnboundedReceiver<Uuid>,
        shutdown: &CancellationToken,
    ) -> Result<(), ()> {
        let asset = sqlx::query_as::<_, (String, i32, i16, i64)>("SELECT stored_path, sample_rate, channels, frame_count FROM audio_assets WHERE id = $1")
            .bind(job.audio_asset_id).fetch_one(&self.db).await;
        let (stored_path, sample_rate, channels, frame_count) = match asset {
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
            .join(job.id.to_string());
        let _ = sqlx::query("DELETE FROM job_chunks WHERE job_id=$1")
            .bind(job.id)
            .execute(&self.db)
            .await;
        let _ = sqlx::query("DELETE FROM beat_analyses WHERE separation_job_id=$1")
            .bind(job.id)
            .execute(&self.db)
            .await;
        let _ = tokio::fs::remove_dir_all(&stage).await;
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
            let (mut analysis, elapsed_ms) = match mix_analysis {
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
            let boundaries = match crate::services::timeline::bar_boundaries(
                &analysis,
                sample_rate as u32,
                frame_count as u64,
                job.request_metadata
                    .pointer("/quantization/beats_per_bar")
                    .and_then(Value::as_u64)
                    .unwrap_or(4) as u8,
            ) {
                Ok(boundaries) => boundaries,
                Err(error) => {
                    self.fail(job.id, error).await;
                    return Ok(());
                }
            };
            let boundary_value = match serde_json::to_value(&boundaries) {
                Ok(value) => value,
                Err(error) => {
                    self.fail(job.id, format!("serialize musical boundaries: {error}"))
                        .await;
                    return Ok(());
                }
            };
            if let Some(object) = analysis.as_object_mut() {
                object.insert("quantized_boundaries".into(), boundary_value.clone());
            }
            if let Err(error) = sqlx::query("UPDATE beat_analyses SET status='SUCCEEDED',result=$2,error_message=NULL,finished_at=now(),updated_at=now() WHERE separation_job_id=$1 AND input_stem='mix'")
                .bind(job.id).bind(&analysis).execute(&self.db).await {
                self.fail(job.id, format!("could not persist mix analysis: {error}")).await; return Ok(());
            }
            self.emit_analysis_ready(
                job.id,
                analysis.clone(),
                boundary_value,
                format!("Mix analysis complete ({elapsed_ms} ms)"),
            )
            .await;
        }
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
                            if let Some(stem) = message.stem {
                                self.emit_stem_completed(
                                    job.id,
                                    stem,
                                    message.stem_metadata.unwrap_or(Value::Null)
                                    ).await
                            }
                        }
                        "cancelled" if message.job_id == Some(job.id) && message.acknowledged == Some(true) =>{
                            self.cancel_acknowledged(job.id).await;
                            return Ok(());
                        }
                        "completed" if message.job_id == Some(job.id) => {
                            let result = message.result.unwrap_or(Value::Null);
                            if let Err(error) = self.finalize_chunks(&job, result).await {
                                self.fail(job.id, error).await;
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
        if let Ok(Some(job)) = sqlx::query_as::<_, ProcessingJob>("UPDATE processing_jobs SET progress = $2, updated_at = now() WHERE id = $1 AND status = 'RUNNING' RETURNING *").bind(id).bind(progress.clamp(0.0, 99.0)).fetch_optional(&self.db).await { self.emit(&job, Some("HTDemucs inference progress".into())); }
    }

    async fn finalize_chunks(&self, job: &ProcessingJob, mut result: Value) -> Result<(), String> {
        let job_id = job.id;
        let strategy = job
            .request_metadata
            .get("analysis_strategy")
            .and_then(Value::as_str)
            .unwrap_or("drums_first");
        let beats_per_bar = job
            .request_metadata
            .pointer("/quantization/beats_per_bar")
            .and_then(Value::as_u64)
            .unwrap_or(4) as u8;
        let original = result
            .get("original")
            .ok_or("HTDemucs result is missing source metadata")?;
        let sample_rate = original
            .get("sample_rate")
            .and_then(Value::as_u64)
            .ok_or("missing source sample rate")? as u32;
        let channels = original
            .get("channels")
            .and_then(Value::as_u64)
            .ok_or("missing source channel count")? as u16;
        let frame_count = original
            .get("frame_count")
            .and_then(Value::as_u64)
            .ok_or("missing source frame count")?;
        let job_dir = self
            .config
            .upload_dir
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .join("jobs")
            .join(job_id.to_string());
        let analysis = if strategy == "mix_first" {
            sqlx::query_scalar::<_, Value>("SELECT result FROM beat_analyses WHERE separation_job_id=$1 AND input_stem='mix' ORDER BY created_at DESC LIMIT 1")
                .bind(job_id).fetch_optional(&self.db).await
                .map_err(|e| format!("read mix-first analysis: {e}"))?
                .ok_or("mix-first analysis result is missing")?
        } else {
            self.emit_named(
                job_id,
                "beat_analysis.started",
                "Analyzing HTDemucs drums timeline",
            )
            .await;
            sqlx::query("INSERT INTO beat_analyses (separation_job_id,status,input_stem,sample_rate,channels,frame_count,model_version) VALUES ($1,'RUNNING','drums',$2,$3,$4,$5)")
                .bind(job_id).bind(sample_rate as i32).bind(channels as i16).bind(frame_count as i64).bind(&self.config.beat_this_model_version)
                .execute(&self.db).await.map_err(|e| format!("create drums analysis state: {e}"))?;
            let mel = self.config.beat_this_mel_model.clone();
            let model = self.config.beat_this_model.clone();
            let version = self.config.beat_this_model_version.clone();
            let path = job_dir.join("drums.wav");
            let result = tokio::task::spawn_blocking(move || {
                crate::routes::beat_analysis::analyze_wav(
                    &path,
                    sample_rate,
                    channels,
                    frame_count,
                    "drums",
                    &mel,
                    &model,
                    version,
                )
            })
            .await
            .map_err(|e| format!("Beat This task failed: {e}"))?;
            let (mut analysis, elapsed_ms) = match result {
                Ok(result) => result,
                Err(error) => {
                    let _ = sqlx::query("UPDATE beat_analyses SET status='FAILED',error_message=$2,finished_at=now(),updated_at=now() WHERE separation_job_id=$1 AND input_stem='drums'").bind(job_id).bind(&error).execute(&self.db).await;
                    return Err(format!("Beat This analysis failed: {error}"));
                }
            };
            let boundaries = crate::services::timeline::bar_boundaries(
                &analysis,
                sample_rate,
                frame_count,
                beats_per_bar,
            )?;
            let boundary_value = serde_json::to_value(&boundaries)
                .map_err(|e| format!("serialize musical boundaries: {e}"))?;
            if let Some(object) = analysis.as_object_mut() {
                object.insert("quantized_boundaries".into(), boundary_value.clone());
            }
            sqlx::query("UPDATE beat_analyses SET status='SUCCEEDED',result=$2,error_message=NULL,finished_at=now(),updated_at=now() WHERE separation_job_id=$1 AND input_stem='drums'")
                .bind(job_id).bind(&analysis).execute(&self.db).await.map_err(|e| format!("persist Beat This result: {e}"))?;
            self.emit_analysis_ready(
                job_id,
                analysis.clone(),
                boundary_value,
                format!("Drums analysis complete ({elapsed_ms} ms)"),
            )
            .await;
            analysis
        };
        let boundaries: Vec<crate::services::timeline::ChunkBoundary> = serde_json::from_value(
            analysis
                .get("quantized_boundaries")
                .cloned()
                .ok_or("stored analysis has no shared boundary schedule")?,
        )
        .map_err(|e| format!("stored shared boundary schedule is invalid: {e}"))?;
        let mut manifest = Vec::with_capacity(boundaries.len());
        for boundary in &boundaries {
            if self
                .cancel_finalization_if_requested(job_id, &job_dir)
                .await
            {
                return Ok(());
            }
            let dir = job_dir.clone();
            let boundary_owned = boundary.clone();
            let chunk_id = job_id;
            let built = tokio::task::spawn_blocking(move || {
                crate::services::chunk_files::write_chunks(
                    &dir,
                    chunk_id,
                    std::slice::from_ref(&boundary_owned),
                    sample_rate,
                    channels,
                )
            })
            .await
            .map_err(|e| format!("chunk writer task failed: {e}"))?
            .map_err(|e| format!("chunk {} could not be finalized: {e}", boundary.index))?;
            let chunk = built
                .into_iter()
                .next()
                .ok_or("chunk writer returned no chunk")?;
            if chunk.boundary != *boundary || chunk.files.len() != 4 {
                return Err(format!(
                    "chunk {} did not preserve the complete shared boundary",
                    boundary.index
                ));
            }
            if self
                .cancel_finalization_if_requested(job_id, &job_dir)
                .await
            {
                return Ok(());
            }
            let stem_urls = chunk.stem_metadata.clone();
            let event_payload = json!({
                "job_id": job_id,
                "chunk_id": chunk.chunk_id,
                "chunk_index": boundary.index,
                "bar_number": boundary.bar_number,
                "start_seconds": boundary.start_seconds(),
                "end_seconds": boundary.end_seconds(),
                "duration_seconds": boundary.end_seconds() - boundary.start_seconds(),
                "source_sample_start": boundary.start_sample,
                "source_sample_end": boundary.end_sample,
                "sample_rate": sample_rate,
                "stems": stem_urls,
                "status": "COMPLETED"
            });
            sqlx::query("INSERT INTO job_chunks (job_id,chunk_index,chunk_id,bar_number,start_seconds,end_seconds,source_sample_start,source_sample_end,sample_rate,stem_metadata) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10) ON CONFLICT (job_id,chunk_index) DO UPDATE SET chunk_id=EXCLUDED.chunk_id,bar_number=EXCLUDED.bar_number,start_seconds=EXCLUDED.start_seconds,end_seconds=EXCLUDED.end_seconds,source_sample_start=EXCLUDED.source_sample_start,source_sample_end=EXCLUDED.source_sample_end,sample_rate=EXCLUDED.sample_rate,stem_metadata=EXCLUDED.stem_metadata,completed_at=now()")
                .bind(job_id).bind(boundary.index as i32).bind(&chunk.chunk_id).bind(boundary.bar_number.map(|v| v as i32)).bind(boundary.start_seconds()).bind(boundary.end_seconds()).bind(boundary.start_sample as i64).bind(boundary.end_sample as i64).bind(sample_rate as i32).bind(&chunk.stem_metadata)
                .execute(&self.db).await.map_err(|e| format!("persist chunk manifest row: {e}"))?;
            self.emit_chunk(job_id, event_payload).await;
            manifest.push(json!({"chunk_id": chunk.chunk_id, "chunk_index": boundary.index, "bar_number": boundary.bar_number, "start_seconds": boundary.start_seconds(), "end_seconds": boundary.end_seconds(), "source_sample_start": boundary.start_sample, "source_sample_end": boundary.end_sample, "sample_rate": sample_rate, "stems": chunk.stem_metadata}));
        }
        if self
            .cancel_finalization_if_requested(job_id, &job_dir)
            .await
        {
            return Ok(());
        }
        let final_manifest = Value::Array(manifest);
        self.emit_manifest(job_id, final_manifest.clone()).await;
        if let Some(object) = result.as_object_mut() {
            object.insert("beat_analysis".into(), analysis);
            object.insert("analysis_strategy".into(), json!(strategy));
            object.insert("chunks".into(), final_manifest);
        }
        self.succeed(job_id, result).await;
        Ok(())
    }

    async fn cancel_finalization_if_requested(&self, id: Uuid, job_dir: &std::path::Path) -> bool {
        let requested = sqlx::query_scalar::<_, bool>(
            "SELECT cancel_requested FROM processing_jobs WHERE id=$1 AND status='RUNNING'",
        )
        .bind(id)
        .fetch_optional(&self.db)
        .await
        .ok()
        .flatten()
        .unwrap_or(false);
        if !requested {
            return false;
        }
        let _ = sqlx::query("DELETE FROM job_chunks WHERE job_id=$1")
            .bind(id)
            .execute(&self.db)
            .await;
        let _ = tokio::fs::remove_dir_all(job_dir.join("chunks")).await;
        if let Ok(Some(job)) = sqlx::query_as::<_, ProcessingJob>("UPDATE processing_jobs SET status='CANCELLED',finished_at=now(),updated_at=now() WHERE id=$1 AND status='RUNNING' AND cancel_requested RETURNING *")
            .bind(id).fetch_optional(&self.db).await {
            self.emit(&job, Some("Cancelled during synchronized chunk finalization".into()));
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
                stem: None,
                stem_metadata: None,
                chunk: None,
                job_id: id,
                status: job.status,
                progress: job.progress,
                message: Some(message.into()),
            });
        }
    }

    async fn emit_analysis_ready(
        &self,
        id: Uuid,
        analysis: Value,
        boundaries: Value,
        message: String,
    ) {
        if let Ok(Some(job)) = sqlx::query_as::<_, ProcessingJob>(
            "SELECT * FROM processing_jobs WHERE id=$1 AND status='RUNNING'",
        )
        .bind(id)
        .fetch_optional(&self.db)
        .await
        {
            let _ = self.events.send(JobEvent {
                event: "beat_analysis.completed".into(),
                stem: None,
                stem_metadata: None,
                chunk: Some(
                    json!({ "job_id": id, "analysis": analysis, "boundaries": boundaries }),
                ),
                job_id: id,
                status: job.status,
                progress: job.progress,
                message: Some(message),
            });
        }
    }

    async fn emit_chunk(&self, id: Uuid, payload: Value) {
        if let Ok(Some(job)) =
            sqlx::query_as::<_, ProcessingJob>("SELECT * FROM processing_jobs WHERE id=$1")
                .bind(id)
                .fetch_optional(&self.db)
                .await
        {
            let _ = self.events.send(JobEvent {
                event: "chunk.completed".into(),
                stem: None,
                stem_metadata: None,
                chunk: Some(payload),
                job_id: id,
                status: job.status,
                progress: job.progress,
                message: Some("Synchronized four-stem chunk ready".into()),
            });
        }
    }

    async fn emit_manifest(&self, id: Uuid, chunks: Value) {
        if let Ok(Some(job)) =
            sqlx::query_as::<_, ProcessingJob>("SELECT * FROM processing_jobs WHERE id=$1")
                .bind(id)
                .fetch_optional(&self.db)
                .await
        {
            let _ = self.events.send(JobEvent {
                event: "chunks.manifest.completed".into(),
                stem: None,
                stem_metadata: None,
                chunk: Some(json!({"job_id": id, "chunks": chunks, "status": "COMPLETED"})),
                job_id: id,
                status: job.status,
                progress: job.progress,
                message: Some("Final synchronized chunk manifest ready".into()),
            });
        }
    }
    async fn succeed(&self, id: Uuid, result: Value) {
        if let Ok(Some(job)) = sqlx::query_as::<_, ProcessingJob>("UPDATE processing_jobs SET status = 'SUCCEEDED', cancel_requested = false, progress = 100, result_metadata = $2, finished_at = now(), expires_at = now() + make_interval(secs => $3::double precision), updated_at = now() WHERE id = $1 AND status = 'RUNNING' RETURNING *")
            .bind(id).bind(result).bind(self.config.audio_job_ttl.as_secs_f64()).fetch_optional(&self.db).await {
            let _ = sqlx::query("UPDATE audio_assets SET expires_at = now() + make_interval(secs => $2::double precision), updated_at = now() WHERE id = (SELECT audio_asset_id FROM processing_jobs WHERE id=$1)")
                .bind(id).bind(self.config.audio_job_ttl.as_secs_f64()).execute(&self.db).await;
            self.emit(&job, Some("HTDemucs separation completed".into()));
        }
    }
    async fn cancel_acknowledged(&self, id: Uuid) {
        if let Ok(Some(job)) = sqlx::query_as::<_, ProcessingJob>("UPDATE processing_jobs SET status = 'CANCELLED', finished_at = now(), updated_at = now() WHERE id = $1 AND status = 'RUNNING' AND cancel_requested RETURNING *").bind(id).fetch_optional(&self.db).await { self.emit(&job, Some("Worker acknowledged cancellation".into())); }
    }
    async fn fail(&self, id: Uuid, message: String) {
        if let Ok(Some(job)) = sqlx::query_as::<_, ProcessingJob>("UPDATE processing_jobs SET status = 'FAILED', error_message = $2, finished_at = now(), updated_at = now() WHERE id = $1 AND status = 'RUNNING' RETURNING *").bind(id).bind(&message).fetch_optional(&self.db).await { self.emit(&job, Some(message)); }
    }
    pub fn emit(&self, job: &ProcessingJob, message: Option<String>) {
        let _ = self.events.send(JobEvent {
            event: "job.updated".into(),
            job_id: job.id,
            status: job.status.clone(),
            progress: job.progress,
            message,
            stem: None,
            stem_metadata: None,
            chunk: None,
        });
    }
}
