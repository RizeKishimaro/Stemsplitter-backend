use std::{env, net::SocketAddr, path::PathBuf, str::FromStr, time::Duration};

#[derive(Clone, Debug)]
pub struct Config {
    pub address: SocketAddr,
    pub database_url: String,
    pub database_max_connections: u32,
    pub upload_dir: PathBuf,
    pub max_upload_bytes: usize,
    pub job_concurrency: usize,
    pub job_poll_interval: Duration,
    pub cors_allow_origins: Vec<String>,
    pub demucs_python: PathBuf,
    pub demucs_worker_script: PathBuf,
    pub demucs_cancel_timeout: Duration,
    pub demucs_startup_timeout: Duration,
    pub beat_this_mel_model: PathBuf,
    pub beat_this_model: PathBuf,
    pub beat_this_model_version: Option<String>,
    pub beat_this_threads: usize,
    pub audio_job_ttl: Duration,
}

impl Config {
    pub fn from_env() -> Result<Self, String> {
        dotenvy::dotenv().ok();
        let host = env::var("HOST").unwrap_or_else(|_| "127.0.0.1".into());
        let port = value("PORT", 8787_u16)?;
        let address = SocketAddr::from_str(&format!("{host}:{port}"))
            .map_err(|error| format!("Invalid HOST/PORT: {error}"))?;
        let database_url =
            env::var("DATABASE_URL").map_err(|_| "DATABASE_URL must be set".to_string())?;
        if database_url.trim().is_empty() {
            return Err("DATABASE_URL must not be empty".into());
        }
        let max_upload_bytes = value("MAX_UPLOAD_BYTES", 1_073_741_824_usize)?;
        let job_concurrency = value("JOB_CONCURRENCY", 1_usize)?;
        let beat_this_threads = value(
            "BEAT_THIS_THREADS",
            std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1)
                .min(16),
        )?;
        if max_upload_bytes == 0 || job_concurrency == 0 || beat_this_threads == 0 {
            return Err(
                "MAX_UPLOAD_BYTES, JOB_CONCURRENCY and BEAT_THIS_THREADS must be positive".into(),
            );
        }
        if job_concurrency != 1 {
            return Err(
                "JOB_CONCURRENCY must be 1 while the persistent HTDemucs worker is enabled".into(),
            );
        }
        let poll_ms = value("JOB_POLL_INTERVAL_MS", 500_u64)?;
        if poll_ms == 0 {
            return Err("JOB_POLL_INTERVAL_MS must be positive".into());
        }
        let audio_job_ttl_secs = value("AUDIO_JOB_TTL_SECS", 86_400_u64)?;
        if audio_job_ttl_secs == 0 {
            return Err("AUDIO_JOB_TTL_SECS must be positive".into());
        }
        let cors_allow_origins = env::var("CORS_ALLOW_ORIGINS")
            .unwrap_or_else(|_| "http://localhost:5173".into())
            .split(',')
            .map(str::trim)
            .filter(|origin| !origin.is_empty())
            .map(str::to_string)
            .collect();

        Ok(Self {
            address,
            database_url,
            database_max_connections: value("DATABASE_MAX_CONNECTIONS", 10_u32)?,
            upload_dir: PathBuf::from(
                env::var("UPLOAD_DIR").unwrap_or_else(|_| "./data/uploads".into()),
            ),
            max_upload_bytes,
            job_concurrency,
            job_poll_interval: Duration::from_millis(poll_ms),
            cors_allow_origins,
            demucs_python: PathBuf::from(
                env::var("DEMUCS_PYTHON")
                    .unwrap_or_else(|_| "/home/rizekishimaro/.venvs/demucs/bin/python".into()),
            ),
            demucs_worker_script: PathBuf::from(
                env::var("DEMUCS_WORKER_SCRIPT")
                    .unwrap_or_else(|_| "./worker/demucs_worker.py".into()),
            ),
            demucs_cancel_timeout: Duration::from_secs(value(
                "DEMUCS_CANCEL_TIMEOUT_SECS",
                60_u64,
            )?),
            demucs_startup_timeout: Duration::from_secs(value(
                "DEMUCS_STARTUP_TIMEOUT_SECS",
                600_u64,
            )?),
            beat_this_mel_model: PathBuf::from(
                env::var("BEAT_THIS_MEL_MODEL")
                    .unwrap_or_else(|_| "./models/mel_spectrogram.onnx".into()),
            ),
            beat_this_model: PathBuf::from(
                env::var("BEAT_THIS_MODEL").unwrap_or_else(|_| "./models/beat_this.onnx".into()),
            ),
            beat_this_model_version: env::var("BEAT_THIS_MODEL_VERSION")
                .ok()
                .filter(|value| !value.trim().is_empty()),
            beat_this_threads,
            audio_job_ttl: Duration::from_secs(audio_job_ttl_secs),
        })
    }
}

fn value<T: FromStr>(key: &str, default: T) -> Result<T, String> {
    match env::var(key) {
        Ok(value) => value
            .parse()
            .map_err(|_| format!("{key} has an invalid value")),
        Err(_) => Ok(default),
    }
}
