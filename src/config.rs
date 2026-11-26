use std::{
    env,
    net::SocketAddr,
    path::{Path, PathBuf},
    process::Command,
    str::FromStr,
    time::Duration,
};

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
        let application_root = application_root()?;
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

        let demucs_python = resolve_demucs_python(&application_root)?;
        let demucs_worker_script = resolve_app_path(
            env::var_os("DEMUCS_WORKER_SCRIPT")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("worker/demucs_worker.py")),
            &application_root,
        );
        let beat_this_mel_model = resolve_app_path(
            env::var_os("BEAT_THIS_MEL_MODEL")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("models/mel_spectrogram.onnx")),
            &application_root,
        );
        let beat_this_model = resolve_app_path(
            env::var_os("BEAT_THIS_MODEL")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("models/beat_this.onnx")),
            &application_root,
        );

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
            demucs_python,
            demucs_worker_script,
            demucs_cancel_timeout: Duration::from_secs(value(
                "DEMUCS_CANCEL_TIMEOUT_SECS",
                60_u64,
            )?),
            demucs_startup_timeout: Duration::from_secs(value(
                "DEMUCS_STARTUP_TIMEOUT_SECS",
                600_u64,
            )?),
            beat_this_mel_model,
            beat_this_model,
            beat_this_model_version: env::var("BEAT_THIS_MODEL_VERSION")
                .ok()
                .filter(|value| !value.trim().is_empty()),
            beat_this_threads,
            audio_job_ttl: Duration::from_secs(audio_job_ttl_secs),
        })
    }
}

fn application_root() -> Result<PathBuf, String> {
    let root = if let Some(root) = env::var_os("DUBPAD_APP_ROOT") {
        PathBuf::from(root)
    } else {
        env::current_exe()
            .ok()
            .and_then(|path| {
                path.ancestors().find_map(|ancestor| {
                    (ancestor.join("worker/demucs_worker.py").is_file()
                        && ancestor.join("scripts/setup_demucs_runtime.py").is_file())
                    .then(|| ancestor.to_path_buf())
                })
            })
            .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")))
    };
    root.canonicalize().map_err(|error| {
        format!(
            "Could not resolve DubPad application root {}: {error}",
            root.display()
        )
    })
}

fn resolve_app_path(path: PathBuf, application_root: &Path) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        application_root.join(path)
    }
}

fn resolve_demucs_python(application_root: &Path) -> Result<PathBuf, String> {
    let setup_script = application_root.join("scripts/setup_demucs_runtime.py");
    let configured_bootstrap = env::var_os("DEMUCS_PYTHON").and_then(|value| {
        let configured = PathBuf::from(value);
        if configured.is_absolute() || configured.components().count() > 1 {
            let path = if configured.is_absolute() {
                configured
            } else {
                application_root.join(configured)
            };
            path.is_file().then_some(path)
        } else {
            find_on_path(configured.to_str()?)
        }
    });
    let bootstrap_python = configured_bootstrap.or_else(|| {
        ["python3", "python"]
        .into_iter()
        .find_map(|name| find_on_path(name))
    }).ok_or_else(|| {
            "Python 3 is required to detect or set up the Demucs runtime. Install Python 3 or set DEMUCS_PYTHON to an existing interpreter.".to_string()
        })?;
    let output = Command::new(&bootstrap_python)
        .arg(&setup_script)
        .arg("--ensure")
        .arg("--json")
        .current_dir(application_root)
        .output()
        .map_err(|error| format!("Could not run Demucs runtime setup script: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "Demucs Python setup failed: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let result: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("Demucs setup returned invalid JSON: {error}"))?;
    let python = result
        .get("python")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            "Demucs setup response did not include the selected Python path".to_string()
        })?;
    let action = result
        .get("action")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("resolved");
    let report = result
        .get("report")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    tracing::info!(python, action, report, "Demucs Python runtime configured");
    Ok(PathBuf::from(python))
}

fn find_on_path(executable: &str) -> Option<PathBuf> {
    env::var_os("PATH").and_then(|paths| {
        env::split_paths(&paths)
            .map(|directory| directory.join(executable))
            .find(|candidate| candidate.is_file())
    })
}

fn value<T: FromStr>(key: &str, default: T) -> Result<T, String> {
    match env::var(key) {
        Ok(value) => value
            .parse()
            .map_err(|_| format!("{key} has an invalid value")),
        Err(_) => Ok(default),
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::resolve_app_path;

    #[test]
    fn relative_worker_and_model_paths_are_anchored_to_app_root() {
        let root = Path::new("/opt/dubpad");
        assert_eq!(
            resolve_app_path("worker/demucs_worker.py".into(), root),
            root.join("worker/demucs_worker.py")
        );
    }

    #[test]
    fn absolute_worker_and_model_overrides_are_preserved() {
        let path = Path::new("/srv/models/beat.onnx");
        assert_eq!(
            resolve_app_path(path.to_path_buf(), Path::new("/opt/dubpad")),
            path
        );
    }
}
