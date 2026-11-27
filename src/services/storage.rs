use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

use sqlx::PgPool;
use tokio::{fs, io::AsyncWriteExt};
use uuid::Uuid;

#[derive(Default)]
struct StreamState {
    readers: usize,
    deleting: bool,
}
static ACTIVE_STREAMS: OnceLock<Mutex<HashMap<Uuid, StreamState>>> = OnceLock::new();

fn stream_states() -> &'static Mutex<HashMap<Uuid, StreamState>> {
    ACTIVE_STREAMS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub struct AudioFileLease {
    job_id: Uuid,
}
pub struct AudioCleanupLease {
    job_id: Uuid,
}

pub fn acquire_audio_stream(job_id: Uuid) -> Option<AudioFileLease> {
    let mut states = stream_states().lock().ok()?;
    let state = states.entry(job_id).or_default();
    if state.deleting {
        return None;
    }
    state.readers += 1;
    Some(AudioFileLease { job_id })
}

pub fn begin_audio_cleanup(job_id: Uuid) -> Option<AudioCleanupLease> {
    let mut states = stream_states().lock().ok()?;
    let state = states.entry(job_id).or_default();
    if state.deleting || state.readers != 0 {
        return None;
    }
    state.deleting = true;
    Some(AudioCleanupLease { job_id })
}

impl Drop for AudioFileLease {
    fn drop(&mut self) {
        if let Ok(mut states) = stream_states().lock() {
            let mut remove = false;
            if let Some(state) = states.get_mut(&self.job_id) {
                state.readers = state.readers.saturating_sub(1);
                remove = state.readers == 0 && !state.deleting;
            }
            if remove {
                states.remove(&self.job_id);
            }
        }
    }
}

impl Drop for AudioCleanupLease {
    fn drop(&mut self) {
        if let Ok(mut states) = stream_states().lock() {
            let mut remove = false;
            if let Some(state) = states.get_mut(&self.job_id) {
                state.deleting = false;
                remove = state.readers == 0;
            }
            if remove {
                states.remove(&self.job_id);
            }
        }
    }
}

pub async fn prepare_root(root: &Path) -> Result<(), std::io::Error> {
    fs::create_dir_all(root).await?;
    fs::create_dir_all(crate::services::project_storage::projects_root(root)).await?;
    cleanup_incomplete(root).await
}

async fn cleanup_incomplete(directory: &Path) -> Result<(), std::io::Error> {
    let mut entries = fs::read_dir(directory).await?;
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        let kind = entry.file_type().await?;
        if kind.is_dir() {
            Box::pin(cleanup_incomplete(&path)).await?;
        } else if path.extension().and_then(|extension| extension.to_str()) == Some("uploading") {
            fs::remove_file(path).await?;
        }
    }
    Ok(())
}

pub fn safe_extension(filename: &str) -> Option<&'static str> {
    let extension = Path::new(filename)
        .extension()?
        .to_str()?
        .to_ascii_lowercase();
    match extension.as_str() {
        "wav" | "wave" => Some("wav"),
        "mp3" => Some("mp3"),
        "flac" => Some("flac"),
        "ogg" | "oga" => Some("ogg"),
        "m4a" | "mp4" => Some("m4a"),
        "aac" => Some("aac"),
        _ => None,
    }
}

pub fn upload_path(root: &Path, project_id: Uuid, asset_id: Uuid, extension: &str) -> PathBuf {
    root.join(project_id.to_string())
        .join(format!("{asset_id}.{extension}"))
}

pub async fn write_chunk(file: &mut tokio::fs::File, bytes: &[u8]) -> Result<(), std::io::Error> {
    file.write_all(bytes).await
}

/// The inference worker stores outputs beside UPLOAD_DIR in `jobs/{job_id}`.
/// The job UUID and fixed stem filename are appended by the stem route only.
pub fn jobs_root(upload_dir: &Path) -> PathBuf {
    upload_dir
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("jobs")
}

/// Remove expired projects and their owned assets. A deletion marker is claimed
/// under a row lock first, so concurrent cleanup loops and project writes cannot
/// race. Files are removed before the cascading database delete; failures clear
/// the marker so a later sweep can retry.
pub async fn cleanup_expired_projects(pool: &PgPool, upload_dir: &Path) -> Result<(), String> {
    cleanup_incomplete(upload_dir)
        .await
        .map_err(|error| format!("remove incomplete upload files: {error}"))?;
    let mut tx = pool
        .begin()
        .await
        .map_err(|error| format!("begin project cleanup claim: {error}"))?;
    let projects = sqlx::query_as::<_, (Uuid,)>("SELECT p.id FROM projects p WHERE p.expires_at<=now() AND (p.deleting_at IS NULL OR p.deleting_at<now()-interval '10 minutes') AND NOT EXISTS (SELECT 1 FROM processing_jobs j WHERE j.project_id=p.id AND j.status IN ('QUEUED','RUNNING')) ORDER BY p.expires_at LIMIT 100 FOR UPDATE SKIP LOCKED")
        .fetch_all(&mut *tx).await.map_err(|error| format!("claim expired projects: {error}"))?;
    for (project_id,) in &projects {
        sqlx::query("UPDATE projects SET deleting_at=now() WHERE id=$1")
            .bind(project_id)
            .execute(&mut *tx)
            .await
            .map_err(|error| format!("mark project for cleanup: {error}"))?;
    }
    tx.commit()
        .await
        .map_err(|error| format!("commit project cleanup claims: {error}"))?;
    for (project_id,) in projects {
        let Some(_cleanup_lease) = begin_audio_cleanup(project_id) else {
            let _ = sqlx::query("UPDATE projects SET deleting_at=NULL WHERE id=$1")
                .bind(project_id)
                .execute(pool)
                .await;
            continue;
        };
        let jobs =
            sqlx::query_as::<_, (Uuid,)>("SELECT id FROM processing_jobs WHERE project_id=$1")
                .bind(project_id)
                .fetch_all(pool)
                .await
                .map_err(|error| format!("read project jobs for cleanup: {error}"))?;
        let mut cleanup_ok = true;
        for (job_id,) in jobs {
            match fs::remove_dir_all(jobs_root(upload_dir).join(job_id.to_string())).await {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    tracing::warn!(%project_id, %job_id, %error, "project job directory cleanup failed");
                    cleanup_ok = false;
                }
            }
        }
        for directory in [
            upload_dir.join(project_id.to_string()),
            crate::services::project_storage::projects_root(upload_dir)
                .join(project_id.to_string()),
        ] {
            match fs::remove_dir_all(&directory).await {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    tracing::warn!(%project_id, path=%directory.display(), %error, "project asset directory cleanup failed");
                    cleanup_ok = false;
                }
            }
        }
        if cleanup_ok {
            sqlx::query("DELETE FROM projects WHERE id=$1 AND deleting_at IS NOT NULL AND expires_at<=now() AND NOT EXISTS (SELECT 1 FROM processing_jobs WHERE project_id=$1 AND status IN ('QUEUED','RUNNING'))")
                .bind(project_id).execute(pool).await.map_err(|error| format!("delete expired project metadata: {error}"))?;
            tracing::info!(%project_id, "expired project and files cleaned");
        } else {
            sqlx::query("UPDATE projects SET deleting_at=NULL WHERE id=$1")
                .bind(project_id)
                .execute(pool)
                .await
                .map_err(|error| format!("release project cleanup claim: {error}"))?;
        }
    }
    Ok(())
}

pub fn validate_upload_size(size: u64, maximum: u64) -> Result<(), ()> {
    if size > maximum { Err(()) } else { Ok(()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upload_validation_accepts_supported_audio_extensions_case_insensitively() {
        assert_eq!(safe_extension("mix.WAV"), Some("wav"));
        assert_eq!(safe_extension("audio.flac"), Some("flac"));
        assert_eq!(safe_extension("../unsafe.exe"), None);
        assert_eq!(safe_extension("no-extension"), None);
    }

    #[test]
    fn upload_validation_enforces_streamed_size_cap() {
        assert!(validate_upload_size(1024, 1024).is_ok());
        assert!(validate_upload_size(1025, 1024).is_err());
    }

    #[test]
    fn temporary_job_cleanup_waits_for_stream_leases_and_blocks_new_readers() {
        let job = Uuid::new_v4();
        let stream = acquire_audio_stream(job).unwrap();
        assert!(begin_audio_cleanup(job).is_none());
        drop(stream);
        let cleanup = begin_audio_cleanup(job).unwrap();
        assert!(acquire_audio_stream(job).is_none());
        drop(cleanup);
        assert!(acquire_audio_stream(job).is_some());
    }
}
