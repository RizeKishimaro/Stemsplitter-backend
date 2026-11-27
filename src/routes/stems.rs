use std::{
    io,
    path::{Path, PathBuf},
};

use axum::{
    body::{Body, Bytes},
    extract::{Path as AxumPath, State},
    http::{HeaderMap, HeaderValue, Response, StatusCode, header},
};
use futures_util::stream;
use serde_json::Value;
use tokio::{
    fs,
    io::{AsyncReadExt, AsyncSeekExt, SeekFrom},
};
use uuid::Uuid;

use crate::{
    error::AppError,
    models::{
        job::ProcessingJob,
        stem::{ProjectStem, StemListResponse, StemMetadata},
    },
    services::storage,
    state::AppState,
};

const STEM_NAMES: [&str; 4] = ["drums", "bass", "other", "vocals"];

#[utoipa::path(
    get,
    path = "/api/v1/jobs/{job_id}/stems",
    params(("job_id" = Uuid, Path, description = "Successful separation job UUID")),
    responses(
        (status = 200, description = "Available stems and audio metadata", body = StemListResponse),
        (status = 404, description = "Job or stem files not found"),
        (status = 409, description = "Job has not succeeded")
    )
)]
pub async fn list_stems(
    State(state): State<AppState>,
    AxumPath(job_id): AxumPath<String>,
) -> Result<axum::Json<StemListResponse>, AppError> {
    let job_id = parse_job_id(&job_id)?;
    let job = get_job(&state, job_id).await?;
    require_succeeded(Some(job.status.as_str()))?;
    let rows = sqlx::query_as::<_, ProjectStem>("SELECT project_id,job_id,stem_name,storage_key,sample_rate,channels,frame_count,file_size_bytes,created_at FROM project_stems WHERE job_id=$1 ORDER BY stem_name")
        .bind(job_id).fetch_all(&state.db).await?;
    let response = if rows.is_empty() {
        let root = storage::jobs_root(&state.config.upload_dir);
        list_from_result(job_id, &job.status, &job.result_metadata, &root).await?
    } else {
        project_stem_list(job_id, &job.status, rows)
    };
    Ok(axum::Json(response))
}

#[utoipa::path(
    get,
    path = "/api/v1/jobs/{job_id}/stems/{stem_name}",
    params(
        ("job_id" = Uuid, Path, description = "Successful separation job UUID"),
        ("stem_name" = String, Path, description = "One of drums, bass, other, vocals"),
        ("Range" = Option<String>, Header, description = "Optional single byte range, for example bytes=0-1023 or bytes=-1024")
    ),
    responses(
        (status = 200, description = "WAV audio stream; full file when Range is absent", body = String, content_type = "audio/wav"),
        (status = 206, description = "Requested WAV byte range; includes Content-Range, Accept-Ranges, and Content-Length", body = String, content_type = "audio/wav"),
        (status = 400, description = "Invalid stem name"),
        (status = 404, description = "Job or stem file not found"),
        (status = 409, description = "Job has not succeeded"),
        (status = 416, description = "Invalid or unsatisfiable Range; Content-Range is bytes */file-size")
    )
)]
pub async fn get_stem(
    State(state): State<AppState>,
    AxumPath((job_id, stem_name)): AxumPath<(String, String)>,
    headers: HeaderMap,
) -> Result<Response<Body>, AppError> {
    let job_id = parse_job_id(&job_id)?;
    let stem_name = validate_stem_name(&stem_name)?;
    let job = get_job(&state, job_id).await?;
    require_succeeded(Some(job.status.as_str()))?;
    let path = if let Some(row) = sqlx::query_as::<_, ProjectStem>("SELECT project_id,job_id,stem_name,storage_key,sample_rate,channels,frame_count,file_size_bytes,created_at FROM project_stems WHERE job_id=$1 AND stem_name=$2")
        .bind(job_id).bind(stem_name).fetch_optional(&state.db).await? {
        resolve_project_stem(&state, &row).await?
    } else {
        resolve_stem_file(
            &storage::jobs_root(&state.config.upload_dir),
            job_id,
            stem_name,
        )
        .await?
    };
    let range_header = headers
        .get(header::RANGE)
        .map(|value| value.to_str().unwrap_or(""));
    let lease = storage::acquire_audio_stream(job_id).ok_or(AppError::NotFound)?;
    stream_stem_with_lease(&path, range_header, Some(lease)).await
}

#[utoipa::path(
    get,
    path = "/api/v1/jobs/{job_id}/stems/{stem_name}/partial",
    params(
        ("job_id" = Uuid, Path, description = "Running or completed separation job UUID"),
        ("stem_name" = String, Path, description = "One of drums, bass, other, vocals"),
        ("Range" = Option<String>, Header, description = "Optional single byte range")
    ),
    responses(
        (status = 200, description = "Incrementally available WAV stem", content_type = "audio/wav"),
        (status = 206, description = "Partial WAV response"),
        (status = 404, description = "Job or partial stem not found"),
        (status = 409, description = "Job is not running or succeeded")
    )
)]
pub async fn get_partial_stem(
    State(state): State<AppState>,
    AxumPath((raw_job, stem_name)): AxumPath<(String, String)>,
    headers: HeaderMap,
) -> Result<Response<Body>, AppError> {
    let job_id = parse_job_id(&raw_job)?;
    let stem_name = validate_stem_name(&stem_name)?;
    let job = get_job(&state, job_id).await?;
    if !matches!(job.status.as_str(), "RUNNING" | "SUCCEEDED") {
        return Err(AppError::Conflict("partial stems are only available while a job is running or after it succeeds".into()));
    }

    let root = storage::jobs_root(&state.config.upload_dir);
    let canonical_root = fs::canonicalize(&root).await.map_err(map_missing_file)?;
    let partial_dir = fs::canonicalize(canonical_root.join(job_id.to_string()).join("partial"))
        .await
        .map_err(map_missing_file)?;
    if !partial_dir.starts_with(&canonical_root) {
        return Err(AppError::NotFound);
    }
    let path = fs::canonicalize(partial_dir.join(format!("{stem_name}.wav")))
        .await
        .map_err(map_missing_file)?;
    if !path.starts_with(&partial_dir) {
        return Err(AppError::NotFound);
    }
    let range_header = headers
        .get(header::RANGE)
        .map(|value| value.to_str().unwrap_or(""));
    let lease = storage::acquire_audio_stream(job_id).ok_or(AppError::NotFound)?;
    stream_stem_with_lease(&path, range_header, Some(lease)).await
}

#[utoipa::path(
    get,
    path = "/api/v1/projects/{project_id}/stems/{stem_name}",
    params(("project_id" = Uuid, Path, description = "Project UUID"), ("stem_name" = String, Path, description = "One of drums, bass, other, vocals"), ("Range" = Option<String>, Header, description = "Optional single byte range")),
    responses((status = 200, description = "Persistent full-length WAV stem", content_type = "audio/wav"), (status = 206, description = "Partial WAV response"), (status = 404, description = "Project stem not found"), (status = 416, description = "Invalid or unsatisfiable range"))
)]
pub async fn get_project_stem(
    State(state): State<AppState>,
    AxumPath((raw_project, stem_name)): AxumPath<(String, String)>,
    headers: HeaderMap,
) -> Result<Response<Body>, AppError> {
    let project_id = Uuid::parse_str(&raw_project)
        .map_err(|_| AppError::BadRequest("project_id must be a UUID".into()))?;
    let stem_name = validate_stem_name(&stem_name)?;
    let row = sqlx::query_as::<_, ProjectStem>("SELECT project_id,job_id,stem_name,storage_key,sample_rate,channels,frame_count,file_size_bytes,created_at FROM project_stems WHERE project_id=$1 AND stem_name=$2")
        .bind(project_id).bind(stem_name).fetch_optional(&state.db).await?.ok_or(AppError::NotFound)?;
    let path = resolve_project_stem(&state, &row).await?;
    let range_header = headers
        .get(header::RANGE)
        .map(|value| value.to_str().unwrap_or(""));
    let lease = storage::acquire_audio_stream(row.job_id).ok_or(AppError::NotFound)?;
    stream_stem_with_lease(&path, range_header, Some(lease)).await
}

async fn resolve_project_stem(state: &AppState, stem: &ProjectStem) -> Result<PathBuf, AppError> {
    let root = crate::services::project_storage::projects_root(&state.config.upload_dir);
    let candidate = crate::services::project_storage::resolve_storage_key(&root, &stem.storage_key)
        .map_err(AppError::Internal)?;
    let canonical_root = fs::canonicalize(root).await.map_err(map_missing_file)?;
    let canonical_path = fs::canonicalize(candidate)
        .await
        .map_err(map_missing_file)?;
    if !canonical_path.starts_with(&canonical_root) {
        return Err(AppError::NotFound);
    }
    Ok(canonical_path)
}

fn project_stem_list(job_id: Uuid, status: &str, rows: Vec<ProjectStem>) -> StemListResponse {
    StemListResponse {
        job_id,
        job_status: status.to_owned(),
        stems: rows
            .into_iter()
            .map(|row| StemMetadata {
                name: row.stem_name,
                sample_rate: row.sample_rate as u32,
                channels: row.channels as u16,
                frame_count: row.frame_count as u64,
                duration_seconds: row.frame_count as f64 / f64::from(row.sample_rate),
                audio_format: "wav".into(),
                file_size_bytes: Some(row.file_size_bytes as u64),
            })
            .collect(),
    }
}

#[cfg(test)]
pub(crate) async fn stream_stem(
    path: &Path,
    range_header: Option<&str>,
) -> Result<Response<Body>, AppError> {
    stream_stem_with_lease(path, range_header, None).await
}

pub(crate) async fn stream_stem_with_lease(
    path: &Path,
    range_header: Option<&str>,
    lease: Option<storage::AudioFileLease>,
) -> Result<Response<Body>, AppError> {
    let file_size = fs::metadata(&path).await.map_err(map_missing_file)?.len();
    let byte_range = match parse_byte_range(range_header, file_size) {
        Ok(value) => value,
        Err(()) => return Ok(range_not_satisfiable(file_size)),
    };
    let (start, length, status) = match byte_range {
        Some(range) => (
            range.start,
            range.end - range.start + 1,
            StatusCode::PARTIAL_CONTENT,
        ),
        None => (0, file_size, StatusCode::OK),
    };
    let mut file = fs::File::open(path).await.map_err(map_missing_file)?;
    file.seek(SeekFrom::Start(start)).await?;
    let body = stream::try_unfold(
        (file, length, lease),
        |(mut file, remaining, lease)| async move {
            if remaining == 0 {
                return Ok::<_, io::Error>(None);
            }
            let mut buffer = vec![0_u8; remaining.min(64 * 1024) as usize];
            let read = file.read(&mut buffer).await?;
            if read == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "stem file ended before its reported length",
                ));
            }
            buffer.truncate(read);
            Ok(Some((
                Bytes::from(buffer),
                (file, remaining - read as u64, lease),
            )))
        },
    );
    let mut response = Response::new(Body::from_stream(body));
    *response.status_mut() = status;
    let response_headers = response.headers_mut();
    response_headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("audio/wav"));
    response_headers.insert(header::ACCEPT_RANGES, HeaderValue::from_static("bytes"));
    response_headers.insert(header::CONTENT_LENGTH, header_value(length)?);
    if let Some(range) = byte_range {
        response_headers.insert(
            header::CONTENT_RANGE,
            header_value(format!("bytes {}-{}/{}", range.start, range.end, file_size))?,
        );
    }
    Ok(response)
}

async fn get_job(state: &AppState, id: Uuid) -> Result<ProcessingJob, AppError> {
    sqlx::query_as::<_, ProcessingJob>("SELECT * FROM processing_jobs WHERE id = $1")
        .bind(id)
        .fetch_optional(&state.db)
        .await?
        .ok_or(AppError::NotFound)
}

fn parse_job_id(raw: &str) -> Result<Uuid, AppError> {
    Uuid::parse_str(raw).map_err(|_| AppError::BadRequest("job_id must be a UUID".into()))
}

fn validate_stem_name(name: &str) -> Result<&'static str, AppError> {
    STEM_NAMES
        .iter()
        .copied()
        .find(|allowed| *allowed == name)
        .ok_or_else(|| {
            AppError::BadRequest("stem_name must be drums, bass, other, or vocals".into())
        })
}

fn require_succeeded(status: Option<&str>) -> Result<(), AppError> {
    match status {
        None => Err(AppError::NotFound),
        Some("SUCCEEDED") => Ok(()),
        Some(_) => Err(AppError::Conflict(
            "stems are available only after a job succeeds".into(),
        )),
    }
}

async fn list_from_result(
    job_id: Uuid,
    status: &str,
    result: &Value,
    root: &Path,
) -> Result<StemListResponse, AppError> {
    require_succeeded(Some(status))?;
    let mut stems = Vec::new();
    for name in STEM_NAMES {
        let path = match resolve_stem_file(root, job_id, name).await {
            Ok(path) => path,
            Err(AppError::NotFound) => continue,
            Err(error) => return Err(error),
        };
        let Some(metadata) = result.get("stems").and_then(|items| items.get(name)) else {
            tracing::warn!(%job_id, stem = name, "stem file exists without result metadata");
            continue;
        };
        let sample_rate = json_u64(metadata, "sample_rate")?;
        let channels = json_u64(metadata, "channels")?;
        let frame_count = json_u64(metadata, "frame_count")?;
        if sample_rate == 0 || channels == 0 || frame_count == 0 {
            return Err(AppError::Internal(format!(
                "invalid metadata stored for stem {name}"
            )));
        }
        let file_size_bytes = fs::metadata(path).await.map_err(map_missing_file)?.len();
        stems.push(StemMetadata {
            name: name.to_owned(),
            sample_rate: u32::try_from(sample_rate)
                .map_err(|_| AppError::Internal("invalid stem sample rate".into()))?,
            channels: u16::try_from(channels)
                .map_err(|_| AppError::Internal("invalid stem channel count".into()))?,
            frame_count,
            duration_seconds: frame_count as f64 / sample_rate as f64,
            audio_format: "wav".into(),
            file_size_bytes: Some(file_size_bytes),
        });
    }
    Ok(StemListResponse {
        job_id,
        job_status: status.to_owned(),
        stems,
    })
}

fn json_u64(value: &Value, key: &str) -> Result<u64, AppError> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| AppError::Internal(format!("stem metadata is missing valid {key}")))
}

async fn resolve_stem_file(
    root: &Path,
    job_id: Uuid,
    stem_name: &str,
) -> Result<PathBuf, AppError> {
    let canonical_root = fs::canonicalize(root).await.map_err(map_missing_file)?;
    let candidate_job_dir = canonical_root.join(job_id.to_string());
    let job_dir = fs::canonicalize(candidate_job_dir)
        .await
        .map_err(map_missing_file)?;
    if !job_dir.starts_with(&canonical_root) {
        return Err(AppError::NotFound);
    }
    let candidate = job_dir.join(format!("{stem_name}.wav"));
    let path = fs::canonicalize(candidate)
        .await
        .map_err(map_missing_file)?;
    if !path.starts_with(&job_dir)
        || !fs::metadata(&path)
            .await
            .map_err(map_missing_file)?
            .is_file()
    {
        return Err(AppError::NotFound);
    }
    Ok(path)
}

fn map_missing_file(error: io::Error) -> AppError {
    if error.kind() == io::ErrorKind::NotFound {
        AppError::NotFound
    } else {
        AppError::Io(error)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ByteRange {
    start: u64,
    end: u64,
}

fn parse_byte_range(header_value: Option<&str>, size: u64) -> Result<Option<ByteRange>, ()> {
    let Some(value) = header_value else {
        return Ok(None);
    };
    let value = value.strip_prefix("bytes=").ok_or(())?;
    if value.contains(',') || size == 0 {
        return Err(());
    }
    let (first, last) = value.split_once('-').ok_or(())?;
    if first.is_empty() {
        let suffix = last.parse::<u64>().map_err(|_| ())?;
        if suffix == 0 {
            return Err(());
        }
        return Ok(Some(ByteRange {
            start: size.saturating_sub(suffix),
            end: size - 1,
        }));
    }
    let start = first.parse::<u64>().map_err(|_| ())?;
    if start >= size {
        return Err(());
    }
    let end = if last.is_empty() {
        size - 1
    } else {
        last.parse::<u64>().map_err(|_| ())?
    };
    if end < start {
        return Err(());
    }
    Ok(Some(ByteRange {
        start,
        end: end.min(size - 1),
    }))
}

fn range_not_satisfiable(size: u64) -> Response<Body> {
    let mut response = Response::new(Body::empty());
    *response.status_mut() = StatusCode::RANGE_NOT_SATISFIABLE;
    response
        .headers_mut()
        .insert(header::ACCEPT_RANGES, HeaderValue::from_static("bytes"));
    response.headers_mut().insert(
        header::CONTENT_RANGE,
        HeaderValue::from_str(&format!("bytes */{size}")).expect("numeric header is valid"),
    );
    response
        .headers_mut()
        .insert(header::CONTENT_LENGTH, HeaderValue::from_static("0"));
    response
}

fn header_value(value: impl ToString) -> Result<HeaderValue, AppError> {
    HeaderValue::from_str(&value.to_string()).map_err(|error| AppError::Internal(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use http_body_util::BodyExt;
    use std::{
        fs as std_fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn temp_root() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("soniclab-stems-{unique}"));
        std_fs::create_dir_all(&root).unwrap();
        root
    }

    fn write_stem(root: &Path, job_id: Uuid, name: &str, data: &[u8]) {
        let directory = root.join(job_id.to_string());
        std_fs::create_dir_all(&directory).unwrap();
        std_fs::write(directory.join(format!("{name}.wav")), data).unwrap();
    }

    #[test]
    fn validates_only_the_four_known_stem_names() {
        assert_eq!(validate_stem_name("vocals").unwrap(), "vocals");
        assert!(validate_stem_name("../secret").is_err());
        assert!(validate_stem_name("drums.wav").is_err());
    }

    #[test]
    fn missing_jobs_and_non_successful_jobs_are_rejected() {
        assert!(matches!(require_succeeded(None), Err(AppError::NotFound)));
        assert!(matches!(
            require_succeeded(Some("RUNNING")),
            Err(AppError::Conflict(_))
        ));
        assert!(require_succeeded(Some("SUCCEEDED")).is_ok());
    }

    #[test]
    fn parses_full_open_suffix_and_clamped_ranges_and_rejects_bad_ranges() {
        assert_eq!(parse_byte_range(None, 10).unwrap(), None);
        assert_eq!(
            parse_byte_range(Some("bytes=2-5"), 10).unwrap(),
            Some(ByteRange { start: 2, end: 5 })
        );
        assert_eq!(
            parse_byte_range(Some("bytes=7-"), 10).unwrap(),
            Some(ByteRange { start: 7, end: 9 })
        );
        assert_eq!(
            parse_byte_range(Some("bytes=-3"), 10).unwrap(),
            Some(ByteRange { start: 7, end: 9 })
        );
        assert_eq!(
            parse_byte_range(Some("bytes=8-99"), 10).unwrap(),
            Some(ByteRange { start: 8, end: 9 })
        );
        for bad in [
            "items=0-1",
            "bytes=10-",
            "bytes=7-3",
            "bytes=-0",
            "bytes=0-1,4-5",
            "bytes=nope",
        ] {
            assert!(parse_byte_range(Some(bad), 10).is_err(), "{bad}");
        }
    }

    #[tokio::test]
    async fn lists_existing_stems_with_audio_metadata_and_size() {
        let root = temp_root();
        let id = Uuid::new_v4();
        write_stem(&root, id, "drums", b"RIFFexample");
        let result = serde_json::json!({"stems":{"drums":{"sample_rate":48000,"channels":2,"frame_count":96000,"path":"/must/not/be/trusted"}}});
        let response = list_from_result(id, "SUCCEEDED", &result, &root)
            .await
            .unwrap();
        assert_eq!(response.job_id, id);
        assert_eq!(response.job_status, "SUCCEEDED");
        assert_eq!(response.stems.len(), 1);
        assert_eq!(response.stems[0].sample_rate, 48_000);
        assert_eq!(response.stems[0].channels, 2);
        assert_eq!(response.stems[0].frame_count, 96_000);
        assert_eq!(response.stems[0].duration_seconds, 2.0);
        assert_eq!(response.stems[0].file_size_bytes, Some(11));
        std_fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn safely_resolves_and_streams_full_and_partial_wav_responses() {
        let root = temp_root();
        let id = Uuid::new_v4();
        let bytes = b"0123456789";
        write_stem(&root, id, "bass", bytes);
        let file_path = resolve_stem_file(&root, id, "bass").await.unwrap();
        let full = stream_stem(&file_path, None).await.unwrap();
        assert_eq!(full.status(), StatusCode::OK);
        assert_eq!(full.headers()[header::CONTENT_TYPE], "audio/wav");
        assert_eq!(full.headers()[header::ACCEPT_RANGES], "bytes");
        assert_eq!(full.headers()[header::CONTENT_LENGTH], "10");
        assert_eq!(
            &full.into_body().collect().await.unwrap().to_bytes()[..],
            bytes
        );

        let response = stream_stem(&file_path, Some("bytes=3-6")).await.unwrap();
        assert_eq!(response.status(), StatusCode::PARTIAL_CONTENT);
        assert_eq!(response.headers()[header::CONTENT_TYPE], "audio/wav");
        assert_eq!(response.headers()[header::ACCEPT_RANGES], "bytes");
        assert_eq!(response.headers()[header::CONTENT_RANGE], "bytes 3-6/10");
        assert_eq!(response.headers()[header::CONTENT_LENGTH], "4");
        let body = response.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(&body[..], b"3456");
        std_fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn missing_stem_file_is_not_found_and_symlink_escape_is_blocked() {
        let root = temp_root();
        let id = Uuid::new_v4();
        assert!(matches!(
            resolve_stem_file(&root, id, "vocals").await,
            Err(AppError::NotFound)
        ));
        std_fs::create_dir_all(root.join(id.to_string())).unwrap();
        assert!(matches!(
            resolve_stem_file(&root, id, "vocals").await,
            Err(AppError::NotFound)
        ));
        let outside = root.join("outside.wav");
        std_fs::write(&outside, b"secret").unwrap();
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&outside, root.join(id.to_string()).join("vocals.wav"))
                .unwrap();
            assert!(matches!(
                resolve_stem_file(&root, id, "vocals").await,
                Err(AppError::NotFound)
            ));
        }
        std_fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn invalid_and_unsatisfiable_ranges_return_416() {
        let root = temp_root();
        let id = Uuid::new_v4();
        write_stem(&root, id, "other", b"0123456789");
        let path = resolve_stem_file(&root, id, "other").await.unwrap();
        let response = stream_stem(&path, Some("bytes=99-100")).await.unwrap();
        assert_eq!(response.status(), StatusCode::RANGE_NOT_SATISFIABLE);
        assert_eq!(response.headers()[header::CONTENT_RANGE], "bytes */10");
        let malformed = stream_stem(&path, Some("bytes=bad")).await.unwrap();
        assert_eq!(malformed.status(), StatusCode::RANGE_NOT_SATISFIABLE);
        std_fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn successful_job_is_required_for_stem_listing() {
        let root = temp_root();
        let error = list_from_result(Uuid::new_v4(), "RUNNING", &Value::Null, &root)
            .await
            .unwrap_err();
        assert!(matches!(error, AppError::Conflict(_)));
        std_fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn unsatisfiable_response_advertises_unsatisfied_range_and_zero_length() {
        let response = range_not_satisfiable(10);
        assert_eq!(response.status(), StatusCode::RANGE_NOT_SATISFIABLE);
        assert_eq!(response.headers()[header::CONTENT_RANGE], "bytes */10");
        assert_eq!(response.headers()[header::ACCEPT_RANGES], "bytes");
        assert_eq!(response.headers()[header::CONTENT_LENGTH], "0");
        let _ = Body::empty();
    }
}
