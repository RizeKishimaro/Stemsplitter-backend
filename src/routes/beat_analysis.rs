use std::{
    io::{Read, Seek},
    path::PathBuf,
    sync::{Mutex, OnceLock},
    time::Instant,
};

use axum::{
    Json,
    extract::{Path, Query, State},
    http::StatusCode,
};
use beat_this::{BeatStream, BeatThis, Model, RtenRuntime, Runtime};
use serde_json::{Value, json};
use sqlx::PgPool;
use tokio::sync::Semaphore;
use uuid::Uuid;

use crate::{
    error::AppError,
    models::{beat_analysis::BeatAnalysisJob, job::ProcessingJob},
    services::storage,
    state::AppState,
};

static BEAT_ANALYSIS_LIMIT: Semaphore = Semaphore::const_new(1);
type BeatModel = <RtenRuntime as Runtime>::Model;
#[derive(serde::Deserialize)]
pub struct BeatAnalysisQuery {
    input_stem: Option<String>,
}

struct CachedBeatDetector {
    mel_path: PathBuf,
    beat_path: PathBuf,
    detector: BeatThis<BeatModel>,
}
static BEAT_DETECTOR: OnceLock<Mutex<Option<CachedBeatDetector>>> = OnceLock::new();

#[utoipa::path(
    post,
    path = "/api/v1/jobs/{job_id}/beat-analysis",
    params(
        ("job_id" = Uuid, Path, description = "Successful HTDemucs separation job UUID"),
        ("input_stem" = Option<String>, Query, description = "Analysis source: drums (default) or mix")
    ),
    responses(
        (status = 202, description = "Beat analysis queued against the selected audio source", body = BeatAnalysisJob),
        (status = 404, description = "Separation job or selected audio source not found"),
        (status = 409, description = "Separation job has not succeeded")
    )
)]
pub async fn create_beat_analysis(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
    Query(query): Query<BeatAnalysisQuery>,
) -> Result<(StatusCode, Json<BeatAnalysisJob>), AppError> {
    let job_id = parse_uuid(&job_id)?;
    let input_stem = query.input_stem.unwrap_or_else(|| "drums".into());
    if !matches!(input_stem.as_str(), "drums" | "mix") {
        return Err(AppError::BadRequest(
            "input_stem must be either drums or mix".into(),
        ));
    }
    let job = sqlx::query_as::<_, ProcessingJob>("SELECT * FROM processing_jobs WHERE id = $1")
        .bind(job_id)
        .fetch_optional(&state.db)
        .await?
        .ok_or(AppError::NotFound)?;
    if job.status != "SUCCEEDED" {
        return Err(AppError::Conflict(
            "separation job must succeed before beat analysis".into(),
        ));
    }

    let (path, sample_rate, channels, frame_count) = if input_stem == "mix" {
        let (stored_path, rate, channel_count, frames) =
            sqlx::query_as::<_, (String, i32, i16, i64)>(
                "SELECT stored_path,sample_rate,channels,frame_count FROM audio_assets WHERE id=$1 AND project_id=$2",
            )
            .bind(job.audio_asset_id)
            .bind(job.project_id)
            .fetch_optional(&state.db)
            .await?
            .ok_or(AppError::NotFound)?;
        let path = tokio::fs::canonicalize(stored_path)
            .await
            .map_err(|_| AppError::NotFound)?;
        let upload_root = tokio::fs::canonicalize(&state.config.upload_dir)
            .await
            .map_err(|_| AppError::NotFound)?;
        let project_root = crate::services::project_storage::projects_root(&state.config.upload_dir);
        let project_root = tokio::fs::canonicalize(project_root)
            .await
            .map_err(|_| AppError::NotFound)?;
        if !path.starts_with(upload_root) && !path.starts_with(project_root) {
            return Err(AppError::NotFound);
        }
        (path, rate, channel_count, frames)
    } else {
        let project_stem = sqlx::query_as::<_, (String, i32, i16, i64)>(
            "SELECT storage_key,sample_rate,channels,frame_count FROM project_stems WHERE job_id=$1 AND stem_name='drums'",
        )
        .bind(job_id)
        .fetch_optional(&state.db)
        .await?;
        if let Some((key, rate, channel_count, frames)) = project_stem {
            let root = crate::services::project_storage::projects_root(&state.config.upload_dir);
            let path = crate::services::project_storage::resolve_storage_key(&root, &key)
                .map_err(AppError::Internal)?;
            let canonical_root = tokio::fs::canonicalize(root)
                .await
                .map_err(|_| AppError::NotFound)?;
            let path = tokio::fs::canonicalize(path)
                .await
                .map_err(|_| AppError::NotFound)?;
            if !path.starts_with(canonical_root) {
                return Err(AppError::NotFound);
            }
            (path, rate, channel_count, frames)
        } else {
            let drums = job
                .result_metadata
                .get("stems")
                .and_then(|stems| stems.get("drums"))
                .ok_or_else(|| {
                    AppError::Internal("successful separation has no drums metadata".into())
                })?;
            let sample_rate = drums
                .get("sample_rate")
                .and_then(Value::as_u64)
                .and_then(|value| i32::try_from(value).ok())
                .or_else(|| {
                    drums
                        .get("sampleRate")
                        .and_then(Value::as_u64)
                        .and_then(|value| i32::try_from(value).ok())
                })
                .ok_or_else(|| {
                    AppError::Internal("drums metadata has an invalid sample rate".into())
                })?;
            let frame_count = drums
                .get("frame_count")
                .and_then(Value::as_u64)
                .and_then(|value| i64::try_from(value).ok())
                .or_else(|| {
                    drums
                        .get("frames")
                        .and_then(Value::as_u64)
                        .and_then(|value| i64::try_from(value).ok())
                })
                .ok_or_else(|| {
                    AppError::Internal("drums metadata has an invalid frame count".into())
                })?;
            let channels = drums
                .get("channels")
                .and_then(Value::as_u64)
                .and_then(|value| i16::try_from(value).ok())
                .ok_or_else(|| {
                    AppError::Internal("drums metadata has an invalid channel count".into())
                })?;
            let path = storage::jobs_root(&state.config.upload_dir)
                .join(job_id.to_string())
                .join("drums.wav");
            (path, sample_rate, channels, frame_count)
        }
    };
    if sample_rate <= 0 || channels <= 0 || frame_count <= 0 {
        return Err(AppError::Internal(
            format!("{input_stem} metadata contains non-positive audio dimensions"),
        ));
    }
    tokio::fs::metadata(&path)
        .await
        .map_err(|_| AppError::NotFound)?;
    // Project cleanup is coordinated by project ID, not the separation job ID.
    // Hold this lease for the entire asynchronous analysis so expired-project
    // cleanup cannot remove the selected audio file while Beat This is reading it.
    let audio_lease = storage::acquire_audio_stream(job.project_id).ok_or(AppError::NotFound)?;

    let analysis = sqlx::query_as::<_, BeatAnalysisJob>(
        "INSERT INTO beat_analyses (separation_job_id, input_stem, sample_rate, channels, frame_count, model_version) VALUES ($1, $2, $3, $4, $5, $6) RETURNING *",
    )
    .bind(job_id)
    .bind(&input_stem)
    .bind(sample_rate)
    .bind(channels)
    .bind(frame_count)
    .bind(&state.config.beat_this_model_version)
    .fetch_one(&state.db)
    .await?;

    let db = state.db.clone();
    let project_id = job.project_id;
    let beats_per_bar = job
        .request_metadata
        .pointer("/quantization/beats_per_bar")
        .and_then(Value::as_u64)
        .and_then(|value| u8::try_from(value).ok())
        .filter(|value| (2..=12).contains(value))
        .unwrap_or(4);
    let mel_model = state.config.beat_this_mel_model.clone();
    let beat_model = state.config.beat_this_model.clone();
    let model_version = state.config.beat_this_model_version.clone();
    let channels = channels as u16;
    tokio::spawn(async move {
        execute_analysis(
            db,
            analysis.id,
            job_id,
            path,
            sample_rate as u32,
            channels,
            frame_count as u64,
            project_id,
            input_stem,
            beats_per_bar,
            mel_model,
            beat_model,
            model_version,
            audio_lease,
        )
        .await;
    });
    Ok((StatusCode::ACCEPTED, Json(analysis)))
}

#[utoipa::path(
    get,
    path = "/api/v1/jobs/{job_id}/beat-analysis",
    params(("job_id" = Uuid, Path, description = "Successful HTDemucs separation job UUID")),
    responses(
        (status = 200, description = "Latest beat analysis state and detector output", body = BeatAnalysisJob),
        (status = 404, description = "Beat analysis not found")
    )
)]
pub async fn get_beat_analysis(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<BeatAnalysisJob>, AppError> {
    let job_id = parse_uuid(&job_id)?;
    let analysis = sqlx::query_as::<_, BeatAnalysisJob>(
        "SELECT * FROM beat_analyses WHERE separation_job_id = $1 ORDER BY created_at DESC LIMIT 1",
    )
    .bind(job_id)
    .fetch_optional(&state.db)
    .await?
    .ok_or(AppError::NotFound)?;
    Ok(Json(analysis))
}

#[utoipa::path(
    get,
    path = "/api/v1/jobs/{job_id}/analysis",
    params(("job_id" = Uuid, Path, description = "Separation job UUID")),
    responses((status = 200, description = "Saved, versioned complete beat grid"), (status = 404, description = "Saved analysis not found"))
)]
pub async fn get_job_analysis(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<Value>, AppError> {
    let job_id = parse_uuid(&job_id)?;
    let result = sqlx::query_scalar::<_, Value>("SELECT result FROM beat_analyses WHERE separation_job_id=$1 AND status='SUCCEEDED' AND result IS NOT NULL ORDER BY finished_at DESC NULLS LAST, created_at DESC LIMIT 1")
        .bind(job_id).fetch_optional(&state.db).await?.ok_or(AppError::NotFound)?;
    Ok(Json(result))
}

fn parse_uuid(value: &str) -> Result<Uuid, AppError> {
    Uuid::parse_str(value).map_err(|_| AppError::BadRequest("job_id must be a UUID".into()))
}

async fn execute_analysis(
    db: PgPool,
    analysis_id: Uuid,
    job_id: Uuid,
    audio_path: PathBuf,
    sample_rate: u32,
    channels: u16,
    frame_count: u64,
    project_id: Uuid,
    input_stem: String,
    beats_per_bar: u8,
    mel_model: PathBuf,
    beat_model: PathBuf,
    model_version: Option<String>,
    audio_lease: storage::AudioFileLease,
) {
    let _audio_lease = audio_lease;
    let _permit = match BEAT_ANALYSIS_LIMIT.acquire().await {
        Ok(permit) => permit,
        Err(error) => {
            let message = format!("beat analysis worker is unavailable: {error}");
            tracing::error!(%analysis_id, %message, "could not acquire beat analysis slot");
            if let Err(update_error) = sqlx::query(
                "UPDATE beat_analyses SET status='FAILED', error_message=$2, finished_at=now(), updated_at=now() WHERE id=$1",
            )
            .bind(analysis_id)
            .bind(&message)
            .execute(&db)
            .await
            {
                tracing::error!(%analysis_id, %update_error, "could not persist beat analysis failure");
            }
            return;
        }
    };
    let _ = sqlx::query(
        "UPDATE beat_analyses SET status = 'RUNNING', updated_at = now() WHERE id = $1",
    )
    .bind(analysis_id)
    .execute(&db)
    .await;

    let inference_started = Instant::now();
    let work = tokio::task::spawn_blocking(move || {
        analyze_wav(
            &audio_path,
            sample_rate,
            channels,
            frame_count,
            &input_stem,
            &mel_model,
            &beat_model,
            model_version,
        )
    })
    .await;

    match work {
        Ok(Ok((result, decode_and_inference_ms))) => {
            tracing::info!(%analysis_id, input_stem = %input_stem, analysis_ms = decode_and_inference_ms, total_ms = inference_started.elapsed().as_millis(), stage_ms = %result["analysis_metadata"]["stages_ms"], "beat analysis complete");
            match crate::services::job_manager::build_beat_grid(
                job_id,
                project_id,
                sample_rate,
                channels,
                frame_count,
                beats_per_bar,
                &result,
            ) {
                Ok(beat_grid) => {
                    let _ = sqlx::query("UPDATE beat_analyses SET status = 'SUCCEEDED', result = $2, error_message = NULL, finished_at = now(), updated_at = now() WHERE id = $1")
                        .bind(analysis_id).bind(beat_grid).execute(&db).await;
                }
                Err(message) => {
                    let _ = sqlx::query("UPDATE beat_analyses SET status = 'FAILED', error_message = $2, finished_at = now(), updated_at = now() WHERE id = $1")
                        .bind(analysis_id).bind(message).execute(&db).await;
                }
            }
        }
        Ok(Err(message)) => {
            tracing::warn!(%analysis_id, %message, "drum beat analysis failed");
            let _ = sqlx::query("UPDATE beat_analyses SET status = 'FAILED', error_message = $2, finished_at = now(), updated_at = now() WHERE id = $1")
                .bind(analysis_id).bind(message).execute(&db).await;
        }
        Err(error) => {
            tracing::error!(%analysis_id, %error, "beat analysis task panicked");
            let _ = sqlx::query("UPDATE beat_analyses SET status = 'FAILED', error_message = 'analysis worker failed', finished_at = now(), updated_at = now() WHERE id = $1")
                .bind(analysis_id).execute(&db).await;
        }
    }
}

pub(crate) fn analyze_wav(
    path: &std::path::Path,
    sample_rate: u32,
    channels: u16,
    frame_count: u64,
    input_stem: &str,
    mel_model: &std::path::Path,
    beat_model: &std::path::Path,
    model_version: Option<String>,
) -> Result<(Value, u128), String> {
    if !mel_model.is_file() || !beat_model.is_file() {
        return Err(format!(
            "Beat This! model files are unavailable (mel: {}, beat: {})",
            mel_model.display(),
            beat_model.display()
        ));
    }
    let started = Instant::now();
    let validation_started = Instant::now();
    validate_wav_metadata(path, sample_rate, channels, frame_count)?;
    let validation_ms = validation_started.elapsed().as_millis();
    let detector_lock = BEAT_DETECTOR.get_or_init(|| Mutex::new(None));
    let cache_lock_started = Instant::now();
    let mut cached = detector_lock
        .lock()
        .map_err(|_| "Beat This! model cache lock was poisoned".to_string())?;
    let cache_lock_wait_ms = cache_lock_started.elapsed().as_millis();
    let load_started = Instant::now();
    let cache_matches = cached
        .as_ref()
        .is_some_and(|entry| entry.mel_path == mel_model && entry.beat_path == beat_model);
    if !cache_matches {
        let detector = BeatThis::new(&RtenRuntime, mel_model, beat_model)
            .map_err(|error| format!("Beat This! model load failed: {error:#}"))?;
        *cached = Some(CachedBeatDetector {
            mel_path: mel_model.to_owned(),
            beat_path: beat_model.to_owned(),
            detector,
        });
    }
    let model_load_ms = if cache_matches {
        0
    } else {
        load_started.elapsed().as_millis()
    };
    let detector = &mut cached.as_mut().expect("detector initialized").detector;
    let mut stream = detector
        .stream(sample_rate)
        .map_err(|error| format!("Beat This! stream initialization failed: {error:#}"))?;
    let mut wav = hound::WavReader::open(path)
        .map_err(|error| format!("drums WAV decode failed: {error}"))?;
    let spec = wav.spec();
    let normalization = if spec.sample_format == hound::SampleFormat::Int {
        (2_f64).powi(i32::from(spec.bits_per_sample) - 1)
    } else {
        1.0
    };
    let input_started = Instant::now();
    let (decoded_frames, decode_downmix_ms, push_ms) = match spec.sample_format {
        hound::SampleFormat::Float => {
            push_wav_mono::<_, f32, _>(&mut wav, &mut stream, channels, normalization)?
        }
        hound::SampleFormat::Int => {
            push_wav_mono::<_, i32, _>(&mut wav, &mut stream, channels, normalization)?
        }
    };
    if decoded_frames != frame_count {
        return Err(format!(
            "decoded drums frame count {decoded_frames} differs from separation metadata {frame_count}"
        ));
    }
    let timed = stream
        .finish_timed()
        .map_err(|error| format!("Beat This! inference failed: {error:#}"))?;
    let elapsed_ms = started.elapsed().as_millis();
    let inference_ms = timed.timing.predict.as_millis();
    let output_decode_ms = timed.timing.decode.as_millis();
    let mel_ms = timed.timing.mel.as_millis();
    let analysis = timed.analysis;
    let beats: Vec<f64> = analysis.beats.iter().map(|time| f64::from(*time)).collect();
    let downbeats: Vec<f64> = analysis
        .downbeats
        .iter()
        .map(|time| f64::from(*time))
        .collect();
    let bpm = median_bpm(&beats);
    let beat_numbers = count_beats_from_downbeats(&beats, &downbeats);
    let beat_events = beats
        .iter()
        .enumerate()
        .map(|(index, time)| {
            Ok(json!({
                "time_seconds": time,
                "sample_position": timestamp_to_sample(*time, sample_rate)?,
                "beat_number_in_bar": beat_numbers[index]
            }))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let downbeat_events = downbeats.iter().map(|time| {
        Ok(json!({"time_seconds": time, "sample_position": timestamp_to_sample(*time, sample_rate)?}))
    }).collect::<Result<Vec<_>, String>>()?;
    Ok((
        json!({
            "bpm": bpm,
            "beats": beat_events,
            "downbeats": downbeat_events,
            "input_stem": input_stem,
        "sample_rate": sample_rate,
        "channels": channels,
            "source_frame_count": frame_count,
            "detection_engine": "beat-this-rs",
            "engine_version": "1.1.0",
            "model_version": model_version,
            "analysis_metadata": {
                "detector_output_time_unit": "seconds",
            "sample_position_policy": "round(timestamp_seconds * original_sample_rate)",
            "detector_internal_sample_rate": 22050,
            "elapsed_ms": elapsed_ms,
            "stages_ms": {
                "wav_validation": validation_ms,
                "model_cache_lock_wait": cache_lock_wait_ms,
                "model_load": model_load_ms,
                "wav_decode_and_downmix": decode_downmix_ms,
                "stream_push_including_mel": push_ms,
                "mel_total_including_stream_push": mel_ms,
                "beat_model_inference": inference_ms,
                "peak_decode": output_decode_ms,
                "input_to_analysis_complete": input_started.elapsed().as_millis()
            }
            }
        }),
        elapsed_ms,
    ))
}

fn timestamp_to_sample(time_seconds: f64, sample_rate: u32) -> Result<u64, String> {
    if !time_seconds.is_finite() || time_seconds < 0.0 {
        return Err("detector returned an invalid timestamp".into());
    }
    let sample = (time_seconds * f64::from(sample_rate)).round();
    if sample > u64::MAX as f64 {
        return Err("detector timestamp exceeds sample-position range".into());
    }
    Ok(sample as u64)
}

fn validate_wav_metadata(
    path: &std::path::Path,
    expected_sample_rate: u32,
    expected_channels: u16,
    expected_frames: u64,
) -> Result<(), String> {
    let (actual_sample_rate, actual_channels, actual_frames) = read_wav_metadata(path)?;
    if actual_sample_rate != expected_sample_rate
        || actual_channels != expected_channels
        || actual_frames != expected_frames
    {
        return Err(format!(
            "drums WAV header ({actual_sample_rate}, {actual_channels} channels, {actual_frames} frames) differs from separation metadata ({expected_sample_rate}, {expected_channels} channels, {expected_frames} frames)"
        ));
    }
    Ok(())
}

/// Read actual WAV source metadata for analysis paths that run before HTDemucs.
pub(crate) fn read_wav_metadata(path: &std::path::Path) -> Result<(u32, u16, u64), String> {
    let wav = hound::WavReader::open(path)
        .map_err(|error| format!("source WAV header could not be read: {error}"))?;
    let spec = wav.spec();
    let actual_frames = u64::from(wav.duration());
    if !(8_000..=384_000).contains(&spec.sample_rate)
        || !(1..=2).contains(&spec.channels)
        || actual_frames == 0
    {
        return Err(format!(
            "source WAV has unsupported metadata: {} Hz, {} channels, {actual_frames} frames",
            spec.sample_rate, spec.channels,
        ));
    }
    Ok((spec.sample_rate, spec.channels, actual_frames))
}

fn normalize_mono_frame(channel_sum: f64, channels: u16, normalization: f64) -> f32 {
    (channel_sum / normalization / f64::from(channels)).clamp(-1.0, 1.0) as f32
}

fn push_wav_mono<R, T, M>(
    reader: &mut hound::WavReader<R>,
    stream: &mut BeatStream<'_, M>,
    channels: u16,
    normalization: f64,
) -> Result<(u64, u128, u128), String>
where
    R: Read + Seek,
    T: hound::Sample + Into<f64>,
    M: Model,
{
    const CHUNK_FRAMES: usize = 16_384;
    let mut mono = Vec::with_capacity(CHUNK_FRAMES);
    let mut channel_sum = 0.0_f64;
    let mut channel_index = 0_u16;
    let mut source_frames = 0_u64;
    let started = Instant::now();
    let mut push_ms = 0_u128;
    for sample in reader.samples::<T>() {
        let sample = sample.map_err(|error| format!("drums WAV sample decode failed: {error}"))?;
        channel_sum += sample.into();
        channel_index += 1;
        if channel_index == channels {
            mono.push(normalize_mono_frame(channel_sum, channels, normalization));
            source_frames += 1;
            channel_sum = 0.0;
            channel_index = 0;
            if mono.len() == CHUNK_FRAMES {
                let push_started = Instant::now();
                stream
                    .push(&mono)
                    .map_err(|error| format!("Beat This! audio stream failed: {error:#}"))?;
                push_ms += push_started.elapsed().as_millis();
                mono.clear();
            }
        }
    }
    if channel_index != 0 {
        return Err("drums WAV ends in an incomplete audio frame".into());
    }
    if !mono.is_empty() {
        let push_started = Instant::now();
        stream
            .push(&mono)
            .map_err(|error| format!("Beat This! audio stream failed: {error:#}"))?;
        push_ms += push_started.elapsed().as_millis();
    }
    if source_frames == 0 {
        return Err("drums WAV contains no audio frames".into());
    }
    let elapsed = started.elapsed().as_millis();
    Ok((source_frames, elapsed.saturating_sub(push_ms), push_ms))
}

fn median_bpm(beats: &[f64]) -> Option<f64> {
    if beats.len() < 2 {
        return None;
    }
    let mut intervals: Vec<f64> = beats
        .windows(2)
        .map(|window| window[1] - window[0])
        .filter(|interval| interval.is_finite() && *interval > 0.0)
        .collect();
    if intervals.is_empty() {
        return None;
    }
    intervals.sort_by(f64::total_cmp);
    let middle = intervals.len() / 2;
    let interval = if intervals.len() % 2 == 0 {
        (intervals[middle - 1] + intervals[middle]) * 0.5
    } else {
        intervals[middle]
    };
    Some(60.0 / interval)
}

fn count_beats_from_downbeats(beats: &[f64], downbeats: &[f64]) -> Vec<Option<u32>> {
    let reset_indices: Vec<usize> = downbeats
        .iter()
        .filter_map(|downbeat| {
            beats
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    (**a - *downbeat).abs().total_cmp(&(**b - *downbeat).abs())
                })
                .filter(|(_, beat)| (**beat - *downbeat).abs() <= 0.05)
                .map(|(index, _)| index)
        })
        .collect();
    let mut result = vec![None; beats.len()];
    let mut count: Option<u32> = None;
    for index in 0..beats.len() {
        if reset_indices.contains(&index) {
            count = Some(1);
        } else if let Some(previous) = count {
            count = Some(previous.saturating_add(1));
        }
        result[index] = count;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::{
        count_beats_from_downbeats, median_bpm, normalize_mono_frame, timestamp_to_sample,
        validate_wav_metadata,
    };
    use uuid::Uuid;

    #[test]
    fn detector_times_map_to_integer_samples_at_source_rate() {
        assert_eq!(timestamp_to_sample(1.25, 48_000).unwrap(), 60_000);
        assert_eq!(timestamp_to_sample(0.000_01, 48_000).unwrap(), 0);
        assert_eq!(
            timestamp_to_sample(f64::NAN, 48_000),
            Err("detector returned an invalid timestamp".into())
        );
    }

    #[test]
    fn bpm_uses_median_detected_beat_interval() {
        let bpm = median_bpm(&[0.0, 0.5, 1.0, 1.6, 2.0]).unwrap();
        assert!((bpm - 120.0).abs() < 1.0);
    }

    #[test]
    fn beat_numbers_reset_at_detected_downbeats_without_assuming_meter() {
        assert_eq!(
            count_beats_from_downbeats(&[0.1, 0.6, 1.1, 1.6, 2.1, 2.6], &[0.1, 1.6]),
            vec![Some(1), Some(2), Some(3), Some(1), Some(2), Some(3)],
        );
        assert_eq!(
            count_beats_from_downbeats(&[0.1, 0.6], &[]),
            vec![None, None]
        );
    }

    #[test]
    fn stereo_wav_header_rate_channels_and_frame_count_are_verified() {
        let path = std::env::temp_dir().join(format!("dubpad-beats-{}.wav", Uuid::new_v4()));
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for _ in 0..16 {
            writer.write_sample(0_i16).unwrap();
            writer.write_sample(0_i16).unwrap();
        }
        writer.finalize().unwrap();
        assert!(validate_wav_metadata(&path, 48_000, 2, 16).is_ok());
        assert!(validate_wav_metadata(&path, 44_100, 2, 16).is_err());
        assert!(validate_wav_metadata(&path, 48_000, 1, 16).is_err());
        assert!(validate_wav_metadata(&path, 48_000, 2, 15).is_err());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn stereo_channels_are_downmixed_without_changing_frame_count() {
        assert_eq!(normalize_mono_frame(0.25 + 0.75, 2, 1.0), 0.5);
        assert_eq!(normalize_mono_frame(-0.5 + 0.5, 2, 1.0), 0.0);
        assert_eq!(normalize_mono_frame(2.0 + 2.0, 2, 1.0), 1.0);
        assert_eq!(normalize_mono_frame(16_384.0 + 16_384.0, 2, 32_768.0), 0.5);
    }

    /// Opt-in full-model benchmark; set BEAT_BENCH_WAV to a fixed local fixture.
    #[test]
    #[ignore = "runs full Beat This model inference on a local audio fixture"]
    fn benchmark_beat_this_fixture() {
        let path = std::path::PathBuf::from(std::env::var("BEAT_BENCH_WAV").unwrap());
        let reader = hound::WavReader::open(&path).unwrap();
        let spec = reader.spec();
        let frames = u64::from(reader.duration());
        drop(reader);
        let (result, elapsed_ms) = super::analyze_wav(
            &path,
            spec.sample_rate,
            spec.channels,
            frames,
            "benchmark",
            &std::path::PathBuf::from(std::env::var("BEAT_THIS_MEL_MODEL").unwrap()),
            &std::path::PathBuf::from(std::env::var("BEAT_THIS_MODEL").unwrap()),
            Some("1.1.0".into()),
        )
        .unwrap();
        println!(
            "elapsed_ms={elapsed_ms} stages_ms={} beat_count={} downbeat_count={}",
            result["analysis_metadata"]["stages_ms"],
            result["beats"].as_array().map_or(0, Vec::len),
            result["downbeats"].as_array().map_or(0, Vec::len),
        );
    }
}
