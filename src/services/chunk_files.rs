//! Slices finalized HTDemucs outputs at one shared integer sample schedule.

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use serde_json::{Value, json};
use uuid::Uuid;

use super::timeline::ChunkBoundary;

const STEMS: [&str; 4] = ["drums", "bass", "other", "vocals"];

#[derive(Clone, Debug)]
pub struct CompletedChunk {
    pub boundary: ChunkBoundary,
    pub chunk_id: String,
    pub files: BTreeMap<String, PathBuf>,
    pub stem_metadata: Value,
}

/// Produce all four files before returning a chunk to its caller. The caller
/// persists readiness and emits the event only after this function validates them.
pub fn write_chunks(
    job_dir: &Path,
    job_id: Uuid,
    boundaries: &[ChunkBoundary],
    sample_rate: u32,
    channels: u16,
) -> Result<Vec<CompletedChunk>, String> {
    if channels == 0 || sample_rate == 0 {
        return Err("invalid chunk audio format".into());
    }
    let final_dir = job_dir.join("chunks");
    fs::create_dir_all(&final_dir).map_err(|e| format!("create chunk directory: {e}"))?;
    let mut completed = Vec::with_capacity(boundaries.len());
    for boundary in boundaries {
        if boundary.sample_rate != sample_rate || boundary.start_sample >= boundary.end_sample {
            return Err(format!("invalid shared boundary {}", boundary.index));
        }
        let chunk_id = format!("{}-{:06}", job_id.simple(), boundary.index);
        let temp_dir = final_dir.join(format!(".{chunk_id}.partial"));
        fs::create_dir_all(&temp_dir).map_err(|e| format!("create temporary chunk: {e}"))?;
        let mut temp_files = BTreeMap::new();
        let result = (|| {
            for stem in STEMS {
                let source = job_dir.join(format!("{stem}.wav"));
                let temp = temp_dir.join(format!("{stem}.wav"));
                write_one(&source, &temp, boundary, sample_rate, channels)?;
                validate_wav(&temp, boundary, sample_rate, channels)?;
                temp_files.insert(stem.to_owned(), temp);
            }
            Ok::<(), String>(())
        })();
        if let Err(error) = result {
            let _ = fs::remove_dir_all(&temp_dir);
            return Err(error);
        }
        let chunk_dir = final_dir.join(&chunk_id);
        fs::rename(&temp_dir, &chunk_dir).map_err(|e| format!("publish chunk atomically: {e}"))?;
        let files = STEMS
            .into_iter()
            .map(|stem| (stem.to_owned(), chunk_dir.join(format!("{stem}.wav"))))
            .collect::<BTreeMap<_, _>>();
        let duration =
            (boundary.end_sample - boundary.start_sample) as f64 / f64::from(sample_rate);
        let stem_metadata = Value::Object(STEMS.into_iter().map(|stem| {
            (stem.to_owned(), json!({
                "url": format!("/api/v1/jobs/{job_id}/chunks/{}/stems/{stem}", boundary.index),
                "sample_rate": sample_rate,
                "channels": channels,
                "frame_count": boundary.end_sample - boundary.start_sample,
                "duration_seconds": duration,
                "audio_format": "wav"
            }))
        }).collect());
        completed.push(CompletedChunk {
            boundary: boundary.clone(),
            chunk_id,
            files,
            stem_metadata,
        });
    }
    Ok(completed)
}

fn write_one(
    source: &Path,
    target: &Path,
    boundary: &ChunkBoundary,
    sample_rate: u32,
    channels: u16,
) -> Result<(), String> {
    let reader = hound::WavReader::open(source)
        .map_err(|e| format!("open separated stem {}: {e}", source.display()))?;
    let spec = reader.spec();
    let total_frames = u64::from(reader.duration());
    if spec.sample_rate != sample_rate
        || spec.channels != channels
        || boundary.end_sample > total_frames
    {
        return Err(format!(
            "stem {} does not cover shared boundary sample range",
            source.display()
        ));
    }
    let out_spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer =
        hound::WavWriter::create(target, out_spec).map_err(|e| format!("create chunk WAV: {e}"))?;
    let first = boundary.start_sample * u64::from(channels);
    let count = (boundary.end_sample - boundary.start_sample) * u64::from(channels);
    let scale = if spec.sample_format == hound::SampleFormat::Int {
        2_f64.powi(i32::from(spec.bits_per_sample) - 1)
    } else {
        1.0
    };
    match spec.sample_format {
        hound::SampleFormat::Int => {
            let mut samples = reader.into_samples::<i32>().skip(first as usize);
            for _ in 0..count {
                let sample = samples
                    .next()
                    .ok_or("stem ended while reading chunk samples")?
                    .map_err(|e| format!("decode separated stem: {e}"))?;
                writer
                    .write_sample((f64::from(sample) / scale).clamp(-1.0, 1.0) as f32)
                    .map_err(|e| format!("write chunk WAV: {e}"))?;
            }
        }
        hound::SampleFormat::Float => {
            let mut samples = reader.into_samples::<f32>().skip(first as usize);
            for _ in 0..count {
                let sample = samples
                    .next()
                    .ok_or("stem ended while reading chunk samples")?
                    .map_err(|e| format!("decode separated stem: {e}"))?;
                writer
                    .write_sample(sample)
                    .map_err(|e| format!("write chunk WAV: {e}"))?;
            }
        }
    }
    writer
        .finalize()
        .map_err(|e| format!("finalize chunk WAV: {e}"))
}

fn validate_wav(
    path: &Path,
    boundary: &ChunkBoundary,
    sample_rate: u32,
    channels: u16,
) -> Result<(), String> {
    let reader = hound::WavReader::open(path).map_err(|e| format!("validate chunk WAV: {e}"))?;
    let spec = reader.spec();
    let expected_frames = boundary.end_sample - boundary.start_sample;
    if spec.sample_rate != sample_rate
        || spec.channels != channels
        || u64::from(reader.duration()) != expected_frames
    {
        return Err(format!(
            "chunk WAV validation failed for {}",
            path.display()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_four_valid_stereo_chunks_from_one_shared_range() {
        let root = std::env::temp_dir().join(format!("dubpad-chunk-test-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let rate = 8_000;
        let frames = 20;
        for stem in STEMS {
            let path = root.join(format!("{stem}.wav"));
            let spec = hound::WavSpec {
                channels: 2,
                sample_rate: rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::create(path, spec).unwrap();
            for frame in 0..frames {
                writer.write_sample(frame as i16).unwrap();
                writer.write_sample(-(frame as i16)).unwrap();
            }
            writer.finalize().unwrap();
        }
        let job_id = Uuid::new_v4();
        let boundary = ChunkBoundary {
            index: 1,
            bar_number: Some(2),
            start_sample: 5,
            end_sample: 13,
            sample_rate: rate,
        };
        let chunks = write_chunks(&root, job_id, &[boundary], rate, 2).unwrap();
        let chunk = &chunks[0];
        assert_eq!(chunk.files.len(), 4);
        for stem in STEMS {
            let wav = hound::WavReader::open(&chunk.files[stem]).unwrap();
            assert_eq!(wav.spec().sample_rate, rate);
            assert_eq!(wav.spec().channels, 2);
            assert_eq!(wav.duration(), 8);
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_a_stem_that_does_not_cover_the_master_end_sample() {
        let root = std::env::temp_dir().join(format!("dubpad-short-stem-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        for (index, stem) in STEMS.iter().enumerate() {
            let path = root.join(format!("{stem}.wav"));
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: 8_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::create(path, spec).unwrap();
            for _ in 0..if index == 3 { 9 } else { 10 } {
                writer.write_sample(0_i16).unwrap();
            }
            writer.finalize().unwrap();
        }
        let boundary = ChunkBoundary {
            index: 0,
            bar_number: Some(1),
            start_sample: 0,
            end_sample: 10,
            sample_rate: 8_000,
        };
        assert!(write_chunks(&root, Uuid::new_v4(), &[boundary], 8_000, 1).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
