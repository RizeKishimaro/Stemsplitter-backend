use std::path::Path;

use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::{
    fs,
    io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt, SeekFrom},
};
use uuid::Uuid;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentRange {
    pub start_sample: u64,
    pub end_sample: u64,
}

pub fn segment_id(job_id: Uuid, index: usize) -> String {
    format!("{}-segment-{index:06}", job_id.simple())
}

/// Use actual detected downbeat frame positions. Zero and the exact final
/// frame are structural endpoints, not inferred musical markers.
pub fn ranges_from_beat_grid(grid: &Value, frames: u64) -> Vec<SegmentRange> {
    let mut boundaries = vec![0, frames];
    if let Some(downbeats) = grid
        .pointer("/analysis/downbeats")
        .and_then(Value::as_array)
    {
        boundaries.extend(
            downbeats
                .iter()
                .filter_map(|beat| beat.get("samplePosition").and_then(Value::as_u64))
                .filter(|sample| *sample > 0 && *sample < frames),
        );
    }
    boundaries.sort_unstable();
    boundaries.dedup();
    boundaries
        .windows(2)
        .filter_map(|pair| {
            (pair[0] < pair[1]).then_some(SegmentRange {
                start_sample: pair[0],
                end_sample: pair[1],
            })
        })
        .collect()
}

/// Write a frame-aligned PCM/float WAV range without decoding or re-encoding
/// samples. The returned SHA-256 covers the complete output WAV bytes.
pub async fn write_wav_segment(
    source: &Path,
    destination: &Path,
    range: SegmentRange,
    expected_rate: u32,
    expected_channels: u16,
) -> Result<(u64, String), String> {
    if range.start_sample >= range.end_sample {
        return Err("segment range is empty".into());
    }
    let mut input = fs::File::open(source)
        .await
        .map_err(|e| format!("open source WAV: {e}"))?;
    let length = input
        .metadata()
        .await
        .map_err(|e| format!("stat source WAV: {e}"))?
        .len();
    let header_len = usize::try_from(length.min(65_536)).unwrap_or(65_536);
    let mut header = vec![0; header_len];
    input
        .read_exact(&mut header)
        .await
        .map_err(|e| format!("read WAV header: {e}"))?;
    let info = parse_wav(&header, length)?;
    if info.sample_rate != expected_rate || info.channels != expected_channels {
        return Err("source WAV does not match project sample metadata".into());
    }
    if range.end_sample > info.frames {
        return Err("segment boundary exceeds source WAV frames".into());
    }
    let data_len = range
        .end_sample
        .checked_sub(range.start_sample)
        .and_then(|frames| frames.checked_mul(u64::from(info.block_align)))
        .ok_or("segment byte length overflow")?;
    let padding = data_len & 1;
    let riff_size = 36_u64
        .checked_add(data_len)
        .and_then(|size| size.checked_add(padding))
        .ok_or("segment WAV size overflow")?;
    if data_len > u32::MAX as u64 || riff_size > u32::MAX as u64 {
        return Err("segment exceeds RIFF/WAV size limit".into());
    }
    let parent = destination
        .parent()
        .ok_or("segment destination has no parent directory")?;
    fs::create_dir_all(parent)
        .await
        .map_err(|e| format!("create segment directory: {e}"))?;
    let temp = parent.join(format!(".{}.tmp", Uuid::new_v4()));
    let result = async {
        let mut output = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
            .await
            .map_err(|e| format!("create segment temp file: {e}"))?;
        let wav_header = make_wav_header(info, data_len, riff_size);
        let mut digest = Sha256::new();
        output
            .write_all(&wav_header)
            .await
            .map_err(|e| format!("write segment header: {e}"))?;
        digest.update(&wav_header);
        let sample_offset = range
            .start_sample
            .checked_mul(u64::from(info.block_align))
            .ok_or("segment source offset overflow")?;
        let byte_offset = info
            .data_offset
            .checked_add(sample_offset)
            .ok_or("segment source offset overflow")?;
        input
            .seek(SeekFrom::Start(byte_offset))
            .await
            .map_err(|e| format!("seek source WAV: {e}"))?;
        let mut remaining = data_len;
        let mut buffer = vec![0; 1024 * 1024];
        while remaining > 0 {
            let take = remaining.min(buffer.len() as u64) as usize;
            input
                .read_exact(&mut buffer[..take])
                .await
                .map_err(|e| format!("read source segment audio: {e}"))?;
            output
                .write_all(&buffer[..take])
                .await
                .map_err(|e| format!("write segment audio: {e}"))?;
            digest.update(&buffer[..take]);
            remaining -= take as u64;
        }
        if padding != 0 {
            output
                .write_all(&[0])
                .await
                .map_err(|e| format!("write WAV data padding: {e}"))?;
            digest.update([0]);
        }
        output
            .flush()
            .await
            .map_err(|e| format!("flush segment WAV: {e}"))?;
        output
            .sync_all()
            .await
            .map_err(|e| format!("sync segment WAV: {e}"))?;
        drop(output);
        fs::rename(&temp, destination)
            .await
            .map_err(|e| format!("publish segment WAV: {e}"))?;
        Ok::<_, String>((
            wav_header.len() as u64 + data_len + padding,
            format!("{:x}", digest.finalize()),
        ))
    }
    .await;
    if result.is_err() {
        let _ = fs::remove_file(&temp).await;
    }
    result
}

#[derive(Clone, Copy)]
struct WavInfo {
    format: u16,
    channels: u16,
    sample_rate: u32,
    block_align: u16,
    bits: u16,
    frames: u64,
    data_offset: u64,
}

fn parse_wav(bytes: &[u8], file_len: u64) -> Result<WavInfo, String> {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("invalid RIFF/WAVE source".into());
    }
    let mut offset = 12_usize;
    let mut format = None;
    let mut channels = None;
    let mut rate = None;
    let mut align = None;
    let mut bits = None;
    let mut data = None;
    while offset + 8 <= bytes.len() {
        let id = &bytes[offset..offset + 4];
        let size = u32::from_le_bytes(
            bytes[offset + 4..offset + 8]
                .try_into()
                .map_err(|_| "invalid WAV chunk size")?,
        ) as usize;
        let body = offset + 8;
        let end = body.checked_add(size).ok_or("WAV chunk offset overflow")?;
        if id == b"data" {
            data = Some((body as u64, size as u64));
            break;
        }
        if end > bytes.len() {
            return Err("WAV header chunk is incomplete or exceeds the header limit".into());
        }
        match id {
            b"fmt " if size >= 16 => {
                format = Some(u16::from_le_bytes(
                    bytes[body..body + 2].try_into().unwrap(),
                ));
                channels = Some(u16::from_le_bytes(
                    bytes[body + 2..body + 4].try_into().unwrap(),
                ));
                rate = Some(u32::from_le_bytes(
                    bytes[body + 4..body + 8].try_into().unwrap(),
                ));
                align = Some(u16::from_le_bytes(
                    bytes[body + 12..body + 14].try_into().unwrap(),
                ));
                bits = Some(u16::from_le_bytes(
                    bytes[body + 14..body + 16].try_into().unwrap(),
                ));
            }
            _ => {}
        }
        offset = end + (size & 1);
    }
    let format = format.ok_or("WAV fmt chunk is missing")?;
    let channels = channels.ok_or("WAV channel count is missing")?;
    let sample_rate = rate.ok_or("WAV sample rate is missing")?;
    let block_align = align.ok_or("WAV block alignment is missing")?;
    let bits = bits.ok_or("WAV sample width is missing")?;
    let (data_offset, data_len) = data.ok_or("WAV data chunk is missing")?;
    if !matches!(format, 1 | 3)
        || channels == 0
        || !matches!(bits, 8 | 16 | 24 | 32)
        || u32::from(block_align) != u32::from(channels) * u32::from(bits) / 8
        || data_offset + data_len > file_len
        || data_len % u64::from(block_align) != 0
    {
        return Err("unsupported or malformed PCM WAV layout".into());
    }
    Ok(WavInfo {
        format,
        channels,
        sample_rate,
        block_align,
        bits,
        frames: data_len / u64::from(block_align),
        data_offset,
    })
}

fn make_wav_header(info: WavInfo, data_len: u64, riff_len: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(44);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(riff_len as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16_u32.to_le_bytes());
    out.extend_from_slice(&info.format.to_le_bytes());
    out.extend_from_slice(&info.channels.to_le_bytes());
    out.extend_from_slice(&info.sample_rate.to_le_bytes());
    let byte_rate = info.sample_rate.saturating_mul(u32::from(info.block_align));
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&info.block_align.to_le_bytes());
    out.extend_from_slice(&info.bits.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::{SampleFormat, WavSpec, WavWriter};
    use serde_json::json;

    #[test]
    fn uses_detected_frame_downbeats_and_exact_final_frame() {
        let grid = json!({"analysis":{"downbeats":[{"samplePosition":100},{"samplePosition":200},{"samplePosition":900}]}});
        assert_eq!(
            ranges_from_beat_grid(&grid, 250),
            vec![
                SegmentRange {
                    start_sample: 0,
                    end_sample: 100
                },
                SegmentRange {
                    start_sample: 100,
                    end_sample: 200
                },
                SegmentRange {
                    start_sample: 200,
                    end_sample: 250
                }
            ]
        );
        assert_eq!(
            ranges_from_beat_grid(&json!({"analysis":{"downbeats":[]}}), 25),
            vec![SegmentRange {
                start_sample: 0,
                end_sample: 25
            }]
        );
    }

    #[tokio::test]
    async fn copies_stereo_pcm_ranges_without_changing_sample_bytes() {
        let dir = std::env::temp_dir().join(format!("dubpad-segment-test-{}", Uuid::new_v4()));
        fs::create_dir_all(&dir).await.unwrap();
        let source = dir.join("source.wav");
        let target = dir.join("slice.wav");
        let spec = WavSpec {
            channels: 2,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(&source, spec).unwrap();
        for value in 0_i16..20 {
            writer.write_sample(value).unwrap();
        }
        writer.finalize().unwrap();
        let (size, _) = write_wav_segment(
            &source,
            &target,
            SegmentRange {
                start_sample: 2,
                end_sample: 7,
            },
            48_000,
            2,
        )
        .await
        .unwrap();
        let reader = hound::WavReader::open(&target).unwrap();
        assert_eq!(size, std::fs::metadata(&target).unwrap().len());
        assert_eq!(reader.duration(), 5);
        assert_eq!(
            reader
                .into_samples::<i16>()
                .map(Result::unwrap)
                .collect::<Vec<_>>(),
            (4_i16..14).collect::<Vec<_>>()
        );
        fs::remove_dir_all(dir).await.unwrap();
    }
}
