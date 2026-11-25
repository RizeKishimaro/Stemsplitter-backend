//! Integer-sample musical chunk boundaries shared by every separated stem.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChunkBoundary {
    pub index: u32,
    pub bar_number: Option<u32>,
    pub start_sample: u64,
    pub end_sample: u64,
    pub sample_rate: u32,
}

impl ChunkBoundary {
    pub fn start_seconds(&self) -> f64 {
        self.start_sample as f64 / f64::from(self.sample_rate)
    }
    pub fn end_seconds(&self) -> f64 {
        self.end_sample as f64 / f64::from(self.sample_rate)
    }
}

/// Builds bar-sized chunks from detector downbeats. The final chunk is clipped
/// to the source frame count; detector timestamps remain untouched in storage.
pub fn bar_boundaries(
    result: &Value,
    sample_rate: u32,
    frame_count: u64,
    beats_per_bar: u8,
) -> Result<Vec<ChunkBoundary>, String> {
    if sample_rate == 0 || frame_count == 0 || beats_per_bar == 0 {
        return Err("sample rate, source frames, and meter must be positive".into());
    }
    let downbeats = result
        .get("downbeats")
        .and_then(Value::as_array)
        .ok_or("beat analysis has no downbeat array")?;
    let mut starts = Vec::with_capacity(downbeats.len() + 1);
    let mut first_detected = None;
    let mut detected_count = 0_usize;
    for event in downbeats {
        let sample = event
            .get("sample_position")
            .and_then(Value::as_u64)
            .ok_or("downbeat is missing an integer source sample position")?;
        if sample < frame_count && starts.last().copied() != Some(sample) {
            first_detected.get_or_insert(sample);
            detected_count += 1;
            starts.push(sample);
        }
    }
    if starts.first().copied() != Some(0) {
        starts.insert(0, 0);
    }
    if detected_count < 2 {
        let bpm = result
            .get("bpm")
            .and_then(Value::as_f64)
            .filter(|bpm| bpm.is_finite() && *bpm > 0.0)
            .ok_or("analysis has neither usable downbeats nor BPM")?;
        let bar_samples =
            (f64::from(sample_rate) * 60.0 * f64::from(beats_per_bar) / bpm).round() as u64;
        if bar_samples == 0 {
            return Err("computed bar duration is zero".into());
        }
        let mut cursor = first_detected.unwrap_or(0).saturating_add(bar_samples);
        while cursor < frame_count {
            starts.push(cursor);
            cursor = cursor.saturating_add(bar_samples);
        }
    }
    starts.sort_unstable();
    starts.dedup();
    let mut boundaries = Vec::with_capacity(starts.len());
    for (index, start) in starts.iter().copied().enumerate() {
        let end = starts
            .get(index + 1)
            .copied()
            .unwrap_or(frame_count)
            .min(frame_count);
        if start < end {
            let pickup = index == 0 && first_detected.is_some_and(|first| first > 0);
            boundaries.push(ChunkBoundary {
                index: index as u32,
                bar_number: if pickup {
                    None
                } else {
                    Some(
                        index as u32 + 1 - u32::from(first_detected.is_some_and(|first| first > 0)),
                    )
                },
                start_sample: start,
                end_sample: end,
                sample_rate,
            });
        }
    }
    Ok(boundaries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn uses_shared_integer_downbeats_and_trims_partial_final_bar() {
        let analysis = json!({"downbeats":[
            {"sample_position": 0}, {"sample_position": 192000},
            {"sample_position": 384000}
        ]});
        let result = bar_boundaries(&analysis, 48_000, 500_000, 4).unwrap();
        assert_eq!(
            result
                .iter()
                .map(|b| (b.start_sample, b.end_sample))
                .collect::<Vec<_>>(),
            vec![(0, 192_000), (192_000, 384_000), (384_000, 500_000)]
        );
    }

    #[test]
    fn refuses_missing_grid_instead_of_fabricating_tempo() {
        assert!(bar_boundaries(&serde_json::json!({}), 48_000, 100_000, 4).is_err());
    }

    #[test]
    fn marks_intro_before_first_detected_downbeat_as_pickup_and_keeps_it_separate() {
        let analysis = json!({"bpm": 120.0, "downbeats":[
            {"sample_position": 24_000}, {"sample_position": 120_000}
        ]});
        let result = bar_boundaries(&analysis, 48_000, 300_000, 4).unwrap();
        assert_eq!(result[0].start_sample, 0);
        assert_eq!(result[0].end_sample, 24_000);
        assert_eq!(result[0].bar_number, None);
        assert_eq!(result[1].start_sample, 24_000);
        assert_eq!(result[1].bar_number, Some(1));
    }
}
