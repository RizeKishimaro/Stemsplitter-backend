#!/usr/bin/env python3
"""Read-only waveform alignment measurements for an original and Demucs stems.

Positive offsets mean the stem's signal is later than the original. This tool
measures local normalized waveform correlation; it never edits audio or applies
compensation. Low correlation is reported as low confidence, not converted to
an invented offset.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

STEMS = ("drums", "bass", "other", "vocals")


def estimate_offset(reference: np.ndarray, candidate: np.ndarray, max_lag: int):
    """Return integer-sample lag and normalized correlation using FFT xcorr."""
    if reference.ndim != 1 or candidate.ndim != 1 or reference.size != candidate.size:
        raise ValueError("alignment arrays must be equal-length mono sample vectors")
    if reference.size < 32 or max_lag < 0 or max_lag >= reference.size // 2:
        raise ValueError("invalid alignment window or lag bound")

    x = np.asarray(reference, dtype=np.float64)
    y = np.asarray(candidate, dtype=np.float64)
    x -= x.mean()
    y -= y.mean()
    if np.sqrt(np.mean(x * x)) < 1e-7 or np.sqrt(np.mean(y * y)) < 1e-7:
        return None, 0.0

    fft_size = 1 << (2 * x.size - 1).bit_length()
    circular = np.fft.irfft(np.fft.rfft(y, fft_size) * np.conj(np.fft.rfft(x, fft_size)), fft_size)
    # Prefix energy sums keep each lag normalization O(1). Recomputing the
    # overlap dot products for every lag makes a 10-second audio window
    # needlessly quadratic in its sample count.
    x_energy = np.concatenate(([0.0], np.cumsum(x * x)))
    y_energy = np.concatenate(([0.0], np.cumsum(y * y)))
    best_lag = 0
    best_score = -1.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            overlap_end = x.size - lag
            left_energy = x_energy[overlap_end]
            right_energy = y_energy[x.size] - y_energy[lag]
        else:
            overlap_start = -lag
            overlap_end = y.size - overlap_start
            left_energy = x_energy[x.size] - x_energy[overlap_start]
            right_energy = y_energy[overlap_end]
        energy = float(np.sqrt(left_energy * right_energy))
        if energy <= 0:
            continue
        numerator = float(circular[lag if lag >= 0 else fft_size + lag])
        score = numerator / energy
        if score > best_score:
            best_lag, best_score = lag, score
    return best_lag, max(0.0, best_score)


def mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples.astype(np.float32, copy=False)
    if samples.ndim == 2:
        return samples.mean(axis=1, dtype=np.float32)
    raise ValueError("audio must be mono or interleaved multichannel frames")


def measure_arrays(reference, stems, sample_rate, *, window_seconds=10.0, max_lag_ms=250.0):
    """Measure each stem against the same original timeline in local windows."""
    ref = mono(np.asarray(reference))
    if not sample_rate or sample_rate <= 0 or ref.size == 0:
        raise ValueError("invalid reference sample rate or frame count")
    window_frames = max(32, round(window_seconds * sample_rate))
    max_lag = round(max_lag_ms * sample_rate / 1000)
    report = {}
    for name in STEMS:
        if name not in stems:
            raise ValueError(f"missing {name} stem")
        candidate = mono(np.asarray(stems[name]))
        if candidate.size != ref.size:
            raise ValueError(f"{name} has {candidate.size} frames; original has {ref.size}")
        windows = []
        for start in range(0, ref.size, window_frames):
            end = min(ref.size, start + window_frames)
            if end - start < max(32, 2 * max_lag + 1):
                continue
            lag, confidence = estimate_offset(ref[start:end], candidate[start:end], max_lag)
            if lag is None:
                windows.append({"start_sample": start, "end_sample": end, "offset_samples": None,
                                "offset_ms": None, "confidence": 0.0, "low_energy": True})
            else:
                windows.append({"start_sample": start, "end_sample": end, "offset_samples": lag,
                                "offset_ms": lag * 1000 / sample_rate,
                                "confidence": confidence, "low_energy": False})
        confident = [item["offset_samples"] for item in windows if not item["low_energy"] and item["confidence"] >= 0.1]
        report[name] = {
            "sample_rate": sample_rate,
            "frame_count": int(ref.size),
            "median_offset_samples": int(np.median(confident)) if confident else None,
            "median_offset_ms": float(np.median(confident) * 1000 / sample_rate) if confident else None,
            "offset_spread_samples": int(max(confident) - min(confident)) if confident else None,
            "constant_offset_candidate": bool(confident) and max(confident) - min(confident) <= 1,
            "windows": windows,
        }
    return {"sample_rate": sample_rate, "frame_count": int(ref.size), "lag_sign": "positive means stem is late", "stems": report}


def read_audio(path: Path):
    info = sf.info(path)
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    return audio, int(rate), info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("stem_directory", type=Path)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--max-lag-ms", type=float, default=250.0)
    args = parser.parse_args()
    original, rate, original_info = read_audio(args.original)
    stems = {}
    stem_metadata = {}
    for name in STEMS:
        path = args.stem_directory / f"{name}.wav"
        audio, stem_rate, info = read_audio(path)
        if stem_rate != rate or audio.shape[0] != original.shape[0]:
            raise SystemExit(f"{name}: sample rate/frame count differs from original")
        stems[name] = audio
        stem_metadata[name] = {"channels": int(info.channels), "subtype": info.subtype}
    result = measure_arrays(original, stems, rate, window_seconds=args.window_seconds, max_lag_ms=args.max_lag_ms)
    result["original"] = {"channels": int(original_info.channels), "subtype": original_info.subtype}
    result["stem_metadata"] = stem_metadata
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
