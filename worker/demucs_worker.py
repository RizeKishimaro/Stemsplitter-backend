#!/usr/bin/env python3
"""Persistent, JSON-lines HTDemucs worker. No model downloads are permitted."""

import json
import os
import queue
import sys
import threading
import traceback
from pathlib import Path


def emit(payload):
    """Send one JSON message to the Rust process."""
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


commands = queue.Queue()
active_id = None
cancel_event = threading.Event()


def read_commands():
    """Read commands from Rust without blocking audio processing."""
    global active_id

    for raw in sys.stdin:
        try:
            command = json.loads(raw)

            if command.get("type") == "cancel" and command.get("job_id") == active_id:
                cancel_event.set()
            else:
                commands.put(command)

        except Exception as error:
            emit(
                {
                    "type": "protocol_error",
                    "message": str(error),
                }
            )

    commands.put({"type": "shutdown"})


def cached_model_files():
    """Verify that every required HTDemucs model file is already cached."""
    from huggingface_hub import try_to_load_from_cache
    import yaml

    config = try_to_load_from_cache(
        "adefossez/HTDemucs",
        "htdemucs.yaml",
    )

    if not isinstance(config, str) or not Path(config).is_file():
        raise RuntimeError(
            "Cached HTDemucs config is unavailable; automatic downloads are disabled"
        )

    data = yaml.safe_load(Path(config).read_text())
    signatures = data.get("models", [])

    if not signatures:
        raise RuntimeError("Cached HTDemucs config has no model signatures")

    for signature in signatures:
        cached = try_to_load_from_cache(
            "adefossez/HTDemucs",
            f"{signature}.safetensors",
        )

        if not isinstance(cached, str) or not Path(cached).is_file():
            raise RuntimeError(
                f"Cached HTDemucs checkpoint {signature} is unavailable; downloads are disabled"
            )

    return config, signatures


def load_separator():
    """Select and validate an available runtime by running real model inference."""
    import torch
    from demucs.api import Separator
    import platform
    import time

    requested = os.environ.get("DEMUCS_DEVICE", "auto").lower()
    cuda_available = bool(torch.cuda.is_available())
    is_rocm = bool(getattr(torch.version, "hip", None))
    mps_backend = getattr(getattr(torch.backends, "mps", None), "is_available", None)
    mps_available = bool(mps_backend and mps_backend())
    machine = platform.machine().lower()
    system = platform.system()

    if requested not in ("auto", "cuda", "mps", "rocm", "cpu"):
        raise RuntimeError("DEMUCS_DEVICE must be auto, cuda, mps, rocm, or cpu")

    devices = []
    if requested == "auto":
        if cuda_available and is_rocm:
            devices.append(("rocm", "cuda"))
        elif cuda_available:
            devices.append(("cuda", "cuda"))
        if system == "Darwin" and machine in ("arm64", "aarch64") and mps_available:
            devices.append(("mps", "mps"))
        devices.append(("cpu", "cpu"))
    elif requested == "cuda":
        if cuda_available and not is_rocm:
            devices.append(("cuda", "cuda"))
    elif requested == "rocm":
        if cuda_available and is_rocm:
            devices.append(("rocm", "cuda"))
    elif requested == "mps":
        if system == "Darwin" and machine in ("arm64", "aarch64") and mps_available:
            devices.append(("mps", "mps"))
    else:
        devices.append(("cpu", "cpu"))
    if not devices:
        raise RuntimeError(f"requested Demucs backend '{requested}' is not available in this PyTorch runtime")

    cache_config, signatures = cached_model_files()

    # Prevent Demucs from downloading checkpoints through torch.hub.
    def no_remote_download(*args, **kwargs):
        raise RuntimeError("Remote checkpoint download blocked by worker policy")

    torch.hub.load_state_dict_from_url = no_remote_download

    failures = []
    for backend, device in devices:
        separator = None
        try:
            load_started = time.perf_counter()
            separator = Separator(
                model="htdemucs",
                device=device,
                shifts=1,
                split=True,
                overlap=0.25,
                progress=False,
            )
            model_init_ms = round((time.perf_counter() - load_started) * 1000, 2)
            probe = torch.zeros((separator.audio_channels, separator.samplerate), dtype=torch.float32)
            inference_started = time.perf_counter()
            _, output = separator.separate_tensor(probe, sr=separator.samplerate)
            inference_test_ms = round((time.perf_counter() - inference_started) * 1000, 2)
            expected = set(separator.model.sources)
            if set(output) != expected or not all(
                tuple(value.shape) == tuple(probe.shape) and bool(torch.isfinite(value).all())
                for value in output.values()
            ):
                raise RuntimeError("HTDemucs hardware validation produced invalid stem output")
            gpu_name = None
            memory_bytes = None
            if device == "cuda":
                gpu_name = torch.cuda.get_device_name(0)
                try:
                    memory_bytes = int(torch.cuda.mem_get_info(0)[0])
                except Exception:
                    pass
            elif device == "mps":
                gpu_name = "Apple Silicon MPS"
                try:
                    memory_bytes = int(torch.mps.driver_allocated_memory())
                except Exception:
                    pass
            return separator, {
                "device": device,
                "backend": backend,
                "device_available": True,
                "cuda_available": cuda_available,
                "mps_available": mps_available,
                "rocm_available": is_rocm and cuda_available,
                "device_name": gpu_name or "CPU",
                "device_memory_bytes": memory_bytes,
                "platform": system,
                "architecture": machine,
                "model": "htdemucs",
                "model_initialization_ms": model_init_ms,
                "inference_test_ms": inference_test_ms,
                "inference_test_passed": True,
                "fallback_reason": "; ".join(failures) or None,
                "sample_rate": separator.samplerate,
                "channels": separator.audio_channels,
                "sources": separator.model.sources,
                "cache_config": cache_config,
                "checkpoint_signatures": signatures,
            }
        except Exception as error:
            failures.append(f"{backend} initialization/inference failed: {error}")
            del separator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            if requested != "auto":
                raise RuntimeError(failures[-1]) from error
    raise RuntimeError("all Demucs backends failed validation: " + "; ".join(failures))


def resample_exact(wave, source_rate, target_rate, frames):
    """
    Resample audio and align its ending to an exact frame count.

    The beginning of the audio remains aligned to sample zero.
    """
    import julius
    import torch

    if source_rate == target_rate:
        if int(wave.shape[-1]) != frames:
            raise ValueError(
                "same-rate audio must retain its decoded frame count "
                f"({int(wave.shape[-1])} != {frames})"
            )
        return wave, 0

    converted = julius.resample_frac(
        wave,
        source_rate,
        target_rate,
    )

    raw_frames = int(converted.shape[-1])

    if raw_frames > frames:
        converted = converted[..., :frames]

    elif raw_frames < frames:
        converted = torch.nn.functional.pad(
            converted,
            (0, frames - raw_frames),
        )

    return converted, raw_frames - frames


def validate_metadata(path, expected):
    """Validate source metadata while treating decoded frames as authoritative."""
    import soundfile as sf

    info = sf.info(path)
    actual = {
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frame_count": int(info.frames),
    }

    if (
        not 8_000 <= actual["sample_rate"] <= 384_000
        or actual["frame_count"] <= 0
        or actual["channels"] not in (1, 2)
    ):
        raise ValueError(
            "Only non-empty mono or stereo audio with a sample rate "
            "between 8000 and 384000 Hz is supported"
        )

    # Browser metadata is advisory: browser decoders can report a different
    # decoded sample rate and may not expose the encoded channel/frame values.
    # The source container and the exact waveform decoded below are authoritative.

    return actual


def process(separator, command):
    """Process one separation job and emit incremental stem events."""
    global active_id

    import soundfile as sf
    from demucs.audio import AudioFile

    job_id = command["job_id"]
    active_id = job_id

    cancel_event.clear()

    stage = Path(command["output_dir"])
    stage.mkdir(parents=True, exist_ok=True)

    input_path = Path(command["input_path"])
    expected = command["expected"]

    try:
        # ---------------------------------------------------------
        # 1. Validate the original audio file.
        # ---------------------------------------------------------

        actual = validate_metadata(input_path, expected)

        # ---------------------------------------------------------
        # 2. Decode the source audio.
        # Demucs requires two channels, so mono is duplicated.
        # ---------------------------------------------------------

        if actual["channels"] == 1:
            source = AudioFile(input_path).read(
                streams=0,
                samplerate=actual["sample_rate"],
                channels=1,
            )

            source = source.expand(2, -1).contiguous()

        else:
            source = AudioFile(input_path).read(
                streams=0,
                samplerate=actual["sample_rate"],
            )
        diagnostic = {
            "job_id": job_id,
            "soundfile_sample_rate": actual["sample_rate"],
            "soundfile_channels": actual["channels"],
            "soundfile_frames": actual["frame_count"],
            "audiofile_shape": list(source.shape),
        }

        diagnostic_path = Path("/tmp") / f"dubpad-decoder-{job_id}.json"
        diagnostic_path.write_text(
            json.dumps(diagnostic, indent=2),
            encoding="utf-8",
        )

        # AudioFile's decoded waveform is the authoritative timeline
        # because this exact waveform is passed to HTDemucs.
        if source.ndim != 2 or source.shape[0] != 2:
            raise ValueError(
                f"Decoded waveform must be stereo; got shape {tuple(source.shape)}"
            )

        actual["frame_count"] = int(
            source.shape[-1]
        )  # ---------------------------------------------------------
        # 3. Resample source audio to the HTDemucs sample rate.
        # ---------------------------------------------------------

        model_wave, input_adjustment = resample_exact(
            source,
            actual["sample_rate"],
            separator.samplerate,
            round(actual["frame_count"] * separator.samplerate / actual["sample_rate"]),
        )

        # ---------------------------------------------------------
        # 4. Report inference progress from Demucs.
        # ---------------------------------------------------------

        def callback(progress):
            if cancel_event.is_set():
                raise KeyboardInterrupt(
                    "Cancellation acknowledged at Demucs segment boundary"
                )

            if progress.get("state") == "end":
                models = max(
                    1,
                    int(progress.get("models", 1)),
                )

                segment_fraction = min(
                    1.0,
                    int(progress.get("segment_offset", 0))
                    / max(
                        1,
                        int(progress.get("audio_length", 1)),
                    ),
                )

                percent = min(
                    99.0,
                    100.0
                    * (int(progress.get("model_idx_in_bag", 0)) + segment_fraction)
                    / models,
                )

                emit(
                    {
                        "type": "progress",
                        "job_id": job_id,
                        "progress": percent,
                        "segment_sample": int(progress.get("segment_offset", 0)),
                        "sample_rate": separator.samplerate,
                    }
                )

        separator.update_parameter(callback=callback)

        # ---------------------------------------------------------
        # 5. Run HTDemucs inference.
        #
        # IMPORTANT:
        # HTDemucs returns all four stems together.
        # Incremental events happen during the output-writing phase,
        # not while the model is still performing inference.
        # ---------------------------------------------------------

        _, stems = separator.separate_tensor(
            model_wave,
            separator.samplerate,
        )

        # ---------------------------------------------------------
        # 6. Prepare the final job result.
        # ---------------------------------------------------------

        result = {
            "original": actual,
            "model_sample_rate": separator.samplerate,
            "model_frame_count": int(model_wave.shape[-1]),
            "input_resample_frame_adjustment": input_adjustment,
            "timing_authority": (
                "integer_samples_origin_preserved_tail_aligned_explicitly"
            ),
            "inference_windowing": {
                "input_scope": "whole_track",
                "demucs_split": True,
                "overlap_fraction": 0.25,
                "musical_chunks_finalized_after_inference": True,
            },
            "resample_alignment_validation": (
                "synthetic_impulse_tested; per-output measured offset not available"
            ),
            "stems": {},
        }

        # ---------------------------------------------------------
        # 7. Process every stem individually.
        #
        # Each stem is restored, validated, written and announced
        # before moving on to the next stem.
        # ---------------------------------------------------------

        for name, stem in stems.items():
            if cancel_event.is_set():
                raise KeyboardInterrupt("Cancellation acknowledged before stem output")

            # Restore the original sample rate and frame count.
            restored, adjustment = resample_exact(
                stem,
                separator.samplerate,
                actual["sample_rate"],
                actual["frame_count"],
            )

            # Convert stereo model output back to mono if necessary.
            if actual["channels"] == 1:
                restored = restored.mean(
                    dim=0,
                    keepdim=True,
                )

            # Validate the restored audio dimensions.
            if (
                int(restored.shape[-1]) != actual["frame_count"]
                or int(restored.shape[0]) != actual["channels"]
            ):
                raise ValueError(f"Output shape validation failed for {name}")

            # Define this stem's output WAV path.
            target = stage / f"{name}.wav"

            # Write the stem as 24-bit PCM WAV.
            sf.write(
                str(target),
                restored.cpu().numpy().T,
                actual["sample_rate"],
                subtype="PCM_24",
            )

            # Read the metadata of the written WAV file.
            info = sf.info(target)

            # Validate the actual written file.
            if (
                int(info.frames) != actual["frame_count"]
                or int(info.samplerate) != actual["sample_rate"]
                or int(info.channels) != actual["channels"]
            ):
                raise ValueError(f"Written output validation failed for {name}")

            # Save complete metadata for the final job result.
            result["stems"][name] = {
                "path": str(target),
                "sample_rate": int(info.samplerate),
                "frame_count": int(info.frames),
                "channels": int(info.channels),
                "source_sample_start": 0,
                "source_sample_end": actual["frame_count"],
                "measured_processing_offset_samples": None,
                "resample_frame_adjustment": adjustment,
            }

            # Prepare the incremental stem completion metadata.
            stem_metadata = {
                "sample_rate": int(info.samplerate),
                "frame_count": int(info.frames),
                "channels": int(info.channels),
                "source_sample_start": 0,
                "source_sample_end": actual["frame_count"],
            }

            # Tell Rust that this particular stem is ready.
            emit(
                {
                    "type": "stem.completed",
                    "job_id": job_id,
                    "stem": name,
                    "stem_metadata": stem_metadata,
                }
            )

        # ---------------------------------------------------------
        # 8. All stems have been written successfully.
        # Return the complete result to Rust.
        # ---------------------------------------------------------

        emit(
            {
                "type": "completed",
                "job_id": job_id,
                "result": result,
            }
        )

    except KeyboardInterrupt:
        # Remove partially generated output files after cancellation.
        for path in stage.glob("*"):
            path.unlink(missing_ok=True)

        emit(
            {
                "type": "cancelled",
                "job_id": job_id,
                "acknowledged": True,
            }
        )

    except Exception as error:
        # Remove incomplete output files after a processing failure.
        for path in stage.glob("*"):
            path.unlink(missing_ok=True)

        emit(
            {
                "type": "error",
                "job_id": job_id,
                "message": str(error),
                "trace": traceback.format_exc(limit=8),
            }
        )

    finally:
        active_id = None
        cancel_event.clear()


def main():
    """Start the persistent worker and process incoming jobs."""

    threading.Thread(
        target=read_commands,
        name="stdin-command-reader",
        daemon=True,
    ).start()

    # Load HTDemucs once. Keep it in memory for subsequent jobs.
    try:
        separator, info = load_separator()

    except Exception as error:
        try:
            import torch

            cuda_visible = bool(torch.cuda.is_available())

            gpu_name = torch.cuda.get_device_name(0) if cuda_visible else None

        except Exception:
            cuda_visible = False
            gpu_name = None

        emit(
            {
                "type": "startup_error",
                "message": str(error),
                "cuda_available": cuda_visible,
                "gpu_name": gpu_name,
            }
        )

        return 2

    emit(
        {
            "type": "ready",
            **info,
        }
    )

    # Keep the process alive and reuse the loaded model.
    while True:
        command = commands.get()

        if command.get("type") == "shutdown":
            return 0

        if command.get("type") == "job":
            process(separator, command)


if __name__ == "__main__":
    raise SystemExit(main())
