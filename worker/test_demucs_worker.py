import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from demucs_worker import load_separator, resample_exact, validate_metadata


class AudioIntegrityTests(unittest.TestCase):
    def test_auto_selects_cpu_when_no_accelerator_is_available_and_validates_inference(self):
        import demucs.api

        class FakeSeparator:
            def __init__(self, **_kwargs):
                self.audio_channels = 2
                self.samplerate = 44_100
                self.model = SimpleNamespace(sources=["drums", "bass", "other", "vocals"])

            def separate_tensor(self, wave, sr=None):
                self.asserted_rate = sr
                return wave, {name: wave.clone() for name in self.model.sources}

        old_loader = torch.hub.load_state_dict_from_url
        try:
            with patch("demucs_worker.cached_model_files", return_value=("cached.yaml", ["weights"])), \
                 patch.object(demucs.api, "Separator", FakeSeparator), \
                 patch.object(torch.cuda, "is_available", return_value=False), \
                 patch.dict("os.environ", {"DEMUCS_DEVICE": "auto"}, clear=False):
                separator, info = load_separator()
            self.assertEqual(info["backend"], "cpu")
            self.assertTrue(info["inference_test_passed"])
            self.assertEqual(separator.samplerate, 44_100)
        finally:
            torch.hub.load_state_dict_from_url = old_loader

    def test_auto_falls_back_when_detected_cuda_fails_real_model_probe(self):
        import demucs.api

        class FakeSeparator:
            def __init__(self, device, **_kwargs):
                if device == "cuda":
                    raise RuntimeError("CUDA probe failed")
                self.audio_channels = 2
                self.samplerate = 44_100
                self.model = SimpleNamespace(sources=["drums", "bass", "other", "vocals"])

            def separate_tensor(self, wave, sr=None):
                return wave, {name: wave.clone() for name in self.model.sources}

        old_loader = torch.hub.load_state_dict_from_url
        try:
            with patch("demucs_worker.cached_model_files", return_value=("cached.yaml", ["weights"])), \
                 patch.object(demucs.api, "Separator", FakeSeparator), \
                 patch.object(torch.cuda, "is_available", return_value=True), \
                 patch.object(torch.cuda, "empty_cache"), \
                 patch.dict("os.environ", {"DEMUCS_DEVICE": "auto"}, clear=False):
                separator, info = load_separator()
            self.assertEqual(separator.model.sources, ["drums", "bass", "other", "vocals"])
            self.assertEqual(info["backend"], "cpu")
            self.assertIn("CUDA probe failed", info["fallback_reason"])
        finally:
            torch.hub.load_state_dict_from_url = old_loader

    def test_44100_round_trip_has_exact_frames_and_preserves_origin(self):
        rate = 44_100
        frames = 44_101
        source = torch.zeros(2, frames)
        source[:, 1234] = 1
        model, _ = resample_exact(source, rate, rate, frames)
        restored, adjustment = resample_exact(model, rate, rate, frames)
        self.assertEqual(tuple(restored.shape), (2, frames))
        self.assertEqual(int(torch.argmax(restored[0]).item()), 1234)
        self.assertEqual(adjustment, 0)

    def test_resampling_back_aligns_exact_frame_count_for_common_rates(self):
        for source_rate in (16_000, 22_050, 32_000, 44_100, 48_000, 96_000):
            with self.subTest(source_rate=source_rate):
                frames = 44_101
                wave = torch.zeros(2, frames)
                wave[:, 1200] = 1
                model_frames = round(frames * 44_100 / source_rate)
                model, _ = resample_exact(wave, source_rate, 44_100, model_frames)
                restored, adjustment = resample_exact(model, 44_100, source_rate, frames)
                self.assertEqual(tuple(restored.shape), (2, frames))
                self.assertLessEqual(abs(adjustment), 2)
                peak = int(torch.argmax(restored[0]).item())
                self.assertLessEqual(abs(peak - 1200), 2)

    def test_decoded_file_metadata_is_authoritative_over_browser_advisory_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.wav"
            sf.write(path, np.zeros((441, 2), dtype=np.float32), 44_100)
            self.assertEqual(validate_metadata(path, {"sample_rate": 48_000, "channels": 2, "frame_count": 441}),
                             {"sample_rate": 44_100, "channels": 2, "frame_count": 441})
            self.assertEqual(validate_metadata(path, {"sample_rate": 48_000, "channels": 1, "frame_count": 1}),
                             {"sample_rate": 44_100, "channels": 2, "frame_count": 441})

    def test_decoded_sample_rate_must_be_within_supported_limits(self):
        expected = {"sample_rate": 48_000, "channels": 2, "frame_count": 10}
        for rate in (7_999, 384_001):
            with self.subTest(rate=rate), patch(
                "soundfile.info",
                return_value=SimpleNamespace(samplerate=rate, channels=2, frames=10),
            ):
                with self.assertRaisesRegex(ValueError, "8000 and 384000"):
                    validate_metadata("unused.wav", expected)

        for rate in (8_000, 384_000):
            with self.subTest(rate=rate), patch(
                "soundfile.info",
                return_value=SimpleNamespace(samplerate=rate, channels=2, frames=10),
            ):
                actual = validate_metadata("unused.wav", expected)
                self.assertEqual(actual["sample_rate"], rate)

    def test_same_rate_path_does_not_resample_or_change_frame_count(self):
        wave = torch.zeros(2, 32)
        result, adjustment = resample_exact(wave, 44_100, 44_100, 32)
        self.assertIs(result, wave)
        self.assertEqual(adjustment, 0)
        with self.assertRaisesRegex(ValueError, "same-rate audio"):
            resample_exact(wave, 44_100, 44_100, 31)


if __name__ == "__main__":
    unittest.main()
