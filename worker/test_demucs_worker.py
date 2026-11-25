import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from demucs_worker import resample_exact, validate_metadata


class AudioIntegrityTests(unittest.TestCase):
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

    def test_decoded_metadata_must_match_expected_source_and_channel_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.wav"
            sf.write(path, np.zeros((441, 2), dtype=np.float32), 44_100)
            self.assertEqual(validate_metadata(path, {"sample_rate": 44_100, "channels": 2, "frame_count": 441}),
                             {"sample_rate": 44_100, "channels": 2, "frame_count": 441})
            with self.assertRaisesRegex(ValueError, "frame_count"):
                validate_metadata(path, {"sample_rate": 44_100, "channels": 2, "frame_count": 440})
            with self.assertRaisesRegex(ValueError, "channels"):
                validate_metadata(path, {"sample_rate": 44_100, "channels": 1, "frame_count": 441})


if __name__ == "__main__":
    unittest.main()
