import unittest

import numpy as np

from stem_alignment import estimate_offset, measure_arrays


class StemAlignmentTests(unittest.TestCase):
    def test_sample_offset_sign_and_precision(self):
        rng = np.random.default_rng(17)
        original = rng.normal(0, 0.1, 16_000).astype(np.float32)
        delay = 137
        shifted = np.pad(original[:-delay], (delay, 0))
        offset, confidence = estimate_offset(original, shifted, 500)
        self.assertEqual(offset, delay)
        self.assertGreater(confidence, 0.99)

    def test_report_marks_constant_and_varying_alignment_separately(self):
        rng = np.random.default_rng(29)
        original = rng.normal(0, 0.1, 32_000).astype(np.float32)
        delay = 21
        constant = np.pad(original[:-delay], (delay, 0))
        varying = original.copy()
        varying[16_000:] = np.pad(original[16_000:-delay], (delay, 0))
        stems = {"drums": constant, "bass": constant, "other": varying, "vocals": constant}
        report = measure_arrays(original, stems, 16_000, window_seconds=1, max_lag_ms=10)
        self.assertTrue(report["stems"]["drums"]["constant_offset_candidate"])
        self.assertEqual(report["stems"]["drums"]["median_offset_samples"], delay)
        self.assertFalse(report["stems"]["other"]["constant_offset_candidate"])

    def test_low_energy_windows_are_not_given_fabricated_offsets(self):
        original = np.ones(8_000, dtype=np.float32)
        stems = {name: np.zeros_like(original) for name in ("drums", "bass", "other", "vocals")}
        report = measure_arrays(original, stems, 8_000, window_seconds=1, max_lag_ms=10)
        self.assertIsNone(report["stems"]["drums"]["median_offset_samples"])
        self.assertTrue(all(window["low_energy"] for window in report["stems"]["drums"]["windows"]))


if __name__ == "__main__":
    unittest.main()
