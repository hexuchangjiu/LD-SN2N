"""Bounded E69 estimator/contract tests, not a deployment validation study."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fiji_calibrated_plugin.backend.noise_calibration import (
    NoiseCalibrationError, QUALIFIED, NOT_QUALIFIED, calibrate_stack, sample_noise,
)


def fixture(seed=69, sigma=3.0, correlated=False):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:24, :32]
    scene = 100.0 + 0.7 * x + 0.3 * y
    noise = rng.normal(size=(98, 24, 32))
    if correlated:
        noise = (noise + np.roll(noise, 1, axis=2)) / np.sqrt(2.0)
    return scene, scene + sigma * noise


class FijiNoiseCalibrationTests(unittest.TestCase):
    def test_white_noise_moments_heldout_and_json(self):
        scene, stack = fixture()
        report, bank = calibrate_stack(stack)
        self.assertEqual(report["qualification"]["status"], QUALIFIED)
        self.assertFalse(report["qualification"]["deployment_qualified"])
        self.assertAlmostEqual(report["measured_noise_sigma_native"], 3.0, delta=0.15)
        self.assertAlmostEqual(report["heldout_validation"]["variance_ratio"], 1.0, delta=0.12)
        self.assertEqual(bank.dtype, np.float32)
        self.assertLess(float(np.max(np.abs(bank.mean(axis=0)))), 3e-7)
        json.dumps(report, allow_nan=False)

    def test_correlated_noise_is_preserved_not_whitened(self):
        scene, stack = fixture(correlated=True)
        report, bank = calibrate_stack(stack)
        self.assertEqual(report["qualification"]["status"], QUALIFIED)
        c = report["heldout_validation"]["neighbor_correlation"]["x"]
        self.assertGreater(c["fit"], 0.40)
        self.assertLess(c["absolute_difference"], 0.1)
        self.assertGreater(np.corrcoef(bank[:, :, 1:].ravel(), bank[:, :, :-1].ravel())[0, 1], 0.4)

    def test_affine_brightness_units_and_no_random_normalization(self):
        scene, stack = fixture()
        first, bank1 = calibrate_stack(stack)
        second, bank2 = calibrate_stack(stack * 2.5 + 37.0)
        self.assertEqual(first["qualification"]["status"], second["qualification"]["status"])
        self.assertAlmostEqual(second["measured_noise_sigma_native"], first["measured_noise_sigma_native"] * 2.5, places=10)
        np.testing.assert_allclose(bank2, bank1 * 2.5, atol=3e-6)
        a = sample_noise(scene, np.random.default_rng(1), first, bank1)
        b = sample_noise(scene * 2.5 + 37.0, np.random.default_rng(1), second, bank2)
        np.testing.assert_allclose(b, 2.5 * a + 37.0, atol=3e-6)
        self.assertGreater(float(b.max()), 1.0)

    def test_query_pair_excluded_and_partition_disjoint(self):
        _, stack = fixture()
        first, bank1 = calibrate_stack(stack, query_index=4)
        stack[4:6] = 1e6
        second, bank2 = calibrate_stack(stack, query_index=4)
        np.testing.assert_array_equal(bank1, bank2)
        self.assertEqual(first["measured_noise_sigma_native"], second["measured_noise_sigma_native"])
        fit = set(second["source"]["fit_frame_indices"])
        held = set(second["source"]["heldout_frame_indices"])
        self.assertFalse(fit & held)
        self.assertFalse((fit | held) & {4, 5})
        self.assertEqual(first["qualification"], second["qualification"])

    def test_bleaching_drift_and_nonstationary_noise_not_qualified(self):
        scene, stack = fixture()
        variants = [stack + np.linspace(0, 30, len(stack))[:, None, None],
                    scene + (stack - scene) * np.linspace(1, 5, len(stack))[:, None, None],
                    stack + np.linspace(0, 20, len(stack))[:, None, None] * np.cos(np.arange(32))[None, None, :]]
        for value in variants:
            report, bank = calibrate_stack(value)
            self.assertEqual(report["qualification"]["status"], NOT_QUALIFIED)
            self.assertTrue(report["qualification"]["reasons"])
            with self.assertRaises(NoiseCalibrationError):
                sample_noise(scene, np.random.default_rng(4), report, bank)

    def test_contamination_is_not_silently_removed(self):
        _, stack = fixture()
        # Original pair (10,11) belongs to the fit split after query pair
        # (0,1) exclusion; do not accidentally assert that heldout leaks in.
        stack[10, 4:16, 8:24] += 1000.0
        report, bank = calibrate_stack(stack)
        self.assertEqual(report["qualification"]["status"], NOT_QUALIFIED)
        self.assertGreater(float(np.max(np.abs(bank))), 100.0)

    def test_heldout_contamination_does_not_change_the_fitted_bank(self):
        _, stack = fixture()
        original, bank1 = calibrate_stack(stack)
        stack[12, 4:16, 8:24] += 1000.0
        changed, bank2 = calibrate_stack(stack)
        np.testing.assert_array_equal(bank1, bank2)
        self.assertEqual(original["measured_noise_sigma_native"], changed["measured_noise_sigma_native"])
        self.assertEqual(changed["qualification"]["status"], NOT_QUALIFIED)

    def test_temporal_dependence_not_certified_by_disjoint_split(self):
        scene, _ = fixture()
        rng = np.random.default_rng(698)
        for rho in (-0.65, 0.65):
            z = rng.normal(size=(98, 24, 32))
            for t in range(1, len(z)):
                z[t] = rho * z[t - 1] + np.sqrt(1 - rho * rho) * z[t]
            report, _ = calibrate_stack(scene + 3 * z)
            self.assertEqual(report["qualification"]["status"], NOT_QUALIFIED)
            self.assertTrue(any("lag_" in r for r in report["qualification"]["reasons"]))

    def test_replay_input_preservation_and_local_rng(self):
        scene, stack = fixture()
        saved = stack.copy()
        report, bank = calibrate_stack(stack)
        state = np.random.get_state()
        a = sample_noise(scene, np.random.default_rng(99), report, bank)
        b = sample_noise(scene, np.random.default_rng(99), report, bank)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(stack, saved)
        np.testing.assert_array_equal(state[1], np.random.get_state()[1])

    def test_domain_shape_and_numeric_guards(self):
        _, stack = fixture()
        for domain in ("z", "phase", "live", "multichannel", "static_repeats "):
            with self.assertRaises(NoiseCalibrationError):
                calibrate_stack(stack, domain=domain)
        invalid = [stack[:10], stack[0], stack[:, None], np.zeros_like(stack),
                   np.full_like(stack, np.nan), np.full_like(stack, np.inf), stack.astype(bool)]
        for value in invalid:
            with self.assertRaises(NoiseCalibrationError):
                calibrate_stack(value)
        for index in (-1, True, 0.5, len(stack)):
            with self.assertRaises(NoiseCalibrationError):
                calibrate_stack(stack, query_index=index)

    def test_artifact_identity_replay_and_no_overwrite(self):
        scene, stack = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            report, bank = calibrate_stack(stack, output_dir=tmp)
            saved = json.loads((Path(tmp) / "noise_calibration_report.json").read_text())
            reloaded = np.load(Path(tmp) / "noise_bank.npy", allow_pickle=False)
            np.testing.assert_array_equal(sample_noise(scene, np.random.default_rng(8), report, bank),
                                          sample_noise(scene, np.random.default_rng(8), saved, reloaded))
            with self.assertRaises(NoiseCalibrationError):
                calibrate_stack(stack, output_dir=tmp)
        altered = bank.copy()
        altered[0, 0, 0] += 1
        with self.assertRaises(NoiseCalibrationError):
            sample_noise(scene, np.random.default_rng(0), report, altered)
        with self.assertRaises(NoiseCalibrationError):
            sample_noise(scene + 1e6, np.random.default_rng(0), report, bank)


if __name__ == "__main__":
    unittest.main()
