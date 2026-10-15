"""Engineering-only adapter tests; no dataset images or scientific claims."""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fiji_calibrated_plugin.backend import model_adapter as adapter
from fiji_calibrated_plugin.backend.vendor import aunet


def e64_reference_forward(model, image):
    """The frozen E64 full_infer equations, independent of training imports."""
    image = np.asarray(image, np.float32)
    h, w = image.shape
    hh = 16 * int(np.ceil((h + 64) / 16))
    ww = 16 * int(np.ceil((w + 64) / 16))
    padded = np.pad(image, ((32, hh-h-32), (32, ww-w-32)), mode="reflect")
    with torch.inference_mode():
        return model(torch.from_numpy(padded.copy())[None, None])[0, 0, 32:32+h, 32:32+w].numpy().copy()


class ModelAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory(prefix="fiji_aunet_qa_")
        cls.root = Path(cls.temp.name)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(69001)
            cls.model = aunet.AUnet(1, 1, True).eval()
        cls.chart = {"offset": -7.0, "scale": 32.0}
        cls.safe = cls.root / "e64_safe.pt"
        torch.save({"schema": "e64-inference-only-v1", "model_state_dict": cls.model.state_dict(),
                    "chart": cls.chart}, cls.safe)
        cls.manual = cls.root / "e54_state.pt"
        torch.save({"schema": "e54-fixed-chart-training-v1", "state_dict": cls.model.state_dict()}, cls.manual)
        cls.raw = cls.root / "raw_state.pt"
        torch.save(cls.model.state_dict(), cls.raw)
        cls.legacy = cls.root / "legacy_module.pth"
        package = types.ModuleType("SelfN2N")
        module = types.ModuleType("SelfN2N.AUnet")
        package.AUnet = module
        for name, klass in adapter._LEGACY_CLASSES.items():
            setattr(module, name, klass)
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {"SelfN2N": package, "SelfN2N.AUnet": module}))
            for klass in adapter._LEGACY_CLASSES.values():
                stack.enter_context(patch.object(klass, "__module__", "SelfN2N.AUnet"))
            torch.save(cls.model, cls.legacy)
        cls.numpy_container = cls.root / "legacy_rng.pth"
        torch.save({"model_state_dict": cls.model.state_dict(), "numpy_random_state": np.arange(4)}, cls.numpy_container)

    @classmethod
    def tearDownClass(cls):
        del cls.model
        cls.temp.cleanup()
        torch.set_num_threads(cls.threads)

    def test_frozen_source_digest(self):
        self.assertEqual(len(adapter._check_vendor()), 64)

    def test_old_torch_rejects_untrusted_load(self):
        with patch.object(torch,'__version__','2.5.1'):
            with self.assertRaisesRegex(adapter.CheckpointTrustError,'PyTorch >=2.6'):
                adapter._load_payload(self.safe,False)

    def test_repeat_exact_full_shape_bn_unchanged_and_e64_parity(self):
        predictor = adapter.load_predictor(self.safe, device="cpu")
        image = np.linspace(-20, 50, 17*33, dtype=np.float32).reshape(17, 33)
        bn_before = {name: value.clone() for name, value in predictor._model.named_buffers()}
        first = predictor.predict(image)
        np.testing.assert_array_equal(first, predictor.predict(image))
        model_domain = ((image + 7.0) / 32.0).astype(np.float32)
        expected = -7.0 + 32.0 * e64_reference_forward(self.model, model_domain)
        np.testing.assert_array_equal(first, expected)
        self.assertEqual(first.shape, image.shape)
        self.assertEqual(first.dtype, np.float32)
        for name, value in predictor._model.named_buffers():
            self.assertTrue(torch.equal(value, bn_before[name]), name)
        self.assertTrue(predictor.assert_unchanged())

    def test_fixed_affine_units_and_no_per_call_minmax(self):
        native = np.linspace(-2, 3, 9*21, dtype=np.float32).reshape(9, 21)
        identity_chart = adapter.load_predictor(self.manual, "cpu", chart={"offset": 0, "scale": 1})
        affine_chart = adapter.load_predictor(self.manual, "cpu", chart=self.chart)
        base = identity_chart.predict(native)
        output = affine_chart.predict(-7.0 + 32.0 * native)
        np.testing.assert_allclose(output, -7.0 + 32.0 * base, rtol=0, atol=4e-6)
        # Different loop states are transformed with the same chart, not refit.
        shifted = affine_chart.predict(-7.0 + 32.0 * (native + 1.0))
        self.assertGreater(float(np.max(np.abs(shifted-output))), 1e-5)
        self.assertFalse(affine_chart.manifest["model_chart_qualified"])
        self.assertFalse(affine_chart.manifest["per_call_normalization"])

    def test_safe_loading_and_manifest_snapshot(self):
        predictor = adapter.load_predictor(self.safe, "cpu")
        manifest = predictor.manifest
        self.assertEqual(manifest["load_mode"], "weights_only")
        self.assertEqual(manifest["chart"], self.chart)
        self.assertTrue(manifest["model_chart_qualified"])
        json.dumps(manifest, allow_nan=False)
        manifest["chart"]["scale"] = -1
        self.assertEqual(predictor.manifest["chart"], self.chart)

    def test_missing_chart_is_distinct_and_conflicting_chart_rejected(self):
        with self.assertRaises(adapter.MissingChartError):
            adapter.load_predictor(self.manual, "cpu")
        with self.assertRaisesRegex(ValueError, "conflicts"):
            adapter.load_predictor(self.safe, "cpu", chart={"offset": 0, "scale": 255})
        for bad in ({"offset": 0, "scale": 0}, {"offset": 0, "scale": float("nan")}):
            with self.assertRaises(ValueError) as found:
                adapter.load_predictor(self.manual, "cpu", chart=bad)
            self.assertNotIsInstance(found.exception, adapter.MissingChartError)

    def test_raw_state_dict_supported_with_manual_chart(self):
        predictor = adapter.load_predictor(self.raw, "cpu", chart=self.chart)
        self.assertEqual(predictor.manifest["chart_source"], "manual_unqualified")

    def test_auto_falls_back_to_cpu(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            predictor = adapter.load_predictor(self.safe)
            self.assertEqual(predictor.manifest["device"], "cpu")
            with self.assertRaisesRegex(ValueError, "unavailable"):
                adapter.load_predictor(self.safe, "cuda:0")

    def test_legacy_module_explicit_trust_and_local_class_mapping(self):
        with self.assertRaises(adapter.CheckpointTrustError):
            adapter.load_predictor(self.legacy, "cpu", chart=self.chart)
        self.assertNotIn("SelfN2N.AUnet", sys.modules)
        predictor = adapter.load_predictor(self.legacy, "cpu", trust_checkpoint=True, chart=self.chart)
        self.assertEqual(predictor.manifest["load_mode"], "trusted_legacy_pickle")
        self.assertNotIn("SelfN2N.AUnet", sys.modules)
        self.assertTrue(predictor.assert_unchanged())

    def test_numpy_rng_legacy_container_requires_trust(self):
        with self.assertRaises(adapter.CheckpointTrustError):
            adapter.load_predictor(self.numpy_container, "cpu", chart=self.chart)
        predictor = adapter.load_predictor(self.numpy_container, "cpu", trust_checkpoint=True, chart=self.chart)
        self.assertTrue(predictor.assert_unchanged())

    def test_arbitrary_architecture_fails_before_chart_fallback(self):
        path = self.root / "bad_architecture.pt"
        torch.save({"model_state_dict": torch.nn.Conv2d(1, 1, 1).state_dict()}, path)
        with self.assertRaisesRegex(ValueError, "architecture") as found:
            adapter.load_predictor(path, "cpu")
        self.assertNotIsInstance(found.exception, adapter.MissingChartError)

    def test_corruption_and_ambiguous_state_rejected(self):
        state = dict(self.model.state_dict())
        state["outc.conv.bias"] = torch.full_like(state["outc.conv.bias"], float("nan"))
        path = self.root / "nonfinite.pt"
        torch.save({"model_state_dict": state}, path)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            adapter.load_predictor(path, "cpu", chart=self.chart)
        path = self.root / "ambiguous.pt"
        torch.save({"model_state_dict": self.model.state_dict(), "state_dict": self.model.state_dict()}, path)
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            adapter.load_predictor(path, "cpu", chart=self.chart)

    def test_assert_unchanged_detects_parameter_bn_and_mode_changes(self):
        predictor = adapter.load_predictor(self.safe, "cpu")
        predictor._model.inc.double_conv[1].running_mean.add_(1)
        with self.assertRaisesRegex(AssertionError, "changed"):
            predictor.assert_unchanged()
        predictor = adapter.load_predictor(self.safe, "cpu")
        predictor._model.train()
        with self.assertRaisesRegex(AssertionError, "eval"):
            predictor.predict(np.zeros((3, 5), np.float32))
        with self.assertRaisesRegex(AssertionError, "eval"):
            predictor.assert_unchanged()

    def test_layer_configuration_and_checkpoint_immutability(self):
        predictor = adapter.load_predictor(self.safe, "cpu")
        predictor._model.inc.double_conv[1].eps = 0.02
        with self.assertRaisesRegex(AssertionError, "configuration"):
            predictor.assert_unchanged()
        path = self.root / "replace_detect.pt"
        torch.save({"model_state_dict": self.model.state_dict(), "chart": self.chart}, path)
        predictor = adapter.load_predictor(path, "cpu")
        torch.save({"fixture": "changed"}, path)
        with self.assertRaisesRegex(AssertionError, "Checkpoint file changed"):
            predictor.assert_unchanged()

    def test_invalid_input_fail_closed_and_small_reflect_input(self):
        predictor = adapter.load_predictor(self.safe, "cpu")
        for bad in (np.zeros((1, 4)), np.zeros((2, 3, 4)), np.full((3, 4), np.inf),
                    np.ones((3, 4), np.complex64)):
            with self.assertRaises(ValueError):
                predictor.predict(bad)
        self.assertEqual(predictor.predict(np.zeros((2, 3), np.float32)).shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
