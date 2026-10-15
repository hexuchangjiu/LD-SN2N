"""Bounded CPU checks for the selected algorithm interfaces."""

import unittest

import numpy as np
import torch
from torch import nn

from fiji_calibrated_plugin.backend.vendor import energy, loop_map
from loopdrift import evaluate_loop, loop_signature, signature_map_from_Y
from loopdrift.drifting import drifting_field, drifting_loss
from loopdrift.features import image_feature_groups, spatial_feature_groups
from loopdrift.objectives import (
    combine_objectives, feature_objectives, pixel_objective, sn2n_constraint,
)


class LoopTests(unittest.TestCase):
    def setUp(self):
        frame = np.arange(64, dtype=float).reshape(8, 8) / 63
        self.trajectory = np.broadcast_to(frame, (5, 3, 8, 8)).copy()

    def test_signature_exactly_reuses_vendor(self):
        self.trajectory[2, 2] *= 0.99
        expected = energy.signature_from_Y(self.trajectory)
        actual = loop_signature(self.trajectory)
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name])
        self.assertEqual((actual["K"], actual["T"]), (5, 3))

    def test_verdict_exactly_reuses_vendor(self):
        result = evaluate_loop(self.trajectory)
        self.assertEqual(result["checks"], energy.verdict(result["signature"]))
        self.assertEqual(result["decision"], energy.decide_from_energy(result["checks"])[0])
        self.assertEqual(result["decision"], "reliable")
        self.assertEqual(result["scope"], "single_trajectory_diagnostic_only")

    def test_map_is_same_function_and_finite(self):
        self.assertIs(signature_map_from_Y, loop_map.signature_map_from_Y)
        maps = signature_map_from_Y(self.trajectory, eps=0.01)
        for name in ("C_stab", "gamma", "R"):
            self.assertEqual(maps[name].shape, (8, 8))
            self.assertTrue(np.isfinite(maps[name]).all())

    def test_invalid_trajectory(self):
        for array in (np.zeros((2, 8, 8)), np.zeros((1, 1, 8, 8)),
                      np.full((2, 3, 8, 8), np.nan), np.zeros((0, 3, 8, 8))):
            with self.subTest(shape=array.shape), self.assertRaises(ValueError):
                loop_signature(array)
        for options in ({"gain_mode": "unknown"}, {"gain_contrast_norm": "unknown"},
                        {"var_cap": -1}, {"gain_defer_q": 0.5}, {"horizon": 1.5}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                evaluate_loop(self.trajectory, **options)


class DriftingTests(unittest.TestCase):
    def test_hand_computed_field_and_gradient(self):
        generated = torch.tensor([[[0.0], [2.0]]], requires_grad=True)
        positive = torch.tensor([[[4.0]]], requires_grad=True)
        field = drifting_field(generated, positive)
        torch.testing.assert_close(field, torch.tensor([[[2.0], [4.0]]]))
        loss, details = drifting_loss(generated, positive)
        self.assertEqual(float(loss), 10.0)
        loss.backward()
        torch.testing.assert_close(generated.grad, torch.tensor([[[-2.0], [-4.0]]]))
        self.assertIsNone(positive.grad)
        self.assertFalse(field.requires_grad)
        self.assertTrue(all(not value.requires_grad for value in details.values()))

    def test_explicit_negative_detaches_and_finite_backward(self):
        gen = torch.tensor([[[0.1, 0.3], [0.4, 0.2]]], requires_grad=True)
        pos = torch.tensor([[[0.8, 0.4], [0.7, 0.9]]], requires_grad=True)
        neg = torch.tensor([[[0.0, 0.2]]], requires_grad=True)
        loss, info = drifting_loss(gen, pos, neg, step_size=0.25)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(gen.grad).all())
        self.assertGreater(float(gen.grad.abs().sum()), 0)
        self.assertIsNone(pos.grad)
        self.assertIsNone(neg.grad)
        torch.testing.assert_close(gen.grad, 2 * (gen.detach() - info["target"]) / gen.numel())

    def test_swap_banks_reverses_field(self):
        generator = torch.Generator().manual_seed(1)
        gen = torch.rand((2, 3, 4), generator=generator)
        pos = torch.rand((2, 5, 4), generator=generator)
        neg = torch.rand((2, 6, 4), generator=generator)
        torch.testing.assert_close(drifting_field(gen, pos, neg), -drifting_field(gen, neg, pos))
        torch.testing.assert_close(drifting_field(gen, pos, pos), torch.zeros_like(gen))

    def test_positive_permutation_invariance(self):
        gen = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
        pos = gen + 0.3
        torch.testing.assert_close(drifting_field(gen, pos), drifting_field(gen, pos[:, [2, 0, 1]]))

    def test_double_precision_and_half_input(self):
        for dtype in (torch.float16, torch.float32, torch.float64):
            gen = torch.tensor([[[0.1], [0.3]]], dtype=dtype, requires_grad=True)
            loss, _ = drifting_loss(gen, gen.detach() + 0.2)
            loss.backward()
            self.assertTrue(torch.isfinite(gen.grad).all())
            self.assertEqual(loss.dtype, torch.float64 if dtype == torch.float64 else torch.float32)

    def test_coordinate_scaling_is_explicit(self):
        gen = torch.tensor([[[0.1, 0.2], [0.4, 0.5]]], dtype=torch.float64)
        pos = gen + 0.3
        expected = drifting_field(gen, pos, temperature=0.2)
        scaled = drifting_field(gen * 3, pos * 3, temperature=0.6)
        torch.testing.assert_close(scaled, expected * 3)

    def test_invalid_banks_and_parameters(self):
        valid = torch.zeros((1, 2, 3))
        cases = [torch.zeros((1, 1, 3)), torch.zeros((1, 2)),
                 torch.zeros((1, 2, 3), dtype=torch.int64), torch.full((1, 2, 3), float("nan"))]
        for gen in cases:
            with self.subTest(shape=gen.shape), self.assertRaises(ValueError):
                drifting_loss(gen, valid)
        for pos in (torch.zeros((2, 2, 3)), torch.zeros((1, 0, 3)), torch.zeros((1, 2, 4))):
            with self.assertRaises(ValueError):
                drifting_loss(valid, pos)
        for options in ({"temperature": 0}, {"temperature": float("nan")},
                        {"temperature": 1e-300}, {"step_size": -1}, {"step_size": float("inf")}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                drifting_loss(valid, valid, **options)


class FeatureTests(unittest.TestCase):
    def test_pool_matches_average_pool_and_layout(self):
        image = torch.arange(128, dtype=torch.float32).reshape(2, 1, 8, 8)
        groups = spatial_feature_groups({"x": image}, 2)["x"]
        expected = torch.nn.functional.avg_pool2d(image, 4).flatten(2).permute(2, 0, 1)
        torch.testing.assert_close(groups, expected)
        self.assertEqual(groups.shape, (4, 2, 1))

    def test_raw_moments_keep_photometric_coordinate(self):
        image = torch.full((2, 1, 8, 8), 3.0, requires_grad=True)
        groups = image_feature_groups(image, pool_size=2)
        torch.testing.assert_close(groups["raw_mean"], torch.full((4, 2, 1), 3.0))
        torch.testing.assert_close(groups["raw_rms"], torch.full((4, 2, 1), 3.0))
        sum(value.mean() for value in groups.values()).backward()
        self.assertTrue(torch.isfinite(image.grad).all())

    def test_invalid_feature_layout(self):
        for image, pool in ((torch.ones(2, 1, 7, 8), 2), (torch.ones(2, 1, 8, 8), 0),
                            (torch.ones(2, 1, 8, 8), True)):
            with self.assertRaises(ValueError):
                image_feature_groups(image, pool_size=pool)
        with self.assertRaises(ValueError):
            image_feature_groups(torch.ones(2, 3, 8, 8))
        with self.assertRaises(ValueError):
            image_feature_groups(torch.ones(2, 1, 8, 8), {"raw_mean": torch.ones(2, 1, 8, 8)})


class ObjectiveTests(unittest.TestCase):
    def test_pixel_teacher_detaches_and_shape_guard(self):
        output = torch.zeros(2, 1, 4, 4, requires_grad=True)
        teacher = torch.ones_like(output, requires_grad=True)
        loss = pixel_objective(output, teacher)
        self.assertEqual(float(loss), 1)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(output.grad).all())
        with self.assertRaises(ValueError):
            pixel_objective(output, teacher[:, :, :1])

    def test_sn2n_formula_and_two_prediction_gradients(self):
        a = torch.full((2, 1, 4, 4), 0.2, requires_grad=True)
        b = torch.full((2, 1, 4, 4), 0.7, requires_grad=True)
        pred_a = torch.full_like(a, 0.4, requires_grad=True)
        pred_b = torch.full_like(b, 0.5, requires_grad=True)
        actual, detail = sn2n_constraint(pred_a, pred_b, a, b, reg=0.5)
        expected = ((pred_a - b.detach()).abs().mean() + (pred_b - a.detach()).abs().mean()
                    + 0.5 * (pred_a - pred_b).abs().mean()) / 2.5
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertGreater(float(pred_a.grad.abs().sum()), 0)
        self.assertGreater(float(pred_b.grad.abs().sum()), 0)
        self.assertIsNone(a.grad)
        self.assertIsNone(b.grad)
        self.assertTrue(all(not value.requires_grad for value in detail.values()))

    def test_sn2n_zero_reg_is_forward_only(self):
        a, b = torch.zeros(2, 1, 4, 4), torch.ones(2, 1, 4, 4)
        pred_a = torch.full_like(a, 0.3, requires_grad=True)
        pred_b = torch.full_like(b, 0.5, requires_grad=True)
        loss, _ = sn2n_constraint(pred_a, pred_b, a, b, reg=0)
        torch.testing.assert_close(loss, (pred_a - b).abs().mean())
        loss.backward()
        self.assertIsNone(pred_b.grad)
        with self.assertRaises(ValueError):
            sn2n_constraint(pred_a, None, a, b)
        with self.assertRaises(ValueError):
            sn2n_constraint(pred_a, pred_b, a, b, reg=-1)

    def test_shared_feature_paths_and_detachment(self):
        gen = torch.zeros(3, 1, 8, 8, requires_grad=True)
        teacher = torch.ones_like(gen, requires_grad=True)
        positive = torch.full((5, 1, 8, 8), 0.6, requires_grad=True)
        losses, diagnostics = feature_objectives(gen, teacher, positive, pool_size=2)
        self.assertEqual(set(losses), {"feature", "drift"})
        self.assertEqual(set(diagnostics), {"raw_mean", "raw_rms"})
        sum(losses.values()).backward()
        self.assertTrue(torch.isfinite(gen.grad).all())
        self.assertIsNone(teacher.grad)
        self.assertIsNone(positive.grad)

    def test_frozen_encoder_retains_image_gradient(self):
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 2, 1, bias=False)

            def features(self, image):
                return {"learned": self.conv(image)}

        encoder = Encoder()
        gen = torch.full((2, 1, 8, 8), 0.2, requires_grad=True)
        teacher = torch.ones_like(gen)
        with self.assertRaises(ValueError):
            feature_objectives(gen, teacher, teacher, encoder)
        encoder.eval().requires_grad_(False)
        losses, _ = feature_objectives(gen, teacher, teacher, encoder)
        sum(losses.values()).backward()
        self.assertGreater(float(gen.grad.abs().sum()), 0)
        self.assertTrue(all(parameter.grad is None for parameter in encoder.parameters()))

    def test_objective_selection_and_missing_inputs(self):
        gen = torch.zeros(2, 1, 8, 8, requires_grad=True)
        losses, _ = feature_objectives(gen, torch.ones_like(gen), None, need_drift=False)
        self.assertEqual(set(losses), {"feature"})
        losses, _ = feature_objectives(gen, None, torch.ones_like(gen), need_feature=False)
        self.assertEqual(set(losses), {"drift"})
        for options in ({"paired_teacher": None, "positive": gen},
                        {"paired_teacher": gen, "positive": None},
                        {"paired_teacher": gen, "positive": gen, "feature_scale": 0}):
            with self.assertRaises(ValueError):
                feature_objectives(gen, **options)

    def test_combination_and_guards(self):
        losses = {"pixel": torch.tensor(2.0, requires_grad=True),
                  "sn2n": torch.tensor(3.0, requires_grad=True)}
        value = combine_objectives(losses, family="pixel", family_scale=4, sn2n_weight=0.5)
        self.assertEqual(float(value), 9.5)
        value.backward()
        self.assertEqual(float(losses["pixel"].grad), 4)
        self.assertEqual(float(losses["sn2n"].grad), 0.5)
        self.assertIs(combine_objectives(losses, family="sn2n"), losses["sn2n"])
        for options in ({"family": "other"}, {"family": "drift"},
                        {"family": "pixel", "sn2n_weight": -1},
                        {"family": "sn2n", "sn2n_weight": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                combine_objectives(losses, **options)

    def test_finite_inputs_cannot_silently_return_infinite_objectives(self):
        huge = torch.full((2, 1, 8, 8), 1e30)
        with self.assertRaises(FloatingPointError):
            pixel_objective(huge, torch.zeros_like(huge))
        with self.assertRaises(FloatingPointError):
            combine_objectives({"pixel": torch.tensor(1e30)}, family="pixel", family_scale=1e30)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

