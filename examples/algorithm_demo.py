"""Small synthetic, CPU-only examples; no training assets or weights required.

Run from the repository root: python -m examples.algorithm_demo
"""

import json

import numpy as np
import torch

from loopdrift import evaluate_loop, signature_map_from_Y
from loopdrift.objectives import (
    combine_objectives,
    feature_objectives,
    pixel_objective,
    sn2n_constraint,
)


def main():
    torch.set_num_threads(1)
    random = torch.Generator().manual_seed(17)
    query = torch.rand((4, 1, 16, 16), generator=random)
    paired_teacher = query * 0.9
    independent_positive = torch.rand((6, 1, 16, 16), generator=random) * 0.9
    output = (query * 0.95).requires_grad_()
    losses, _ = feature_objectives(output, paired_teacher, independent_positive)
    losses["pixel"] = pixel_objective(output, paired_teacher)
    pair_a = torch.rand((4, 1, 16, 16), generator=random)
    pair_b = torch.rand((4, 1, 16, 16), generator=random)
    prediction_a = (pair_a * 0.9).requires_grad_()
    prediction_b = (pair_b * 0.9).requires_grad_()
    losses["sn2n"], _ = sn2n_constraint(prediction_a, prediction_b, pair_a, pair_b)
    combined = combine_objectives(losses, family="drift", sn2n_weight=0.5)
    combined.backward()

    # Fixed analytic images illustrate the reducer, not a trained denoiser.
    plane = np.linspace(0, 1, 256).reshape(16, 16)
    trajectory = np.broadcast_to(plane, (5, 3, 16, 16)).copy()
    diagnostic = evaluate_loop(trajectory)
    maps = signature_map_from_Y(trajectory, eps=0.01)
    summary = {
        "scope": "synthetic algorithm smoke test; not scientific efficacy",
        "losses": {name: float(value.detach()) for name, value in losses.items()},
        "combined_loss": float(combined.detach()),
        "finite_student_gradient": bool(torch.isfinite(output.grad).all()),
        "loop_diagnostic_decision": diagnostic["decision"],
        "map_shape": list(maps["C_stab"].shape),
        "map_is_error_probability": False,
    }
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

