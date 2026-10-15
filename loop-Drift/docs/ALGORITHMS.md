# Algorithm interfaces

The `loopdrift` module provides a small set of diagnostic and learning primitives. It does not include a complete trainer, datasets, checkpoints or the historical experiment pipeline.

## Loop diagnostics

`loop_signature(Y)` accepts finite trajectories with shape `[K,T,H,W]`, where K ≥ 1, T ≥ 2 and H/W ≥ 2. It returns residual energy, stochastic dispersion and structural gain. `evaluate_loop(Y)` applies the shared P1–P4 implementation. `signature_map_from_Y` and `render_2d_colormap` directly expose the shared Map implementation.

These interfaces do not perform noise calibration or the full Fiji assessment. One seed cannot estimate stochastic dispersion. A flat image is not evidence of structural fidelity. P2 checks all supplied residual transitions; the other checks retain their defined horizon semantics.

The Map represents loop-response dispersion and structural direction. It is not a pixelwise reconstruction error, a probability of correctness or a posterior sample. Comparisons between images require shared numerical and display scales.

## Drifting objective

The compact implementation follows the field construction in Deng et al., [Generative Modeling via Drifting](https://arxiv.org/html/2602.04770v1), Eqs. 6 and 8–12. For query x, positive samples p and negative samples n:

```text
a_j(x) = softmax_j(-||x-p_j||_2 / temperature)
b_j(x) = softmax_j(-||x-n_j||_2 / temperature)
V(x)   = sum_j a_j(x)*p_j - sum_j b_j(x)*n_j
target = stopgrad(x + step_size*V(x))
loss   = mean((x-target)^2)
```

`drifting_field` returns a detached field. `drifting_loss` returns the loss and diagnostics. Features use shape `[groups,samples,channels]`. By default, negatives come from the generated batch with self-pairs excluded, requiring at least two samples. Explicit negative banks are used as supplied. Only predictions receive gradients; bank, field and target paths are detached.

This is a minimal equation-level implementation, not the complete paper training code. It omits bidirectional affinities, affinity floors, adaptive batch scaling, force-RMS normalization, multi-temperature aggregation, queues and pretrained encoders. Temperature and step size are expressed in the supplied feature coordinates. Identical explicit positive and negative banks give a zero field; leave-one-out sampling need not.

## Comparison and constraint losses

| Function | Definition |
|---|---|
| `pixel_objective` | Paired image MSE against a detached teacher output |
| `image_feature_groups` | Spatial bins mapping [B,C,H,W] to [P²,B,C], with optional photometric moments |
| `feature_objectives` | Feature MSE and Drifting on the same groups and fixed feature scale |
| `sn2n_constraint` | Spatial SN2N L1 objective described below |
| `combine_objectives` | Fixed `family_scale * family_loss + sn2n_weight * sn2n_loss` |

For paired inputs a and b and positive regularization coefficient r, the spatial constraint is

```text
L = [MAE(D(a), b) + MAE(D(b), a) + r*MAE(D(a), D(b))] / (2+r).
```

At r = 0 the historical forward-only branch is retained: `L = MAE(D(a), b)`. Both predicted images receive gradients in the consistency term; input targets are detached. The selected implementation does not include wavelet, sparsity or pair-generation code.

An optional encoder must be frozen and in evaluation mode; gradients through its input still reach the student. Photometric RMS uses `sqrt(mean(x²)+1e-8)`. Feature scales are fixed caller-supplied values, not automatically matched training scales.

Distribution matching alone does not establish image correspondence or structural fidelity. Data independence, teacher choice and train/test separation remain the caller's responsibility.

## Tests and source attribution

From `loop-Drift`:

```console
python -m examples.algorithm_demo
python -m unittest discover -s tests -p test_algorithms.py -v
```

Tests cover hand-calculated fields and gradients, detached banks, antisymmetry, invalid inputs, pooling axes, frozen encoders, both SN2N branches and consistency with the shared Loop engine.

The compact Drifting module was written from the cited equations rather than copied from the reference training repository. No software license was found in the inspected reference revision, so no permission to redistribute that reference implementation is claimed. See [licensing and attribution](../LICENSE_STATUS.md).
