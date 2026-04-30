# Morphing of and writing with a scissor linkage mechanism
---
## Reference

If you use these notebooks, please cite:

> Mohanraj, A. & Prasath, S. G. (2026). *Morphing of and writing with a scissor linkage mechanism*. arXiv:2602.14958
Two notebooks, two inverse-design tasks. Both use the same differentiable kinematic engine; what differs is what they ask the optimizer to do. If you have not read the thesis, the short version is: a scissor mechanism is a chain of $N$ units, each unit is two rigid arms crossed at a pin joint, and the only thing you tune per unit is where along the arm that pin sits  the **aspect ratio** $\alpha_j$. Changing $\alpha_j$ changes how much that unit curves, which propagates to everything downstream. The whole mechanism deploys from a single actuated angle $\Psi$.

---

## `ShapeMorphingTask.ipynb`

**What it does:** Given a target planar curve, find $\{\alpha_j\}$ and a target actuation angle $\Psi^*$ so the deployed *shape* of the mechanism  meaning its full centerline  approximates that curve when $\Psi = \Psi^*$.

This is the static task (§4.1 of the thesis): one snapshot, one actuation angle, match the entire body of the mechanism to the curve.

### How the pipeline works

**Target processing.** The class `StarTargetCurve` generates a smooth 5-pointed polar star ($N = 55$ units by default) and does three things: arc-length parameterizes it, resamples to $N+1$ equidistant nodes, and computes the signed curvature profile $\kappa^t_j$ at each node via finite differences. This gives you the reference curvature values the optimizer is trying to match.

**Forward model.** `ScissorInverseDesign` is a `torch.nn.Module` whose `forward()` method runs the forward kinematicd and returns two things: the center positions $\{\mathbf{r}_j\}$ and the per-unit curvatures $\{\kappa_j\}$. The forward pass does, in order:

1. *Angle recursion* — propagates $\phi_j$ from unit to unit using the law-of-cosines compatibility relation (each pair of neighboring units forms a constrained quadrilateral).
2. *Effective rotation* — converts $\phi_j$ and $\alpha_j$ to $\phi^*_j$, the rotation that unit $j$ contributes to the mechanism's global orientation.
3. *Global orientations* — accumulates $\phi^*_j$ to get the absolute direction of each member.
4. *Positions* — integrates those directions scaled by $l$ to get $\{\mathbf{r}_j\}$.
5. *Curvature* — evaluates the closed-form expression $\kappa_j = 2(2\alpha_j-1) / [4\alpha_j l (1-\alpha_j) \sin(\phi_j/2)]$ at each unit.

**Design variables.** All four are `nn.Parameter`s living in unconstrained space:

| Variable | Parameterization | Range |
|---|---|---|
| $\alpha_j$ (`alpha_logits`) | sigmoid | $(0.3,\; 0.7)$ |
| $l$ (`log_l`) | exp | $(0, \infty)$ |
| $\Psi^*$ (`psi_star`) | direct (radians) | — |
| $\beta_0$ (`init_rot`) | direct (radians) | — |

The sigmoid bound $(0.3, 0.7)$ keeps units away from the degenerate limits while covering both positive and negative curvature regimes ($\alpha > 0.5$ and $\alpha < 0.5$ respectively).

**Loss.** Four terms, equal weights in the default setup:

- *Curvature mismatch* $\sum_j (\kappa_j - \kappa^t_j)^2$ — the core signal.
- *Tip loss* $\|\mathbf{r}_N - \mathbf{p}_N\|^2$ — pins the endpoint.
- *Rotation loss* $(\beta_0 - \theta_{\mathrm{target}})^2$ — aligns the initial tangent.
- *Length constraint* $(L - L^t)^2$ — prevents the optimizer from trivially scaling the mechanism to zero.

**Optimization.** Adam, lr = 0.1, 1000 iterations. Prints a loss breakdown table every 200 steps. Converges fast for the star target.

### Output

Three plots: deployed mechanism overlaid on the target curve, curvature profiles compared side by side, and the optimized $\alpha^*_j$ sequence with shading indicating sign of curvature. You also get printed values of $\Psi^*$, $\beta_0$, and $l$.

### Changing the target

Replace `StarTargetCurve` with any class exposing `target_coords` ($(N+1)\times 2$ array), `target_curvatures` (length-$N$ tensor), `s_total` (float), and `initial_tangent_angle` (float). For analytically defined curves $y = f(x)$, compute $\kappa^t_j = y'' / (1 + y'^2)^{3/2}$ directly. For point-set inputs, fit a spline first.

---

## `WritingTask.ipynb`

**What it does:** Given a target planar curve, find $\{\alpha_j\}$ and scale $l$ so the **distal tip** traces that curve as $\Psi$ is swept from $\Psi_{\max}$ down to $\Psi_{\min}$.

This is the dynamic task (§4.2): the mechanism is no longer evaluated at one angle  every point on the target curve must be hit by the tip at some moment during the actuation sweep.

### How the pipeline works

**Target processing.** Two functions are provided. `get_sine_target_curvature` generates an analytical sine curve. `get_real_target_curvature` takes the hardcoded pixel coordinates of the character `j` (from Fig. 4.3E of the thesis), normalizes them to a unit box, arc-length parameterizes, and uses a `UnivariateSpline` to extract a smoothed curvature profile $\kappa^t(s)$ on 100 uniformly spaced arc-length samples. The smoothing is important: raw finite-difference curvature from sparse discrete coordinates is noisy and makes the loss misleading.

**Forward model.** `StaticDesignOptimizer` runs the full $K$-step actuation sweep in a **single vectorized forward pass** — the entire $K \times N$ grid of angles is solved simultaneously, no Python loop over $\Psi$. This is necessary for reasonable speed at 24000 training epochs. The pass computes:

1. *Internal angles* $\phi_j(\Psi_k)$ — the compatibility system is solved over the full $(K \times N)$ grid using batched law-of-cosines. The first unit's angle is pinned to $\Psi_k$ exactly.
2. *Effective rotations* $\phi^*_j(\Psi_k)$ via the same half-angle relation as the shape-morphing notebook.
3. *Cumulative orientations* via `torch.cumsum` over $\phi^*$.
4. *Tip position* $\mathbf{r}_{\mathrm{tip}}(\Psi_k)$ — reduced to a dot product over the $N$ units, returning a $K \times 2$ trajectory tensor.

**Arc-length alignment.** This is the non-obvious step. The mechanism trajectory is parameterized by $\Psi$, but the target is parameterized by arc length $s$. These don't coincide, and the mapping between them shifts as $\{\alpha_j\}$ changes during optimization. `resample_to_target` performs differentiable linear interpolation (via `torch.searchsorted`) of the mechanism's curvature onto the fixed target arc-length grid $\{s^t_k\}$. The gradients pass cleanly through this interpolation back to $\alpha_j$ and $l$.

**Loss.** The composite objective has four terms:

- *Curvature mismatch* $\|\tilde{\boldsymbol{\kappa}}_{\mathrm{tip}} - \boldsymbol{\kappa}^t\|^2$ — the resampled mechanism curvature vs. the target.
- *Smoothness* $\lambda_{\mathrm{sm}}\|\nabla\boldsymbol{\alpha}\|_2^2$ — penalizes large $\alpha$ jumps between adjacent units. Weight 0.001 in the default setup.
- *Arc-length consistency* $100 \cdot (L - L^t)^2 / L^{t2}$ — prevents scale collapse.
- *Steric penalty* $10^5 \cdot \sum_{k,j} \max(0,\, \phi_{\min} - \phi_j(\Psi_k))^2$ — ReLU-based soft constraint. Fires if any internal angle drops below 10° at any actuation step, preventing kinematic lock-out.

**Optimization.** Adam with separate learning rates: `alpha_params` at lr = 0.005, `log_l` at lr = 0.01. `ReduceLROnPlateau` halves the learning rate when loss stagnates for 500 steps. Gradient clipping at norm 1.0 is applied before each step  necessary because the Jacobian entries for proximal units are systematically large (the exponential sensitivity result from §3.6 of the thesis), which can produce explosive gradients early in training.

### Output

Two plots: tip trajectory overlaid on the target curve, and curvature profile comparison.

### A word on local minima

The trajectory optimization landscape is highly non-convex. A single run will often converge to a geometrically wrong solution that has a plausible loss value. The thesis runs 15 independent random restarts per candidate $N$ and picks the best. For the `j` character, `N = 30` with 24000 epochs per run is a reasonable starting point; for cleaner targets like the sine curve, 8000 epochs at `N = 30` is sufficient. If your result looks like a garbled curve, re-run with a different random seed  it is rarely a bug.

---

## Why autograd instead of finite differences

The sensitivity analysis (§3.6) shows that tip displacement variance decays approximately exponentially from base to tip: a unit-level perturbation near the base produces orders of magnitude more tip displacement than the same perturbation near the tip. Finite-difference gradient estimates along the proximal direction are therefore numerically unreliable precisely where the design leverage is largest. Implementing forward kinematics in PyTorch gives exact gradients from a single backward pass at no additional computational cost relative to the forward evaluation.

---

## Dependencies

```
torch    numpy    scipy    matplotlib
```

Both notebooks detect CUDA and fall back to CPU automatically.
