# Stabilized Column Generation

Dual-price smoothing (stabilization) is a technique for improving the convergence of column generation by interpolating between the current dual solution and a stabilized "center" point. Without stabilization, column generation often oscillates — generating columns that cycle in and out of the basis without making progress on the bound. Stabilization dampens these oscillations by constraining pricing to a trusted neighborhood.

This document describes the mathematical framework, the algorithmic choices, and the exact implementation in `src/master/stabilization.rs`.

---

## 1. Mathematical Background

### 1.1 Lagrangian Dual and Subgradient

Let the master LP be:

```
min cᵀx    subject to   Ax ≥ b,  x ≥ 0
```

with dual variables π (one per constraint). The Lagrangian dual value at dual point π is:

```
L(π) = πᵀb + multiplicity · min{ reduced cost over pricing universe }
```

The **subgradient** of the concave Lagrangian dual at π is:

```
g(π) = b − multiplicity · A·x*
```

where x* is the column (or combination) returned by the pricing oracle. The subgradient points in the direction that maximizes L(π).

### 1.2 Stabilization Concept

Instead of pricing at the raw dual point π_raw, stabilization prices at a **smoothed** dual point:

```
π_smooth(i) = (1 − α) · π_outward(i) + α · c_center(i)
```

where:
- `α ∈ [0, 1]` is the step-size parameter (small α → more smoothing, large α → closer to raw).
- `c_center` is the stabilized center point from previous iterations.
- `π_outward` is the "outward" point — either the raw duals or a directional projection (see §2.3).

The smoothed point π_smooth is used as input to the pricing oracle. After pricing, α may be adjusted based on whether new columns were found and the geometric relationship between the center, the pricing point, and the subgradient.

---

## 2. Algorithm

### 2.1 High-Level Flow

Stabilization is invoked inside `optimize()` in `column_generation.rs` around each pricing loop:

```
for each iteration:
    solve master LP → get raw duals π_raw

    if stabilize:
        smoothing.after_master(π_raw, senses, gap, stage, initial)

    converged = check_stopping_criterion()

    if not converged:
        current_duals = stabilize ? smoothing.pricing : π_raw

        while pricing_loop:
            answer = price(stage, current_duals)

            if stabilize:
                feedback = Feedback{
                    negative_columns: fresh_new_cols,
                    subgradient: g,
                    lagrangian: L(current_duals),
                    phase: stage
                }
                again = smoothing.after_pricing(feedback)
                if again:
                    current_duals = smoothing.pricing  // retry with smoothed
                else:
                    break

        if stabilize:
            smoothing.after_iteration()
```

### 2.2 Restart Schedule (`restart_schedule`)

Called at the start of a cut round or after enumeration resets the master:

```rust
self.alpha = (alpha_factor < 1.) ? alpha_factor : initial_alpha;
self.effective_alpha = self.alpha;
self.misprices = 0;
self.active = false;
self.center_phase = phase;
```

- `alpha_factor` (< 1): forces a fixed α for the entire run.
- `alpha_factor` ≥ 1: uses `initial_alpha` and allows dynamic adjustment.
- Resets misprice counter and deactivates smoothing until gap is acceptable.

### 2.3 Publish Step (`publish`) — The Core Smoothing Equation

This is the heart of stabilization. Given the current α, it computes the smoothed duals:

```rust
// outward point: directional_point() if beta_factor > 0, else raw.clone()
outward = self.directional_point()   // or self.raw

self.pricing[i] = (1 - alpha) * outward[i] + alpha * center.get(i).unwrap_or(0.)

// Zero out infeasible duals (sign violation)
for each i:
    if sense[i] == LessEqual and pricing[i] > tolerance:
        pricing[i] = 0.
    if sense[i] == GreaterEqual and pricing[i] < -tolerance:
        pricing[i] = 0.
```

**Key properties:**
- When α = 0: pricing = outward (no center influence).
- When α = 1: pricing = center (full smoothing, ignores current duals).
- The sign-correction step ensures smoothed duals respect constraint direction.

### 2.4 Directional Point (`directional_point`) — β-Projection

When `beta_factor > 0`, the outward point is projected onto the subgradient direction for faster convergence:

```rust
diff[i] = raw[i] - center[i]
length  = ||diff||

if beta_factor == 1.:
    // Cosine projection of diff onto normalized subgradient
    sub_norm = ||g||
    beta     = dot(diff, g) / (length * sub_norm)   // cosine similarity
else:
    beta = beta_factor   // user-specified constant

beta = clamp(beta, 0, 1)

out[i] = center[i] + (1 - beta) * diff[i] - beta * length * g[i]/||g||

// Normalization step: scale to match the outward direction's length
step_norm = ||out - center||
if step_norm > tolerance:
    out[i] = center[i] + (out[i] - center[i]) * (length / step_norm)
```

This is a **projected gradient step**: it moves from the center toward the raw duals, but projects out the component along the subgradient. The normalization ensures the outward point has the same length as the raw diff vector.

When `beta_factor == 1`, β becomes the cosine of the angle between the raw-diff direction and the subgradient — this is the classic convex combination used in RSDA (Relativized Subgradient Damped Algorithm).

### 2.5 α-Update Rule (`after_pricing`)

After each pricing call, α is adjusted:

**Case A: Dynamic α (`alpha_factor >= 1`)**

```rust
if misprices == 0:
    // Good iteration — no negative reduced cost columns found
    if (negative_columns == 0 or angle > tolerance):
        alpha -= 0.1                          // increase smoothing
    else if (angle < -tolerance and alpha < 0.999):
        alpha += 0.1 * (1 - alpha)            // decrease smoothing, asymptotic
```

**Case B: Fixed α schedule (`alpha_factor < 1`)**

```rust
misprices += 1
effective_alpha = 1 - misprices * (1 - alpha)
```

This linearly decreases effective α as more iterations fail to find columns, eventually reaching tolerance.

**Both cases — when new columns exist:**

```rust
if active and misprices > 0:
    if alpha_factor >= 1:
        effective_alpha = 2 * effective_alpha - 1   // exponential increase
    else:
        effective_alpha = 1 - misprices * (1 - alpha)

    if misprices <= max_updates and effective_alpha > tolerance:
        publish(effective_alpha)
        return true  // retry pricing
    else:
        active = false
```

The exponential update `2α − 1` doubles the step-size each time, quickly returning to the raw dual point when progress is being made.

### 2.6 Center Update (`consider_center`)

The center is updated only when the Lagrangian bound improves:

```rust
if not lagrangian.is_finite():
    return

if center_phase <= phase:
    if center_bound is finite and lagrangian <= center_bound + tolerance:
        return  // no improvement

// Accept new center
center_bound = lagrangian
c = pricing.clone()   // the smoothed dual point used for pricing
g = subgradient       // the subgradient from this iteration
next_center = (c, g)
```

The center is only advanced when the Lagrangian value at the current pricing point exceeds the previous center bound — ensuring monotone improvement.

### 2.7 Gap Acceptance (`gap_accepts`)

Stabilization activates only when the optimality gap is within bounds:

```rust
fn gap_accepts(gap):
    return (max_gap <= 0 or gap <= max_gap) and (min_gap <= gap or min_gap == 0)
```

This prevents stabilization from interfering during early iterations where the gap is still large.

---

## 3. Parameters Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha_factor` | 1.0 | If < 1, fixed α for entire run. If ≥ 1, dynamic adjustment with `initial_alpha`. |
| `initial_alpha` | 0.5 | Starting α when `alpha_factor >= 1`. |
| `beta_factor` | 0.0 | Directional projection strength: 0 = no projection (use raw), 1 = cosine projection, other = constant β. |
| `max_updates` | 10 | Maximum number of misprice updates before deactivating stabilization. |
| `min_gap` | 0.0 | Minimum gap required before stabilization activates. |
| `max_gap` | 0.0 | Maximum gap; 0 means no upper limit. |
| `minimum_stage` | 0 | Minimum phase/stage before stabilization is allowed. |
| `tolerance` | 1e-6 | Numerical tolerance for comparisons and sign corrections. |

---

## 4. Data Structures

### 4.1 `Parameters`

Holds the configuration parameters listed above.

### 4.2 `Feedback`

Returned by the pricing oracle after each call:

```rust
struct Feedback {
    negative_columns: usize,   // count of columns with negative reduced cost
    subgradient: Vec<f64>,     // g = b - multiplicity * A * x*
    lagrangian: f64,           // L(current_duals)
    phase: i32,                // current phase/stage index
}
```

### 4.3 `DualPriceSmoothing`

The main stateful struct:

| Field | Purpose |
|-------|---------|
| `alpha` | Current step-size parameter (before effective scaling) |
| `effective_alpha` | α actually used in `publish()` |
| `angle` | Cosine similarity between center-direction and subgradient |
| `center_bound` | Best Lagrangian value seen at the center |
| `center_phase` | Phase index of the current center |
| `misprices` | Counter for consecutive iterations without new columns |
| `active` | Whether stabilization is currently interpolating |
| `center_moved` | Whether a new center was accepted this iteration |
| `center` | The stabilized center point (Vec<f64>) |
| `pricing` | The smoothed duals returned for pricing |
| `raw` | The raw master duals from the last solve |
| `subgradient` | Normalized subgradient (when beta_factor > 0) |
| `senses` | Row constraint directions (LessEqual/GreaterEqual) |
| `next_center` | Pending center update, applied at next `after_master()` |

---

## 5. Lifecycle Methods

### 5.1 `restart_schedule(phase)`

Called when a cut round or enumeration resets the master. Resets all state except keeps track of the new phase.

### 5.2 `remap_rows(map)`

After rows (cuts) are added or removed, compacts all internal vectors according to the row mapping.

### 5.3 `after_master(raw, senses, gap, stage, initial)`

Called after the master LP is solved but before pricing:

1. If `next_center` exists, apply it as the new center (and normalize subgradient if β > 0).
2. Store raw duals and senses.
3. Initialize pricing = raw.clone().
4. If `initial == true`: set effective_alpha = 0 (no smoothing on first iteration).
5. Check gap acceptance, stage threshold, and α > tolerance.
6. Call `publish(effective_alpha)` and set `active = true`.

### 5.4 `after_pricing(feedback) -> bool`

Called after each pricing call:

1. Update center via `consider_center()`.
2. Compute angle between center-direction and subgradient.
3. Adjust α based on the update rules (§2.5).
4. If new columns exist (negative_columns > 0) or not active: return false.
5. Increment misprices, compute effective_alpha.
6. If within max_updates and effective_alpha > tolerance: call `publish()` and return true (retry pricing).
7. Otherwise deactivate and return true (one final iteration with raw duals).

### 5.5 `after_iteration()`

Called after each outer iteration (whether or not columns were added):

1. Reset `center_moved = false`.
2. If active: deactivate and set pricing = raw.clone().

---

## 6. Utility Functions

```rust
fn dot(a, b) → f64          // Dot product of two vectors
fn norm(v) → f64            // L2 norm (sqrt of max(0, dot(v,v)))
fn normalize(v)             // In-place L2 normalization
fn greater(value, bound, tol) → bool  // Toleranced comparison: value > bound + tol
```

### `cosine(center, priced, g)` — Angle Helper

Computes the cosine of the angle between the direction from center to the pricing point and the subgradient:

```rust
toward[i] = priced.get(i).unwrap_or(0.) - center[i]
result = dot(toward, g) / (norm(g) * norm(toward))
```

This angle is used in the α-update logic to decide whether to increase or decrease smoothing. A positive angle means the subgradient points roughly toward the pricing point (good direction); a negative angle suggests oscillation.

---

## 7. Integration with Column Generation

In `column_generation.rs::optimize()`, stabilization interacts with:

1. **Phase management**: Stabilization restarts at full α whenever phases advance or cuts are added.
2. **Penalty updates**: Stabilization does not interfere with artificial variable penalty scaling.
3. **Column cleanup**: When columns are pruned, `remap_rows()` keeps the center consistent.
4. **Enumeration**: After enumeration resets the master, `restart_schedule()` is called.
5. **Cut separation**: Each new cut round triggers `restart_schedule()`, giving stabilization a fresh start.

---

## 8. Convergence Properties

The stabilization scheme follows the classic convex optimization framework for Lagrangian dual maximization:

- **α schedule**: The step-size α controls the trade-off between convergence speed and stability. Large α allows aggressive moves toward the optimum; small α provides damping against oscillation.
- **Center tracking**: By maintaining a running center point, stabilization effectively performs a form of **discounted averaging** over dual points, which is known to converge faster than raw subgradient methods for ill-conditioned problems.
- **Directional projection (β > 0)**: When β = 1, the scheme becomes equivalent to RSDA (Relativized Subgradient Damped Algorithm), which has provable convergence guarantees even when exact optimality cannot be reached.

---

## 9. Command-Line Interface

```
--no-stabilization    Disable dual-price smoothing entirely
```

When disabled, pricing uses raw duals directly with no interpolation or center tracking — this is the classic (unstable) column generation algorithm.

---

## 10. Python Bindings

The stabilization parameters are exposed through PyO3 in `python/rust/src/root.rs`:

```python
baldes_root(
    ...,
    stabilize=True,   # Enable/disable stabilization
    ...
)
```

Additional parameters (alpha_factor, beta_factor, etc.) can be configured through the `smoothing` options when building the solver.
