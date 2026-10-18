# Dual-price stabilization

[Documentation index](README.md) · [Tree lifecycle](tree.md)

Column generation can repeatedly produce similar master duals and columns.
BALDES optionally smooths the dual point sent to pricing, using a stored
center and bounded retries when smoothed pricing finds no fresh columns.
It is enabled by default in `column_generation::Options`.

The implementation is
[`DualPriceSmoothing`](../src/master/stabilization.rs), integrated by the
[column-generation loop](../src/master/column_generation.rs). The sections below
describe the implemented update and retry rules.

## Smoothing weight

For raw master duals `raw`, stored `center`, and an outward point `outward`:

```text
pricing[i] = (1 - effective_alpha) * outward[i]
           + effective_alpha * center[i]
```

By default `outward = raw`. **Larger alpha means more weight on the center.**
At alpha zero pricing uses the outward point; at alpha one it uses the center.
Reducing alpha moves pricing back toward the current master duals.

After interpolation, a positive dual on a `LessEqual` row or a negative dual
on a `GreaterEqual` row is set to zero when its magnitude exceeds the
configured tolerance. Equality rows have no such sign restriction.

## Parameters

These are Rust `master::stabilization::Parameters`, configured through
`RootOptions::cg.smoothing` or directly in `column_generation::Options`.
They are separate from the named BALDES pricing options.

| Parameter | Default | Behavior |
| --- | --- | --- |
| `alpha_factor` | `1.0` | Below 1: use that base alpha and the linear retry rule; otherwise adapt alpha dynamically |
| `initial_alpha` | `0.5` | Initial base alpha for the dynamic mode |
| `beta_factor` | `0.0` | Zero: use raw outward point; positive: enable the directional construction below |
| `max_updates` | `10` | Maximum smoothed retry updates before returning to raw duals |
| `min_gap` | `0.0` | Minimum accepted gap, using the comparison tolerance |
| `max_gap` | `0.0` | Maximum accepted gap; nonpositive disables the upper gate |
| `minimum_stage` | `0` | Minimum stage accepted by `after_master` |
| `tolerance` | `1e-6` | General comparison and sign-correction tolerance |

The current column-generation caller passes stage `0` to `after_master` and
passes the actual pricing phase in feedback. Raising `minimum_stage` above
zero therefore disables activation in that caller. Its gap argument is the
master LP value minus the current finite bound, or zero when no finite bound
is available.

## Lifecycle

| Method | Effect |
| --- | --- |
| `restart_schedule(phase)` | Reset base/effective alpha, retry count, active flag, and center phase; retain the stored center/bound vectors |
| `after_master(...)` | Apply a pending center, store raw duals/senses, reset retries, and activate smoothing if the stage/gap/alpha gates allow it |
| `after_pricing(feedback)` | Consider a center update, adjust base alpha, and decide whether pricing should retry |
| `after_iteration()` | Clear the moved-center marker and return active pricing to raw duals |
| `remap_rows(map)` | Compact dual, center, sense, and pending-center vectors after row removal |

The first master iteration uses raw duals. Cut/enumeration transitions restart
the schedule; row cleanup applies the row map. Removing columns alone does
not change row coordinates and does not require a center remap.

## Pricing retries and alpha updates

Feedback includes the fresh negative-column count, a subgradient vector,
Lagrangian value, and pricing phase. In the column-generation driver, the
subgradient starts at the row RHS and subtracts multiplicity times the best
acceptable column's coefficients when its reduced-cost magnitude passes the
numeric test.

On the first pricing pass in dynamic mode (`alpha_factor >= 1`):

```text
if no fresh negative columns or angle > 1e-12:
    alpha -= 0.1
else if angle < -1e-12 and alpha is below the near-one gate:
    alpha += 0.1 * (1 - alpha)
```

The angle threshold is **1e-12**, independently of `Parameters::tolerance`.
The near-one gate includes the general tolerance and a relative margin.
Updating base alpha affects subsequent master iterations; retries update the
current effective alpha separately.

If fresh negative columns exist, or smoothing is inactive, `after_pricing`
returns false and stops retrying. Otherwise it increments `misprices` and uses:

```text
dynamic mode: effective_alpha = 2 * effective_alpha - 1
fixed-base mode: effective_alpha = 1 - misprices * (1 - alpha)
```

It republishes a smoothed point only while the retry count is at most
`max_updates` and effective alpha exceeds the tolerance. When either gate
fails, it deactivates smoothing and requests a final retry at raw duals.
“Fixed” describes the base alpha, not an unchanging effective retry weight.

## Center updates

A nonfinite Lagrangian value is ignored. If the current center phase is less
than or equal to the feedback phase, a finite center bound is replaced only
when the new value improves it beyond tolerance. A lower-numbered feedback
phase can reset the center phase and accept a different bound without that
same improvement requirement.

The accepted pricing point and feedback subgradient become a **pending**
center, installed by the next `after_master` call. Consequently the stored
center bound is not unconditionally monotone across phase changes.

The angle helper computes:

```text
toward = pricing - center
angle = dot(toward, feedback_subgradient)
      / (norm(toward) * norm(feedback_subgradient))
```

The division is unguarded. A zero denominator may yield `NaN`; comparisons
with it then fail. This behavior differs from forcing an angle of zero.

## Optional directional construction

With positive `beta_factor`, let `d = raw - center`, `l = norm(d)`, and `g`
be the stored center subgradient. A negligible `l` returns raw duals.
For `beta_factor == 1`, beta is the cosine between `d` and `g`; a negligible
subgradient also returns raw duals. Otherwise the configured beta is used.
Beta is clamped to `[0, 1]` before constructing:

```text
outward = center + (1 - beta) * d - beta * l * g / norm(g)
```

A negligible subgradient contributes zero to its directional term. When the
constructed step has norm above `1e-12`, it is rescaled to length `l`. This
changes the outward point before the alpha interpolation.

## Configure or disable

A complete Rust example, requiring `clp`:

```rust
use baldes::{master::root::RootOptions, vrptw::{Instance, Rounding}};

fn main() -> baldes::Result<()> {
    let mut instance = Instance::read("instances/R101.txt")?;
    instance.truncate(10)?;
    let mut options = RootOptions::default();
    options.cg.smoothing.initial_alpha = 0.5;
    options.cg.smoothing.beta_factor = 0.0;
    options.cg.smoothing.max_updates = 10;
    let result = instance.solve_root(8, Rounding::None, options)?;
    println!("{:?}: root LP={}", result.master.status, result.master.lp);
    Ok(())
}
```

Set `options.cg.stabilize = false` to disable it in Rust. The CLI exposes:

```sh
cargo run --release --features clp --bin baldes -- \
  vrptw instances/R101.txt --customers 10 --no-stabilization
```

Python exposes the enable/disable switch through the public root API:

```python
import pybaldes

result = pybaldes.solve_root(
    "instances/R101.txt", customers=10, stabilize=False,
)
print(result.status, result.lp)
```

The Python root API does not expose the individual smoothing parameters.
