# Dual-price stabilization

[Documentation index](../README.md) · [Tree lifecycle](tree.md)

Column generation can repeatedly produce similar master duals and columns.
BALDES optionally smooths the dual point sent to pricing, using a stored
center and bounded retries when smoothed pricing finds no fresh columns.
It is enabled by default in `cg::Options`.

An explicit `cg::Options.proximal` configuration replaces smoothing with
box-step, three-piece or five-piece dual regularization on a private LP copy.
Its optional multi-point epigraph also stays private; ordinary master duals are
retried before accepting a no-column result. Auxiliary objectives never become
original bounds or incumbents. See [network proximal controls](../extensions/mp.md#proximal-pricing).
Automatic three/five-piece curvature is available through
`ProximalParameters::automatic_three_piece()` / `automatic_five_piece()` or
JSON/Python `proximal={"curvature": {}}`. The solve loop supplies initialization,
physical-row participation, raw-dual measurements, iteration updates, activity
retries and compatible warm/child inheritance. See
[automatic curvature controls](../extensions/mp.md#automatic-curvature).

The implementation is
[`DualPriceSmoothing`](../../src/master/stabilization/mod.rs), integrated by the
[column-generation loop](../../src/master/cg/driver.rs). The sections below
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
`RootOptions::cg.smoothing` or directly in `cg::Options`.
They are separate from the named BALDES pricing options.

| Parameter | Default | Behavior |
| --- | --- | --- |
| `alpha_factor` | `1.0` | Below 1: fixed base and linear retries; exactly 1: angle-adaptive base and doubled retries; above 1: initial base and doubled retries |
| `initial_alpha` | `0.5` | Initial base alpha for the dynamic mode |
| `beta_factor` | `0.0` | Zero: use raw outward point; positive: enable the directional construction below |
| `max_updates` | `10` | Maximum smoothed retry updates before returning to raw duals |
| `min_gap` | `0.0` | Minimum accepted gap, using the comparison tolerance |
| `max_gap` | `0.0` | Maximum accepted gap; nonpositive disables the upper gate |
| `minimum_stage` | `0` | Minimum stage accepted by `after_master` |
| `tolerance` | `1e-6` | General comparison and sign-correction tolerance |

The reference stores `alpha_factor`, `beta_factor`, `min_gap` and `max_gap`
as single-precision controls. The constructor rounds these fields through `f32`
before using them as `f64`. A factor that rounds to zero disables stabilization.

The column-generation caller passes the current pricing stage. Its gap compares
the inherited node lower bound with the best artificial-free restricted-LP
objective accepted before this solve, using the reference comparison tolerance.
It tracks these separately from the global integer incumbent. A new cut or
enumeration round carries the certified lower bound and resets the LP bound.
Unknown bounds use finite -1e12/+1e12 sentinels; a known lower bound with an
unknown LP bound therefore gives a gap slightly below one. Warm starts retain
the parent certificate.

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
returned acceptable column's coefficients. A near-zero or rounded-zero reduced
cost does not suppress this contribution.

On the first pricing pass in angle-adaptive mode (`alpha_factor == 1`):

```text
if no fresh negative columns or angle > 1e-12:
    alpha -= 0.1
else if angle < -1e-12 and alpha is below the near-one gate:
    alpha += 0.1 * (1 - alpha)
```

The angle threshold is **1e-12**, independently of `Parameters::tolerance`.
The near-one gate includes the general tolerance and a relative margin.
Updating base alpha affects subsequent master iterations; retries update the
current effective alpha separately. The angle is calculated only on that first
adaptive pricing pass and retained during subsequent mispricing retries.

If fresh negative columns exist, or smoothing is inactive, `after_pricing`
returns false and stops retrying. Otherwise it increments `misprices` and uses:

```text
dynamic mode: effective_alpha = 1 - 2 * (1 - effective_alpha)
fixed-base mode: effective_alpha = 1 - misprices * (1 - alpha)
```

It republishes a smoothed point only while the retry count is at most
`max_updates` and effective alpha exceeds the absolute-plus-relative zero gate
(`tolerance + abs(effective_alpha) * 1e-10`). When either gate
fails, it deactivates smoothing and requests a final retry at raw duals.
“Fixed” describes the base alpha, not an unchanging effective retry weight.
The dynamic formula preserves the compiled reference's floating-point operation
order; rewriting it as `2 * effective_alpha - 1` can change the result.

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
be the stored normalized center subgradient. A negligible `l` returns raw duals.
For `beta_factor == 1`, beta is the cosine between `d` and `g`; a negligible
subgradient also returns raw duals. Otherwise the configured beta is used.
Beta is clamped to `[0, 1]` before constructing:

```text
outward = center + (1 - beta) * d + beta * l * g
```

A negligible subgradient contributes zero to its directional term. When the
constructed step has norm above `1e-12`, it is rescaled to length `l`. This
changes the outward point before the alpha interpolation. Mispricing retries use
the raw outward point. Restart invalidates the automatic direction until a subsequent accepted center
provides a new normalized subgradient. Retaining the old vector alone does not
make it valid. The 54-case parameter corpus covers this capacity-row transition.

## Configure or disable

A complete Rust example, requiring `clp`:

```rust
use baldes::{master::routing::RootOptions, inputs::{Instance, Rounding}};

fn main() -> baldes::Result<()> {
    let mut instance = Instance::read("instances/Solomon/R101.txt")?;
    instance.truncate(10)?;
    let mut options = RootOptions::default();
    options.cg.smoothing.initial_alpha = 0.5;
    options.cg.smoothing.beta_factor = 0.0;
    options.cg.smoothing.max_updates = 10;
    let result = instance.solve(8, Rounding::None, options)?;
    println!("{:?}: root LP={}", result.master.status, result.master.lp);
    Ok(())
}
```

Set `options.cg.stabilize = false` to disable it in Rust. The CLI exposes:

```sh
cargo run --release --features clp --bin baldes -- \
  vrptw instances/Solomon/R101.txt --customers 10 --no-stabilization
```

Python exposes the enable/disable switch through the public root API:

```python
import pybaldes

result = pybaldes.solve(
    "instances/Solomon/R101.txt", customers=10, stabilize=False,
)
print(result.status, result.lp)
```

The Python root API does not expose the individual smoothing parameters.

An unavailable best pricing solution differs from a zero subgradient. Feedback
carries that availability explicitly; the adaptive angle retains its preceding
value when a best solution is unavailable, matching the native early return.
Completed CG rounds use the LP objective accepted within convergence tolerance
for subsequent stabilization gap bookkeeping, while the published pricing bound
remains conservative.

When an unavailable pricing solution accompanies an accepted pending center,
the center uses raw LP duals. The saved normalized subgradient remains intact;
the fallback RHS vector does not replace the direction or make it valid.
Customer-only elementarity IDs and canonical retained-path export order
preserve ordering through directional enumeration.
