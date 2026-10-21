# Multiple time windows

Build with `--features multi-time-windows` to attach a union of closed
service-start intervals to any disposable scalar resource. For example:

```rust
use baldes::{Graph, Window};
let mut graph = Graph::<2>::new(3, 1)?;
graph.set_time_windows(1, 0, &[
    Window::new(8., 12.)?,
    Window::new(16., 20.)?,
])?;
```

A forward arrival at 13 waits until 16. A backward latest arrival of 15 projects
to 12. Waiting is allowed between visits, including across service-window gaps;
these windows constrain service starts, not the entire travel or waiting period.
Service durations remain part of arc consumption, as in the single-window model.

`Graph::set_time_windows(vertex, resource, &[Window])` validates all intervals,
sorts them and merges touching or overlapping intervals. It replaces the ordinary
vertex bound with their hull. Empty unions, nonfinite/reversed intervals, invalid
IDs, and equality resources are rejected without changing the graph. A zero-width
interval is valid. `time_windows` returns the normalized configured union;
`clear_time_windows` removes it and retains the current ordinary bound.

Subsequent `set_bounds` calls restrict the retained union. Incoming arc bounds
also intersect that union, rather than filling its gaps. An explicitly specified
initial source value must belong to its domain. Without one, initialization uses
the earliest permitted source value. Backward initialization uses the latest
permitted sink value within the effective bucket bounds.

The feature does not add a new resource kind. Ordinary disposable dominance
remains valid because forward projection is monotone and waiting is permitted.
The existing scalar resource dimensions, NG/elementarity, arc-window constraints
and resource-branch windows continue to apply.

## Python and network JSON

Build Python with `maturin develop --features multi-time-windows` (or add that
feature to the existing wheel build). The stateful solver exposes:

```python
solver.set_vertex_time_windows(1, 0, [(8.0, 12.0), (16.0, 20.0)])
```

Network JSON, built with `clp,multi-time-windows`, accepts `time_windows` on a
resource: one entry per vertex, each either null (ordinary bounds) or a nonempty
list of [lower, upper] pairs. For a three-vertex network:

```json
{
  "kind": "disposable",
  "time_windows": [null, [[8, 12], [16, 20]], null],
  "bounds": [[0, 30], [0, 18], [0, 30]]
}
```

When both fields are supplied, `bounds` intersects the union; the second interval
in this example is effectively [16, 18]. Without the build feature the JSON field
is rejected as unknown, rather than silently ignored. Solomon and CVRPLIB file
formats retain their existing input semantics; use the graph API or network JSON
for disjoint windows.

## Implementation and validation

Forward transitions choose the first admissible value at or after arrival;
backward transitions search from the last interval and clip at the head before
subtracting consumption and projecting onto the tail domain. Both intersect
vertex and arc restrictions. These transitions are shared by pricing, replay,
canonical path collection and elementary enumeration. Exact incumbent validation
also applies union membership using its exact arithmetic.

Bucket propagation projects across gaps, and component/reachability construction
includes the image of a whole bucket cell. A discontinuous waiting jump can thus
reach a later interval without being pruned by an adjacency built for the hull.
If a far cell endpoint is infeasible, the image uses a conservative hull bound.
Backward bucket arcs retain both phase flags, since a gap can cross the
halfway boundary inside one input cell. Actual label resources select the phase.
This can create extra dependencies and increase preparation work; no performance
improvement is claimed. Consumption-symmetric prefix reuse is disabled on graphs
with configured unions, because hull symmetry alone does not justify time
reversal. Ordinary forward/backward labeling remains available.

Tests compare bucket pricing with independent exhaustive integer path schedules
on randomized DAGs and elementary cyclic graphs. Regressions cover earliest and
latest projection, gap rejection, arc intersections, terminal domains, effective
sink bounds, zero-width windows, nonindexed resource coordinates, normalization,
topology invalidation, halfway continuation, enumeration, JSON loading, and exact
incumbent witnesses.

Validation commands:

```sh
cargo test -p baldes --features clp,multi-time-windows,simd --lib --test multi_time_windows --test main_pricing --test fixing_enumeration --test accumulated_resource --test porting_regressions
cargo test -p baldes --lib --test main_pricing --test fixing_enumeration --test accumulated_resource --test porting_regressions
cargo clippy -p baldes --lib --test multi_time_windows --features clp,multi-time-windows,simd
```

The Python regression in `python/tests/test_multi_time_windows.py` passes against
the feature-enabled extension and skips on wheels without it. Test builds used
`CARGO_PROFILE_DEV_OPT_LEVEL=0`, `CARGO_PROFILE_DEV_DEBUG=0`, and
`CARGO_INCREMENTAL=0`. Modified Rust files pass rustfmt. The broader Clippy run
including all unit tests hits existing denied `approx_constant` and
`absurd_extreme_comparisons` lints in unrelated tests; targeted Clippy completes
with repository warnings.
