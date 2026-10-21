# Scalar resources

[Documentation index](../README.md) · [Branching](../solver/branching.md)

Scalar capacity is a Rust const generic: `Graph<const D: usize>`,
`Solver<D>`, and the labeling and enumeration engines carry the same dimension.
The default is 2. Any `usize` count can be instantiated explicitly; there is no
runtime dimension dispatch or heap-backed scalar fallback. Each scalar state,
vertex window, arc consumption vector, and hot dominance cache uses a fixed array.

```rust
use baldes::{Graph, Controls, price};

let mut graph = Graph::<4>::new(3, 4)?;
for vertex in 0..3 {
    for resource in 0..4 {
        graph.set_bounds(vertex, resource, 0., 10.)?;
    }
}
let first = graph.add_arc(0, 1, -2.)?;
let second = graph.add_arc(1, 2, -3.)?;
graph.arc_mut(first)?.consumption = [1., 2., 1., 3.].into();
graph.arc_mut(second)?.consumption = [2., 1., 4., 2.].into();
let result = price(&graph, &Controls::default())?;
assert_eq!(result.columns[0].resources, vec![3., 3., 5., 5.]);
# Ok::<(), baldes::core::error::OrError>(())
```

The constructor's resource count is the active count, at most `D`. This keeps
zero- and one-resource models usable with capacity 2. `add_resource(kind)`
initializes another coordinate within the compiled capacity and rejects overflow.
Type annotations can use the default (`let graph: Graph = Graph::new(3, 2)?;`);
standalone constructors should select it explicitly with `Graph::<2>::new`.
`ScalarValues<T, D>` and `ResourceValues<D>` support indexing, slices, and array
conversion with `.into()`. Exported routes contain only the active coordinates.

## Python and default Rust builds

With `make config`, edit **Scalar resource dimensions**; the saved key is
`CONFIG_RESOURCE_DIMENSIONS` (default `2`).

These APIs use one capacity fixed when the library is compiled. Set
`BALDES_RESOURCE_DIMENSIONS` to change it; leaving it unset selects 2:

```sh
BALDES_RESOURCE_DIMENSIONS=5 cargo build --release --features clp
BALDES_RESOURCE_DIMENSIONS=5 maturin develop --features clp
```

Use `baldes::DEFAULT_RESOURCE_DIMENSIONS` in Rust or
`pybaldes.resource_dimensions` in Python to inspect that capacity. Creating or registering more resources returns
an error. Rebuilding is required to change the capacity; there is no finite menu
of runtime specializations. Explicit Rust types such as `Graph::<9>` work
independently of this setting.

## Resource semantics

Every active coordinate participates in extension, dominance, forward/backward
joins, route reconstruction, enumeration, branching, resource knapsack charges,
conservative path domains, and exact incumbent checks.

A disposable resource accumulates nonnegative arc consumption and may wait to
reach a lower bound. An equality resource accumulates signed consumption without
waiting. Arc windows constrain the head state after extension. Initial values
default to source lower bounds. Backward equality search requires fixed sink
windows; `price` selects forward search when those are not fixed.

Bucket indexing uses at most two selected axes, independently of scalar capacity.
A resource above index one can be the main axis through `Graph::main_resource`.
Positive `EngineControls::steps` or `Solver::bucket_steps` select additional
disposable axes; at most one joins the main axis. Unselected resources still
participate in feasibility and dominance. More active resources can retain more
nondominated labels and increase time and memory. Existing resource-kind,
NG, budget, and certificate-eligibility restrictions still apply.
