# Unified solve interface

Use `baldes solve INPUT [OPTIONS]` for routing instances and JSON pricing models.
The CLI has one entry point and one parser for time, iteration and node limits,
stabilization, bound mode, logging, pricing options and JSON output. Routing and
network adapters handle their model-specific controls.

```sh
baldes solve instances/Solomon/R101.txt --customers 5 --time 10
baldes solve examples/networks/triangle.json --tree --json -
```

A pricing JSON input can be a network object, an array of networks, or a model
containing `rows` and `networks`. Within a model, `networks` accepts either one
object or an array. Compilation always creates a collection. A single network
has the same solver path and defaults as an array containing that network.
Shared rows and original-variable declarations belong to the enclosing model.

Python exposes `pybaldes.solve` for instances, paths, complete network models,
scalar pricing networks and arrays of pricing networks:

```python
from pybaldes import PricingArc, PricingNetwork, solve

pricing = PricingNetwork(
    id=7, vertices=3, multiplicity=(0, 1),
    arcs=(PricingArc(0, 1, -2), PricingArc(1, 2, 0)),
)
single = solve(pricing, stabilize=False)
collection = solve([pricing], stabilize=False)
assert single.lp == collection.lp == -2
assert single.oracle_calls == collection.oracle_calls
```

For shared constraints, use `solve(pricings, rows=rows)` or construct a
`NetworkProblem`. Its `networks` constructor argument also accepts a scalar.
`Instance.solve()` and `NetworkProblem.solve()` call the same public operation.
Routing data automatically selects capacity-only or time-window constraints;
`problem_type` can explicitly override that inference.

## Rust model and tree entry points

Use `master::model` for the canonical network collection and `master::tree`
for integer search. This example needs the `clp` feature:

```rust
use baldes::{Graph, graph::Window, master::{model::{self, Network, Problem}, tree}};

fn main() -> baldes::Result<()> {
    let mut graph = Graph::<2>::new(3, 1)?;
    for vertex in 0..3 {
        graph.set_bounds(vertex, 0, 0., 10.)?;
    }
    graph.add_arc(0, 1, -2.)?;
    graph.add_arc(1, 2, 0.)?;
    let problem = Problem {
        rows: vec![],
        networks: vec![Network {
            id: 7, graph, multiplicity: Window::new(0., 1.)?,
            arc_entries: vec![vec![], vec![]], route_entries: vec![], activity_entries: vec![],
        }],
    };
    let root = model::solve(problem.clone(), model::Options::default())?;
    assert_eq!(root.master.lp, -2.);
    let integer = tree::solve(problem, tree::ModelTreeOptions::default())?;
    assert_eq!(integer.incumbent, -2.);
    Ok(())
}
```

`Problem::routing(graph, vehicles)` produces a `RoutingContext` with routing
registration metadata. `master::routing::solve` runs its root relaxation;
`tree::solve(context, TreeOptions::default())` uses the same tree driver.
See [network models](../solver/models.md), [routing tree](../solver/tree.md)
and [architecture](architecture.md) for the registration-specific contracts.

This is a breaking modernization. The old routing/network command aliases,
separate Python solve functions, native solve exports, and Rust `master::root`
and `master::mp` paths are removed. Use `master::routing`, `master::model`,
and the common master construction and search modules.

Master construction, dynamic-row insertion, artificial columns, pricing, and
tree selection/evaluation dispatch are shared. Model hooks supply coefficients,
cut families, branch candidates, witnesses and certification. Different model
policies can still select different LP bases or search trajectories; scalar
versus array shape does not select a different policy.
