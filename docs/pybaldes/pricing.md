# Stateful pricing with Solver

[PyBaldes documentation](README.md) · [Quickstart](quickstart.md) · [Resources](resources.md)

`Solver` owns a directed graph and retained pricing state. Use it to obtain
resource-feasible paths, or as an oracle inside a master that you implement.
It does not create or optimize a shared master automatically.

## Construction order and IDs

1. Construct `Solver(n_vertices, n_resources)`. The active scalar count cannot
   exceed `pybaldes.resource_dimensions`.
2. Set source/sink and add directed arcs, retaining the IDs from `new_arc`.
3. Configure scalar kinds, initial values, vertex bounds, and arc consumption.
4. Configure NG neighborhoods or elementarity sets.
5. Install any coupled-resource model after the physical graph is complete.
6. Set pricing options, optionally prepare buckets, and price.

Vertex IDs are zero-based. Arc IDs follow insertion order. Per-arc vectors must
use that order, including arcs not used by a returned route. `new_resource(name,
kind)` configures the next pre-created scalar coordinate; it does not grow the
constructor's resource count. The name is an interface label, not a lookup key.

## Scalar resource semantics

| `kind` | Meaning |
| --- | --- |
| `0` | Disposable resource: nonnegative consumption; may wait to reach lower bounds |
| `1` | Equality resource: signed accumulation without waiting |

Initial values default to source lower bounds unless explicitly set. Lower and
upper bounds are inclusive. Equality here describes propagation, not a master
row's equality constraint. Backward equality pricing needs fixed sink windows;
otherwise the engine selects forward search. See [scalar resources](../reference/resources.md).

The [quickstart](quickstart.md#price-a-path-directly) builds a complete load example.
For a time resource, arc consumption typically includes travel plus tail service;
vertex bounds constrain service start. The low-level API does not infer these
quantities from coordinates.

## NG and elementarity

`set_ng_neighborhood(vertex, nodes)` requires distinct valid IDs and inclusion
of the vertex itself. Actual preparation is limited by the compiled
`pybaldes.ng_memory_capacity` (15 by default), including merged arc memory.
The binding's preliminary size check is not a promise that 64 entries fit.
NG relaxation permits some revisits; it is not full elementarity.

`add_elementarity_set(nodes)` returns a set ID;
`attach_elementarity_to_node(vertex, set_id)` associates an existing set with
a vertex. These are low-level elementarity controls. In declarative networks,
`PricingNetwork(elementary=True)` requests the network's elementary domain.

## Repeated pricing in column generation

Build the graph once and update reduced costs as master duals change:

```python
from pybaldes import Solver

solver = Solver(3, 0)
solver.set_source(0)
solver.set_sink(2)
solver.new_arc(0, 1, 2.0)
solver.new_arc(1, 2, 1.0)
solver.build_default_resource_buckets()

# Example reduced arc costs computed by your master integration.
solver.set_reduced_costs([-3.0, 1.0])
minimum, routes, exact = solver.price_with_status(30)
if not routes and not exact:
    minimum, routes, exact = solver.price_with_status(30, exact=True)
for route in routes:
    print(route.cost, route.original_cost, route.arcs())
```

`set_reduced_costs(costs)` requires one finite value per physical arc. It preserves
original costs and prepared graph state. Derive reduced costs using your own
row coefficients; this API cannot infer the rows in an external master.

`update_reduced_costs(costs, duals)` instead replaces original arc costs and
applies vertex dual prices. Supply one cost per arc and one dual per vertex.
Use it when original costs change or when the vertex-dual representation matches
your master. Do not feed already reduced costs into it and subtract duals twice.

## Pricing returns and exactness

| Call | Return | Contract |
| --- | --- | --- |
| `solve()` | `(cost, Route)` or `None` | Single-route pricing; `None` is not a master infeasibility proof |
| `price(maximum)` | `(minimum, routes)` | Staged pricing without an aggregate exactness flag |
| `price_with_status(maximum, exact=False)` | `(minimum, routes, exact)` | Staged pricing; can return heuristic improving columns |
| `price_with_status(maximum, exact=True)` | Same triple | Run unlimited-label pricing phase once; can include nonnegative columns |
| `enumerate(maximum)` | `(1e12, 0, routes)` | Current compatibility wrapper around phase-0 pricing; first two values are placeholders |

`maximum` must be a positive integer. Only use a minimum as a pricing bound when
the returned exactness flag establishes it for the configured domain. Unlimited
labels do not remove time, arithmetic, memory, or model-domain restrictions.
An NG-domain minimum is not the same as an elementary-path minimum.

The low-level `enumerate()` method does not itself transfer a complete universe
into a master or populate the retained enumeration store. Do not interpret its
placeholder values as a bound or route count. `enumerated_count` and
`get_enumerated(index)` inspect an existing retained universe; calling
`enumerate()` alone does not establish one. For reported route coverage and MIP
transfer use the [network workflow](networks.md#enumeration-and-integer-transfer).

## Route and Phase

A low-level `Route` owns its data and remains valid after the solver is released.

| Member | Meaning |
| --- | --- |
| `cost` | Priced route objective, including configured coupled charges |
| `original_cost` | Original objective before reduced-cost adjustments |
| `nb_nodes` | Number of entries in the node sequence |
| `nodes()` / `arcs()` | Copies of ordered vertex/physical-arc IDs |
| `resource_consumption(resource_id)` | Final scalar resource value |
| `is_certified` | Whether the producing pricing result was exact; not a directed arithmetic certificate |

`get_phase(index)` returns `Phase` with `check_ng`, `allow_jump_arcs`, and
`labels_per_bucket`. `phase_count` reports the configured schedule size.

## Configuration method reference

| Method | Purpose |
| --- | --- |
| `set_source(v)`, `set_sink(v)` | Select terminal vertices |
| `set_graph_id(id)` | Set diagnostic graph identity |
| `new_arc(tail, head, cost)` | Add physical arc; return its ID |
| `new_resource(name, kind)`, `set_resource_type(resource_id, kind)` | Configure pre-created scalar coordinates |
| `set_arc_consumption(arc_id, resource_id, value)` | Set one finite consumption |
| `set_vertex_bounds(vertex_id, resource_id, lower, upper)` | Set inclusive scalar bounds |
| `set_initial_resource(resource_id, value)` | Set finite source state |
| `set_vertex_time_windows(vertex_id, resource_id, windows)` | Optional disjoint windows; feature-gated |
| `set_ng_neighborhood(vertex_id, nodes)` | Replace a vertex's NG memory |
| `add_elementarity_set(nodes)`, `attach_elementarity_to_node(vertex_id, set_id)` | Configure elementarity membership |
| `set_pricing_controls(cutoff, max_labels, max_columns)` | Set cutoff and positive work/output limits |
| `set_option(name, value)`, `get_option(name)` | Set/read a numeric BALDES option |
| `option_names()`, `option_support(name)` | Static supported-name list and availability/scope explanation |
| `build_default_resource_buckets()` | Prepare bucket labeling using current graph/options |
| `set_bidirectional_mode(mode)` | 0 off, 1 normal, 2 halfway |

Choose options before preparing buckets. If topology, elementarity, or arc order
changes after coupled resources are installed, reinstall their definitions so
prepared domains match the graph. Prefer rebuilding the solver for topology
changes between iterations. Cost-only updates have a dedicated reuse interface.
