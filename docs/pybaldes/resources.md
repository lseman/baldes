# Coupled resources and function encodings

[pyBaldes documentation](README.md) · [Pricing](pricing.md)

Scalar coordinates cover additive consumption and vertex bounds. `ResourceModel`
adds physical-graph resource families with selection domains, simultaneous
pickup/delivery, numeric dependencies, and state-dependent objective terms.
It attaches definitions to the same solver; it does not expand your vertex IDs.

## Declare and install

```python
from pybaldes import ResourceModel, Solver

solver = Solver(3, 0)
solver.new_arc(0, 1, -20.0)
solver.new_arc(1, 2, -20.0)
model = ResourceModel()
parent = model.add_custom_resource(
    increments=[2.0, 1.0], weights=[3.0, 4.0],
    forward_bound=7.0, backward_bound=3.0,
)
model.add_piecewise_linear_cost(
    parent, functions=[[(0.0, 0.0), (10.0, 10.0)]] * 2,
)
model.install(solver)
minimum, routes, exact = solver.price_with_status(10, exact=True)
assert exact and minimum == -17.0
assert routes[0].original_cost == -17.0
```

Each increment/weight/function vector has one entry per physical arc in insertion
order. A returned resource ID refers to this declaration list. Dependent
resources reference an earlier compatible numeric parent, not a scalar ID from
`Solver.new_resource`. Keep IDs returned by methods instead of hard-coding them.
Install after graph construction. Reinstall after topology/arc-order changes.
Installation invalidates prepared labels, fixing, and enumeration state.

In this example the custom state moves from 0 to 2 to 3. Its own objective charge
is `2*3 + 3*4 = 18`; the dependent piecewise charges add `2 + 3 = 5`.
Adding those to base arc cost -40 gives -17. Coupled charges appear in
`Route.original_cost` as well as its priced objective.

## ResourceModel methods

The [API reference](api.md#resourcemodel-signatures) lists exact signatures.

| Method | Inputs and meaning |
| --- | --- |
| `add_selection` | `initial`, per-arc `allowed` bitmasks, optional `vertices`, `relaxed=False`; intersect selected-value domains |
| `add_simultaneous_pickup_delivery` | `capacity`, per-arc `deliveries`, `pickups`, `tolerance=1e-6`; enforce physical route load |
| `add_cumulative_cost` | `initial`, per-arc `increments`, `weights`, optional `sink_upper` and `source_upper`; accumulate state-dependent cost |
| `add_custom_resource` | Per-arc `increments`, `weights`, optional `forward_bound`, `backward_bound`; zero-initial numeric state with directional cost bounds |
| `add_dependent` | Earlier `parent`, `initial`, `lower`, `upper`, per-arc `offsets` and `factors`; affine dependency |
| `add_piecewise_linear_resource` | `initial`, `lower`, `upper`, per-arc knot `functions`; direct state transitions |
| `add_piecewise_linear_cost` | Earlier `parent` and per-arc knot functions; objective charges evaluated on parent state |
| `add_piecewise_linear_dependent` | Earlier `parent`, own initial/bounds, per-arc knot functions; dependent numeric transitions |
| `add_encoded_piecewise_linear_resource` | Direct transitions using `EncodedFunction` objects |
| `add_encoded_piecewise_linear_cost` | Parent-state charges using encoded functions |
| `add_encoded_piecewise_linear_dependent` | Numeric dependency using encoded functions |
| `add_interval_cost` | Initial interval, global bounds, per-arc increments and knot functions; preserve a shared continuous initial choice |
| `add_slope_interval_cost` | Interval cost with explicit slope encodings and front policy |
| `add_paired_front` | Declare a paired front resource and its feasibility/cost mode |
| `set_paired_front_program` | Attach source/terminal windows, root and per-arc base/dependent functions |
| `install` | Validate and attach the declaration list to a `Solver` |

Selection masks encode allowed values as bits, rather than a list of scalar
values. Simultaneous pickup/delivery checks the combined physical load along the
route; independent total delivery and pickup bounds do not capture that contract.
Numeric transition equations, directional bounds, and feasibility tolerances are
specified in the [coupled-resource model guide](../extensions/coupled-resources.md).

## Knot and slope encodings

```python
from pybaldes import EncodedFunction

knots = EncodedFunction.knots([(0.0, 2.0), (4.0, 10.0)])
slope = EncodedFunction.slope_segments([(2.0, 0.0), (2.0, 4.0)])
affine = EncodedFunction.affine(2.0, 2.0, 0.0, 4.0)
assert knots.value(3.0) == slope.value(3.0) == affine.value(3.0) == 8.0
assert knots.value(5.0) is None
restored = EncodedFunction.from_dict(slope.to_dict())
assert restored.value(3.0) == 8.0
```

Knot pairs are `(argument, value)`. Slope encoding instead begins with
`(initial_value, initial_coordinate)`; each later pair is
`(segment_slope, ending_coordinate)`. They are different representations with
different evaluation order. Do not pass knot pairs to `slope_segments`.
`affine(intercept, slope, lower, upper)` prepares an anchored slope representation.

`.encoding` and `.policy` identify the stored encoding and evaluation policy.
`.value(argument)` returns `None` outside the accepted domain; invalid arithmetic
raises `ValueError`. `to_dict()`/`from_dict()` retain explicit wire encoding.
Use them when inserting encoded functions into network JSON descriptors.

Knot functions use strict closed domains. Slope functions accept evaluation
policy `"strict_domain"` (default) or `"endpoint_tolerance"`. Slope interval
resources separately choose front policy `"strict_domain"` or
`"front_algebra"`. Strict fronts require strict slope functions;
front-algebra fronts require endpoint-tolerance slope functions. Knot
functions cannot be substituted where lossless slope encodings are required.

Paired modes are `"feasibility"`, `"cost_bounded"`, and `"cost_unbounded"`.
Paired resources use an explicit front program rather than an `EncodedFunction`
resource declaration. Its base function uses prepared slope pairs, while root,
dependent and bound functions are explicit affine or vector dictionaries. See
[paired front programs](../extensions/coupled-resources.md#paired-front-programs-and-waiting)
for the schema and a complete builder example. Cost modes charge the selected
joined dependent value; no initial-minimum normalization is applied. Different
meeting splits can produce different values for a physical path.

## Disjoint time windows

Requires a build with `multi-time-windows`:

```python
from pybaldes import Solver

if hasattr(Solver, "set_vertex_time_windows"):
    solver = Solver(3, 1)
    for vertex in range(3):
        solver.set_vertex_bounds(vertex, 0, 0.0, 20.0)
    a = solver.new_arc(0, 1, -2.0)
    b = solver.new_arc(1, 2, -1.0)
    solver.set_arc_consumption(a, 0, 4.0)
    solver.set_arc_consumption(b, 0, 2.0)
    solver.set_vertex_time_windows(1, 0, [(0.0, 2.0), (8.0, 12.0)])
    minimum, routes, exact = solver.price_with_status(10, exact=True)
    assert routes[0].resource_consumption(0) == 10.0
```

Arrival at 4 waits until 8; the final state is 10. Ordinary vertex bounds
intersect the window union. Bounds `[3, 7]` at the middle vertex would make that
visit infeasible; the gap is not filled by the windows' convex hull. Supply
nonempty finite closed intervals on disposable resources.

For declarative networks use `PricingResource(time_windows=...)`, with one
entry per vertex; `None` uses ordinary bounds for that vertex. See
[multiple time windows](../extensions/multi-time-windows.md) for validation rules.

## Declarative network resources

`PricingNetwork(coupled_resources=(descriptor, ...))` accepts JSON-style mappings
with a `kind` tag. It is separate from `ResourceModel.install(solver)`, which
operates on a low-level solver. Per-arc vectors and parent declaration indices
retain the same ordering requirements. Use the descriptor schema in the
[coupled-resource guide](../extensions/coupled-resources.md#python), including
explicit bounds objects and function encodings. Routing text readers do not
infer coupled resource definitions.

Unsupported certificate operators or exhausted work can leave bounds unknown.
Resource replay, pricing exactness, integer witness validation, and independent
bound certificates are separate operations; inspect the corresponding result.
