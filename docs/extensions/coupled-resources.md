# Coupled resources

Coupled resources run in the existing physical-graph labeling framework. Install
selected/binary value domains, simultaneous pickup/delivery, cumulative costs,
bounded affine or piecewise-linear dependencies, base-state maps, and
state-dependent costs, and free-entry interval-cost fronts with `Graph::set_coupled_resources`,
`Solver::set_coupled_resources`, or Python `ResourceModel.install`.
No graph expansion or separate pricing front is used by this path.
Paired pickup-before-delivery constraints are outside the requested scope.

## Resource definitions

Paired resources use `CoupledResource::PairedFront { mode }` with an explicit
prepared front program. Cost-bearing paired resources accept finite nonnegative
objective coefficients through `set_coupled_objective_coefficients`; the default
is one. Their objective is the selected joined dependent value, without an
initial-minimum subtraction. Phase I suppresses the charge and retains the
resource transitions and feasibility checks.

An original `accumulated_resource` variable can bind a cost-bearing paired
resource with a nonnegative cost. Its selected value is reported independently
of that cost. Paired activity rows, integer activity registrations, negative
objective coefficients and explicit source-control choices are unsupported.
Custom/cumulative registrations retain their own row and branch contracts.

Routes retain `coupled_values`, pairs of resource index and selected dependent
value. These values participate in identity and survive snapshots and reporting.
Pool and incumbent validation check them against the retained front program.
Different forward/backward splits can produce different values on the same
physical path; a stored selected value must match a successful split join.
A path-only certificate encloses the successful split values rather than
claiming an optimum over continuous source controls.

For graphs containing only custom/cumulative coupled descriptors, forward
dominance also uses a finite-horizon signed proof when the physical graph is
acyclic, or an explicit walk cap or complete elementarity proves a finite
continuation horizon. Bounds and dominance share that prepared horizon.
Consistently signed weights determine the state order, even with signed
increments or negative prefix costs. Directed magnitude bounds must prove every
future state, charge and cost sum finite. Mixed-sign weights require equal
states unless an integer-lattice envelope proves every intermediate operation
exact; then a bounded continuation charge may consume the numeric prefix gap.
Cycles without a proved horizon, weighted activity objectives and uncertain overflow retain the
conservative comparison. This proof does not spend a physical-cost gap.

For signed custom/cumulative programs on DAGs, pricing supplies the current
vertex to the same comparison. Prepared descendant summaries exclude arcs that
can only occur before that vertex, preserving both signs of genuinely reachable
future weights. Rust callers can use `dominance_penalty_at_vertex` for this
contract; the context-free `dominance_penalty` keeps its global envelope.
Unsupported cyclic or heterogeneous programs retain the global proof.

Backward comparisons with vertex context use the existing propagated
source-to-vertex numeric entry envelopes in their outward suffix correction.
This can prove fractional comparisons when the global entry box is too broad;
it does not assume that signed prefixes are nonnegative or suppress rounding
allowances. Route-cap updates refresh these propagated domains together with
the horizon for both original and auxiliary pricing kernels.

Backward custom/cumulative comparisons also tighten suffix corrections on
proved integral programs. Reachable entry-state envelopes are intersected with
the integer lattice, and both correction dot products and their difference must
fit the exact binary64 integer range. Those comparisons need no floating-point
rounding allowance. Fractional coefficients, unknown domains and larger
intermediate sums retain the outward correction bound.

All per-arc vectors use the original graph's **arc vector order**, including
arcs into the sink. They do not use physical arc IDs as array indices.
Multiple resources compose in declaration order.

| Resource | Transition |
| --- | --- |
| `Charging` | Retain continuous earliest-departure time over remaining energy, consume travel energy, enforce service windows, and optimize partial charging with station-specific PWL curves. |
| `PairedFront` | Prepare and execute an explicit paired base/dependent front program with directional extension, waiting, bounds and split joins. |
| `IntervalCost` | Choose one shared initial value in the initial interval; on every arc add its increment, enforce state/charge domains, and charge the PWL function. Minimize the full route sum over that choice. |
| `SlopeIntervalCost` | Retain prepared slope segments through translation, restriction and addition, with an explicit front-operation policy. |
| `Selection` | Intersect the remaining value domain with arc and vertex masks; reject an empty domain. Up to 64 values are supported. Optional `vertices` gives one mask per vertex. Forward extension checks the head; backward extension checks the tail. With `relaxed`, an unremembered vertex resets the domain to `initial` after checking feasibility, forgetting earlier arc and vertex eliminations. By default (`BALDESselectionTighteningSchedule` 1) the routing master follows the reference schedule: per resource it takes the minimal violating interval ending at an improper path's first violation, remembers those no longer than `BALDESmaxNGaverNeighbourhoodSize` (or else the path's shortest one), and skips tightening while `BALDESmaxNGneighbourhoodSize` is negative. Schedule 0 remembers all minimal violating intervals. Remembered intervals include both endpoints. This also applies with omitted vertex masks. Incumbents always use the exact rule. |
| `SimultaneousPickupDelivery` | Deliver before picking up at each stop, with all route deliveries loaded at departure. Enforce capacity at departure and between all stops, admitting loads up to `capacity + tolerance` (default 1e-6, also used by load dominance; 0 is exact). |
| `CumulativeCost` | Update `x += increment[arc]`, then add `x * weight[arc]` to the objective. Optional `sink_upper` and `source_upper` are the resource's upper bounds at the sink and source. With nonnegative data, dominance reads `sink_upper` as a bound on the weight of any completion (a forward label with a larger `x` pays `(x_a - x_b) * sink_upper`) and `source_upper` as a bound on the increments of any prefix (backward labels). |
| `CustomResource` | Zero-initialized cumulative state and cost: `x += increment[arc]`, then charge `x * weight[arc]`. Optional `forward_bound` and `backward_bound` provide independent dominance constants. No vertex transition or hard state bounds. |
| `Dependent` | Update `y += offset[arc] + factor[arc] * parent_after`; reject values outside the declared closed bounds. The parent must be an earlier numeric resource. |
| `PiecewiseLinearDependent` | Update `y += function[arc](parent_after)` within the function's finite argument domain, then enforce the dependent state bounds. |
| `PiecewiseLinearResource` | Replace `x` with `function[arc](x_before)`, enforcing the input domain and output window. |
| `PiecewiseLinearCost` | Add `function[arc](parent_after)` to the objective without changing the parent. The function domain remains a constraint in phase I. |

For simultaneous pickup/delivery the state is total deliveries `D`, net pickups
minus deliveries `N`, and the greatest prefix value `M` of `N`, including zero.
After a service `(d,p)`, update `D += d`, `N += p-d`, `M = max(M,N)` and require
`D+M <= capacity`. This is sufficient for the completed route: its departure
load is `D`, and its maximum load is `D+M`. Increasing `D` at a later stop also
checks the load required at earlier stops. Two independent delivery/pickup
totals would miss that constraint. Deliveries and pickups must be nonnegative.

Cumulative weights, increments and affine dependency coefficients may be signed
but must be finite. Arithmetic overflow is reported as an error. Dependencies use
the parent's value after its transition on the same arc, and do not wait or
clamp to bounds. These are explicit modeling contracts; arbitrary
external dependency-type conventions are not asserted to have identical semantics.

`CustomResource` exposes the fixed cumulative-style custom family separately from
`CumulativeCost`. It shares the numeric kernel, joins, dependency handling and
directed path certification, but always starts its state and objective at zero.
Signed finite increments and weights are allowed. `forward_bound` promises an
upper bound on the total weight of any completion; `backward_bound` promises an
upper bound on the increments of any prefix. These optional constants only help
dominance with nonnegative increments and weights. They do not impose hard
feasibility limits, and omitting them keeps conservative dominance. Both must be
finite and nonnegative when supplied.

Rust uses `CoupledResource::CustomResource { increments, weights, forward_bound,
backward_bound }`. Python uses
`model.add_custom_resource(increments, weights, forward_bound=None, backward_bound=None)`.
For example, increments `[2, 1]` and weights `[3, 4]` charge `2*3 + 3*4 = 18`.
The JSON descriptor retains its own kind:

```json
{"kind":"custom_resource","increments":[2,1],"weights":[3,4],"forward_bound":7,"backward_bound":3}
```

There is no `initial` parameter. This descriptor is a built-in numeric cost
resource; use `CoupledPlugin` for extension-defined feasibility callbacks.

Piecewise-linear functions have at least two knots `[argument,value]`, strictly
increasing finite arguments, and finite adjacent argument/value differences.
They are continuous and may increase, decrease or have flat segments. Exact
knots return the supplied value. Interpolation uses `(x-x0)/(x1-x0)`, then
`y0 + t*(y1-y0)`, clamped to the segment's endpoint value range to preserve
monotonicity despite rounding. Outside-domain arguments are infeasible; no
extrapolation, resource waiting or state clamping is introduced.

## Explicit slope encodings for numeric resources

The encoded numeric variants preserve the supplied function representation:
`EncodedPiecewiseLinearResource`, `EncodedPiecewiseLinearDependent`, and
`EncodedPiecewiseLinearCost`. Existing knot-based variants and their constructors
keep their current behavior. A slope function is evaluated with ordered segment
multiply/add operations, rather than converting it into argument/value knots.
Those two evaluation methods can differ for fractional arguments.

Prepared slope pairs have a distinct layout: the first pair is
`(value_at_lower, lower_argument)`; each remaining pair is
`(slope, segment_end)`. A single prepared pair is a point function. An affine
input pair instead means `(intercept, slope)` and must be prepared over a domain;
it is not a point or a knot pair.

```python
from pybaldes import EncodedFunction, ResourceModel

function = EncodedFunction.slope_segments(
    [(2.0, 0.0), (3.0, 5.0), (-1.0, 10.0)],
    policy="strict_domain",
)
assert function.value(2.0) == 8.0
assert function.value(7.0) == 15.0
assert function.value(11.0) is None
copy = EncodedFunction.from_dict(function.to_dict())

model = ResourceModel()
resource = model.add_encoded_piecewise_linear_resource(
    initial=0.0, lower=0.0, upper=100.0,
    functions=[function],  # One function per physical arc, in insertion order.
)
```

`strict_domain` rejects arguments outside the exact stored domain.
`endpoint_tolerance` explicitly selects the recovered endpoint comparison policy,
which may snap arguments just inside or outside the endpoints. It does not apply
that tolerance to other resources or interior knots. Both policies report
nonfinite/intermediate overflow as an error. An outside-domain result is `None`;
a genuine function value of `1e12` remains a value. Dictionary/JSON round trips
retain the encoding and policy and validate their contents.

Raw-domain image, preimage and directed-certificate proofs remain conservative.
Unknown bounds retain candidates; endpoint-tolerance preprocessing must not discard
accepted near-boundary states using the nominal endpoints. Phase I still enforces
function domains while suppressing objective contributions. Raw interval and
paired-front support is separate work and is not provided by the numeric variants.
The implementation and tests are written, with compile checks completed; runtime
validation of this new API is deferred at the user's request during concurrent work.

`SlopeIntervalCost` provides a separate free-entry interval resource using prepared
slope functions. The Python builder is
`add_slope_interval_cost(initial_lower, initial_upper, lower, upper, increments,
functions, policy="strict_domain")`. Functions must be slope-encoded; knot
conversion is rejected. The alternate operation policy `front_algebra`
requires functions with leaf policy `endpoint_tolerance`; `strict_domain` requires
matching strict leaves. This keeps function evaluation and front operation
choices explicit.

Each forward extension translates the coordinate, restricts the allowed domain,
adds the charge front and scans its segment endpoints for a minimum. Backward
labels retain ordered suffix replay for the final joined objective. Unproved
operations produce an explicit incomplete result. Forward dominance accepts
bitwise-equal fronts, ordered by the independent scalar prefix costs. Under
`strict_domain` it otherwise accepts only identically zero fronts when every
charge is zero or costs are suppressed in phase I; such a front dominates when
each of the other's components lies in one of its components. Unequal costly
strict fronts never compare, because translation and addition re-round segment
spans and can reverse a current pointwise order. Under `front_algebra` a
covering front may exceed the other by delta, with values within the reference
tolerance counted as equal, and pays `delta` plus the difference of current
minima; this follows the reference's tolerant front comparison and is a parity
rule rather than a binary64 proof. Backward labels compare only with identical
suffix programs. Completion bounds, fixing and certificates
remain conservative. The result is a
minimum of the computed front algebra, not a promise of a global minimum over
every sequential binary64 evaluation of the source program. Compile checks pass;
new runtime regressions remain deferred.

## Free-entry interval costs

`IntervalCost { initial, bounds, increments, functions }` retains a continuous
piecewise-linear cost function in each physical label. Unlike a deterministic
base state, its initial value is chosen over the whole route. Minimizing each arc
or each half independently is invalid: charges `x` and `-x` on `[0,4]` jointly
cost zero, although their independent minima sum to -4.

Forward extension shifts the cost frontier by the arc increment and adds the
charge at the arrival state. Backward extension pulls the suffix frontier and
charge back through that increment. Joins intersect domains and minimize the sum
at the common state, correcting the independently minimized prefix/suffix costs.
Signed increments, signed/nonmonotone continuous charges, singleton feasible
intersections, and multiple independent interval resources are supported. Initial
and global windows must be finite, nested closed intervals. Every arc has one
increment and one charge function. Charge domains remain constraints in phase I.

These fronts participate in pricing, original-cost replay, cuts, fixing,
enumeration, retained repricing and node forks. `CoupledLabelState::interval_cost_fronts`
exposes the current knots in declaration order. A frontier cannot be a numeric
dependency parent because it has no single arrival value. Reversal reuse remains
disabled for this family. Production dominance normally requires equal cost
fronts. Domain inclusion is sufficient for the restricted all-zero front with
zero future charges; frozen-front comparisons additionally require independent
scalar and resource accounting. A current normalized front difference alone is
not a certificate for future rounded additions. Completion caches do not
approximate missing frontier correlations.

Floating-point operations can report arithmetic failure. A translation that merges
distinct frontier knots is rejected rather than silently dropping a cheaper
initial choice. Directed witness replay supports conservative objective intervals
for exact integral state translations, argument knots and window endpoints within
its bounded exact domain; other cases return unknown. These intervals may be
loose and do not establish an original-domain pricing certificate.

This is an explicit additive-transition interval-front contract. Use the paired
relation below for function-valued transitions with waiting. Raw external
descriptor import and tolerance parity are separate work.

```python
from pybaldes import ResourceModel

model = ResourceModel()
model.add_interval_cost(
    initial_lower=0.0, initial_upper=4.0, lower=0.0, upper=4.0,
    increments=[0.0, 0.0],
    functions=[[(0.0, 0.0), (4.0, 4.0)],
               [(0.0, 0.0), (4.0, -4.0)]],
)
# Install on a two-arc physical graph: the joint resource charge is zero.
```

## Paired front programs and waiting

Declare `CoupledResource::PairedFront { mode }`, then call
`Graph::prepare_paired_front_program` with one `MachineArcFunctionInput` per
physical arc, a `MachineRootInput`, and source/terminal base windows. The
preparation builds both directional operands. All per-arc inputs use insertion
order. Replacing resource definitions or topology requires reinstallation.

The program contains:

| Field | Meaning |
| --- | --- |
| `source_bounds`, `terminal_bounds` | Finite ordered base-coordinate windows |
| `root` | Initial dependent function |
| `arcs[].tail_bounds` | Base window at the arc's tail |
| `arcs[].waiting_bounds` | Waiting window at its head |
| `arcs[].waiting_rate` | Finite coefficient converted toward zero to a signed 32-bit runtime rate |
| `arcs[].base` | Prepared base-consumption function, not a total-arrival function |
| `arcs[].dependent` | Dependent increment function |
| `arcs[].dependent_lower`, `arcs[].dependent_upper` | Dependent bound functions |

`base` uses `{"pairs": [[value, coordinate], [slope, end], ...]}`.
The first pair anchors a value; subsequent pairs give segment slopes and ending
coordinates. `root`, `dependent` and both bound functions use either
`{"encoding": "affine", "intercept": value, "slope": slope}` or
`{"encoding": "vector", "function": {"pairs": [...]}}`.
These pairs are not `(argument, value)` knots. Preparation restricts functions
to their declared windows and returns an explicit unknown/error for unsupported
arithmetic or exhausted work.

The modes are `feasibility`, `cost_unbounded` and `cost_bounded`. Feasibility
mode contributes no dependent objective; both cost modes retain their literal
joined dependent cost. Forward and backward extension execute distinct front
recurrences. A joined value depends on the meeting split, including on some
waiting programs. It must not be substituted with the minimum of a separate
continuous-control problem. `replay` selects the split-zero join; forward
`replay_state` executes the all-forward recurrence. Phase I preserves the
program while suppressing its objective contribution.

Python declares and installs the same program:

```python
from pybaldes import ResourceModel

model = ResourceModel()
resource = model.add_paired_front(mode="cost_unbounded")
affine = lambda value: {"encoding": "affine", "intercept": value, "slope": 0.0}
arc = {
    "tail_bounds": [0.0, 8.0], "waiting_bounds": [0.0, 8.0],
    "waiting_rate": 0.0, "base": {"pairs": [[1.0, 0.0], [0.0, 8.0]]},
    "dependent": affine(1.0), "dependent_lower": affine(0.0),
    "dependent_upper": affine(10.0),
}
model.set_paired_front_program(resource, {
    "source_bounds": [0.0, 8.0], "terminal_bounds": [0.0, 8.0],
    "root": affine(0.0), "arcs": [arc, arc],
})
# Install on a physical graph with exactly two arcs.
# model.install(solver)
```

Network JSON declares `{"kind": "paired_front", "mode": "cost_unbounded"}`
in `coupled_resources` and supplies `paired_front_programs` as a list of the
same program dictionaries, each with an additional `resource` index. Python
`PricingNetwork` accepts the same field. See the complete
[paired example](../../examples/networks/paired_piecewise_linear.json), whose two
required routes have selected values 3 and 5 and total LP cost 8.

For precomputed operands, the lower-level `paired_machine_inputs`,
`paired_machine_roots` and `paired_machine_terminal_bounds` fields remain
available. A complete program cannot coexist with separate operands for the
same resource. `paired_waiting_base_resources` derives head waiting windows
from scalar vertex/arc bounds; use it only when that scalar coordinate represents
the paired base state. Disconnected scalar time windows are unsupported here.

Paired fronts automatically select the front-program domain during preparation.
They have no alternate source-control or normalization API. Raw interval fronts
retain their separate `front_program_domain` switch and policies.
A successful numerical join and an independent bound certificate remain
different contracts. Unknown arithmetic, stale preparation and exhausted work
must not be interpreted as infeasibility.

Raw paired-base domains can be processed explicitly before installing the
front program. Bind the scalar base with
`set_paired_waiting_windows_from_resource`, then call
`Graph::preprocess_paired_base_domains` with the raw arc functions. Its
`PairedBaseDomains` result exposes the derived vertex and arc windows and can
install them together with the front program through `install_front_program`.
Installation is atomic and rejects changed base inputs or topology.

This raw step currently requires a disposable scalar base whose additive
consumption matches the supplied arrival functions. It returns removed arc
indices as data; automatic installation declines a result requiring physical
arc exclusions. Existing prepared-input APIs keep their supplied geometry.
Do not preprocess already prepared windows again: the recovered backward pass
reads the original arc caps while staging its updates, so another pass can
produce a different domain.

A paired program can run alone or with standard graph resources within the
configured compile-time resource capacity. The reference binary has at most
two ordinary standard slots; that is a native comparison boundary rather than
a limit on BALDES scalar resource composition. A second
paired program and combinations with other coupled descriptors (including
`CumulativeCost`, `CustomResource`, selection, pickup/delivery, plugins and
other fronts) are rejected with an `unsupported paired resource combination`
error. Rejected installation leaves the previous program intact. These are
solver composition limitations; additional resource families are not installed
as extension-only substitutes.

Paired fixing retains its reduced bucket domain when enumeration times out.
Repricing that retained domain can omit a route accepted at another front split,
as in the reference implementation. Call `Solver::restore_arcs` to rebuild the
original pricing domain. An interrupted enumeration does not install a complete
retained route universe.

Paired owner objective coefficients accept every finite sign. Raw front pricing
keeps its unit charge and minimum-front selection; original route costs apply
the owner coefficient to the recorded selected activity. Activity rows likewise
use that selected activity without changing the raw front callback. When these
owner terms differ from raw pricing, raw completion alone cannot certify the
owner objective, and the solve may report `Stalled` with an unknown bound.

Integer owner registration is retained as metadata. It does not impose integer
values on each path's paired activity or provide a projected integer branching
rule. Owner registration bounds are separate from resource windows and shared
activity rows. Positive propagated lower bounds can fail the pricing callback
with an explicit unsupported-bound error.

Recovered degenerate backward compositions retain generated zero-width NaN
slopes internally; public input descriptors still require finite values.
Unresolved retained shapes report unknown. Captured operator and path-split
tests validate bounded cases, not general solver parity.

## Rust

```rust
use baldes::Graph;
use baldes::graph::coupled::CoupledResource;

let mut physical = Graph::<2>::new(4, 0)?;
for (tail, head) in [(0,1), (1,2), (2,3), (0,2), (2,1), (1,3)] {
    physical.add_arc(tail, head, -1.)?;
}
physical.elementary = true;
let deliveries = physical.arcs.iter()
    .map(|a| if a.head == 2 { 4. } else { 0. }).collect();
let pickups = physical.arcs.iter()
    .map(|a| if a.head == 1 { 4. } else { 0. }).collect();
physical.set_coupled_resources(&[
    CoupledResource::SimultaneousPickupDelivery {
        capacity: 4., deliveries, pickups,
        tolerance: baldes::graph::coupled::DEFAULT_LOAD_TOLERANCE,
    },
])?;
let result = baldes::price(&physical, &Default::default())?;
assert_eq!(result.columns[0].nodes, [0,2,1,3]);
```

For an existing stateful solver, use `solver.set_coupled_resources(&definitions)`.
It keeps physical IDs and options and invalidates labels, fixing and enumeration.
Reinstall after changing the physical arc vector order. Set up the complete graph
before installing resources. Resource validation also runs during preparation.

With `clp`, pass the attached physical graph to the ordinary
`RoutingContext::set_partitioning` or `master::model::Network` constructor.
Customer covering rows and cuts continue to refer to physical customers. No
expanded-node adapter, duplicated rows, or route projection is needed.
Coupled charges are included in priced and enumerated route objectives, master
column reconstruction, retained-universe repricing, fixed-route objective
validation and warm-start replay. Private phase I suppresses cumulative objective
charges while retaining their state transitions and feasibility constraints.

## Python

For Python installation, declaration order, method signatures, and complete examples, see [pyBaldes resource modeling](../pybaldes/resources.md).

```python
from pybaldes import ResourceModel, Solver

solver = Solver(4, 0)
for tail, head in [(0,1), (1,2), (2,3), (0,2), (2,1), (1,3)]:
    solver.new_arc(tail, head, -1.0)
model = ResourceModel()
model.add_simultaneous_pickup_delivery(
    capacity=4.0,
    deliveries=[0,4,0,4,0,0],
    pickups=[4,0,0,0,4,0],
)
model.install(solver)
minimum, routes = solver.price(50)
assert routes[0].nodes() == [0,2,1,3]
```

`add_selection`, `add_cumulative_cost`, `add_custom_resource`, `add_dependent`, and
`add_piecewise_linear_dependent` install the other
resource families through the same model. `Route.original_cost` reports the
original coupled objective; `Route.cost` reports its priced cost.

The shared network JSON interface, used by `baldes solve` and Python
`PricingNetwork`, accepts an optional `coupled_resources` list and optional
`elementary` flag. Each descriptor uses a `kind` tag:

```json
{
  "coupled_resources": [
    {"kind": "selection", "initial": 3, "allowed": [3,3,3,3,3,3]},
    {"kind": "simultaneous_pickup_delivery", "capacity": 4,
     "deliveries": [0,4,0,4,0,0], "pickups": [4,0,0,0,4,0]},
    {"kind": "cumulative_cost", "initial": 0,
     "increments": [1,1,1,1,1,1], "weights": [1,1,1,1,1,1]},
    {"kind": "dependent", "parent": 2, "initial": 0,
     "bounds": {"lower": 0, "upper": 100},
     "offsets": [0,0,0,0,0,0], "factors": [1,1,1,1,1,1]}
  ],
  "elementary": true
}
```

This is a fragment of a pricing network, not a complete model. Python accepts
these same descriptors in `PricingNetwork(coupled_resources=(...), ...)`.
The specialized CVRP/VRPTW text instance readers do not infer these quantities.

For example, the additional JSON descriptor below defines one physical arc's
piecewise function. Supply one function for every physical arc:

```json
{"kind": "piecewise_linear_dependent", "parent": 0, "initial": 0,
 "bounds": {"lower": 0, "upper": 100},
 "functions": [{"points": [[0,0], [2,1], [4,3]]}]}
```

The corresponding Python method accepts `functions=[[(0,0), (2,1), (4,3)]]`.
Parent indices always refer to the same resource declaration list. Base maps use
`kind: "piecewise_linear_resource"` with `initial`, `bounds`, and `functions`;
state costs use `kind: "piecewise_linear_cost"` with `parent` and `functions`.
Python provides `add_piecewise_linear_resource` and `add_piecewise_linear_cost`.
Base maps can express state-dependent consumption, waiting, saturation and resets
explicitly by their knots; the engine adds no implicit waiting or reset rule.

## Label operations and current limits

`Charging` uses `ChargingResourceArc` entries in physical arc-vector order.
Each entry supplies travel time/energy, a service-start window, service duration,
and an optional cumulative energy-to-time curve. Curves strictly increase from
`[0,0]` through the battery capacity. The initial energy and departure time are
explicit. Stations also constrain charging completion by closing time. The full
continuous front survives; backward joins replay their retained suffix on the
actual forward state. Front work is bounded per extension and exhaustion returns
an error. Charging adds no cost, is not a numeric dependency parent, uses
conservative equality dominance, and disables reversal reuse. Directed path
certification declines this family, including in phase I. See the
[EVRP adapter](evrp.md#physical-graph-backend) for end-to-end pricing and
schedule reconstruction.

Selection labels carry the remaining domains. Simultaneous load labels carry
composable delivery/net-load/peak summaries. Forward numeric labels carry actual
values. Backward numeric labels carry affine entry conditions and a cumulative
cost function; a join evaluates them at the forward label's actual values.
Kernels containing piecewise dependencies instead retain shared continuation
programs for numeric constraints. A join replays that suffix from its actual
forward entry values. Composition therefore adds a shared link rather than
multiplying piecewise regions, and flat/nonmonotone functions need no inverse.
Cumulative cost remains affine in its independent numeric coordinates; these
kernels cache diagonal cumulative-cost maps in linear per-arc space.
State-dependent nonlinear charges are kept out of the backward baseline and
replayed at the actual forward entry state during joins. Each suffix carries an
exact-sum interval, an absolute charge bound and a count; these provide an outward
roundoff allowance for the ordered charge sum without rescanning the whole
suffix to build a bound. Unknown/overflowing envelopes disable cost pruning.
Affine-only kernels
retain the dense precomputed-map fast path. Shared programs have iterative
release to avoid recursive destruction of long paths.
Dependency constraints are therefore not tested at an invented backward arrival
value. All original scalar coordinates and cut/branch states still participate.

Numeric transitions are prepared once per physical arc when resources are
installed. Forward programs retain declaration order with resolved parent slots;
backward programs cache the triangular affine map, offsets, entry conditions and
cost coefficients. Label extension composes the cached map with its suffix.
This trades per-arc storage (quadratic in numeric resource count) for avoiding
repeated matrix construction. Phase I rebuilds the cached cost terms with zero
weights while preserving the resource transitions and constraints. A symbolic
map overflow is recorded and reported if backward extension uses that arc;
it does not prevent finite forward replay.

Dependency preprocessing first checks each arc's declared entry/exit windows.
It then propagates source-reachable numeric intervals forward and sink-compatible
intervals backward through the full graph. Rounded-addition preimages are
expanded before subtraction. Direct base-map preimages use bounded searches over
ordered binary64 arguments on each segment. They preserve flat portions,
nonmonotone branches and rounding plateaus without requiring an algebraic inverse;
segments already entirely inside the target range bypass the searches. Piecewise argument domains participate in both
passes. Arcs with empty necessary domains are disabled without renumbering the
physical graph. The work limit is at most 64 paired passes; each descending
iterate is conservative, so stopping early only weakens pruning. Unknown
arithmetic retains candidates. `kernel.preprocessing()` reports pass count,
convergence and usable-arc counts; `route_arc_possible` reports the necessary
complete-route filter for an arc. `replay`/`replay_state` still evaluate standalone
partial resource paths without the complete-route filter.

Resource dominance now uses directional orders. A forward simultaneous state
`(D,N,M)` can dominate another when all three coordinates are no larger; backward
feasibility only needs `D` and `M`, so backward net load may differ. Selected
value domains use superset dominance. Forward dependency dominance propagates
possible comparison directions through the earlier-parent chain. A smaller
state is allowed when the lower window is protected by nonnegative increments;
a larger state is allowed when nonpositive increments protect the upper window.
Negative factors reverse the parent comparison. Competing effects retain both
possible directions and must satisfy both protections, otherwise dominance is
rejected. Preparation encloses every intermediate operation over declared and
prepared numeric domains, and unequal comparisons require finite arithmetic.
Both entry states must lie in those domains. Equality remains available when
these sufficient conditions cannot establish an order; backward conditions keep
the existing identical-expression/window-implication rule. Piecewise slope
signs participate in forward order propagation when the prepared parent domain
is contained in every arc function's domain. Backward nonlinear comparisons
require identical continuation programs before applying cost compensation. A
flat function still constrains its parent through the argument domain; output
equality does not erase that constraint. Unbounded cumulative parent comparisons
also retain directional guards against overflow unique to the dominator.
Direct base maps retain equality dominance until a whole-map feasibility order
is established. Parents of arbitrary nonlinear cost functions retain equality
(including differences induced by ancestors), so cost-free numeric comparisons
cannot discard a better future objective.

Charged-resource dominance separately orders physical scalar reduced costs and
accumulated numeric costs before considering the mixed label key. Equal numeric
inputs retain the repaired baseline rule. Unequal inputs can use a prepared
zero-compensation proof for cumulative-cost models, optionally combined with
selection and simultaneous loads: nonnegative increments and weights permit
`0 <= a <= b`; nonpositive increments and weights permit `b <= a <= 0`.
Both cones produce ordered nonnegative future charges, and the dominator's
accumulated numeric charge must also be nonnegative. Zero-weight coordinates
retain directional overflow guards. Signed cases outside those cones retain
conservative equality conditions. Dependency/domain checks remain mandatory.
Composite numeric/dependency models keep the baseline identity rule because the
broader checks did not improve the measured workload. No positive current-key
margin or cost compensation is spent by charged forward dominance.

Backward dominance requires identical continuation programs, compatible entry
windows and numeric state, with its outward coefficient-difference bound where
applicable. Unknown arithmetic retains both labels; an overflow during actual
extension is reported rather than silently proving route infeasibility.

Completion pruning now includes an outward lower bound on the cumulative join
correction. The engine builds direction-wide hulls of actual retained numeric
states and suffix cost coefficients after labeling. Individual bucket scans
use narrower per-bucket hulls, while inherited completion minima keep the
broader hull that includes predecessor candidates. Per-bucket compatibility
checks also use selected-domain unions, simultaneous-load envelopes and
supported backward affine entry conditions. Skipping one bucket's scan never
skips its predecessor traversal. Each completion query fixes
its own state and bounds the opposite states. This works with signed coefficients
without assuming a finite walk length. The two dot products in the implemented
join are enclosed separately, preserving their floating-point evaluation scope.
Missing hulls or arithmetic overflow leave the bound unknown and retain the
candidate. These bounds are used by pricing, completion, fixing and directional
enumeration; they are not scalar-only original-domain certificates.

Hulls are rebuilt after fresh/continued labeling and discarded on new pricing,
continuation or node forks. If elementarity or physical topology changes after
resource installation, reinstall the definitions so prepared dominance domains
cannot become stale. Replacing the resource preparation also invalidates labels
at unchanged scalar arc costs.

Forward-prefix reversal reuse is available for symmetric selection masks and
simultaneous loads with symmetric physical-arc quantities, integral deliveries
and pickups, and capacity at most 2^52. Uniform-sign net increments retain the
original compact reuse rule. Mixed signs additionally track an internal suffix
peak `R = max suffix net load`, updated by `R = max(0, R + pickup - delivery)`.
Reversing the arc sequence swaps prefix and suffix peaks; public
`load_summaries()` remains `(D,N,M)`.

Mixed-sign reuse builds a union of the two feasible prefix orientations. Each
load resource retains a prefix while `D+M <= capacity` or `D+R <= capacity`.
With several resources this may retain extra prefixes whose feasible orientations
differ; every complete join still checks all resources in the actual physical
route orientation. Dominance preserves both roles through ordered `D,N,M,R`.
Standalone extension/replay and ordinary backward labels remain strictly feasible.
Single-orientation completion hulls, completion-existence fixing and arbitrary
ordinary completion callbacks cannot certify this union, so those pruning
operations conservatively decline in the mixed reuse path.

The scalar resource/depot symmetry checks and conservative reduced-cost potential
checks must also pass. Numeric and PWL resources, fractional loads, asymmetric
resource events and presence rows retain ordinary directional labeling.
Phase-I cost suppression does not make a numeric resource reversible. Pricing,
fixing and enumeration use the same eligibility rules. Arithmetic failures
invalidate completion instead of proving that no route exists.

The physical graph retains its ordinary 1024-vertex limit. There is no extra
1024-state expansion limit on this production path. Labels remain subject to
normal label, time and path limits. A cyclic graph without a justified route
length/resource bound can still exhaust those limits.

Integer witnesses use outward interval replay for coupled feasibility and
objective contributions. A straddling boundary or unsupported arithmetic remains
unknown. Encoded knot/slope numeric programs include independently enclosed
endpoint-snap branches. Plugins can opt into `CoupledPlugin::certify`, which
must independently prove original-domain feasibility under the caller's work
and deadline; its default returns unknown. Ordinary numerical plugin transitions
do not supply proof. Charging supplies independently checked explicit targets
and relaxed necessary-condition propagation; failure to find a target remains
unknown rather than infeasible.

For numeric and interval costs, a polynomial directed walk relaxation propagates
state/objective enclosures over the original physical graph up to the integer
route contract's vertex-simple hop limit. It can retain a cost certificate when
literal path enumeration exceeds its limit. Unsupported objective operators,
overflow or exhausted proof work still return unknown. Original-domain scalar
relaxation collectors and additive-cost
certificates do not silently claim support for coupled resources. General
external dependency-type semantics and exact native dominance/bound policies
are not claimed to match the reference binary.

## Validation status

The earlier state-expansion compiler and its public adapters have been removed.
The direct kernel is the only coupled-resource implementation.

The direct-resource integration cases cover both directions, all split joins, standalone replay,
base maps, signed state-dependent objectives, function domains, complete-route
reduction, fixing/enumeration, node forks and root phase-I cost restoration.
Independent physical-load/path and numerical-interval oracles supplement those
integration cases. Python validation must use a wheel built from the same resource API revision.
Operator captures and integration checks cover specific retained programs;
they do not close every paired-resource boundary listed above.
Earlier expansion smoke results do not validate these kernels, and the current
regressions do not claim exact native descriptor, tolerance or performance parity.

## Raw paired base-domain preparation

Raw graph registration and prepared machine installation are separate stages.
`Graph::preprocess_paired_base_domains` takes raw arc functions and returns a
`PairedBaseDomains` artifact with derived vertex domains, arc domains and removed
physical arc indices. It requires the existing explicit scalar base binding;
matching an independent scalar resource by function shape is insufficient.
The bound coordinate must be disposable and its consumption must match the
raw arrival mapping. Other mappings return an explicit unsupported outcome.

For a graph with one scalar base resource, the complete installation pipeline is:

```rust
use baldes::{Graph, graph::Window};
use baldes::graph::coupled::{CoupledResource, MachineArcFunctionInput,
    PairedResourceMode, PreparedSlopeFunction};
use baldes::graph::coupled::descriptor::{FrontierBudget, FrontierOutcome,
    MachineRootInput, RawAffine};

let mut graph = Graph::<2>::new(3, 1)?;
let bounds = Window::new(0., 4.)?;
for vertex in 0..3 { graph.set_bounds(vertex, 0, 0., 4.)?; }
for (tail, head) in [(0, 1), (1, 2)] {
    let arc = graph.add_arc(tail, head, -1.)?;
    graph.arc_mut(arc)?.consumption[0] = 1.;
}
graph.set_coupled_resources(&[CoupledResource::PairedFront {
    mode: PairedResourceMode::CostUnbounded,
}])?;
graph.set_paired_waiting_windows_from_resource(0, 0)?;
let base = PreparedSlopeFunction::from_prepared_pairs(vec![(1., 0.), (0., 4.)]).unwrap();
let affine = |value| MachineRootInput::Affine(RawAffine::new(value, 0.).unwrap());
let inputs: Vec<_> = (0..2).map(|physical_arc| MachineArcFunctionInput {
    physical_arc, tail_bounds: bounds, waiting_bounds: bounds, waiting_rate: 0,
    base: &base, dependent: affine(0.25),
    dependent_lower: affine(0.), dependent_upper: affine(8.),
}).collect();
let domains = match graph.preprocess_paired_base_domains(
    0, &inputs, FrontierBudget::default(),
) {
    FrontierOutcome::Complete(domains) => domains,
    other => panic!("raw base preprocessing unavailable: {other:?}"),
};
assert_eq!(domains.vertex_bounds()[1], Window::new(1., 3.)?);
assert!(matches!(domains.install_front_program(
    &mut graph, &inputs, affine(0.), FrontierBudget::default(),
), FrontierOutcome::Complete(())));
```

Installation is atomic and rejects changed raw inputs, topology or bound scalar
state. Objective associations, owner registration bounds and compatible retained
waiting rates survive installation. Removed arcs are reported by the artifact;
its installer currently declines them until explicit topology exclusions can be
installed. Work or deadline limits return unknown without publishing partial
state.

Already prepared models continue to use `prepare_paired_front_program` or the
lower-level prepared setters directly. They are not automatically reprocessed:
the staged backward rule reads original arc upper bounds, so preprocessing its
own output again can produce a different domain. An explicitly bound prepared
base keeps its supplied source upper in bucket indexing instead of widening it
to the terminal upper. This is numerical domain preparation, not an original
objective certificate or a continuous source-control optimization.
