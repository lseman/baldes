# EVRP implementation status

[Documentation index](../README.md)

`baldes::extensions::evrp` is the initial model and independent schedule replay layer.
It supports deterministic directed travel with separate time, energy and cost,
delivery capacity, customer service-start windows, repeat charging-station
visits, and explicit partial charging with station-specific piecewise-linear
curves. Fixed-sequence charging optimization and reference bidirectional direct
pricing and a two-phase subpath prototype are available. Both feed the same
route-based LP master and column-generation adapter.

Charging curves map energy to cumulative time from empty. Charging from `a`
to `b` takes `T(b) - T(a)`; `charge_for` applies the inverse curve and caps at
full charge. Knots must start at zero and increase strictly in both coordinates.
Non-concave charging profiles are accepted. Energy and time units are supplied
consistently by callers.

`Problem::replay` takes an explicit closed route with an amount charged per
visit and an initial departure time. The vehicle starts full. It reports every
arrival, service start, departure and arrival/departure energy, plus total
travel cost and delivered load. Charging follows station service and must
finish by closing; customers only require service start within their window.
Waiting is allowed and consumes no energy. A route can cover a customer subset,
but cannot repeat customers or contain intermediate depots. Comparisons use
strict floating-point bounds with no implicit tolerance. Travel costs must be
nonnegative; pricing reduced costs are not inputs to this validator.

Replay checks feasibility of a supplied charging schedule, not optimality
of its charging decisions. Tariffs, charger congestion, regeneration, mixed
vehicle compatibility and station insertion are not implemented.

## Fixed stop sequence optimizer

`Problem::optimize_fixed_order(order, departure, ChargingOptions::default())`
chooses continuous partial charging amounts on a supplied sequence of customers
and stations. It minimizes completion time. Travel cost is fixed by that
sequence, and charging has no separate monetary cost in the current model.
Station visits may repeat; customer visits must be elementary.

The algorithm propagates piecewise-affine earliest departure time as a function
of exact remaining energy. At a station, for cumulative charging time `T` and
incoming time function `F`, it evaluates
`T(x) + min { F(q) - T(q) : q <= x }`. On each affine interval the minimum is
attained at an endpoint or at `q = x`. All these candidates are generated;
their lower envelope is split at domain endpoints and function intersections.
The construction permits non-concave charging curves without an energy grid.
Time-window waiting is another piecewise-affine operation. Every retained piece
keeps its predecessor and energy mapping for schedule reconstruction.

The result distinguishes `Optimal`, `Infeasible`, `WorkLimit`, and
`NumericalFailure`. `Optimal` is the optimizer's result in floating-point
arithmetic, not an interval-arithmetic certificate. Every returned schedule
passes independent strict replay. Reconstruction that fails replay reports
`NumericalFailure`, never `Infeasible`. The work budget counts generated pieces
and envelope comparisons; reaching it yields no feasibility or optimality claim.
The envelope implementation prioritizes a transparent reference algorithm over
large-instance speed.

Tests compare charging allocations against independent quarter-energy grid
enumeration on tiny heterogeneous/non-concave instances whose optimum lies on
that grid, and check waiting, repeated stations, exact windows and work limits.
These tests do not establish performance on published benchmarks.

## Direct bidirectional pricing

`Problem::price_direct(duals, fleet_dual, departure, DirectOptions::default())`
prices elementary customer routes with repeatable station visits. The dual
vector has one entry per location and must be zero at stations and depots.
Reduced cost is travel cost minus customer duals minus one fleet dual. The best
replay-validated route is returned even when its reduced cost is nonnegative.
No subset-row cuts, ng relaxation or branching restrictions are accepted yet.

The implementation uses forward earliest-departure functions `F(e)` and
backward latest-departure functions `B(e)` at exact departure energy `e`. Backward
functions exclude processing at the meeting node, already processed by the
forward label. Internally backward times are negated to share lower-envelope
code. Backward extension undoes charging, station closing, service, customer
windows and travel in the appropriate order. At a station the core operation
is `max { B(y) - T(y) : y >= q } + T(q) - service`, with station closing applied
to departure before charging is undone.

Dominance conservatively requires equal visited-customer sets, no greater
reduced cost, no more used arcs and pointwise time dominance over the entire
other function's domain. The function comparison covers domain gaps and
interior crossings; endpoint-only scalar tests are insufficient. Joins require
compatible customer sets, capacity, total arc count and some common energy
with `F(e) <= B(e)`. Both directions retain energy mappings for reconstruction;
every returned complete route passes strict schedule replay.

This is an E1-inspired reference implementation of the coherent function
extension/dominance/join approach described by
[Nafstad, Desaulniers and Stålhane (2025), Sections 3–4](https://publications.polymtl.ca/64567/).
The prefix/suffix affine-envelope construction is specific to Baldes's model;
the paper's faster PWL algorithm, midpoint-by-time stopping, unreachable-customer
sets and acceleration techniques are not reproduced here. The current objective
is travel cost, not the paper's separate route-duration extension.

Pricing statuses are explicit:

- `Bounded`: all customer-serving routes with at most `max_arcs` travel arcs
  were covered. This does not certify that longer routes have no negative cost.
- `Complete`: that bound covers every relevant route, using customer
  elementarity when there are no stations, or a conservative depot-horizon
  bound when all available non-self travel times are strictly positive.
- `WorkLimit` / `NumericalFailure`: no pricing completeness claim. Any returned
  incumbent remains independently replay-validated.

`Complete` describes exhaustive model coverage in f64 arithmetic, not a rigorous
real-number reduced-cost certificate. No station-repeat bound is inferred from
zero-time travel. Nonnegative-cost self loops are omitted because they cannot
improve a schedule or customer coverage. Work/label limits and arc limits are
configurable. Both directions currently grow to half the arc bound and compare
compatible label pairs; this prioritizes correctness evidence on tiny cases
over large-instance speed.

Tests compare minimum reduced costs to exhaustive stop-order enumeration with
the fixed-order optimizer, with dominance enabled and disabled. They also check
capacity, customer elementarity, zero-time travel, work limits, bounded versus
global coverage, and domain-interior dominance failures.

## Two-phase subpath prototype

`Problem::price_subpaths(duals, fleet_dual, departure, SubpathOptions::default())`
uses the same objective, dual convention, route arc limit and completeness
statuses as direct pricing. Phase one generates elementary customer orders
between depot/station endpoints with no interior charging. It retains all such
orders fitting a full battery and load capacity, including direct charger-to-
charger moves with no customers. `max_subpaths` caps this phase explicitly.

Phase two composes these subpaths into complete routes. It propagates the
entire continuous time/energy function through every constituent customer and
endpoint, optimizes charging, and checks customer elementarity and capacity.
It may prune a prefix only with equal customer history, no greater reduced cost,
no more arcs and full function-domain dominance. Subpaths are never supplied
as master columns. Both backends return the same complete-route representation
with a replay-validated charging schedule.

The decomposition is inspired by
[Jacquillat and Lo, v3 (2026)](https://arxiv.org/html/2407.02640v3).
The prototype deliberately retains distinct phase-one orders and carries full
charging functions in phase two. It does not implement the paper's compressed
resource vectors, two-sided subpath domination, adaptive ng relaxation or cuts.
Thus it does not assume that scalar forward domination can replace the paper's
domination requirements. Keeping the full prefix state makes composition an
explicit dynamic program under this static model; the paper's speedups have
not been reproduced.

Tests compare both backends with exhaustive route-order evaluation, under the
same data, duals and arc bounds, with dominance enabled and disabled. Fixtures
include heterogeneous/non-concave charging, repeated stations, and a route with
two consecutive charging stops in both directions. Global versus bounded
coverage and phase-one/phase-two work limits have separate checks.

## Native physical-graph backend

`Problem::price_native(duals, fleet_dual, departure, NativeOptions::default())`
uses the ordinary Baldes labeling engine. Select
`EvPricingBackend::Native(NativeOptions::default())` in the shared EV master to
use it in both feasibility and travel-cost phases. The reference direct and
subpath backends remain available; the default backend is unchanged.

`Problem::native_graph(...)` also exposes the compiled `Graph<3>` for a native
`pricing::Solver<3>`. The graph preserves location IDs and duplicates the depot
at vertex `locations.len()` as the sink. Its active scalar resources track arc
count and delivered load. Only customers receive elementarity sets, so station
visits may repeat. A native PWL boolean state records whether a customer has
been served; the sink requires that state to be one. Customer duals are charged
on incoming arcs and the fleet dual is charged once on source departure.
The delivery-only native simultaneous-load resource additionally guards load
feasibility and dominance without scalar epsilon comparisons.

The native `CoupledResource::Charging` retains earliest departure time as a
continuous function of exact remaining energy. Travel consumes energy; service
starts inside its window. Stations apply the full partial-charging envelope
`T(x) + min_{q<=x}(F(q)-T(q))` and require service/charging to finish by closing.
Heterogeneous and non-concave strictly increasing curves are supported without
an energy grid. Forward labels retain owned front pieces. Backward labels retain
the physical suffix; joins replay that suffix on the actual forward front.
Dominance conservatively requires equal charging fronts, and reversal reuse is
disabled. This resource adds no objective and cannot be a numeric parent.

After native pricing chooses a stop order, the existing fixed-order optimizer
reconstructs its charging amounts and independently replays the schedule. A
failed reconstruction reports numerical failure, never infeasibility or complete
pricing. `max_front_work` bounds each individual frontier extension, including
suffix replay; it is not a total pricing-work budget. `max_reconstruction_work`
separately bounds witness reconstruction. Label limits and `max_arcs` bound the
native search. Resource arithmetic/state failures remain unknown; the native
engine may aggregate a nonzero frontier-budget exhaustion into numerical failure.
The `work` statistic counts witness reconstruction work, not native front work.

This adapter supports at most fifteen customers by default, reflecting native
exact NG local memory (31 or 63 with the `ng-memory-31` or `ng-memory-63`
build feature), and at most 1023 input locations because it adds a sink.
It uses native numerical resource/cost comparisons. `Complete` and `Bounded`
retain their numerical coverage meanings; no directed-rounding certificate is
provided for charging. The largest absolute reduced arc cost times `max_arcs`
must be below `1e12 - 1`, because the native engine reserves `1e12` as a sentinel;
unsupported scales are rejected explicitly and require rescaling cost units.
Native cuts and restrictions can be attached to the
exposed graph/solver; the shared EV master does not add cut separation or tree
search. The tiny regression comparisons establish agreement on the tested
fixtures, with no performance or large-instance scaling claim.

## Shared route master and column generation

`EvMaster<M: LpMaster>` uses Baldes's existing LP backend interface. Supply an
empty backend, the EV problem, its initial departure time, and an optional
nonnegative fleet limit. The master adds one set-partitioning equality per
customer and an optional fleet upper-bound row. Only complete depot-to-depot
customer-serving routes are columns. Every inserted route is independently
replayed against the original problem, and its cost and customer coefficients
are recomputed. The same stop order with different charging amounts is one
travel-cost column.

No seed routes are required. Phase I minimizes artificial customer coverage,
pricing with zero travel cost and the current coverage/fleet duals. When
artificial usage reaches zero to tolerance, artificials are fixed to zero and
the original travel costs are restored. Phase II prices against that objective.
The two EV backends use exactly these same rows, objectives and tolerances.

Before pricing, the adapter checks the LP solution's primal row feasibility,
dual signs, column reduced costs, objective and primal/dual agreement. A
negative replay-validated column can improve the pool even after incomplete
pricing. Convergence requires an optimal restricted LP and `Complete` pricing
with no reduced cost below the requested tolerance. `Bounded`, work limits and
numerical failures cannot establish full-model LP convergence or infeasibility.
Positive phase-I usage establishes numerical infeasibility only after complete
pricing and a positive adjusted dual objective; tolerance-sized negative reduced
costs are accounted for by decreasing customer duals. This is still f64
validation, not an interval-arithmetic certificate.

Results distinguish convergence, phase-I infeasibility, incomplete pricing,
LP stop, iteration/time limits and numerical failures. They retain phase and
pricing statistics per pricing call, plus the last validated LP snapshot. A
snapshot taken before the last column insertion is a feasible point with zero
weight on later columns; it does not claim optimality of the updated pool.
`travel_objective` is available only when artificial coverage is zero to
tolerance. A restricted-pool objective is never labeled a full-model lower
bound. `optimize` can be called again after a work/iteration/time limit, retaining
the route pool and the current phase.

The adapter is generic over `LpMaster`; the integration tests below use Clp.
For example, with an existing `problem`:

```rust,ignore
use baldes::extensions::evrp::{EvMaster, EvColumnGenerationOptions};
use baldes::master::clp::ClpMaster;
let mut master = EvMaster::new(problem, ClpMaster::new()?, 0.0, Some(3.0))?;
let result = master.optimize(EvColumnGenerationOptions::default())?;
// Inspect result.status before interpreting its last LP snapshot.
```

LP solves receive the remaining time budget. Pricing uses its work/label limits,
with elapsed time checked before and after the call; the wall-time setting is
not an interrupt inside an individual pricing operation.

## Reproducible tiny comparisons

Run `cargo run --features clp --example evrp_compare` for a CSV comparison of
all three backends with identical data and tolerances (`1e-7` for LP and reduced
cost). On the checked fixtures:

| Case | Direct LP | Subpath LP | Full route-pool LP | Meaning |
|---|---:|---:|---:|---|
| Three customers, route capacity two, fleet limit two | 4.5 | 4.5 | 4.5 | Fractional weights of 0.5 demonstrate LP rather than integer optimality. |
| Two customers, nonlinear charging required, fleet limit one | 5.0 | 5.0 | 5.0 | Returned routes require partial charging and pass replay. |

The test reference pool enumerates every tiny stop order and constructs charging
schedules independently on an integer-energy grid. For these integral fixtures,
every order with no grid witness is also checked for continuous infeasibility
by the fixed-order optimizer. The full pool LP is built separately from column
generation, and the fractional case also has an analytic objective of 4.5.
Additional tests cover fleet/reachability infeasibility, an empty customer set,
seed validation and deduplication, phase-I initialization without seeds,
resuming after limits, and bounded/work-limited pricing without convergence.

The CSV reports columns, pricing calls, labels, retained subpaths and algebra
work. These tiny reference counts do not establish a runtime advantage or
scalability on published instances.

Run `cargo test --lib evrp::` for model/algebra/pricing tests and
`cargo test --features clp --lib evrp::` to include shared-master comparisons.

## Supported scope and remaining extensions

The raw ElectricRouting.jl multi-depot data is downloaded under
[`instances/evrp/electricrouting-jl/`](../../instances/evrp/electricrouting-jl/README.md):
161 instances and 312 upstream reference solutions, pinned to commit
`102c9c90f6021d2dea664f0d8ad6990661c7668a` with per-file hashes. These files
need a format reader and multi-depot integration before equivalent Baldes runs;
they are not inputs to `evrp_compare`. Their nonlinear charging curves must be
specified separately to match an upstream nonlinear experiment.

This implements the initial static, single-vehicle-type travel-cost EV LP
workflow. It is not an integer branch-and-price solver. Cuts, branching/ng
state, mixed fleets, tariffs, congestion, time-dependent travel, external
benchmark parsers and large-instance performance engineering remain extensions.
The E1 reference backend and E2-inspired prototype do not reproduce every
acceleration from either paper. The native backend uses the explicit charging-front resource; scalar resource
pricing alone does not establish nonlinear charging feasibility.
