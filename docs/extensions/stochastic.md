# Stochastic demands and replenishment recourse

`baldes::extensions::stochastic` implements a CVRPSD extension based on
Tomazella, Munari and Morabito (2026), *Pseudo-Compact Formulations and
Branch-and-Cut Approaches for the Capacitated Vehicle Routing Problem with
Stochastic Demands*:
https://optimization-online.org/wp-content/uploads/2026/04/Tomazella_Munari_Morabito_2026_Formulations-1.pdf

The model assumes independent finite discrete customer demands, integer vehicle
capacity, and demand realizations no larger than capacity. The depot is vertex
0, and demand entry `i` belongs to customer `i + 1`. Costs may be asymmetric;
callers should supply a matrix satisfying the triangle inequality. Construction
checks matrix dimensions, nonnegative finite costs and zero diagonal, but does
not check every triangle. A priori routes visit each assigned customer once,
and their total expected demand cannot exceed capacity.

## Supported policies

- `Policy::Classical` implements P1, equation (9): replenish at the customer on
  failure. If a customer empties the vehicle exactly, replenish before visiting
  the next customer. No replenishment is charged after the last customer.
- `Policy::OptimalRestocking` implements P3, equation (11): at each remaining-load
  state choose whether to replenish before continuing or risk a customer failure.

`Problem::evaluate_route` returns planned travel cost, expected recourse cost,
their sum, and expected demand. It uses a backward dynamic program with `Q + 1`
states and two vectors, taking O(route length * (Q+1) * support size) time.
Probability sums within 1e-12 of one are normalized during `Demand::new`.
Computation uses `f64`; no Monte Carlo sampling is involved.

## BALDES pricing and branch-and-price

`Problem::pricing_graph` compiles to the ordinary BALDES physical graph, using
`CoupledPlugin` objective costs for expected recourse. `Problem::price` runs
`pricing::Solver::price_phase_partial`, including bucket labeling and optional
bidirectional joins. This prices the stochastic objective directly. The compiled graph also works
with `RoutingContext::set_partitioning` and the shared routing root master in
numerical bound mode; regression tests cover its phase-I cost suppression and
restoration.

The default search traverses the physical graph in reverse and carries a
`Q + 1` Bellman continuation vector. Each extension prepends one Bellman
operator in O((Q+1) * support size), avoiding repeated prefix evaluation.
For visited-customer subsets, componentwise continuation-vector dominance
is safe because Bellman expectation and minimum operators are monotone.
Expected capacity is summed in customer-ID order, independently of route order.

`PricingOptions::bidirectional = true` retains the original bidirectional
adapter: forward states retain their complete customer sequence and joins
replay suffixes. `Problem::pricing_graph` also retains that original physical
orientation for shared-master integration. Generic completion bounds and
reversal reuse remain disabled for cost plugins. Other cost plugins retain
equality dominance unless they explicitly supply a cost-preserving dominance
contract. Large instances can still exhaust their label or time budget.

The adapter supplies full elementarity through NG memory and supports at most
`NG_MEMORY_CAPACITY` customers: 15 by default, or 31/63 with `ng-memory-31` /
`ng-memory-63`. The route domain has at most n+1 arcs, including departure and
return. A loose scalar mean-load window is only a necessary filter; canonical
forward resource replay enforces the expected-demand capacity condition.
Cost magnitudes are checked against BALDES's route-pool sentinel; unusually
large units must be rescaled.

Column generation now starts with scalar NG-route prepricing when the customer
count exceeds `ColumnGenerationOptions::ng_size` (default 8; zero disables it).
`Problem::price_ng` builds nearest-customer memories in the ordinary BALDES
engine. Its first sweep retains 16 labels per bucket to find columns quickly;
this heuristic sweep supplies **no lower bound**. A complete scalar sweep can
supply a bound, with the default master allocating at most 100,000 labels and
two seconds to the NG stage before falling back to stochastic Bellman pricing.

The scalar objective charges travel plus each visited customer's minimum
nonpositive preventive detour. Bellman recourse is bounded below by the sum of
these customer charges, including on nonmetric matrices. Hence the minimum of
a completely priced NG relaxation is a global lower bound on true elementary
stochastic reduced costs. It proves convergence only if it excludes improving
columns within tolerance. It does not generally equal true stochastic pricing.

Cyclic walks never enter the master or an incumbent. Candidate elementary routes
are evaluated with the full P1/P3 Bellman program, including all branch duals.
When cycles obscure useful candidates, decremental state-space refinement adds
the repeated customer to memories along its cycle and reprices; every elementary
route remains in the relaxation. Label/time exhaustion retains earlier valid
bounds but cannot turn an observed minimum into a bound. Primal pool/complement
search runs during column generation, and feasible integer restricted-master
snapshots can become incumbents even before pricing converges.

Use `stochastic_instance ... --ng-size 8` for this pipeline and `--ng-size 0`
for the Bellman-only adapter. `Problem::price` itself retains its original exact
stochastic pricing contract. The benchmark runner also accepts `--ng-size`.

`Master<M: LpMaster>` builds exact customer-cover rows and an upper fleet row.
Phase I uses artificial variables and zero physical/recourse costs; phase II
restores total expected costs. Every seed/generated route is independently
re-evaluated. Complete pricing is required before reporting LP convergence or
phase-I infeasibility. Restricted-pool objectives are never reported as global
bounds.

`Problem::solve_with` accepts a factory for fresh LP backends. Its arc
branch-and-price driver branches on fractional directed arc flows: zero removes
the arc from subsequent pricing; one adds a required-arc equality row whose
dual enters pricing. Each child performs two-phase column generation. Solving
only an integer restricted route pool is not used to establish optimality.
`Problem::solve` is the Clp convenience wrapper, available with `features=clp`:

```rust,ignore
use baldes::extensions::stochastic::{Policy, SolveOptions};
let answer = problem.solve(Policy::OptimalRestocking, 3, SolveOptions::default())?;
```

Run the complete example:

```sh
cargo run -p baldes --features clp --example stochastic
```

`SolveStatus::Optimal` means the branch-and-price tree is exhausted under the
configured floating-point tolerances. It is not an outward-rounded arithmetic
certificate. Node, iteration, pricing-work, and time limits retain an incomplete
status. Completed routes and labeling statistics survive pricing interruptions;
partial routes can improve the restricted master but never establish convergence.
Best-fit starts with several nearest-neighbor/shuffled orders, travel 2-opt,
and a bounded exact-cover/complement search over known columns provide feasible incumbents without waiting for complete pricing.
Heuristic packing failure does not prove infeasibility.

Stopped solves return the best incumbent found and a numerical lower bound.
The initial degree relaxation accounts for negative recourse on nonmetric
matrices. A hop-bounded shortest-walk relaxation supplies a global reduced-cost
lower bound even after incomplete pricing; shifting the fleet dual by its
negative part gives a valid full-master bound. Open nodes retain the best such
bound or their inherited bound. `PricingResult::minimum_reduced_cost` is absent
after incomplete pricing; `lower_bound` is the separate relaxation bound.
Generic directed
plugin certification returns unknown for cost-bearing plugins, preventing a
feasibility-only proof from silently assigning zero recourse cost.

## Exact tiny-instance baseline

`Problem::solve_tiny` enumerates elementary route orders, retains the cheapest
order for each customer subset, and solves the resulting set-partitioning problem
by subset dynamic programming. `max_vehicles` is an upper limit, not an equality.
All customers, including zero-demand customers, must be assigned.

This deliberately supports at most nine customers. `route_budget` bounds the
number of evaluated routes; exceeding it returns an error, not a feasible answer
or an optimality certificate. `Ok(None)` means that exhaustive search proved fleet
infeasibility. An empty customer set has a zero-cost solution. Optimality is with
respect to the supplied distributions and ordinary floating-point comparisons.

## Remaining boundaries

The paper's arc-based MIP, rule-based and penalty recourse policies, Python
bindings, stronger stochastic dominance, specialized fast joins, and stochastic
cut separation are not implemented. The branch-and-price driver currently adds
arc branch rows and uses no additional cutting planes.

Verification:

```sh
cargo test -p baldes --features clp --test stochastic --test stochastic_pricing
```

Tests cover hand-calculated costs, comparison against a separate scenario tree,
BALDES pricing against exhaustive route orders in both directions, phase-I cost
suppression, forbidden/required arc handling, full-route LP comparisons, integer
optima against the tiny oracle, a fractional root requiring branching, and
incomplete-work statuses.
