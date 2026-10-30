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

## Conventional routing with Theta cuts

`Problem::solve` uses the ordinary BALDES routing branch-cut-and-price tree.
Pricing uses mean demands and travel costs plus per-customer recourse lower
bounds. Exact P1/P3 route evaluation supplies the excess charged by Theta
optimality cuts. Integral routes are separated before accepting an integral
master solution. The separate stochastic Bellman-pricing master/tree has been
removed.

The routing pipeline includes heuristic pricing phases, exact bidirectional
pricing, dynamic NG refinement, bucket pruning and reduced-cost fixing.
Rank-one cuts (eight per round), rounded capacity cuts on mean demands, and
load-indexed completion bounds are enabled by default. Dynamic NG mode `-1`
pauses memory strengthening while rank-one rows are active; existing NG memories
remain in use. Set `BALDESdynamicNGmode = 1` to continue strengthening, or `0`
to disable it. Explicit pricing options override the default NG limits of 15
customers per vertex and on average. `BALDESloadCompletionResource = -1`
disables load completion bounds. Route enumeration is disabled for recourse.

```rust,ignore
use baldes::extensions::stochastic::{HgsWarmStart, Policy, SolveOptions};
use std::time::Duration;
let mut options = SolveOptions::default();
options.tree.time_limit = Duration::from_secs(30);
options.hgs_warm_start = Some(HgsWarmStart {
    time_limit: Duration::from_secs(3), seed: 0,
});
let answer = problem.solve(Policy::OptimalRestocking, 3, options)?;
println!("{:?}: {}", answer.tree.status, answer.tree.incumbent);
```

`SolveOptions` contains `tree: TreeOptions`, `ng_size` (default 8), and
`hgs_warm_start`. `SolveResult::tree` is the conventional routing `TreeOutput`;
`hgs_warm_start` reports the optional heuristic's validation counts, elapsed time,
and the accepted plan's stochastic `initial_objective`. For callers configuring
all seeds themselves, `solve_theta_cuts(policy, vehicles, ng, tree)` runs the same
routing implementation directly.

HGS searches a deterministic CVRP with the same cost matrix and mean demands.
Its integer loads are scaled and rounded down, then every candidate route is
independently checked for elementarity and original expected-capacity feasibility.
A complete plan must cover each customer exactly once within the fleet limit.
Its incumbent cost is recomputed with the selected stochastic policy, including
expected recourse; HGS's travel score is never used as the stochastic objective.
Both directions of every route in a valid HGS plan are evaluated; an improving
reverse order replaces the original in the seed, and both remain candidate
columns. This also matters with symmetric travel costs because recourse depends
on visit order. The summary reports `proposal_objective` before this check and
`reversed_routes`, alongside the actual seeded `initial_objective`.
Accepted routes seed the root column pool. A validated complete plan also enters
`TreeOptions::initial_solution`, so it is immediately available without a pool
MIP search and survives exhausted time or node budgets.

HGS is optional and runs only when requested. HGS, validation and seed construction
consume the overall `tree.time_limit`; the routing tree receives the remaining
budget. A zero overall budget skips HGS. Heuristic initialization and cleanup can
slightly overrun wall-clock limits. HGS candidates never certify a lower bound.
No previous run's solution is automatically loaded.

```sh
cargo run --release --features clp --example stochastic_instance -- \
  instances/stochastic/generated_salavati_style/synthetic-n40-m3-r01-fr0.94.json \
  --solve --seconds 30 --hgs-seconds 3 --hgs-seed 0 --json
```

`--solve` always uses Theta recourse cuts; `--recourse-cuts` remains a compatibility
alias. `--dynamic-ng-mode -1|0|1`, `--rank-one-cut-limit`, `--log-cg`, and
`--no-load-completion-bound` configure the routing path. Flags for the removed
stochastic pricing implementation are rejected.

The result retains the best incumbent and numerical bound on interruption.
`TreeStatus::Optimal` describes termination at the configured floating-point
tolerances, not an outward-rounded arithmetic certificate. The routing tree
requires nonnegative adjusted arc costs; some rounded nonmetric inputs can fail
that check and require a different cost representation.

## Subset-conditioned RCSP completion bounds

Exact conventional pricing conditions load-indexed completion tables on a small
selected NG-memory subset. The stochastic default requests four elements;
`BALDESloadCompletionSubsetSize` (or `--completion-subset-size`) selects 0–6.
Zero retains the unconditional bound. `--no-load-completion-bound` disables both.
Elements are ranked by negative incoming reduced cost per unit of load.

The table state is `(vertex, remaining load, selected memory subset)`, in each
labeling direction. Each arc follows the corresponding projection of the actual
NG transition, including vertex elements, arc elements, arc memory and forgetting.
A selected customer is unavailable while remembered and can be revisited after
it is forgotten. This adapts the selected-set idea of Florio et al. §4.5.2 to NG
pricing; it does not impose permanent elementarity on an NG label.

Every feasible NG continuation projects into this state graph. Other resources
and branch memories are relaxed, loads are rounded down, available capacity is
rounded up, and future nonnegative packing-cut increments are omitted. Thus the
table is a lower bound on the current pricing objective, including arc-based
Theta, capacity and branch prices. The pruning test also retains packing
penalties already paid by the label. Unsupported cost-bearing resources/cuts
continue to disable these bounds.

Tables are rebuilt at the current reduced costs and NG neighborhoods for each
exact pricing call. The optional subset construction has a 100 ms setup budget,
at most 32 MiB of table cells across both directions, and at most one million
projected transitions. The selected set shrinks to fit the memory limit; an
incomplete construction falls back to the unconditional bound. An NG-conditioned
table can also be finite when a zero-load negative cycle makes the unconditional
walk relaxation unbounded. If neither bound completes, ordinary pricing proceeds.
`BALDES_DEBUG_COMPLETION=1` reports the actual selected subset size per call.

These are pricing accelerators, not additional master inequalities. They do not
by themselves strengthen a fully priced LP relaxation. Benchmark comparisons
must include setup time as well as labeling work and the retained global bound.

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

Rule-based and penalty recourse policies, stochastic Python bindings, and
reproduction of the paper's reported run times are outside this implementation.

```sh
cargo test -p baldes --features clp --test stochastic --test stochastic_pricing
```

Tests cover hand-calculated and scenario-tree route costs, exact tiny-instance
solutions, the routing tree against the tiny oracle with both policies and cut
families, bucket fixing, HGS route validation, seeded incumbents without pool
search, stochastic objective reconstruction, and shared time budgets.
