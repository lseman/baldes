# Column elimination

`baldes::extensions::ce` solves the continuous elementary
route set-partitioning relaxation for CVRP and VRPTW. It uses a layered
decision diagram and an arc-flow LP, followed by infeasible-prefix refinement.
It is an independent implementation inspired by [Karahalios's reference
solver](https://github.com/amkarahalios/dd_vrptw) and the work of Karahalios
and van Hoeve, *Column elimination for capacitated vehicle routing problems*.

The general initial diagram records the number of customer visits and the
last location, with at most `n` visits per route. It contains every feasible
elementary route but also permits non-elementary and resource-infeasible
routes. Transitions that cannot meet the destination time window even from
the source's earliest service time, and customers whose demand exceeds
capacity, are excluded. Forward propagation then computes independent lower
bounds on accumulated load and service-start time. An arc is removed if even
these optimistic bounds violate capacity or a time window. A reverse pass
propagates latest feasible service times with conservative outward rounding,
removes regions that cannot reach the end depot, and compaction physically
reclaims dead nodes and arcs before the LP is built. These reductions also
apply after each refinement batch and preserve feasible routes with directed,
nonmetric travel times and zero-demand customers.

For at least 20 customers with finite, nonnegative time windows, the solver
can instead index states by location and a rounded-down service-time bucket.
The bucket width is a power of two bounded by the smallest customer transition
duration. Every transition must advance its bucket, preserving acyclicity.
Anchors are lower bounds, so all feasible elementary routes remain represented;
exact replay still checks original times, capacity and repeated visits.
This mode is selected only when the estimated state count is at most one
quarter of the layered construction and fits the budgets. Unsupported time
ranges, zero-duration transitions or oversized graphs use the existing fallback.
Identical relaxed suffixes are also merged across layers; topological ordering
is recomputed after refinement rather than relying on the original visit count.

For capacity-only instances with at least 20 customers and positive integer
demands, the solver can instead compile states indexed by load and location.
Loads are divided by their demand GCD without rounding. Each location also
has a two-customer ng neighborhood (itself and its cheapest reachable
neighbor), retaining limited visit memory to prevent some repetitions early.
Positive loads provide an acyclic ordering. This construction is selected
only when every window has an infinite upper bound, capacity fits the bounded
integer representation, scaled capacity is at most `4n`, and conservative
state/arc estimates fit the configured budgets. Otherwise the general
layered construction is used. Capacity states strengthen the initial bound
but can increase the latency of the first LP solve.

The LP imposes customer visit equalities,
flow conservation at each internal node, and an optional fleet upper bound.
Depot-to-depot empty routes are excluded.

After each optimal LP solve, the extension decomposes arc flows into paths
and replays capacity, service-start windows, waiting, service duration and
elementarity. Each infeasible prefix is refined into exact states containing
its last location, visited-customer set, load and service-start time. Equal
states share a node only when their active resource values match bit for bit.
Time is omitted from exact-state keys only if all window upper bounds are
infinite and a conservative forward bound proves that time cannot overflow
on an elementary route. Every
outgoing transition from a new exact state is checked immediately, eliminating
all detectable conflicts rather than just the reported one. Incoming prefixes
with different resources retain their own successors.
Prefixes are resolved against the updated diagram after each split so shared
prefixes and conflicts already removed by the batch are handled safely. The
LP is rebuilt after the batch, rather than after every individual conflict.
Consequently every feasible route survives refinement. If the complete
decomposition is feasible, it supplies a route-LP solution matching the
arc-flow lower bound, establishing LP convergence within numerical tolerances.

```rust,ignore
use baldes::extensions::ce::{Options, Problem, Status};
use baldes::master::solvers::ClpMaster;

let problem = Problem::from_instance(&instance)?;
let result = problem.solve_column_elimination(ClpMaster::new, Options::default())?;
if result.status == Status::Converged {
    println!("Route LP objective: {:?}", result.lower_bound);
    for route in result.routes {
        println!("{} * {:?}", route.flow, route.path);
    }
}
```

The backend factory supplies a fresh empty `LpMaster` each iteration. The
extension loads rows and columns in bulk to avoid repeated native matrix
copies; only disabled arcs need explicit upper-bound changes, since flow and
visit rows already bound active arcs by one. Clp requires `--features clp`;
the refinement engine itself has no Cargo feature
or additional dependency. Matrices use separate start/end depot indices,
`0` and `n+1`, with customers `1..=n`. Costs and travel times may differ and
may be asymmetric. A missing cost (`None`) forbids that transition. Window
upper bounds may be positive infinity. `from_instance` copies the distance
convention, depot window and fleet size; set `max_vehicles = None` to relax
the fleet constraint.

`Converged` means the **continuous route LP** has converged. Returned route
flows can be fractional and do not constitute an integer incumbent. This
extension currently has no integer branch-and-bound, cuts, Lagrangian solver,
primal heuristics, or integration with the existing root/tree driver. It
merges exact resource states during refinement and rebuilds the LP after each
refinement batch. Its capacity-only initial diagram uses exact loads and
limited ng memory; the time-window mode uses conservative initial time buckets.

Iteration, time and diagram limits have explicit statuses. `lower_bound` is
the last optimal relaxed-diagram objective, including on a later stop; it is
absent if no optimal LP was obtained. `routes` is populated only at convergence.
An infeasible relaxed diagram certifies infeasibility of the routing model.
Invalid data, an oversized initial diagram, or failed numerical decomposition
returns an error. The general initial construction has `O(n²)` temporary nodes
and `O(n³)` possible arcs before pruning. The capacity-state construction is
pseudo-polynomial in scaled capacity `Q/g`: at most `O(nQ/g)` states and
`O(n²Q/g)` arcs with two-customer ng memory. Refinement can still grow
exponentially. Reported node and arc counts reflect the compacted graph;
temporary refinement allocations also respect the configured budgets.
The time budget is checked between stages
and passed to the backend; backend support determines solve interruption.

Run an instance from the CLI (Solomon or CVRPLIB):

```sh
cargo run --release --features clp -- ce instances/cvrplib/P-n16-k8.vrp --customers 5 --time 60 --log --json ce.json
baldes ce --help
```

The command respects the instance fleet size; `--unlimited-fleet` removes
that limit. `--nodes`, `--arcs` and `--iterations` set diagram and solve
budgets. JSON includes the last `lower_bound`, the converged `lp_bound`
(null before convergence), status, statistics and fractional route flows.

For VRPTW, pass a Solomon instance directly:

```sh
cargo run --release --features clp -- ce instances/C101.txt --customers 5 --time 60 --log
```

Time windows and service durations are enforced automatically. The command
still solves the continuous route LP, with potentially fractional routes.
`--log` (also `-v` or `--verbose`) writes progress every 10 iterations to
stderr; `--log-interval N` selects a positive interval. The first and last
solved iterations are always logged, followed by the final status, including
limit stops. Columns show the LP lower bound, decomposed path count,
infeasible path count, diagram size and elapsed time. Stdout retains the
final result summary. The Rust API exposes `Options::log_interval` as well.

To use parallel HiGHS, build with both backends and select it explicitly:

```sh
cargo build --release --features clp,highs --bin baldes
target/release/baldes ce instances/C101.txt --solver highs --threads 0 --log
```

`--master` is an alias for `--solver`, matching the main solver CLI spelling.
CE enforces capacity constraints automatically. The main solver's `--rank1`
and `--capacity` cut flags are not implemented in CE and report explicit errors.

For HiGHS, `--threads 0` (also the default when selecting HiGHS) resolves to
all CPUs available to the process. An explicit positive count limits the
worker pool. Clp remains the default backend, and `--threads` requires
`--solver highs`. Missing HiGHS builds report the required Cargo features.
JSON records `solver`, the resolved `threads` count and
`simplex_concurrency`. Verbose output displays the worker count and simplex
concurrency limit.

The linked solver is queried for its maximum simplex concurrency. HiGHS
1.15.1 on the benchmark host permits a 32-thread worker pool but caps PAMI
simplex concurrency at eight. Selecting all CPUs therefore does not promise
32 simultaneously active simplex tasks or faster solves. For example,
`--threads 4` can be preferable for smaller or memory-bound models.

The Rust API provides `HighsMaster::with_threads(0)` (all available CPUs) or
`HighsMaster::with_threads(n)`, and preserves these settings when cloning.
HiGHS models sharing its process-global scheduler must use the same worker
count. This wrapper never resets the shared scheduler to change that count.

Run the small demonstration with:

```sh
cargo run --release --features clp --example ce
```

Regression tests verify route preservation after refinement, exact resource
replay, atomic limits, fractional LP behavior, infeasibility, and agreement
with full route enumeration on small asymmetric VRPTW instances:

```sh
cargo test -p baldes --features clp --lib extensions::ce
```

Performance smoke check (2026-10-21, release build, Clp):

```sh
baldes ce instances/C101.txt --time 15 --log
```

Before the subsequent resource-propagation reductions, one local run on all
100 customers converged to the continuous route-LP bound
`828.936867` in 8.00 seconds, with two LP solves and three prefix refinements.
Local feasibility pruning reduced the initial arc count from 990,200 to
436,988. This is a single-instance check, not a general runtime guarantee.
Capacity-only instances can still require many refinements because the
general layered states do not track exact accumulated load or visit history.
