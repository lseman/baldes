# Column elimination

`baldes::extensions::column_elimination` solves the continuous elementary
route set-partitioning relaxation for CVRP and VRPTW. It uses a layered
decision diagram and an arc-flow LP, followed by infeasible-prefix refinement.
It is an independent implementation inspired by [Karahalios's reference
solver](https://github.com/amkarahalios/dd_vrptw) and the work of Karahalios
and van Hoeve, *Column elimination for capacitated vehicle routing problems*.

The initial diagram records the number of customer visits and the last
location, with at most `n` visits per route. It contains every feasible
elementary route but also permits non-elementary and resource-infeasible
routes. The LP imposes customer visit equalities, flow conservation at each
internal node, and an optional fleet upper bound. Depot-to-depot empty routes
are excluded.

After each optimal LP solve, the extension decomposes arc flows into paths
and replays capacity, service-start windows, waiting, service duration and
elementarity. For the first infeasible prefix, it copies the successor nodes
along the prefix, redirects its incoming arcs, and removes the conflicting
transition. Other incoming prefixes retain their original successors.
Consequently every feasible route survives refinement. If the complete
decomposition is feasible, it supplies a route-LP solution matching the
arc-flow lower bound, establishing LP convergence within numerical tolerances.

```rust,ignore
use baldes::extensions::column_elimination::{Options, Problem, Status};
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

The backend factory supplies a fresh empty `LpMaster` each iteration. Clp
requires `--features clp`; the refinement engine itself has no Cargo feature
or additional dependency. Matrices use separate start/end depot indices,
`0` and `n+1`, with customers `1..=n`. Costs and travel times may differ and
may be asymmetric. A missing cost (`None`) forbids that transition. Window
upper bounds may be positive infinity. `from_instance` copies the distance
convention, depot window and fleet size; set `max_vehicles = None` to relax
the fleet constraint.

`Converged` means the **continuous route LP** has converged. Returned route
flows can be fractional and do not constitute an integer incumbent. This
extension currently has no integer branch-and-bound, cuts, Lagrangian solver,
primal heuristics, or integration with the existing root/tree driver. Unlike
the reference solver, it uses prefix isolation rather than merging equal
exact resource states, and rebuilds the LP after each refinement.

Iteration, time and diagram limits have explicit statuses. `lower_bound` is
the last optimal relaxed-diagram objective, including on a later stop; it is
absent if no optimal LP was obtained. `routes` is populated only at convergence.
An infeasible relaxed diagram certifies infeasibility of the routing model.
Invalid data, an oversized initial diagram, or failed numerical decomposition
returns an error. The initial diagram has `O(n²)` nodes and `O(n³)` arcs, and
refinement can grow exponentially. The time budget is checked between stages
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

Run the small demonstration with:

```sh
cargo run --release --features clp --example column_elimination
```

Regression tests verify route preservation after refinement, exact resource
replay, atomic limits, fractional LP behavior, infeasibility, and agreement
with full route enumeration on small asymmetric VRPTW instances:

```sh
cargo test -p baldes --features clp --lib extensions::column_elimination
```
