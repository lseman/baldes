# Clarke–Wright initial columns

Use `--initial-columns savings` to populate a CVRP or VRPTW restricted master
with routes from parallel Clarke–Wright Savings before column generation:

```sh
./target/release/baldes solve instances/X/X-n101-k25.vrp --initial-columns savings --rounding nearest
./target/release/baldes solve instances/Solomon/R201.txt --initial-columns savings --master clp
```

The default remains `--initial-columns artificial`. Savings starts with singleton
routes and considers endpoint merges in descending savings order, accepting only
cost reductions that pass the target graph's complete resource and arc checks.
This respects time windows, service times and waiting for VRPTW, and capacity
without time windows for CVRP. Route reversals are checked again for feasibility
and cost, including directed distances.

Savings routes are ordinary initial master columns. Artificial variables remain
available for feasibility recovery; the heuristic is not guaranteed to find a
complete partition within the fleet limit. A complete fleet-feasible Savings plan
also supplies an upper bound and, with `--tree`, an initial incumbent. An
incomplete or excess-fleet plan supplies only its feasible columns.

You may combine Savings with `--hgs-time SECONDS`: HGS adds distinct routes and
the better complete incumbent is kept. CLI output reports the number of Savings
columns and its incumbent, if any. Savings initialization is separate from the
subsequent solver time budget and stores a quadratic number of candidate savings.

Rust callers can use `Instance::seed_savings(&graph, rounding)` and populate
`model::Options::seed_routes` with the returned routes.
