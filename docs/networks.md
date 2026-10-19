# Shared master with multiple pricing networks

With `clp`, `master::networks::solve` solves a fresh root whose pricing graphs
share master rows. Each `Network` owns a graph, a unique nonnegative ID and
finite nonnegative multiplicity bounds. Local arc IDs may overlap across
networks: route identities include the network ID and resource schedule.

`Problem.rows` defines the shared constraints. `Network.arc_entries` contains
one list of `(shared_row, coefficient)` pairs per physical arc, in graph order.
`route_entries` adds coefficients charged once per route. The coordinator adds
the network multiplicity rows separately; callers do not include them in the
shared-row indices. Arc costs define original route costs. Variable-cost
mapping used internally by pricing does not modify the supplied graphs.
Physical multiplicity rows appear in input-network order: each upper row is
followed immediately by its lower row when the lower bound is positive. Shared
rows follow these multiplicity rows in their original order.

The coordinator maintains independent pricing state, combines columns into one
restricted master and aggregates each network's multiplicity contribution and
stabilization subgradient. `Output.oracle_calls` counts individual oracle calls;
`Output.master.pricing_calls` counts batches. `Output.values` gives the current
restricted-master quantity for every route in `Output.routes`.

Phase estimates guide stabilization. Published lower bounds additionally use a
resource-free graph-walk relaxation with downward arithmetic. A relevant
negative cycle, arithmetic overflow or unavailable required path leaves this
certificate unknown. If a complete bound is unavailable, the coordinator
returns `Stalled` rather than certifying optimality. Finite artificial penalties
also do not certify original-model infeasibility. A stalled restricted LP can
still contain artificial variables; its objective and route quantities are
not then an original feasible solution.

The current coordinator accepts fresh roots with an empty initial column
registry. Cuts, enumeration/MIP transfer, fixed partial solutions and tree
snapshots still use the existing single-network workflow and are not yet
integrated here. The scalar resource dimension remains capped at two.

Independent checks cover shared covering constraints, fractional and required
multiplicities, disabled networks, overlapping local arc IDs, original lower
bounds on generated assignment models and conservative incomplete outcomes:

```sh
cargo test --features clp --lib master::networks::tests
```
