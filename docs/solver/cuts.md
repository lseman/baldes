# Routing cuts

[Documentation index](../README.md) · [BALDES options](../reference/options.md)

A cut must have consistent coefficients in the master, pricing, retained
routes, and descendants. The root adapter installs cut rows and dual effects;
node snapshots retain active rows and the pricing state needed to use them.

## Available families

| Family | Purpose | Rust root configuration | CLI / Python |
| --- | --- | --- | --- |
| Limited-memory rank-one | Strengthen the customer-cover relaxation | `with_rank_one_cuts()` | `--rank1` / `rank_one_cuts=True` |
| Rounded capacity | Enforce capacity-derived subset inequalities | `with_capacity_cuts(config)` | `--capacity-cuts` / `capacity_cuts=True` |
| Kohl 2-path | Use time-window feasibility to strengthen capacity subsets | Capacity config with time windows | `--two-path-cuts` / `two_path_cuts=True` |
| Strong k-path | Separate route-count inequalities on customer sets | `with_kpath_cuts(config)` or `with_native_kpath_cuts(config)` | `--cuts` / Rust API |
| Route-load knapsack | Separate inequalities on integer resource loads | `with_knapsack_cuts(config)` | `--cuts` / Rust API |
| Fenchel load cuts | Search integer packing inequalities with an auxiliary LP | Knapsack config `.with_fenchel(parameters)` | `--fenchel-cuts` / Rust API |

All routing root integrations require `clp`. HiGHS is needed only for the
optional enumerated integer solve. Cut configuration does not enable that
solve automatically.

## Start with capacity and rank-one cuts

```sh
cargo run --release --features clp --bin baldes -- \
  vrptw instances/Solomon/R101.txt --customers 25 --capacity-cuts --rank1 --time 60
```

An experimental rank-one selection budget can limit how many active cuts touch
any vertex through their support or memory:

```sh
--rank1 --option BALDESrankOneCutsMaxPerVertex=8
```

The default is `0` (disabled). Positive values count each cut once per vertex
and graph, including both endpoints of memory arcs. Existing nonretired cuts
consume the budget across separation rounds and packing/covering passes.
Candidates that exceed it are skipped in the separator's existing score order;
lower-ranked candidates may fill the remaining slots. Existing cuts are retained,
even if already over budget. This trades potential bound strength for cheaper
pricing; `8` is an example to benchmark, not a tuned recommendation. The Rust
registry API exposes the same control as `SeparationParameters::max_per_vertex`.

A complete Rust example:

```rust
use baldes::{master::root::RootOptions, inputs::{Instance, Rounding}};

fn main() -> baldes::Result<()> {
    let mut instance = Instance::read("instances/Solomon/R101.txt")?;
    instance.truncate(25)?;
    let rounding = Rounding::None;
    let capacity = instance.capacity_cut_config(rounding, false);
    let options = RootOptions::default()
        .with_capacity_cuts(capacity)
        .with_rank_one_cuts();
    let result = instance.solve_root(8, rounding, options)?;
    println!("{:?}: root LP={}", result.master.status, result.master.lp);
    Ok(())
}
```

Use `capacity_cut_config(rounding, true)` to attach the instance's ready/due
windows, service times, and travel matrix for Kohl separation. The CLI/Python
2-path switch is for VRPTW; CVRP inputs do not supply time windows.

Capacity/Kohl separation has priority 3 at the root and 1 below it. It shares a
persistent subset pool. Basic cut rows may be retired by cleanup; reported cut
counts can therefore exceed the number of rows still active.

## Strong k-path adapters

`KPathCutConfig::new(demands, capacity)` is the routing convenience adapter.
Its demand vector uses depot index 0 and customer indices `1..n`; the split
sink is `n + 1`. `with_time_windows` supplies optional time-window data.

`NativeKPathCutConfig` accepts explicit graphs, graph IDs, packing/covering
set incidence, and demands in the original zero-based set order without a
depot entry. Use it when the convenience adapter does not represent the input
mapping. `with_kpath_cuts` and `with_native_kpath_cuts` select alternative
adapters; the most recently selected one replaces the other.

Graph IDs, set IDs, and pricing vertex indices have different roles. Keep
those mappings explicit when registering cuts. Full-memory enumeration and
limited-memory pricing can use different coefficient state; the adapter
updates the master when switching modes.

The Kohl subset-growth heuristic within strong k-path separation visits each
customer subset once and has a default work budget of 10,000 unique search
states per separation call. Set `KPathParameters.max_two_path_states`, or use
`set_parameter("max_two_path_states", value)`, to tune it; zero skips this
heuristic, while the other k-path separation procedures remain enabled. This
bounds an otherwise factorial traversal of equivalent insertion orders. The
existing set-size limit remains 15.

When the budget is exhausted, the current subset can be evaluated as a
candidate and recursion unwinds cleanly. Every candidate still passes the
normal violation/time-feasibility checks. This is a bounded heuristic, so it
can find fewer or different cuts than the previous exhaustive growth search;
it does not disable a cut family or relax cut validity checks.

## Route-load knapsack and resource IDs

`KnapsackCutConfig::new(rows)` takes integer resource rows from
`cuts::knapsack::KnapsackRow`. Separation uses route resource loads, and
pricing must charge the same load-dependent contribution. Ordinary arc-only
costs are insufficient for this family.

`KnapsackCutConfig::with_resource_indices(map)` maps resource IDs in those
rows to coordinates in the pricing solver. An empty map uses identity
mapping. For example, an input resource identified as 7 can map to solver
coordinate 1. This changes the coordinate lookup, while preserving resource
IDs in the cut associations. The mapping must remain valid through pricing,
enumeration, and child inheritance.

Keep rows integer-scaled according to their model semantics. The resource
map is not a unit conversion. See the mapped-resource and load-dependent
pricing cases in [the integration tests](../../tests/cuts/kpath_knapsack.rs) for
complete fixtures.

For direct Rust `pricing::Solver` use, set
`knapsack_exact_coefficients = true` to evaluate graph-scaled coefficients with
unrounded nonpositive master duals. The setting survives `fork_for_node()` and
applies to pricing and enumeration. It rejects positive duals and non-monotone
step tables. This option does not turn numerical route costs into directed
certificates or automatically register shared-network rows. The default retains
the existing rounded pricing convention.

## Fenchel separation

```sh
cargo run --release --features clp --bin baldes -- \
  cvrp instances/cvrplib/E-n22-k4.vrp --fenchel-cuts --time 30
```

The CVRP/VRPTW switch builds an integer load row from total customer demand,
divides demands by their greatest common divisor, and enables Fenchel alone.
Demands must be nonnegative integers. The normalized total must not exceed
4096; use explicit Rust resource rows and limits for larger domains.
`Instance::fenchel_cut_config()` exposes the same configuration to Rust.

For custom rows, use `KnapsackCutConfig::new(rows).with_fenchel(parameters)`,
where `parameters` is `cuts::knapsack::FenchelParameters`. This builder retains
the configured rounding/1K families. Each supplied row must be a valid integer
packing inequality on additive, nonnegative route loads.

An auxiliary CLP model solves the dual of the recovered bin-packing covering
formulation; unbounded-knapsack dynamic programming generates pattern constraints.
Extraction waits for auxiliary convergence. Before returning a cut,
integer-profit dynamic programming certifies its integer coefficients and RHS.
The monotone step extension also covers loads absent from the current master.
Fenchel cuts automatically use graph-scaled coefficients and unrounded duals
in pricing and enumeration, including child solvers and integer transfer.

Defaults limit separation to 128 load types, 128 LP iterations, and two seconds
per separation call. A limit or failure to find a representable violated cut
does not certify membership in the integer hull. Separation is opt-in and
requires `clp`.

## Configuration boundaries

Named BALDES options do not create model data. Enabling capacity separation
still requires demands and vehicle capacity; setting a knapsack mode still
requires resource rows. Explicit capacity enablement and per-round overrides
are applied after inherited cut state is installed. Knapsack construction
overrides do not reconfigure an existing inherited separator.

See [BALDES options](../reference/options.md) for supported values and setter errors.

## Pricing, enumeration, and integer solving

1. Separation creates rows and registers their pricing state.
2. Pricing incorporates active cut duals while constructing routes.
3. Complete elementary enumeration uses full-memory coefficients, including
   updates to route columns already in the master.
4. Child snapshots preserve active rows and cut state. The optional
   [HiGHS solve](enumerated-mip.md) rebuilds full-memory route columns against
   every active physical row and omits artificial columns.

## Presence-based costs in the labeling engine

The low-level Rust `LabelingEngine::load_presence(graph, rows)` interface accepts
`cuts::soft_conflicts::PricingInput` descriptors. Each contains a `PresenceRow`,
a dual, and one event list per physical arc in `graph.arcs` order. Events
identify original variables: several arcs can emit the same event, and an arc
can emit several events. Repeated visits emit presence once.

`Mode::PairCount` counts stored pairs whose two events occur on a route;
`Mode::AnyPair` charges at most once across the entire row. Duplicate pairs
count separately in pair-count mode. The engine subtracts dual times the
coefficient from route cost; `original_cost` retains physical arc costs.
Duals are unrounded. Invalid mappings, nonfinite duals or costs outside the
engine's supported finite arithmetic range return an error without replacing
its previous preparation.

Presence incidence participates in dominance, forward/backward joins, completion
bounds and reduced-cost fixing. Symmetric prefix reuse switches to ordinary
bidirectional labeling to preserve physical event orientation. Installing or
clearing rows invalidates cached labels; pass an empty list to clear them.

For stateful pricing, use `pricing::Solver::set_presence_rows(rows)` and inspect
its descriptors with `presence_rows()`. The solver owns these rows across
engine rebuilds, node forks and raw variable-cost pricing. Changed rows or duals
invalidate cost-dependent labels, bucket fixing and retained enumeration;
identical inputs retain working state. Validation failures preserve the prior
rows and caches. Logical branches and physical arc exclusions remain in force.
Pass an empty list to clear the rows.

These interfaces cover pricing, fixing, enumeration and retained-path repricing.
Enumeration preserves distinct physical arc paths when presence rows are active,
so different variable incidence cannot merge through customer-set equivalence.
It follows the configured cutoff, route and work limits; a failed attempt does
not publish a partial universe. Presence costs are replayed when retained paths
are inspected or priced, including in node forks. Symmetric physical graphs use
ordinary directional buckets for these rows.

Use the solver setter to keep owned descriptors and engine costs consistent.
Enumeration rejects conflicting rows installed only on its engine. Rust callers
constructing `enumeration::Label` or `CutTables` literals should initialize the
new presence fields to `None` for models without presence rows.

Shared-master row registration remains under development. These numerical
pricing costs do not by themselves publish a certified master bound.
