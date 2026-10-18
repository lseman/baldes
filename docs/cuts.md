# Routing cuts

[Documentation index](README.md) · [BALDES options](options.md) ·
[Source layout](source-layout.md)

A cut must have consistent coefficients in the master, pricing, retained
routes, and descendants. The root adapter installs cut rows and dual effects;
node snapshots retain active rows and the pricing state needed to use them.

## Available families

| Family | Purpose | Rust root configuration | CLI / Python |
| --- | --- | --- | --- |
| Limited-memory rank-one | Strengthen the customer-cover relaxation | `with_rank_one_cuts()` | `--rank1` / `rank_one_cuts=True` |
| Rounded capacity | Enforce capacity-derived subset inequalities | `with_capacity_cuts(config)` | `--capacity-cuts` / `capacity_cuts=True` |
| Kohl 2-path | Use time-window feasibility to strengthen capacity subsets | Capacity config with time windows | `--two-path-cuts` / `two_path_cuts=True` |
| Strong k-path | Separate route-count inequalities on customer sets | `with_kpath_cuts(config)` or `with_native_kpath_cuts(config)` | Rust API |
| Route-load knapsack | Separate inequalities on integer resource loads | `with_knapsack_cuts(config)` | Rust API |

All routing root integrations require `clp`. HiGHS is needed only for the
optional enumerated integer solve. Cut configuration does not enable that
solve automatically.

## Start with capacity and rank-one cuts

```sh
cargo run --release --features clp --bin baldes -- \
  vrptw instances/R101.txt --customers 25 --capacity-cuts --rank1 --time 60
```

A complete Rust example:

```rust
use baldes::{master::root::RootOptions, vrptw::{Instance, Rounding}};

fn main() -> baldes::Result<()> {
    let mut instance = Instance::read("instances/R101.txt")?;
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
pricing cases in [the integration tests](../tests/cuts/kpath_knapsack.rs) for
complete fixtures.

## Configuration boundaries

Named BALDES options do not create model data. Enabling capacity separation
still requires demands and vehicle capacity; setting a knapsack mode still
requires resource rows. Explicit capacity enablement and per-round overrides
are applied after inherited cut state is installed. Knapsack construction
overrides do not reconfigure an existing inherited separator.

See [BALDES options](options.md) for supported values and setter errors.

## Pricing, enumeration, and integer solving

1. Separation creates rows and registers their pricing state.
2. Pricing incorporates active cut duals while constructing routes.
3. Complete elementary enumeration uses full-memory coefficients, including
   updates to route columns already in the master.
4. Child snapshots preserve active rows and cut state. The optional
   [HiGHS solve](enumerated-mip.md) rebuilds full-memory route columns against
   every active physical row and omits artificial columns.

The cut registry and coefficient logic live under [src/cuts](../src/cuts/).
Root scheduling and row remapping live in the
[root controller](../src/master/root/controller.rs).

## Validation

```sh
cargo test --features clp --test capacity_root --test kpath_knapsack
cargo test --features clp,highs --test highs_mip
```

The suites cover small exhaustive validity checks, separation violations,
bidirectional pricing, enumeration repricing, mapped resource IDs, time
windows, and inherited cut rows. The HiGHS tests exercise complete node
masters across the existing cut-family configurations.
