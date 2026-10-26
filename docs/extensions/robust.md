# Robust vehicle routing

The production robust VRPTW implementation uses the ordinary bucket graph
labeling engine and standard root/tree solvers. Labels carry budgeted demand
and travel-time states in addition to their nominal bucket resources. Forward
and backward extension, dominance, and concatenation use those robust states.

## CLI

```sh
cargo build --release --features clp
target/release/baldes solve instances/Solomon/R112.txt \
  --customers 25 --tree --cuts --robust 1,2,0.5,0.5
```

The format is `--robust GAMMA_DEMAND,GAMMA_TIME,ALPHA_DEMAND,ALPHA_TIME`:

- `GAMMA_DEMAND` bounds how many demands can deviate per route.
- `GAMMA_TIME` bounds how many travel times can deviate per route.
- Demand deviations are `trunc(ALPHA_DEMAND * demand)`.
- Travel-time deviations are `0.1 * trunc(10 * ALPHA_TIME * travel_time)`
  in instance units.

For both budgets equal to one, use `--robust 1,1,0.5,0.5`; for both equal to
two, use `--robust 2,2,0.5,0.5`. A zero budget disables that uncertainty.
Robust Solomon input requires time windows and `--rounding tenth`, the default.
Remove `--tree` for a root relaxation solve.

## Rust and the example

Build robust data with `extensions::robust::Problem::from_instance` and
`Uncertainty`, or supply `Node` and `Travel` data directly. Then call
`Problem::pricing_graph` (or `pricing_graph_with_memory`), construct a
`master::routing::RoutingContext`, and use `master::routing::solve` or
`master::tree::solve`. The graph retains the budgeted resources throughout
pricing; no separate robust master is needed.

The example uses this production path and prints a 36-row root LP sweep:

```sh
cargo run --release --features clp --example robust_c201 -- \
  instances/Solomon/C201.txt 15 0.5
```

Arguments are the instance, customer prefix, and relative deviation for both
uncertainties. Each budget ranges over `0,1,2,3,5,10`. The example uses full
customer NG memories to price elementary routes, with cuts and enumeration
disabled. It shares the CLI's Solomon conversion, including integer-scaled
time resources. CSV status names come from the standard CG solver; `Optimal`
denotes completion of the root LP, not an integer optimality claim. `lp_bound`
is `NaN` when artificial activity remains positive or the solve is incomplete.

## Independent validation reference

`extensions::robust::reference` retains the independent forward-only elementary
labeler and minimal LP master for small-instance checks:

- `reference::price` compares pricing minima with the bucket graph engine.
- `reference::RobustMaster` compares elementary LP objectives.
- `Problem::replay` checks individual routes against the uncertainty recurrences.

Production code should use the bucket graph path above. The former root-level
`RobustMaster`, pricing types, and `Problem::price` remain as deprecated
compatibility entry points. New validation code should import `reference`
explicitly. Removing compatibility entry points is deferred until the full
benchmark sweep and migration checks are complete.

Regression tests compare forward-only and bidirectional bucket pricing against
both brute-force enumeration and the independent reference, including sparse
asymmetric arcs and nonzero depot indices. Separate tests check every route
split for robust concatenation and compare bucket-root LPs with the reference
master.

## Table 7 solver sweep

```sh
cargo build --release --features clp --bin baldes
python3 scripts/benchmarks/benchmark_robust.py \
  --output results/robust-table7 --jobs 4 --time 60
```

The default grid uses the first 25 customers of C1/R1/RC1 Solomon instances,
excluding R101–R104 and RC105 as specified in the local paper
`papers/munari_robust.pdf`, pages 24–25. Each of the 24 remaining instances
gets one nominal solve and 54 uncertainty settings: demand-only, time-only,
and joint uncertainty, each with Γ = 1…6 and α = 0.10, 0.25, 0.50.

This is a solver screening pass with a 60-second limit per case. The paper's
limit was 3,600 seconds; use `--time 3600` to match it. `--instances R112`
selects a smaller smoke sweep. The output directory must be new so earlier
results are preserved.

The runner copies the tested binary and records its hash, source/input hashes,
settings and commands. It writes per-case logs, solver JSON, independent replay
results, `results.csv`, and `summary.json`. Replay checks exact customer cover,
fleet size, route costs, worst-case demand, and worst-case service-start times
in integer tenth units. The summary also detects contradictions between lower
bounds and feasible incumbents as uncertainty increases. `all_optimal` requires
all cases to finish `Optimal` with valid incumbents, closed gaps, and no such
contradictions. Time limits and incomplete solves remain unresolved.

This sweep does not perform the paper's 10,000-realization Monte Carlo risk
simulation and is not a reproduction of its risk estimates.
