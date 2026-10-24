# Your first solve

[Documentation index](../README.md) · [CLI reference](cli.md) · [Python API](python.md)

Build the CLI, solve a small instance, and choose the result your application
needs. Run every command below from the repository root. The example inputs
are included in the checkout.

## Build the CLI

Install Rust/Cargo, Git, and a C++14 compiler, then clone with submodules:

```bash
git clone --recurse-submodules https://github.com/lseman/baldes.git
cd baldes
cargo build --release --features clp
./target/release/baldes --help
```

The `clp` feature supplies the routing master and CLI. A default
`cargo build --release` builds the core library. For optional backends and
Make configuration, see [Build and installation](build.md).

## Choose a solve mode

| Goal | Mode | What to expect |
| --- | --- | --- |
| Get a feasible routing plan | `hgs` | A heuristic solution; no optimality proof |
| Inspect the root LP relaxation | `solve` | Route columns, potentially fractional, and lower-bound information |
| Search for an integer routing solution | `solve --tree` | An incumbent when found, a bound, and a remaining optimality gap |

Start with a small instance or customer subset. A time limit can stop a run
before convergence; it does not promise an optimal solution.

## Find a feasible solution

Run Hybrid Genetic Search on the bundled 32-node CVRP instance:

```bash
./target/release/baldes hgs instances/cvrplib/A-n32-k5.vrp --time 10
```

Use this mode when you need routes quickly. HGS searches for a feasible
integer routing plan; its objective does not prove a lower bound or optimality.

## Solve a root relaxation

Keep ten customers from a Solomon instance with time windows:

```bash
./target/release/baldes solve instances/Solomon/R101.txt \
  --customers 10 --time 60 --verbose --json root-result.json
```

The progress table goes to stderr; `root-result.json` stores the result for
inspection. Look at the solve status, LP objective, selected bound, and route
values. Root route values may be fractional, so these columns are not
necessarily a dispatchable routing plan.

A root status of `Optimal` means the LP relaxation converged. It does not
establish an optimal integer solution. The default bound mode is numerical;
independent bound validation is available with `--bound-mode certified`.
An unknown selected bound is `null` in JSON. See the
[CLI guide](cli.md) for the bound fields and validation behavior.

## Search for an integer solution

Add `--tree` to explore the branch-and-price tree:

```bash
./target/release/baldes solve instances/Solomon/R101.txt \
  --customers 10 --tree --time 60 --verbose
```

An incumbent is a feasible integer solution. The lower bound and incumbent
define the remaining gap. Tree `Optimal` means that gap closed within the
configured tolerances. If the run reaches a limit, inspect its status and gap
before treating the incumbent as optimal. See the [tree guide](../solver/tree.md)
for node limits, branching, and inherited state.

## Keep distance conventions consistent

Routing commands default to `--rounding tenth`, truncating distances to tenths.
Use `--rounding nearest` for TSPLIB nearest-integer distances, or
`--rounding none` for unrounded Euclidean distances. Explicit distance matrices
retain their supplied weights.

When comparing an objective with a published result, match the distance
convention and customer count first. BALDES root benchmarks use
`--rounding none`; see the [benchmark setup](../benchmarks/root.md).

## Common setup fixes

| Symptom | Next step |
| --- | --- |
| Clp or CoinUtils sources are missing | Run `git submodule update --init vendor/Clp vendor/CoinUtils`, then rebuild with `--features clp`. |
| `target/release/baldes` is missing | Run the CLI build above from the repository root. |
| An instance cannot be opened | Run from the repository root and use `instances/Solomon/R101.txt` or an absolute path. |
| A root result contains fractional routes | Use HGS for a heuristic integer plan or `solve --tree` for integer search. |

## Continue with your model

- [Python API](python.md): construct customers in memory and inspect typed root results.
- [CLI reference](cli.md): cuts, logging, distance modes, and backend selection.
- [Multi-pricing](../extensions/mp.md): coordinate multiple route families in one master.
- [Options reference](../reference/options.md): supported pricing controls and their scope.
