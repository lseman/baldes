<div align="center">

<img src="assets/baldes_logo.svg" width="400" alt="BALDES"/>

<br>

<img src="https://img.shields.io/badge/version-2.0-blue" alt="version 2.0" />
<img src="https://img.shields.io/badge/language-Rust-orange" alt="Rust" />
<img src="https://img.shields.io/github/license/lseman/baldes" alt="License: MIT" />

</div>

# BALDES 2.0

BALDES is a branch-cut-and-price toolkit for vehicle routing built around a
bucket-graph labeling algorithm for resource-constrained shortest path (RCSP)
pricing. Version 2.0 is a complete rewrite in Rust.

The 2.0 solver is a Rust port of our C++ RCSP engine, OpenRSCP.

The C++ BALDES 1.x code remains in the git history before the 2.0 commit.

## What is in 2.0

- Bucket-graph labeling: resource and NG preparation, bucket grids, strongly
  connected component ordering, jump arcs, preceding-bucket dominance,
  bidirectional search with a moving half-way point, and symmetric joins.
- Staged pricing with the native heuristic phase schedule.
- Reduced-cost fixing on bucket arcs and elementary route enumeration with
  inspection pricing.
- Limited-memory rank-one (subset-row) cuts: 1- and 3-row separation,
  local-search and tuple-based 4- and 5-row separation, cut memory, and
  limited- and full-memory cut state in pricing and enumeration labels.
- Dual-price smoothing, and a Clp-backed restricted master with cut rounds,
  basic-cut removal and artificial-column penalties.
- Rounded capacity and Kohl 2-path cuts, with time-window feasibility checks.
- An initial best-bound branch-and-price tree with directed arc-flow
  branching, shared elementary columns and a restricted-master integer heuristic.
- Phased strong branching (100 → 3 → 1 candidates), native product and
  tree-depth scores, and bounded strong round-up diving with discrepancy limits.
- Solomon VRPTW and CVRPLIB CVRP input, HGS heuristic, and root-solve.
- Dynamic NG strengthening from master cycles at configurable separator priority.

The core solver uses only the Rust standard library. The `clp` feature links
COIN-OR Clp through its C interface for the restricted master.

## Build

The library without the master LP:

```bash
cargo build --release
```

With the Clp master and the `baldes` command:

```bash
git submodule update --init   # vendor/Clp and vendor/CoinUtils
cargo build --release --features clp
```

The `clp` feature compiles Clp 1.17.10 and CoinUtils 2.11 from the
[`vendor/`](vendor/) submodules; a C++14 compiler is needed.

## Run

```bash
./target/release/baldes vrptw instances/R101.txt --customers 25 --rank1
./target/release/baldes cvrp instances/cvrplib/A-n32-k5.vrp --capacity-cuts
./target/release/baldes vrptw instances/R101.txt --customers 10 --two-path-cuts --tree --time 60
./target/release/baldes hgs instances/R101.txt --population-size 50 --nb-iter 5000
```

The first argument selects the instance type: `vrptw` (time windows), `cvrp`
(capacitated, no time windows), or `hgs` (Hybrid Genetic Search heuristic). Both
commands read Solomon-format files and CVRPLIB (TSPLIB) `.vrp` files; the format
is detected from the content. `vrptw` enforces time windows and service times;
`cvrp` ignores them.
CVRPLIB instances have no time windows, so they are solved as `cvrp`. They use
the file's distance convention: TSPLIB `nint` for `EUC_2D`, `ceil` for
`CEIL_2D`, or the explicit `EDGE_WEIGHT_SECTION` (full or triangular). As for
Solomon files, the root fleet bound is the customer count. Route-length limits (`DISTANCE`) and service times are not supported. Run
`baldes --help` for usage.

| Option | Meaning |
| --- | --- |
| `--customers N` | keep the first N customers |
| `--ng N` | NG neighbourhood size (default 8) |
| `--reference-ng` | the N nearest customers, ties by index (the comparison setup) |
| `--rounding MODE` | distances `none`, `nearest`, `tenth` or `ceil` (default: the file's convention) |
| `--rank1` | separate limited-memory rank-one cuts |
| `--capacity` / `--capacity-cuts` | separate native rounded capacity cuts in the Clp root master |
| `--two-path-cuts` | enable capacity cuts with Kohl's time resource attached (VRPTW only) |
| `--tree`, `--nodes N` | branch-and-price; default node limit 10,000 |
| `--no-strong-branching`, `--no-diving` | disable the corresponding tree mechanism |
| `--no-mip-heuristic`, `--mip-nodes N` | disable or budget restricted-master integer search |
| `--no-enumeration` | no reduced-cost fixing or enumeration |
| `--upper-bound X` | initial incumbent for fixing |
| `--no-stabilization` | no dual-price smoothing |
| `--dynamic-ng P` | strengthen NG memories from master cycles at separator priority P |
| `--option NAME=VALUE` | set an RCSP pricing option (repeatable) |
| `--iterations N`, `--time S` | master iteration and time limits |
| `--json FILE` | write the root result, history and positive routes |
| `--population-size N` | HGS minimum population size (default: 25) |
| `--nb-iter N` | HGS iterations without improvement before stopping/restarting |
| `--seed N` | HGS random seed (default: 0) |

Solomon instances are in [`instances/`](instances/), and a few CVRPLIB ones in
[`instances/cvrplib/`](instances/cvrplib/).

## Comparison with the solver

Root relaxations of Solomon VRPTW instances (100 customers), BALDES against
the the solver binary. Both solve the same split-depot graph with unrounded
Euclidean distances, the same NG neighbourhoods (the 8 nearest customers), the
Clp master and no reduced-cost fixing or enumeration. Times are the solvers' own
root timers, medians of three runs, each solver running alone on an AMD
Ryzen 9 9950X. the solver is single-threaded; BALDES runs the forward and backward
labeling of exact pricing on two threads, so it is timed on one and two cores.

| Instance | Cuts | Root LP | Pricing calls | Cuts added | BaPCod (s) | BALDES 1 core (s) | BALDES 2 cores (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C101 | none | 828.937 | 82 | 0 | 0.09 | 0.058 | 0.057 |
| C102 | none | 828.937 | 133 | 0 | 0.39 | 0.38 | 0.36 |
| C103 | none | 828.065 | 138 | 0 | 1.13 | 1.18 | 1.11 |
| C201 | none | 591.557 | 98 | 0 | 0.16 | 0.12 | 0.12 |
| C202 | none | 591.557 | 1257 | 0 | 14.64 | 13.17 | 12.16 |
| R101 | none | 1636.389 | 95 | 0 | 0.07 | 0.046 | 0.045 |
| R101 | rank-one | 1641.509 | 121 | 225 | 1.07 | 1.10 | 1.09 |
| R102 | none | 1472.815 | 145 | 0 | 0.32 | 0.29 | 0.28 |
| R103 | none | 1211.107 | 220 | 0 | 0.91 | 0.91 | 0.89 |
| R103 | rank-one | 1213.624 | 289 | 343 | 1.98 | 1.90 | 1.83 |
| R201 | none | 1140.130 | 364 | 0 | 1.57 | 1.47 | 1.43 |
| R201 | rank-one | 1147.804 | 636 | 832 | 8.98 | 7.46 | 7.27 |
| R202 | none | 1020.646 | 470 | 0 | 5.92 | 5.98 | 5.66 |

On C101–C103, C201, C202 and R102, rank-one separation finds no violated cut,
so those runs equal the ones without cuts.

## Python: pybaldes

The Python distribution and import are both named `pybaldes`. It contains the
low-level `Solver`, `Route`, and `Phase` bindings and direct Rust-backed VRPTW/CVRP
root solving. The default build includes the Clp master, built from the same
`vendor/` submodules as the Rust CLI:

```bash
git submodule update --init
python -m pip install .
```

```python
import pybaldes

result = pybaldes.solve_root(
    "instances/R101.txt",
    problem_type="vrptw",  # or "cvrp" for CVRPLIB .vrp files
    customers=25,
    capacity_cuts=True,
    rank_one_cuts=True,
    time_limit=120,
)
print(result.status, result.lp, result.bound, result.cuts)
for route in result.routes:
    print(route.value, route.cost, route.nodes)
```

`Instance.from_file(path)` reads Solomon-format and CVRPLIB `.vrp` files. You can also build inputs
in memory:

```python
from pybaldes import Customer, Instance

instance = Instance("small", capacity=2, customers=[
    Customer(0, 0),  # depot, zero demand
    Customer(1, 0, demand=1),
    Customer(0, 1, demand=1),
])
result = instance.solve_root(problem_type="cvrp", capacity_cuts=True)
```

CVRP ignores time windows and service times; in-memory VRPTW data requires finite
`ready`/`due` windows, including the depot. `vehicles` is retained as input
metadata; the current root model uses the customer count as its fleet upper
bound, matching the Rust CLI.

Options include `ng`, `reference_ng`, `rounding` (`none`, `nearest`, `tenth`),
`stabilize`, `reduced_cost_fixing`, `upper_bound`, `max_iterations`,
`max_cut_rounds`, and a `pricing_options` dictionary of native option values.
Disabling reduced-cost fixing also disables enumeration. Results include positive
LP routes, duals, iteration history, cut/fixing counts, and timers;
`result.to_dict()` and `result.to_json()` export them. Unknown/infinite values
become `null` in JSON. Cut counts include generated rows later retired.
These are root LP relaxations; `Optimal` does not certify an integer VRP optimum.

Python root solves enable Kohl with `two_path_cuts=True`. This attaches time
resource 0, corresponding to generic capacity-cut registration;
ordinary capacity cuts retain the default resource -1. Both use the capacity
family's priority 3 at root and 1 below it, complement form and persistent subset pool.

### Initial branch-and-price tree

`--tree` solves the integer routing model and respects the instance's vehicle
limit. Its JSON contains an incumbent with validated routes, a global lower
bound, node counts and status. `Optimal` means the requested numerical gap is
closed; `TimeLimit`, `NodeLimit` and `Incomplete` preserve bounds for unfinished
nodes. The cost-only `--upper-bound` is a root/fixing option; the tree builds
its incumbent from feasible routes instead. Limits are checked between master
iterations and during pricing and integer search.

The Rust API is `master::tree::solve(problem, TreeOptions)`. The integer
heuristic searches the binary set-partitioning model over generated elementary
routes, with per-call time and node budgets. Its restricted optimum is never
used as a lower bound. Children price the duals of inherited arc-flow branch
rows. They rebuild their Clp master and separate cuts anew while sharing
feasible elementary columns. Strong branching screens 100 candidates with
restricted LPs, prices three survivors heuristically, then evaluates one with
exact pricing. Its product and tree-depth score formulas are ported from the
reference; heuristic scores never certify bounds. Strong round-up diving uses
priced route-fixing trials with depth, discrepancy, time and node budgets.
Dive bounds remain private to the heuristic subspace. Both mechanisms are
enabled by default in the tree and configurable through `TreeOptions`.

Full the solver tree-trajectory parity has not been established. Generator-history
ranking, basis/cut inheritance and a general-purpose MIP backend remain work;
the current restricted-master integer search solves the binary routing
set-partitioning model directly.
The Python API currently exposes root solving; use Rust or the CLI for the tree.

For pricing alone, build without Clp:

```bash
python -m pip install . --config-settings=build-args=--no-default-features
```

`pybaldes.has_clp` reports backend availability; root solving raises a clear error
in a pricing-only build. The wheel uses Python's stable ABI for Python 3.9+.
For local development, `maturin develop --release` installs the mixed package;
run `python -m pytest python/tests` to test a build with Clp.

## Not in 2.0 yet

Strong k-path root integration, EVRP front ends, tree basis/cut inheritance, and remaining
1.x Python APIs.

## TODO

- Cumulative resources
- Pickup and delivery

## Repository layout

| Path | Contents |
| --- | --- |
| `src/` | the Rust crate (`baldes`) and `src/bin/baldes.rs` |
| `tests/` | integration tests |
| `python/rust/` | Rust/PyO3 extension crate |
| `python/pybaldes/` | Python package and typed routing API |
| `python/tests/` | installed-package integration tests |
| `tools/` | comparison scripts against the C++ reference |
| `instances/` | Solomon VRPTW instances; `cvrplib/` holds CVRPLIB samples |
| `assets/` | logo |
| `landing/` | project landing page (deployed to GitHub Pages) |

## Citation

If you use BALDES in academic work, please cite:

```bibtex
@Misc{BucketGraphLabeling,
  author       = {Laio Oriel Seman and Pedro Munari and Teobaldo Bulh\~oes and Eduardo Camponogara},
  title        = {BALDES: a modern C++ Bucket Graph Labeling Algorithm for Vehicle Routing},
  howpublished = {\url{https://github.com/lseman/baldes}},
  year         = {2024},
  note         = {GitHub repository},
  urldate      = {2024-09-17},
  month        = sep
}
```

## License

MIT. See [LICENSE](LICENSE).

## References

1. Ruslan Sadykov, François Vanderbeck. "BaPCod - a generic branch-and-price code." Technical report HAL-03340548, Inria Bordeaux Sud-Ouest, 2021.
2. Ruslan Sadykov, Eduardo Uchoa, Artur Alves Pessoa. "A Bucket Graph Based Labeling Algorithm for Vehicle Routing." Transportation Science, 2021. https://doi.org/10.1287/trsc.2020.0985
3. Diego Pecin, Artur Pessoa, Marcus Poggi, Eduardo Uchoa, Haroldo Santos. "Limited memory rank-1 cuts for vehicle routing problems." Operations Research Letters 45.3 (2017): 206-209. https://doi.org/10.1016/j.orl.2017.02.006
