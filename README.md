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
  inspection pricing at the root and descendant nodes.
- Limited-memory rank-one (subset-row) cuts: 1- and 3-row separation,
  local-search and tuple-based 4- and 5-row separation, cut memory, and
  limited- and full-memory cut state in pricing and enumeration labels.
- Dual-price smoothing, and a Clp-backed restricted master with cut rounds,
  basic-cut removal and artificial-column penalties.
- Rounded capacity and Kohl 2-path cuts, with time-window feasibility checks;
  strong k-path and route-load knapsack cuts integrated with pricing,
  enumeration, and inherited master rows.
- A single-network best-bound branch-and-price tree with Ryan–Foster,
  vehicle-count and directed arc-flow branching, packing-set assignment rows,
  shared elementary columns, and a restricted-master integer heuristic.
- Isolated node snapshots carrying the Clp model/basis, active cuts, branching
  rows, fixed bucket arcs, pricing state, and retained enumeration.
- Optional HiGHS MIP solves over complete enumerated node masters, preserving
  inherited branches, active cuts, and route bounds.
- Phased strong branching (100 → 3 → 1 candidates), native product and
  tree-depth scores, and bounded strong round-up diving with discrepancy limits.
- Solomon VRPTW and CVRPLIB CVRP input, HGS heuristic, and root-solve.
- Dynamic NG strengthening from master cycles at configurable separator priority.

The core solver uses only the Rust standard library. The `clp` feature links
COIN-OR Clp through its C interface for the restricted master. The optional
`highs` feature adds a HiGHS LP backend and integer solving; the routing tree
uses Clp for LPs and HiGHS for its optional enumerated MIP.

## Build

For an interactive kernel-style build configuration:

```bash
make config   # Configure backends, CLI, build profile and compiler options
make          # Build with the saved .config
```

Use arrows and Space/Enter to change settings, **S** to save, and **Q** to
cancel. `make menuconfig` is an alias. See [build configuration](docs/build.md)
for all options, prerequisites and noninteractive builds. Direct Cargo builds
remain available as below.

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

To include HiGHS integer solving, install a HiGHS shared library and build both
features:

```bash
cargo build --release --features clp,highs
```

HiGHS linking searches `HIGHS_DIR/lib`, then `vendor/highs/lib`, then
`pkg-config` and the system linker path. The C interface currently expects a
32-bit `HighsInt` build. The `baldes` CLI and routing tree require `clp`;
`highs` alone provides the library's HiGHS backend and generic MIP API.

## Run

```bash
./target/release/baldes vrptw instances/R101.txt --customers 25 --rank1
./target/release/baldes cvrp instances/cvrplib/A-n32-k5.vrp --capacity-cuts
./target/release/baldes vrptw instances/R101.txt --customers 10 --two-path-cuts --tree --time 60
./target/release/baldes hgs instances/R101.txt --population-size 50 --nb-iter 5000
```

The first argument selects the instance type: `vrptw` (time windows), `cvrp`
(capacitated, no time windows), or `hgs` (Hybrid Genetic Search heuristic).
The routing commands read Solomon-format files and CVRPLIB (TSPLIB) `.vrp`
files; the format is detected from the content. `vrptw` enforces time windows and service times;
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
| `--branching MODE` | tree families: `native` (default), `ryan-foster`, `vehicles`, or `arc` |
| `--no-strong-branching`, `--no-diving` | disable the corresponding tree mechanism |
| `--enumerated-mip` | optional HiGHS solve over complete enumerated node masters (`--tree`, `clp,highs` features) |
| `--no-mip-heuristic`, `--mip-nodes N` | disable or budget restricted-master integer search |
| `--no-enumeration` | no reduced-cost fixing or enumeration |
| `--upper-bound X` | initial incumbent for fixing |
| `--no-stabilization` | no dual-price smoothing |
| `--dynamic-ng P` | strengthen NG memories from master cycles at separator priority P |
| `--option NAME=VALUE` | set a supported BALDES pricing option (repeatable) |
| `--iterations N`, `--time S` | master iteration and time limits |
| `--json FILE` | write results: root LP/history, tree incumbent/bounds/node counters, or HGS solution |
| `--population-size N` | HGS minimum population size (default: 25) |
| `--nb-iter N` | HGS iterations without improvement before stopping/restarting |
| `--seed N` | HGS random seed (default: 0) |

Solomon instances are in [`instances/`](instances/), and a few CVRPLIB ones in
[`instances/cvrplib/`](instances/cvrplib/).

For named pricing controls, see the [BALDES option reference](docs/options.md).
Public option names start with `BALDES`, for example
`--option BALDESlabelDominanceStrategy=1`. Rust exposes supported metadata in
`OPTION_TABLE`; Python exposes `Solver.option_names()` and
`Solver.option_support(name)`. The public list contains 39 supported controls,
plus two enumerated-MIP controls when both `clp` and `highs` are enabled.
Scope can depend on the operation or configured cut family.

## Branch-and-price tree

`--tree` solves the integer routing model with the configured fleet bound. Tree JSON includes validated incumbent routes, a global lower bound,
node counts, pricing/fixing statistics, MIP work, and status. `Optimal` means
the requested numerical gap is closed; `TimeLimit`, `NodeLimit`, and
`Incomplete` retain bounds for unfinished nodes. The cost-only `--upper-bound`
is a root/fixing option; the tree obtains its incumbent from feasible routes.

The Rust API is `master::tree::solve(problem, TreeOptions)`. Children fork the
parent's solved Clp model and basis, carry active cuts, branching rows, fixed
bucket arcs, options and retained enumeration, and rebuild labels for the new
duals. Siblings own independent state. Fixing and enumeration run below the
root when enabled; branch-distinct arc paths remain separate.

`--branching native` enables vehicle-count, packing-set assignment,
Ryan–Foster together/apart, and directed arc-flow families in that adapter order.
Ryan–Foster is enforced inside pricing and retained enumeration. The current
model has one pricing network: customer cover equations fix assignment to that
network, so assignment rows are available but provide no fractional assignment
candidate. Multiple vehicle types require multi-network modeling.

Strong branching screens 100 candidates with restricted LPs, prices three
survivors heuristically, then evaluates one with exact pricing. Its product and
tree-depth scores rank the survivors. Strong round-up diving uses
priced route-fixing trials with depth, discrepancy, time and node budgets.
Both mechanisms are enabled by default. Heuristic trial objectives and dive
bounds do not certify the original node.

The restricted-master integer heuristic searches generated elementary routes
with its own time/node budgets. Its optimum is not a lower bound for the full
routing problem. The separate HiGHS path below can certify an enumerated node
within its saved enumeration coverage.

### HiGHS over enumerated routes

After building with `clp,highs`, enable end-of-node MIP explicitly:

```bash
./target/release/baldes vrptw instances/R101.txt --customers 10 \
  --tree --enumerated-mip --time 60 --json tree.json
```

Or configure it in Rust:

```rust
use baldes::master::{solvers::mip::MipOptions, tree::TreeOptions};

let options = TreeOptions {
    enumerated_mip: Some(MipOptions::default()),
    ..TreeOptions::default()
};
```

HiGHS receives an independent integer model with every active branch/cut row,
full-memory route coefficients, and existing route upper bounds. Artificial
columns are excluded, and returned route witnesses are validated. The source
Clp LP and sibling snapshots remain untouched.

The route-count gate requires complete enumeration. Its normal threshold
(`BALDESmaxNumOfEnumSolutionsForMIP`, default 10,000) raises the end-of-node
threshold (`BALDESmaxNumOfEnumSolsForEndOfNodeMIP`, default 0) to at least that
value. `MipOptions` defaults to 2,000,000 nodes and 360,000 seconds, capped in
the tree at 98% of remaining global time. These budgets are separate from
`--mip-nodes`, which controls the restricted-column heuristic.

Optimality, infeasibility, objective cutoff, time/node limits, and incomplete
results remain distinct. A proof over routes restricted by an incumbent is
used only within that saved coverage. Tree JSON reports
`enumerated_mip_calls` and `enumerated_mip_nodes`; `mip_nodes` includes both
integer-solving paths. `--no-enumeration` leaves no complete universe to solve.
The current MIP call runs after node evaluation; the native immediate handoff
inside column generation remains unimplemented.

Node preprocessing propagates original arc-flow bounds and inherited capacity
cut rows, applies transitive Ryan–Foster deductions, rejects contradictory
nodes, and carries zero-arc fixing into pricing and enumeration. The current
model is single-network; candidate-history ranking and cut callbacks during
integer solving remain future work.
See the [enumerated MIP guide](docs/enumerated-mip.md) for implementation
limits. Python currently exposes root LP solving; use Rust or the CLI for
the tree.

## Root benchmarks

Root relaxations of Solomon VRPTW instances (100 customers). Both solvers
use the same split-depot graph with unrounded
Euclidean distances, the same NG neighbourhoods (the 8 nearest customers), the
Clp master and no reduced-cost fixing or enumeration. Times are the solvers' own
root timers, medians of three runs, each solver running alone on an AMD
Ryzen 9 9950X. The reference run uses one thread; BALDES runs the forward and backward
labeling of exact pricing on two threads, so it is timed on one and two cores.

![Root relaxation solve time: BALDES vs VRPSolver / BaPCod](docs/benchmark.svg)

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
`max_cut_rounds`, and a `pricing_options` dictionary of BALDES option values.
Disabling reduced-cost fixing also disables enumeration. Results include positive
LP routes, duals, iteration history, cut/fixing counts, and timers;
`result.to_dict()` and `result.to_json()` export them. Unknown/infinite values
become `null` in JSON. Cut counts include generated rows later retired.
These are root LP relaxations; `Optimal` does not certify an integer VRP optimum.

Python root solves enable Kohl with `two_path_cuts=True`. This attaches time
resource 0 for capacity-cut registration;
ordinary capacity cuts retain the default resource -1. Both use the capacity
family's priority 3 at root and 1 below it, complement form and persistent subset pool.

For pricing alone, build without Clp:

```bash
python -m pip install . --config-settings=build-args=--no-default-features
```

`pybaldes.has_clp` reports backend availability; root solving raises a clear error
in a pricing-only build. The wheel uses Python's stable ABI for Python 3.9+.
For local development, `maturin develop --release` installs the mixed package;
run `python -m pytest python/tests` to test a build with Clp.

Strong k-path and route-load knapsack cuts are available in the Rust root API
through `with_kpath_cuts` and `with_knapsack_cuts`. The latter takes original
integer resource rows; `KnapsackCutConfig::with_resource_indices` maps original
resource IDs to solver coordinates. Load-dependent costs and that mapping pass
through pricing, enumeration, and inherited master rows.

## Validation

```bash
cargo test --workspace --all-features
cargo check --workspace --all-targets --all-features
```

These commands require the HiGHS shared library. The latest all-feature
workspace suite passed **158 tests**, including transfer from Clp to HiGHS,
inherited branches and active cut families, proof/limit handling, and 50 small
integer models checked by exhaustive enumeration. Clp-only option and CLI
checks, HiGHS-only tests, and default/all-feature build checks also passed.

## TODO

- heterogeneous fleets/multiple pricing networks
- accumulated-resource branching
- pickup-and-delivery

## Repository layout

See the [documentation index](docs/README.md) for usage and implementation
guides, and the [source layout guide](docs/source-layout.md) for module
responsibilities and subfolders.

| Path | Contents |
| --- | --- |
| `src/` | the Rust crate (`baldes`) and `src/bin/baldes.rs` |
| `tests/` | integration tests |
| `python/rust/` | Rust/PyO3 extension crate |
| `python/pybaldes/` | Python package and typed routing API |
| `python/tests/` | installed-package integration tests |
| `.github/scripts/` | documentation build and checks |
| `instances/` | Solomon VRPTW instances; `cvrplib/` holds CVRPLIB samples |
| `assets/` | logo |
| `landing/` | landing page and generated documentation website |

Build the website documentation with `.github/scripts/compile_docs.py`; see the
[website publishing guide](docs/website.md) for setup and preview commands.

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
4. Talal Vidal. "Hybrid genetic search for the CVRP: Open-source implementation and SWAP* neighborhood." Computers & Operations Research 140 (2022): 105643. https://doi.org/10.1016/j.cor.2021.105643
