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
  basic-cut removal and artificial-column penalties; optional native IPM master solves.
- Rounded capacity and Kohl 2-path cuts, with time-window feasibility checks;
  strong k-path and route-load knapsack cuts integrated with pricing,
  enumeration, and inherited master rows.
- Solomon VRPTW and CVRPLIB CVRP input, HGS heuristic, initial-column seeding, and root solving.
- Dynamic NG strengthening from master cycles at configurable separator priority.

The core solver uses only the Rust standard library. The `clp` feature links
COIN-OR Clp through its C interface for the restricted master. The optional
`highs` feature adds a HiGHS LP backend and integer solving.

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
| `--no-enumeration` | no reduced-cost fixing or enumeration |
| `--upper-bound X` | initial incumbent for fixing |
| `--no-stabilization` | no dual-price smoothing |
| `--master MODE` | `clp` (default), `highs` (requires HiGHS), or native `ipm`; IPM disables stabilization |
| `--hgs-time S` | positive HGS search budget before routing solve; separate from `--time` |
| `--hgs-columns MODE` | `best` plan (default), `all` distinct feasible routes, or `N` cheapest new routes |
| `--dynamic-ng P` | strengthen NG memories from master cycles at separator priority P |
| `--option NAME=VALUE` | set a supported BALDES pricing option (repeatable) |
| `--verbose`, `-v`, `--log` | solver progress table on stderr, every 10 CG iterations |
| `--log-interval N` | enable progress every N CG iterations; N must be positive |
| `--iterations N` | master iteration limit |
| `--time S` | routing solve or standalone HGS time limit in seconds |
| `--json FILE` | write root LP/history or HGS solution results |
| `--population-size N` | HGS minimum population size (default: 25) |
| `--nb-iter N` | HGS iterations without improvement before stopping/restarting |
| `--seed N` | HGS random seed (default: 0) |
| `--help`, `-h` | show all CLI options, including tree controls |

Solomon instances are in [`instances/`](instances/), and a few CVRPLIB ones in
[`instances/cvrplib/`](instances/cvrplib/).

Enable `--verbose` for an aligned progress table with the restricted-master
objective, certified lower bound, incumbent, gap, reduced cost, column counts,
pricing phase, and elapsed time. Use `--log-interval 5` for more frequent output.

For advanced routing, see the [branch-and-price tree guide](docs/tree.md) and
[enumerated MIP guide](docs/enumerated-mip.md). Build checks and test commands
live in the [validation guide](docs/testing.md); benchmark results are in
[Root benchmarks](docs/benchmarks.md).

For named pricing controls, see the [BALDES option reference](docs/options.md).
Public option names start with `BALDES`, for example
`--option BALDESlabelDominanceStrategy=1`. Rust exposes supported metadata in
`OPTION_TABLE`; Python exposes `Solver.option_names()` and
`Solver.option_support(name)`. The public list contains 39 supported controls,
plus two enumerated-MIP controls when both `clp` and `highs` are enabled.
Scope can depend on the operation or configured cut family.

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
For local development, `maturin develop --release` installs the mixed package.
See the [Python validation guide](docs/testing.md#python) for test commands.

Strong k-path and route-load knapsack cuts are available in the Rust root API
through `with_kpath_cuts` and `with_knapsack_cuts`. The latter takes original
integer resource rows; `KnapsackCutConfig::with_resource_indices` maps original
resource IDs to solver coordinates. Load-dependent costs and that mapping pass
through pricing, enumeration, and inherited master rows.

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
