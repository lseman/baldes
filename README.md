<div align="center">

<img src="assets/baldes_logo.svg" width="400" alt="BALDES"/>

<br>

<img src="https://img.shields.io/badge/version-2.0-blue" alt="version 2.0" />
<img src="https://img.shields.io/badge/language-Rust-orange" alt="Rust" />
<img src="https://img.shields.io/github/license/lseman/baldes.svg" alt="License" />

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
- A Solomon VRPTW model and a root-solve command.

The core solver uses only the Rust standard library. The `clp` feature links
COIN-OR Clp through its C interface for the restricted master.

## Build

The library without the master LP:

```bash
cargo build --release
```

With the Clp master and the `baldes` command:

```bash
export BALDES_CLP_LIB_DIR=/path/to/clp/archives
cargo build --release --features clp
```

`BALDES_CLP_LIB_DIR` must contain the static archives
`libopenrscp_clp_solver.a` and `libopenrscp_coinutils.a`. The C++ OpenRSCP
Python build produces them in `python/build/openrscp_core`. Using the same Clp
build as the reference keeps degenerate master solutions, and therefore
trajectories, comparable.

## Run

```bash
./target/release/baldes vrptw instances/R101.txt --customers 25 --rank1
./target/release/baldes cvrp /path/to/P-n16-k8.txt --capacity-cuts
```

The first argument selects the instance type. Both commands read Solomon-format
files, including CVRP instances converted to that format. `vrptw` enforces time
windows and service times; `cvrp` ignores them. Run `baldes --help` for usage.

| Option | Meaning |
| --- | --- |
| `--customers N` | keep the first N customers |
| `--ng N` | NG neighbourhood size (default 8) |
| `--reference-ng` | the N nearest customers, ties by index (the comparison setup) |
| `--rank1` | separate limited-memory rank-one cuts |
| `--capacity` / `--capacity-cuts` | separate native rounded capacity cuts in the Clp root master |
| `--no-enumeration` | no reduced-cost fixing or enumeration |
| `--upper-bound X` | initial incumbent for fixing |
| `--no-stabilization` | no dual-price smoothing |
| `--iterations N`, `--time S` | master iteration and time limits |
| `--json FILE` | write the root result, history and positive routes |

Solomon instances are in [`instances/`](instances/).

## Python: pybaldes

The Python distribution and import are both named `pybaldes`. It contains the
low-level `Solver`, `Route`, and `Phase` bindings and direct Rust-backed VRPTW/CVRP
root solving. The default build includes the Clp master and requires the same
archives as the Rust CLI:

```bash
export BALDES_CLP_LIB_DIR=/path/to/clp/archives
python -m pip install .
```

```python
import pybaldes

result = pybaldes.solve_root(
    "instances/R101.txt",
    problem_type="vrptw",  # or "cvrp" for converted CVRP files
    customers=25,
    capacity_cuts=True,
    rank_one_cuts=True,
    time_limit=120,
)
print(result.status, result.lp, result.bound, result.cuts)
for route in result.routes:
    print(route.value, route.cost, route.nodes)
```

`Instance.from_file(path)` reads Solomon-format files. You can also build inputs
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
bound, matching the Rust CLI. Native CVRPLIB `.vrp` input is not supported yet.

Options include `ng`, `reference_ng`, `rounding` (`none`, `nearest`, `tenth`),
`stabilize`, `reduced_cost_fixing`, `upper_bound`, `max_iterations`,
`max_cut_rounds`, and a `pricing_options` dictionary of native option values.
Disabling reduced-cost fixing also disables enumeration. Results include positive
LP routes, duals, iteration history, cut/fixing counts, and timers;
`result.to_dict()` and `result.to_json()` export them. Unknown/infinite values
become `null` in JSON. Cut counts include generated rows later retired.
These are root LP relaxations; `Optimal` does not certify an integer VRP optimum.

For pricing alone, build without Clp:

```bash
python -m pip install . --config-settings=build-args=--no-default-features
```

`pybaldes.has_clp` reports backend availability; root solving raises a clear error
in a pricing-only build. The wheel uses Python's stable ABI for Python 3.9+.
For local development, `maturin develop --release` installs the mixed package;
run `python -m pytest python/tests` to test a build with Clp.

## Not in 2.0 yet

Strong k-path root integration, the C API entry points for cuts,
Native CVRPLIB input, EVRP front ends, branching, the heuristics, and remaining
1.x Python APIs. [PORTING.md](PORTING.md)
maps each Rust module to its C++ counterpart.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/` | the Rust crate (`baldes`) and `src/bin/baldes.rs` |
| `tests/` | integration tests |
| `python/rust/` | Rust/PyO3 extension crate |
| `python/pybaldes/` | Python package and typed routing API |
| `python/tests/` | installed-package integration tests |
| `tools/` | comparison scripts against the C++ reference |
| `instances/` | Solomon VRPTW instances |
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

1. Ruslan Sadykov, Eduardo Uchoa, Artur Alves Pessoa. "A Bucket Graph Based Labeling Algorithm for Vehicle Routing." Transportation Science, 2021. https://doi.org/10.1287/trsc.2020.0985
2. Diego Pecin, Artur Pessoa, Marcus Poggi, Eduardo Uchoa, Haroldo Santos. "Limited memory rank-1 cuts for vehicle routing problems." Operations Research Letters 45.3 (2017): 206-209. https://doi.org/10.1016/j.orl.2017.02.006
