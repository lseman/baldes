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

With the Clp master and the `baldes-vrptw` command:

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
./target/release/baldes-vrptw instances/R101.txt --customers 25 --rank1
```

| Option | Meaning |
| --- | --- |
| `--customers N` | keep the first N customers |
| `--ng N` | NG neighbourhood size (default 8) |
| `--reference-ng` | the N nearest customers, ties by index (the comparison setup) |
| `--rank1` | separate limited-memory rank-one cuts |
| `--no-enumeration` | no reduced-cost fixing or enumeration |
| `--upper-bound X` | initial incumbent for fixing |
| `--no-stabilization` | no dual-price smoothing |
| `--iterations N`, `--time S` | master iteration and time limits |
| `--json FILE` | write the root result, history and positive routes |

Solomon instances are in [`instances/`](instances/).

## Not in 2.0 yet

Strong k-path and rounded capacity cuts, the C API entry points for cuts,
CVRP and EVRP front ends, branching, the heuristics, and the Python bindings
of 1.x. [PORTING.md](PORTING.md)
maps each Rust module to its C++ counterpart.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/` | the Rust crate (`baldes`) and `src/bin/vrptw.rs` |
| `tests/` | integration tests |
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
