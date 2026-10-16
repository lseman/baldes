# Porting status

BALDES 2.0 is a Rust port of the C++ OpenRSCP RCSP engine. The C++ OpenRSCP
code is the reference implementation; it is not part of this repository. The previous C++ BALDES code is in `legacy/` and is
not used by the Rust crate.

## Module map

| C++ responsibility | Rust module |
| --- | --- |
| Errors, logging, recovered option defaults | `src/core/` |
| Physical graph and resource bounds | `src/graph/mod.rs` |
| Local NG coordinates and arc transfer | `src/graph/ng.rs` |
| NG cycle detection and strengthening | `src/graph/relaxation.rs` |
| Bucket grids, auxiliary graph, SCCs, jumps, refinement | `src/graph/buckets.rs` |
| Stable ancestry and dominance insertion | `src/labeling/storage.rs` |
| Grouped candidate pool | `src/labeling/pool.rs` |
| Directional bucket scheduling and joins | `src/labeling/engine.rs` |
| Phase schedules and pricing state | `src/pricing/` |
| Pairing frontiers and fixing | `src/pricing/fixing.rs`, `src/labeling/engine.rs` |
| Full-memory elementary enumeration and inspection | `src/enumeration/mod.rs` |
| Rank-one cut registry, separator adapter, route coefficients | `src/cuts/mod.rs` |
| Rank-one separation (1/3-row, neighbourhood 4/5-row) | `src/cuts/separation.rs` |
| Local-search and tuple-based 4+-row separation | `src/cuts/local_search.rs`, `src/cuts/tuple.rs` |
| Limited- and full-memory cut label state | `src/cuts/state.rs` |
| Dual smoothing and misprice updates | `src/master/stabilization.rs` |
| Clp ownership and sparse master operations | `src/master/clp.rs` |
| Column-generation loop with cut rounds | `src/master/column_generation.rs` |
| Route/master integration and rank-one driver | `src/master/root.rs` |
| Solomon model and command | `src/vrptw.rs`, `src/bin/vrptw.rs` |
| No-cut C API entry points | `src/api/mod.rs` |

## Rank-one cuts

Separation follows the C++ `RankOneSeparation.h`, `LocalSearchRankOneSeparation.h`
and `TupleRankOneSeparation.h`. The candidate set reproduces libstdc++'s
red-black tree insertion (`src/cuts/rbtree.rs`), because its 1e-6 tolerant
comparator is not transitive and the insertion position depends on the tree
shape. Ties left by `std::sort` follow a port of libstdc++'s introsort
(`src/cuts/stdsort.rs`). Packing local search uses its own `minstd_rand0`
(seed 1); covering local search calls the C library `rand()`, as the binary
does.

Pricing labels carry twenty packed cut words and an additional cost; full
dominance, bucket keys, joins, fixing and completion bounds read them.
Enumeration labels carry sixty full-memory words, and stored paths are
repriced from physical incidence with the active cuts. The root driver follows
`_solve_root_clp` in the C++ Python package: cut rows, rounded cut duals and
pricing order, basic-cut removal, cut artificials, separation phases and
tailing off.

Strong k-path cuts, capacity cuts and the cut C API entry points are not
ported.

## Not ported

Specialized cumulative and pickup-delivery resources (rejected by the C++
native API as well), arbitrary extension callbacks, Python bindings, and the
C API entry points for cuts.

## Validation

`tools/benchmark_native.py` runs the same Solomon root solves with this crate,
the C++ OpenRSCP build and its reference solver, with identical NG
memories and fixing settings.
