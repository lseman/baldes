# Command-line usage

[Documentation index](README.md) · [Build configuration](build.md) · [Python](python.md)

Commands below run from the repository root after building with `--features clp`.

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
`./target/release/baldes --help` for usage.

## Options

| Option | Meaning |
| --- | --- |
| `--customers N` | keep the first N customers |
| `--ng N` | NG neighbourhood size (default 8) |
| `--reference-ng` | the N nearest customers, ties by index (the comparison setup) |
| `--rounding MODE` | distances `none`, `nearest`, `tenth` or `ceil` (default: the file's convention) |
| `--rank1` | separate limited-memory rank-one cuts |
| `--capacity` / `--capacity-cuts` | separate native rounded capacity cuts in the Clp root master |
| `--two-path-cuts` | enable capacity cuts with Kohl's time resource attached (VRPTW only) |
| `--tree` | run branch-and-price for integer routing |
| `--nodes N` | tree node limit (default 10,000) |
| `--branching MODE` | `native` (default), `ryan-foster`, `vehicles`, or `arc` |
| `--no-strong-branching` | use fractionality instead of phased strong branching |
| `--no-diving` | disable strong diving |
| `--no-mip-heuristic` | disable restricted-master integer search |
| `--mip-nodes N` | node budget per restricted-master integer-search call |
| `--enumerated-mip` | solve complete enumerated node masters with HiGHS; requires `--tree` and the `highs` feature |
| `--no-enumeration` | no reduced-cost fixing or enumeration |
| `--upper-bound X` | initial incumbent for fixing |
| `--no-stabilization` | no dual-price smoothing |
| `--master MODE` | `clp` (default), `highs` (requires HiGHS), or native `ipm`; IPM disables stabilization |
| `--bound-mode MODE` | `numerical` (default) uses pricing bounds with tolerances; `certified` requires independent original-domain bound validation |
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

Root `Optimal` means the LP relaxation converged; tree `Optimal` means the
integer routing gap closed within the configured tolerances. HGS supplies
feasible solutions without an optimality proof. Root LP route values can be
fractional, and NG routes can revisit customers.

Normal root and tree solves use numerical bounds and skip the full certificate
search. Use `--bound-mode certified` for independently validated bounds. That
mode attempts validation when an exact pass can close pricing or the incumbent
gap; failure leaves its selected bound unknown. Route-feasibility checks and
exact pricing remain enabled in both modes.

JSON reports `bound_mode`, the selected `bound`, and separate `numerical_bound`
and `certified_bound` fields. An unknown bound is `null`. A numerical estimate
never substitutes for an unknown certified bound. A tree result labels the
guarantee of its global bound; it does not report a global numerical estimate
in certified mode. Root results retain both estimates when available.

Solomon instances are in [`instances/`](../instances/), and a few CVRPLIB ones in
[`instances/cvrplib/`](../instances/cvrplib/).

Enable `--verbose` for an aligned progress table with the restricted-master
objective, selected lower bound, incumbent, gap, reduced cost, column counts,
pricing phase, and elapsed time. Use `--log-interval 5` for more frequent output.

For advanced routing, see the [branch-and-price tree guide](tree.md) and
[enumerated MIP guide](enumerated-mip.md). Build checks and test commands
live in the [validation guide](testing.md); benchmark results are in
[Root benchmarks](benchmarks.md).

For named pricing controls, see the [BALDES option reference](options.md).
Public option names start with `BALDES`, for example
`--option BALDESlabelDominanceStrategy=1`. Rust exposes supported metadata in
`OPTION_TABLE`; Python exposes `Solver.option_names()` and
`Solver.option_support(name)`. The public list contains 39 supported controls,
plus two enumerated-MIP controls when both `clp` and `highs` are enabled.
Scope can depend on the operation or configured cut family.
