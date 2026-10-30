# Development scripts

Run commands from the repository root. Keep reusable tools here; write generated
binaries to `target/` and experiment output to `results/` or `target/` (both
ignored by Git). Website build automation lives in `.github/scripts/`.

## Build configuration

`build/` contains the Rust sources for the Makefile's configuration frontend.
Use `make config`, `make build`, or `make test-config`; Make compiles the tool
into `target/baldes-build`. No prebuilt executable is needed in this directory.

## Setup

| Command | Purpose |
| --- | --- |
| `python3 scripts/setup/get_instances.py` | Download and extract CVRPLIB instance sets into `instances/`; requires `7z`. Use `--dry-run` to list downloads. |
| `python3 scripts/setup/get_bks.py C101` | Download the current public CVRPLib BKS to `C101.sol`; accepts names or paths such as `instances/X-n101-k25.vrp`. Use `--problem cvrp` or `--problem vrptw` to restrict lookup, `-o FILE` for an output path, and `--force` to replace a file. Requires only Python's standard library and internet access. |
| `python3 scripts/setup/gurobi_downloader` | Download and verify a stable Gurobi Linux release into `external/`. Use `--help` for version, platform, and destination options. |

The solver can fetch a BKS itself with `baldes hgs INSTANCE --get-bks`; add
`--intensify` to re-solve route subsets with threshold acceptance and elite
restarts, and `--workers N` for concurrent disjoint subproblems. See
[HGS intensification](../docs/extensions/hgs-intensify.md).

## Synthetic stochastic-demand instances

`python3 scripts/generate_stochastic_instances.py` creates 440 NEW symmetric
CVRPSD instances using the generation rules of Salavati-Khoshghalb et al.
The output is `instances/stochastic/generated_salavati_style/` (ignored by Git),
with seeds, SHA-256 manifest, CSV index, and source/rounding metadata. These are
synthetic replications of the experimental design, not recovered published
benchmark files. `--help` lists options. Run
`python3 scripts/tests/test_generate_stochastic_instances.py` for checks.

The `stochastic_instance` example (`features=clp`) validates a generated JSON
file through the actual BALDES constructors; add `--solve` to run numerical
conventional routing branch-cut-and-price with Theta recourse cuts, rank-one
cuts, capacity cuts, and dynamic NG mode -1. Add `--hgs-seconds 3 --hgs-seed 0`
to seed columns and the incumbent with validated mean-demand HGS proposals,
within the overall solve budget.

The paper's BC-Theta comparison is implemented in
`benchmarks/stochastic_branch_cut.py` using gurobipy. The paired runner is
`benchmarks/compare_stochastic.py`. See `docs/benchmarks/stochastic.md` for the
implemented equations, separation differences, licensing and fair timing scope.

## Benchmarks

Requires a built solver with the appropriate backend/features. JavaScript
runners require Node.js; Python runners use the standard library.
Inputs must already exist at the paths specified by each script. Routing runners
currently use Solomon files at `instances/C202.txt`, etc., and CVRPLIB files at
`instances/cvrplib/`; the downloader preserves the upstream archive layout, so
prepare these paths before running. Runs can take several minutes.

| Script in `scripts/benchmarks/` | Arguments and purpose |
| --- | --- |
| `benchmark_solvers.mjs` | `before=/path/to/binary after=/path/to/binary`: compare builds/options on mixed VRPTW and CVRP root workloads. |
| `benchmark_vrptw_pair.mjs` | `BEFORE AFTER LABEL`: alternate two binaries on cut-enabled VRPTW workloads and verify matching non-timing results. |
| `benchmark_pstep.mjs` | `[BINARY] [--grid]`: validate p-step bounds against enumeration and compare serial/parallel jobs; `--grid` expands the instance/p combinations. |
| `benchmark_pstep_pricing.mjs` | `[BINARY] [OUTPUT]`: compare bucket and enumeration pricing backends. |
| `benchmark_ce.mjs` | `name=/path/to/binary [...]`: compare column-elimination solves. |
| `benchmark_ce_lp_threads.mjs` | `MODEL.mps [OUTPUT] [THREAD_COUNTS]`: compare HiGHS simplex/HiPO thread counts on an exported LP; requires the `highs` executable. |
| `benchmark_hgs_intensify.py` | `BINARY INSTANCE SEED.sol --workers 1 2 4 --output FILE`: compare HGS workers with fixed batches and iteration budgets; checks identical routes and generation counts before reporting speedup. |
| `search_hgs_bks.py` | `INSTANCE [INSTANCE ...] --output DIR`: download current CVRPLib solutions and run seeded intensification with multiple random seeds. |
| `benchmark_robust.py` | `--output DIR [--jobs 4] [--time 60]`: run the 1,320-case Table 7 solver grid and independently replay robust incumbents. Uses `instances/Solomon/` and a Clp binary; see [robust routing](../docs/extensions/robust.md). |

For example:

```sh
node scripts/benchmarks/benchmark_solvers.mjs before=/tmp/baldes-before after=./target/release/baldes
node scripts/benchmarks/benchmark_pstep.mjs ./target/release/baldes --grid
python3 scripts/benchmarks/search_hgs_bks.py instances/XL/XL-n3334-k934.vrp instances/XL/XL-n5288-k1246.vrp --output results/hgs-search --time 300 --workers 4 --seeds 41 43 47 --std-similarity
```

The HGS search runs one instance and seed at a time, using workers for its
subproblems. It creates a new output directory with individual `.sol`, JSON and
log files plus `summary.json`; it never overwrites an existing directory. Each
run downloads its own current BKS through `--get-bks` and compares the final
cost with the solver's recomputed seed cost. Candidate improvements are not
automatically submitted or independently certified. Per-run logs retain the
solver's validation and search details. See `--help` for search parameters.

Routing comparisons require Linux `taskset`; `BALDES_PROFILE_OUT`,
`BALDES_PROFILE_CASES`, `BALDES_PROFILE_RUNS`, and `BALDES_PROFILE_CPUS`
control output, workloads, repetitions, and CPU affinity. See each script's
header for additional settings. P-step scripts use `PSTEP_*` settings:
`PSTEP_OUT` selects the validation runner's output directory; the backend
comparison accepts its output directory as the second positional argument.
Column-elimination runners use `CE_BENCH_*` and `CE_LP_*`, respectively.
Routing benchmark/profiling runners invoke `solve`; capacity-only workloads
explicitly add `--ignore-time-windows`. They do not require removed command
aliases. Network JSON uses the same `solve` command.

For the published root benchmark configuration, see
[the benchmark guide](../docs/benchmarks/root.md); explicitly use
`--rounding none` when reproducing those results.

## Profiling

Both tools require Node.js and Linux `taskset`; sampling also requires `perf`
and permission to collect performance counters. Outputs go under `results/`
unless `BALDES_PROFILE_OUT` is set. Use fresh labels to preserve previous runs.

| Script in `scripts/profiling/` | Arguments and purpose |
| --- | --- |
| `profile_routing.mjs` | `BINARY [LABEL]`: mixed VRPTW/CVRP root timings; the default label `baseline` also collects perf samples. |
| `profile_vrptw.mjs` | `BINARY LABEL`: cut-enabled VRPTW timings, optional synthetic capacity, and optional sampling with `BALDES_PROFILE_PERF=1`. |

## Validation

```sh
make test-config
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/tests -p 'test_*.py'
```

Downloader tests use mocked downloads and temporary archives; they do not
download Gurobi or require a license.
