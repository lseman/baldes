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
| `python3 scripts/setup/gurobi_downloader` | Download and verify a stable Gurobi Linux release into `external/`. Use `--help` for version, platform, and destination options. |

## Benchmarks

Requires Node.js and a built solver with the appropriate backend/features.
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

For example:

```sh
node scripts/benchmarks/benchmark_solvers.mjs before=/tmp/baldes-before after=./target/release/baldes
node scripts/benchmarks/benchmark_pstep.mjs ./target/release/baldes --grid
```

Routing comparisons require Linux `taskset`; `BALDES_PROFILE_OUT`,
`BALDES_PROFILE_CASES`, `BALDES_PROFILE_RUNS`, and `BALDES_PROFILE_CPUS`
control output, workloads, repetitions, and CPU affinity. See each script's
header for additional settings. P-step scripts use `PSTEP_*` settings:
`PSTEP_OUT` selects the validation runner's output directory; the backend
comparison accepts its output directory as the second positional argument.
Column-elimination runners use `CE_BENCH_*` and `CE_LP_*`, respectively.
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
