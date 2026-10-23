# Configuring and building BALDES

[Documentation index](../README.md)

## Direct Cargo builds

Install Rust/Cargo and clone the repository. The core library can be built
without an external LP backend:

```sh
cargo build --release
```

For the CLI and routing master, initialize the vendored dependencies and build
with Clp. This requires a C++14 compiler:

```sh
git submodule update --init vendor/Clp vendor/CoinUtils
cargo build --release --features clp
./target/release/baldes --help
```

The `clp` feature builds Clp 1.17.10 and CoinUtils 2.11 from those submodules.
To also enable the optional HiGHS backend and integer solver, install its
shared library and use `cargo build --release --features clp,highs`.
The CLI and routing tree require `clp`; `highs` alone provides the library's
HiGHS backend and generic MIP API. See below for library discovery details.

Disjoint service-start windows are optional: use
`cargo build --release --features multi-time-windows` for the library, or
`--features clp,multi-time-windows` for network JSON and the CLI. See
[multiple time windows](../extensions/multi-time-windows.md) for the Rust,
Python and JSON APIs.

Python installation is covered in the [pybaldes guide](python.md).

### Experimental native pricing

`cargo build --release --features native-pricing` enables the Clp backend and a
C++14 batch preflight for buckets with at least 32 eligible labels. It checks
two disposable resources and NG rejection together, using SSE2 where available
and a scalar C++ fallback elsewhere. Rust retains cut/branch updates, label
construction and ordered insertion. Other resource dimensions, equality or
coupled resources, disjoint time windows and finite label budgets retain scalar
Rust propagation.

This feature is disabled by default: initial comparisons preserved solver
outputs but did not establish an end-to-end speedup, and R202 regressed. See
[the native-kernel experiment](../benchmarks/native-kernels.md) before enabling
it for a workload. The existing Clp paired sparse-matrix kernel remains enabled
with `clp` independently of this experiment.


## Gurobi master backend

Download and extract the latest stable Linux release into the repository's
ignored `external/` directory:

```sh
./scripts/setup/gurobi_downloader
```

The script detects x86-64, ARM64, or PowerPC Linux, verifies Gurobi's published
checksum, and prints the installation path to use in `make config`. It reuses
verified downloads and existing installations. Use `--version 13.0.3` to pin a
release, `--destination PATH` for another directory, or `--help` for options.
Extraction requires Python with `tarfile.data_filter` (Python 3.12+).

Install Gurobi and configure a valid license. Set `GUROBI_HOME` to its platform
installation directory, containing `lib/`, then build:

```sh
export GUROBI_HOME=/opt/gurobi1300/linux64
export LD_LIBRARY_PATH="$GUROBI_HOME/lib:${LD_LIBRARY_PATH:-}"
cargo build --release --features clp,gurobi
./target/release/baldes vrptw instances/Solomon/R101.txt --master gurobi
```

On macOS configure `DYLD_LIBRARY_PATH` as needed instead. The build detects the
versioned C library (for example `libgurobi130.so`) in `$GUROBI_HOME/lib`.
Set `GUROBI_LIB_NAME=gurobi130` explicitly if that directory contains multiple
versions. Gurobi is linked dynamically and is not bundled with BALDES.
License errors are reported when the backend is initialized.

`GurobiMaster` implements the generic Rust `LpMaster` interface with automatic
algorithm selection, warm-basis reoptimization, primal values, duals, reduced costs, basis-aware
cut cleanup, solve limits, and independent model copies. Use `--features gurobi`
for the library backend alone; the routing CLI/tree also require `clp`.
All three native features may be enabled together. Enumerated-node integer
solving continues to use the `highs` feature.

Routing root/tree solves give HiGHS and Gurobi all CPUs available to the process
by default. Use `--master-threads N` to limit their worker budget, or
`--master-threads 0` for all available CPUs. CPU affinity is respected.
Clp and native IPM ignore this setting. HiGHS retains adaptive primal/dual
simplex; Gurobi uses automatic algorithm selection. Both receive the worker
budget and determine actual algorithm parallelism. Warm simplex reoptimization
may remain serial, and more workers do not guarantee a faster solve.
HiGHS models in one process must use a consistent worker budget because its
native scheduler is shared; configure it before the first solve. Forks retain
their parent's budget. In Rust, use `RootOptions::master_threads`; in Python,
use `solve_root(..., master_threads=N)`.

For Python, build with `maturin develop --release --features clp,gurobi` from
`python/`, then pass `master_solver="gurobi"` to `solve_root`. Check
`pybaldes.has_gurobi` to see whether that wheel includes the backend.
In `make config`, enable **Gurobi backend** and set **Gurobi installation**,
or leave its path empty to inherit `GUROBI_HOME` from your shell. Save and
run `make` to build with those settings.

## Interactive configuration

From the repository root:

```sh
make config       # Open the configuration window in your terminal
make              # Compile with the saved settings
```

`make menuconfig` opens the same window. Use **Up/Down** or **J/K** to select
an option, **Space/Enter** to change it, **S** to save and exit, and **Q/Escape**
to discard changes. Text fields accept Enter to confirm, Escape to cancel,
Backspace to delete and Ctrl-U to clear. **Home/End** jump to the first/last
setting; **Page Up/Page Down** move five settings at a time.

Settings are grouped into Build, Performance, Toolchain and Cargo. Wide
terminals show contextual help and a live build summary beside the options;
compact terminals keep help below the scrolling list. The header tracks
unsaved changes. The menu adapts automatically when resized, down to 64 columns
by 20 rows, and respects `NO_COLOR` for a monochrome display.

This is a kernel-style terminal menu, implemented in Rust rather than Linux's
Kconfig tooling. It requires Make, Rust/Cargo, and POSIX `stty` in an interactive
ANSI terminal (Linux/macOS). It opens in the current terminal; no desktop
session, Python packages, ncurses headers or UI library downloads are needed.
The helper is compiled directly with `rustc` into `target/baldes-build` before
configuration, so it does not need to build the solver first.

## Build options

| Setting | Default | Effect |
| --- | --- | --- |
| Profile | `release` | Choose Cargo's `release` or `dev` profile |
| Clp | Off | Enable the `clp` Cargo feature |
| HiGHS | Off | Enable the `highs` Cargo feature |
| Gurobi | Off | Enable the `gurobi` Cargo feature; requires an installation and license |
| Command-line program | Off | Build the `baldes` executable along with the library; selects Clp |
| Optional SIMD | Off | Cycle off / AVX2 / AVX-512; only CPU-supported modes are selectable |
| Native CPU | Off | Add `-C target-cpu=native`, preserving existing Rust flags |
| Release debug symbols | Off | Set `CARGO_PROFILE_RELEASE_DEBUG=1` for profiling |
| Offline | Off | Pass `--offline` to Cargo |
| Locked dependencies | On | Pass `--locked`, preserving `Cargo.lock` |
| Scalar resource dimensions | `2` | Set `BALDES_RESOURCE_DIMENSIONS`; fixed capacity for Python and the default Rust types |
| Parallel jobs | `0` (automatic) | Pass `--jobs N` when nonzero |
| Target triple | Empty (host) | Pass `--target TRIPLE` for cross compilation |
| HiGHS installation | Empty (automatic) | Set `HIGHS_DIR` to the root containing `lib/libhighs` |
| Gurobi installation | Empty (shell environment) | Set `GUROBI_HOME` to the installation root containing `lib/libgurobi` |

The library, including native IPM, pricing, cuts, and HGS, is always included.
These components currently have no Cargo feature switches. Algorithm settings
such as cut priorities and IPM tolerances remain runtime options.

The CLI requires Clp. Selecting the CLI enables Clp; deselecting Clp also
deselects the CLI. Native CPU optimization cannot be combined with a target
triple. Cross compilation requires installing the Rust target and configuring
its linker separately, including any enabled native solver libraries.

Before building Clp, initialize its source submodules and provide a C++14
compiler:

```sh
git submodule update --init vendor/Clp vendor/CoinUtils
```

HiGHS needs a shared library with 32-bit `HighsInt`. Discovery follows the
existing Cargo build: `HIGHS_DIR/lib`, `vendor/highs/lib`, `pkg-config`, then the
system linker path. Leaving the menu's installation field empty preserves
`HIGHS_DIR` from your shell. Configure your runtime library search path as
needed for a custom HiGHS installation.

## Saved settings and automation

The menu saves a local `.config`, ignored by Git. Subsequent Make commands
read it. A missing configuration uses release/library-only defaults with no
optional solver dependencies. Cargo commands invoked directly continue to
use their explicit Cargo flags; they do not read `.config`.

```sh
make showconfig                # Display effective settings
make print-command             # Preview the Cargo build invocation
make defconfig                 # Replace .config with defaults
make defconfig CONFIG=ci.config # Write another configuration file
make CONFIG=ci.config          # Build with it
```

For noninteractive use, run `make defconfig` and edit the file or supply a
saved config via `CONFIG=path`. The format is `CONFIG_NAME=value`, using `y/n`
for booleans and double quotes for strings. Backslashes and quotes in strings
must be escaped. Missing settings take their defaults; unknown/duplicate keys,
invalid values and conflicting dependencies are rejected. The file is parsed
as data and is never included as Make code or sourced by a shell.

An example configuration for the CLI with both LP backends:

```ini
CONFIG_PROFILE=release
CONFIG_CLP=y
CONFIG_HIGHS=y
CONFIG_GUROBI=n
CONFIG_CLI=y
CONFIG_SIMD=off
CONFIG_NATIVE_CPU=n
CONFIG_RELEASE_DEBUG=n
CONFIG_OFFLINE=n
CONFIG_LOCKED=y
CONFIG_JOBS=0
CONFIG_RESOURCE_DIMENSIONS=2
CONFIG_TARGET=""
CONFIG_HIGHS_DIR=""
CONFIG_GUROBI_HOME=""
```

## Other Make targets

| Command | Action |
| --- | --- |
| `make build` / `make all` | Same build as plain `make` |
| `make check` | Cargo check for the selected artifacts and features |
| `make test` | Run library/integration tests with the selected profile/features |
| `make clippy` | Lint library/integration tests and the selected CLI |
| `make fmt` | Check formatting in the `baldes` package |
| `make clean` | Run `cargo clean -p baldes`; retain `.config` |
| `make test-config` | Test config parsing, dependencies and Cargo argument translation |
| `make help` | List targets and tool overrides |

`make test` and `make clippy` select integration tests as well as the library;
Cargo may build the CLI for integration tests when its Clp requirement is
enabled, even if the CLI artifact is disabled for ordinary builds. The
workspace's Python extension is built separately using its existing packaging
commands. `CARGO=path` and `RUSTC=path` override the tool executables.

Existing compiler/linker environment variables remain available to Cargo.
The frontend prints the command and any configuration-specific environment
overrides before running it and propagates Cargo's failure status to Make.

The scalar setting accepts any nonnegative `usize`. For example,
`CONFIG_RESOURCE_DIMENSIONS=5` compiles capacity 5; models may activate fewer
coordinates. The saved setting overrides `BALDES_RESOURCE_DIMENSIONS` inherited
from the shell for Make builds. Direct Cargo or maturin builds use that environment
variable instead and do not read `.config`. Explicit Rust `Graph::<D>` / `Solver::<D>`
counts remain independent of the configured default. See [scalar resources](../reference/resources.md).

Optional SIMD is disabled by default. In `make config`, cycle through `off`,
`avx2`, and `avx512`; unsupported CPU modes are skipped. Saved modes are validated
against the host CPU and require an empty target triple. Legacy `n` / `y` settings
remain accepted as `off` / `avx2`.
Direct Cargo builds use `--features simd` for AVX2 or `--features simd-avx512`
for AVX-512F (requires Rust 1.89 or newer). On x86-64, sufficiently long
disposable-resource dominance scans use the selected kernel after runtime CPU
detection. AVX-512 builds fall back to AVX2 when available. Other CPUs and small scans retain the
portable implementation. Scalar capacities remain arbitrary; the optimized
scan applies when the model activates one or two disposable resources and
the compiled capacity is at least two.
The `simd` feature also enables runtime-dispatched AVX2 rank-one bitset
accumulation for at least 16 words, independently of the resource dimension.
This feature does not require `target-cpu=native` and does not change numerical
comparison tolerances or certification policy.

For eight-label AVX-512F dominance scans, use
`cargo build --release --features clp,simd-avx512` (or pass the same features
to maturin). This feature includes `simd` and requires Rust 1.89 or newer;
the default and AVX2-only builds retain the package's existing Rust requirement.
Runtime dispatch prefers AVX-512F on supported x86-64 CPUs, then AVX2, then
the portable implementation. No global CPU flags are required. The AVX-512
kernel needs only AVX-512F, including for the packed NG masks, and preserves
first-candidate ordering and the original floating-point comparison order.
See [SIMD dispatch](../reference/simd.md) for dispatch behavior and portability details.
