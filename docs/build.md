# Configuring and building BALDES

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

Python installation is covered in the [pybaldes guide](python.md).

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
| Command-line program | Off | Build the `baldes` executable along with the library; selects Clp |
| Optional SIMD | Off | Enable runtime-dispatched AVX2 dominance scans (`simd` feature) |
| Native CPU | Off | Add `-C target-cpu=native`, preserving existing Rust flags |
| Release debug symbols | Off | Set `CARGO_PROFILE_RELEASE_DEBUG=1` for profiling |
| Offline | Off | Pass `--offline` to Cargo |
| Locked dependencies | On | Pass `--locked`, preserving `Cargo.lock` |
| Scalar resource dimensions | `2` | Set `BALDES_RESOURCE_DIMENSIONS`; fixed capacity for C/Python and the default Rust types |
| Parallel jobs | `0` (automatic) | Pass `--jobs N` when nonzero |
| Target triple | Empty (host) | Pass `--target TRIPLE` for cross compilation |
| HiGHS installation | Empty (automatic) | Set `HIGHS_DIR` to the root containing `lib/libhighs` |

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
CONFIG_CLI=y
CONFIG_SIMD=n
CONFIG_NATIVE_CPU=n
CONFIG_RELEASE_DEBUG=n
CONFIG_OFFLINE=n
CONFIG_LOCKED=y
CONFIG_JOBS=0
CONFIG_RESOURCE_DIMENSIONS=2
CONFIG_TARGET=""
CONFIG_HIGHS_DIR=""
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
counts remain independent of the configured default. See [scalar resources](resources.md).

Optional SIMD is disabled by default. Enable `CONFIG_SIMD=y` in `make config`,
use `cargo build --release --features clp,simd`, or pass `--features clp,simd`
to maturin. On x86-64, sufficiently long disposable-resource dominance scans
use AVX2 after runtime CPU detection. Other CPUs and small scans retain the
portable implementation. Scalar capacities remain arbitrary; the optimized
scan applies when the model activates one or two disposable resources and
the compiled capacity is at least two.
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
See [SIMD dispatch and optimization targets](simd.md) for kernel measurements
and additional vectorization opportunities.
