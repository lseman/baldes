# Install PyBaldes

[PyBaldes documentation](README.md) · [Quickstart](quickstart.md)

## Install from a checkout

The source package requires Python 3.9+, Rust/Cargo 1.85+, Git, and a C++14
compiler for the default Clp build. Package construction uses maturin.
Use a virtual environment so the interpreter used to install is also the one
that imports the extension:

```bash
git clone --recurse-submodules https://github.com/lseman/baldes.git
cd baldes
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

For an existing checkout, initialize the two native dependencies before installing:

```bash
git submodule update --init vendor/Clp vendor/CoinUtils
python -m pip install .
```

The mixed package contains Python wrappers and `pybaldes._core`, a PyO3 extension
using Python's stable ABI starting with Python 3.9. ABI compatibility does not
remove platform, CPU, or shared-library requirements. The commands above build
from local source; they do not assume an available binary wheel on a package index.

## Verify the installed build

```python
import pybaldes

print(pybaldes.__version__)
print(pybaldes.__file__)
print("Clp:", pybaldes.has_clp)
print("HiGHS:", pybaldes.has_highs)
print("Gurobi:", pybaldes.has_gurobi)
print("Scalar capacity:", pybaldes.resource_dimensions)
print("NG capacity:", pybaldes.ng_memory_capacity)
print("Multiple windows:", hasattr(pybaldes.Solver, "set_vertex_time_windows"))
```

Clp is enabled by default. Both routing roots and shared-master network solves
require it, even if a routing root selects another LP solve backend.
Low-level `Solver` pricing and `ResourceModel` do not require Clp.

## Build features

Pass features through pip's maturin build arguments:

```bash
python -m pip install . --config-settings=build-args="--features multi-time-windows,ng-memory-31"
```

| Python crate feature | Effect |
| --- | --- |
| `clp` (default) | Build the Clp master and expose routing/network solving |
| `highs` | Link HiGHS; enable its routing LP backend and network enumerated MIP |
| `gurobi` | Link Gurobi; enable its routing LP backend; runtime license required |
| `simd` | Enable optional runtime-dispatched SIMD kernels |
| `multi-time-windows` | Expose disjoint service-start windows |
| `rank-one-node-memory` | Select node memory for generated rank-one cuts |
| `ng-memory-31` / `ng-memory-63` | Increase compiled NG memory from the default 15; 63 takes precedence |

The Python feature list is defined by [python/rust/Cargo.toml](../../python/rust/Cargo.toml).
Do not assume every root-crate Cargo feature is forwarded by the Python crate.
The Python crate currently forwards `simd`, but does not declare `simd-avx512`.
See the [BALDES build guide](../guides/build.md) for native library discovery,
Gurobi environment variables, and backend requirements.

For a pricing-only extension:

```bash
python -m pip install . --config-settings=build-args=--no-default-features
```

For HiGHS or Gurobi, first configure the native installation as described in the
build guide, then build with the corresponding feature:

```bash
python -m pip install . --config-settings=build-args="--features highs"
# Or, with a configured Gurobi installation and license:
python -m pip install . --config-settings=build-args="--features gurobi"
```

These features retain default Clp unless `--no-default-features` is also passed.
A backend flag reports compiled support, not runtime library/license health.

## Scalar capacity

`Solver(n_vertices, n_resources)` chooses the number of active scalar resources
within a compiled capacity. The default capacity is two. Rebuild to change it:

```bash
BALDES_RESOURCE_DIMENSIONS=5 python -m pip install .
```

Inspect `pybaldes.resource_dimensions` afterward. Coupled resources declared with
`ResourceModel` are a separate interface; their IDs are not scalar coordinates.
Direct pip/maturin builds read `BALDES_RESOURCE_DIMENSIONS`, not Make's `.config`.

## Develop and test

Inside an activated virtual environment:

```bash
python -m pip install maturin pytest
maturin develop --release
python -m pytest python/tests -q
```

The root `pyproject.toml` points maturin at `python/rust/Cargo.toml` and installs
wrappers from `python/pybaldes/`. Rebuild after changing Rust code or features.
Restart long-lived interpreters and notebook kernels after replacing the native
extension. Optional-feature tests skip when their required build is unavailable.
