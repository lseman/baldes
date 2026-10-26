# pyBaldes troubleshooting and reproducibility

[pyBaldes documentation](README.md) · [Installation](installation.md)

## Installation and import

| Symptom | Check and action |
| --- | --- |
| `ModuleNotFoundError: pybaldes` | Use the same interpreter for `python -m pip install .` and execution; inspect `python -m pip --version`. |
| `pybaldes._core` cannot be imported | Build/install the native extension; merely adding `python/` to `PYTHONPATH` is insufficient. |
| Clp/CoinUtils source missing during build | Initialize `vendor/Clp` and `vendor/CoinUtils` submodules, then rebuild. |
| Native library cannot be loaded | Check platform-compatible extension and dynamic HiGHS/Gurobi library discovery. |
| Root/network solve says Clp is required | Rebuild with default features or `clp`; selecting IPM does not remove this requirement. |
| HiGHS/Gurobi selection rejected | Check `has_highs` / `has_gurobi`, rebuild with the feature, and verify native installation/license. |
| Old behavior after a Rust edit | Rebuild with maturin and restart the interpreter or notebook kernel. |
| Disjoint-window method absent | Rebuild with `multi-time-windows`; check with `hasattr`. |

Inspect `pybaldes.__file__` to locate the imported package. A stale extension
can coexist with newer Python wrappers, especially with manually managed
`PYTHONPATH`. Install the package as a unit rather than copying `_core` alone.

## Model validation

| Symptom | Likely cause |
| --- | --- |
| VRPTW rejects customer data | Infinite default `due`, invalid bounds, or non-finite data; set explicit finite windows including depot. |
| Resource count rejected | Active count exceeds `resource_dimensions`; change the build capacity or model. |
| NG local memory exceeds capacity | Neighborhood plus merged arc memory exceeds `ng_memory_capacity`; reduce it or rebuild with wider memory. |
| Per-arc vector rejected | Vector length/order does not match all inserted physical arcs. |
| Network identity or row error | Duplicate network IDs or invalid zero-based shared-row indices. |
| Piecewise function rejected | Invalid ordering, domain, representation, or policy mismatch. |
| Snapshot route fixing fails | Wrong snapshot handle, unavailable route index, unsupported domain, or non-integer quantity. |
| JSON refuses infinity/NaN | Network model/option serialization is strict; use finite model bounds and valid options. |

Validation can occur in a Python wrapper, during native preparation, or during
pricing. Low-level errors can be `RuntimeError`; routing/network input errors
often use `ValueError` or `TypeError`. Preserve the full error message and a
small reproducer instead of treating every error as an infeasible model.

## Interpret solve outcomes

- Fractional routes are expected in a root LP. Use the appropriate integer
  workflow for a dispatchable plan.
- Root `Optimal` does not prove an integer optimum.
- Unknown certified bounds remain unknown even if the numerical bound is finite.
- Empty heuristic pricing output does not establish an exact pricing minimum.
- `enumerate()` on low-level Solver does not prove complete universe coverage.
- A network enumeration failure leaves ordinary pricing/search available; inspect
  `failure`, `complete`, and `coverage_cutoff` before using an integer-transfer result.
- A user-supplied upper bound or cutoff is not itself a validated feasible plan.

If objectives disagree across Python and CLI, match customer count, explicit
matrix weights, rounding, NG construction, cuts, bound mode, and backend first.
Python uses the instance's rounding by default; CLI routing uses `tenth`.
Published root benchmark reproduction requires `--rounding none`.

## Concurrency and lifetime

Routing `solve` releases the GIL during native solving. Fresh
`solve(..., tree=True)` also releases it. Shared root solves and calls
through `NetworkNode` retain it; node handles are thread-bound and must remain
on their creating thread. Low-level Solver calls currently retain the GIL, so
Python threads do not imply concurrent low-level native pricing.

Low-level Route owns its returned data and survives solver destruction. A
NetworkNode owns its snapshot and can survive deletion of the root result.
JSON export omits the native node and cannot be used to resume it. For independent
worker processes, construct/install model state in each process rather than
passing native handles as if they were serializable Python records.

## Record a reproducible run

Save the model or instance hash, repository revision, package version, compiler
and platform, backend features, scalar/NG capacities, solve keywords, and HGS
seed. Preserve result JSON and stderr logs. Include both HGS and solve budgets
when reporting elapsed time. Record numerical and certified bounds separately.

```python
import json
import platform
import pybaldes

metadata = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "pybaldes": pybaldes.__version__,
    "has_clp": pybaldes.has_clp,
    "has_highs": pybaldes.has_highs,
    "has_gurobi": pybaldes.has_gurobi,
    "resource_dimensions": pybaldes.resource_dimensions,
    "ng_memory_capacity": pybaldes.ng_memory_capacity,
}
print(json.dumps(metadata, indent=2))
```

For a bug report, include a minimal model, exact command, expected behavior,
actual status/error, and the installed build details. The wrapper source is in
[python/pybaldes](../../python/pybaldes); native bindings are in
[python/rust/src](../../python/rust/src); regression examples are in
[python/tests](../../python/tests).
