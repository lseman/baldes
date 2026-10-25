# pyBaldes documentation

[All BALDES documentation](../README.md)

pyBaldes is the Python distribution of BALDES. Import it as `pybaldes`.
It provides typed routing inputs/results, a stateful resource-constrained
shortest-path pricing solver, and shared-master models with multiple pricing
networks. Solves run in the Rust library; no CLI subprocess is required.

## Choose an interface

| Your task | Interface | Result |
| --- | --- | --- |
| Solve the LP relaxation of a CVRP or VRPTW instance | `solve_root` or `Instance.solve_root` | `RootResult`, potentially fractional routes |
| Price paths for your own column-generation loop | `Solver` | Reduced costs, owned `Route` objects, pricing exactness |
| Combine vehicle types, route families, or independent pricing graphs | `NetworkProblem.solve` or `solve_networks` | `NetworkResult`, with a resumable root node |
| Search for an integer solution of a shared-master model | `solve_networks(..., tree=True)` | Integer witness when found, bound and tree status |
| Add state-dependent feasibility or objective terms | `ResourceModel`, `EncodedFunction` | Resources attached to a low-level solver |

The routing convenience API exposes root solving. It does not expose the CLI's
single-network routing tree or a standalone HGS function. `hgs_time` seeds a
routing root solve with heuristic routes. Network tree search has its own model
and branching rules; it is not an `Instance.solve_root(tree=True)` option.

## Start here

1. [Install and inspect your build](installation.md).
2. [Run a complete Python example](quickstart.md).
3. [Understand objectives, bounds, and exactness](results.md).
4. Choose [routing models](routing.md), [low-level pricing](pricing.md), or
   [shared-master networks](networks.md).

## User guides and reference

- [Installation](installation.md): source builds, features, backends, compiled capacities, and development setup.
- [Quickstart](quickstart.md): self-contained routing and pricing examples.
- [Routing inputs and options](routing.md): `Customer`, `Instance`, file formats, distance conventions, and every `solve_root` keyword.
- [Results and guarantees](results.md): root/network result fields, status interpretation, certificates, and JSON export.
- [Pricing solver](pricing.md): graph construction, scalar resources, NG memory, repeated pricing, route ownership, and method reference.
- [Coupled and encoded resources](resources.md): resource declaration, dependencies, piecewise functions, paired fronts, and build-dependent windows.
- [Shared-master networks](networks.md): model coefficients, solve controls, enumeration, snapshots, branching, and synchronization.
- [API reference](api.md): exported names, constructors, callable signatures, and source locations.
- [Troubleshooting and reproducibility](troubleshooting.md): build failures, validation errors, unknown bounds, concurrency, and experiment records.

Examples use Python 3.9+ syntax. Terminal commands run from the repository root
unless they explicitly change directories. Optional features are identified at
the point of use. The reference describes the source in this checkout; inspect
`pybaldes.__version__` and build flags when using an already installed wheel.
