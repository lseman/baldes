# BALDES documentation

Start with the [project README](../README.md) for installation and quick examples.
Use the [CLI guide](cli.md) and [Python guide](python.md) for detailed usage. Commands here run from the repository
root. These guides cover installing, configuring and using BALDES.

## Guides

| Guide | Use it to |
| --- | --- |
| [Command-line usage](cli.md) | Run routing and HGS, interpret output, and look up CLI flags |
| [Python API](python.md) | Install pybaldes, construct inputs, configure solves, and export results |
| [Algorithms and references](algorithms.md) | Review solver components and research references |
| [Build configuration](build.md) | Configure optional backends and compiler options in the terminal menu, then build with Make |
| [Tree and node lifecycle](tree.md) | Run integer routing, understand node inheritance, bounds, limits, and result status |
| [Multi-pricing](multi-pricing.md) | Model a shared master, resume or fork snapshots, and run assignment branching |
| [Finite column imports](column-imports.md) | Import original-vector patterns, fix columns and constrain multiplicities without routing graphs |
| [P-step CVRP](pstep.md) | Build and solve the compact p-step LP relaxation |
| [Scalar resources](resources.md) | Configure general scalar dimensions and bucket indexing |
| [Branching](branching.md) | Choose vehicle-count, Ryan–Foster, or arc-flow branching and understand the single-network assignment limitation |
| [Cuts](cuts.md) | Configure cut families and keep coefficients consistent across pricing, enumeration, and child nodes |
| [Enumerated MIP](enumerated-mip.md) | Enable HiGHS integer solving and interpret its route-count gate and proof coverage |
| [BALDES options](options.md) | Look up supported controls, defaults, types, and scope |
| [Stabilization](stabilization.md) | Configure dual smoothing and understand center updates and pricing retries |
| [Root benchmarks](benchmarks.md) | Review root relaxation timings and benchmark configuration |

## Capabilities and interfaces

| Operation | Rust | CLI | Python | Backend |
| --- | --- | --- | --- | --- |
| RCSP pricing and enumeration | Yes | Through routing solves | Low-level `Solver` | Core Rust |
| Routing root relaxation | Yes | Default `vrptw` / `cvrp` mode | `pybaldes.solve_root` | Clp model store; Clp/IPM/optional HiGHS solves |
| Branch-and-price tree | Yes | `--tree` | Not exposed | Clp model store; Clp/IPM/optional HiGHS solves |
| Complete enumerated node MIP | Yes, opt-in | `--tree --enumerated-mip` | Not exposed | Clp + HiGHS |
| Generic mixed-integer master solve | `highs::solve_integer` | Not exposed directly | Not exposed | HiGHS |

The `vrptw` and `cvrp` routing tree uses one pricing network. For models with
multiple route families, use the [multi-pricing interface](multi-pricing.md),
including its CLI and Python APIs and assignment tree. Its supported branching
families and solve limitations differ from the single-network routing tree.

Root `Optimal` means the LP relaxation converged. Tree `Optimal` means the
integer routing gap closed within the configured tolerances. A feasible
restricted-column solution supplies an incumbent; its objective alone does
not prove a lower bound for the full routing problem.

## Rust module organization

Core modules implement graphs, labeling, pricing, cuts, enumeration and the
master. `master::multi_pricing` coordinates one master with multiple pricing
subproblems and belongs to the core master framework. Rust callers can also use
`master::multi_pricing::solve_master` for original master variables with no
pricing subproblems.

| Module | Purpose |
| --- | --- |
| `extensions::hgs` | Hybrid Genetic Search and primal heuristics |
| `extensions::ipm` | Native interior-point LP backend |
| `extensions::evrp` | Electric-vehicle routing models and charging schedules |
| `extensions::pstep` | P-step CVRP LP with parallel bucket pricing and reference enumeration |
| `master::multi_pricing` | Shared rows, per-subproblem multiplicities and node lifecycle |

`extensions` groups algorithms and specialized models in the same crate; it
does not imply runtime plugins or new build features. Existing `baldes::hgs`,
`baldes::ipm`, `baldes::evrp` and `master::networks` paths remain compatibility
re-exports. New Rust code should use the module paths above. CLI JSON fields
and Python names such as `NetworkProblem` and `solve_networks` stay compatible.
