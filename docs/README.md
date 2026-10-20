# BALDES documentation

Start with the [project README](../README.md) for installation and quick examples.
Use the [CLI guide](cli.md) and [Python guide](python.md) for detailed usage. Commands here run from the repository
root. The guides below describe the current Rust implementation.

## Guides

| Guide | Use it to |
| --- | --- |
| [Command-line usage](cli.md) | Run routing and HGS, interpret output, and look up CLI flags |
| [Python API](python.md) | Install pybaldes, construct inputs, configure solves, and export results |
| [Algorithms and references](algorithms.md) | Review solver components and research references |
| [Build configuration](build.md) | Configure optional backends and compiler options in the terminal menu, then build with Make |
| [Tree and node lifecycle](tree.md) | Run integer routing, understand node inheritance, bounds, limits, and result status |
| [Scalar resources](resources.md) | Configure general scalar dimensions and bucket indexing |
| [Branching](branching.md) | Choose vehicle-count, Ryan–Foster, or arc-flow branching and understand the single-network assignment limitation |
| [Cuts](cuts.md) | Configure cut families and keep coefficients consistent across pricing, enumeration, and child nodes |
| [Enumerated MIP](enumerated-mip.md) | Enable HiGHS integer solving and interpret its route-count gate and proof coverage |
| [BALDES options](options.md) | Look up supported controls, defaults, types, and scope |
| [Stabilization](stabilization.md) | Configure dual smoothing and understand center updates and pricing retries |
| [Source layout](source-layout.md) | Find the module responsible for a change |
| [Website publishing](website.md) | Compile Markdown to HTML and preview or deploy the site |
| [Root benchmarks](benchmarks.md) | Review root relaxation timings and benchmark configuration |
| [Build and validation](testing.md) | Select feature combinations and run the relevant checks |

## Capabilities and interfaces

| Operation | Rust | CLI | Python | Backend |
| --- | --- | --- | --- | --- |
| RCSP pricing and enumeration | Yes | Through routing solves | Low-level `Solver` | Core Rust |
| Routing root relaxation | Yes | Default `vrptw` / `cvrp` mode | `pybaldes.solve_root` | Clp model store; Clp/IPM/optional HiGHS solves |
| Branch-and-price tree | Yes | `--tree` | Not exposed | Clp model store; Clp/IPM/optional HiGHS solves |
| Complete enumerated node MIP | Yes, opt-in | `--tree --enumerated-mip` | Not exposed | Clp + HiGHS |
| Generic mixed-integer master solve | `highs::solve_integer` | Not exposed directly | Not exposed | HiGHS |

The routing tree currently models **one pricing network**. A network is a
separate routing subproblem, such as a vehicle type or depot. Forward and
backward labeling within one graph do not create multiple networks.
Ryan–Foster branching works with one network. Packing-set assignment rows are
supported, but useful fractional assignment branching requires multiple
networks; see [Branching](branching.md#packing-set-assignment).

Root `Optimal` means the LP relaxation converged. Tree `Optimal` means the
integer routing gap closed within the configured tolerances. A feasible
restricted-column solution supplies an incumbent; its objective alone does
not prove a lower bound for the full routing problem.

## Keeping the guides current

When changing behavior, update the owning guide and its linked source/tests.
For BALDES options, keep `OPTION_TABLE` support metadata and the reference table
in sync. Check documented commands and links rather than treating a historical
test count as a permanent guarantee.
