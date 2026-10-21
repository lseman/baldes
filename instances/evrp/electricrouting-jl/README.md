# ElectricRouting.jl instance snapshot

Downloaded on 2026-10-20 from
[sean-lo/ElectricRouting.jl](https://github.com/sean-lo/ElectricRouting.jl),
pinned to commit `102c9c90f6021d2dea664f0d8ad6990661c7668a`.

- `Instances/`: all 161 upstream instance text files, unchanged.
- `Solutions/`: all 312 upstream reference solution text files, unchanged,
  including files with the `_benchmark` suffix. These are upstream results,
  not solutions computed or independently certified by Baldes.
- `UPSTREAM_README.md`: the upstream repository README at that revision.
- `SOURCE.json`: upstream paths, Git blob hashes, byte counts and SHA-256 hashes
  for every downloaded file, plus the downloaded archive's SHA-256 hash.

Every downloaded file was checked against its Git blob hash from the pinned
GitHub tree. Instance files contain 16–100 customers, four depots, and 11, 16,
21 or 26 charging stations. The smallest example is
`Instances/016_08_big_01.txt` (16 customers, four depots, 11 stations).

The source is Jacquillat and Lo's implementation accompanying
[Subpath-Based Column Generation for Electric Vehicle Routing Problems](https://arxiv.org/abs/2407.02640).
The pinned repository has no LICENSE file; no Baldes license is assigned to
these upstream files.

## Model and format caveats

The node table uses `d` for depots, `f` for charging stations and `c` for
customers. It includes coordinates, demand, time windows and service time.
The tail includes battery/load parameters and separate vehicle start and end
allocations by depot. Routes may start and finish at different depots; the
reference solutions include examples of this.

The files specify a scalar inverse refueling rate, not piecewise-linear
charging-curve knots. Upstream nonlinear charging functions are supplied in
code/experiment settings. Reproducing a nonlinear experiment requires recording
that curve and the original scaling/rounding settings as well as the instance.

These are raw downloads, not directly runnable Baldes input files. Baldes's
current EV interface expects an in-memory, single-depot closed-route problem.
A format reader and multi-depot EV pricing/master integration are required for
an equivalent run of the original instances. Selecting a single depot or
truncating customers would produce a derived instance, not the original
benchmark. Reference costs must not be compared without matching model,
objective, units, scaling and charging assumptions.

The upstream format reader and generator can be inspected at the pinned
revision:

- [read_evrptw_data.jl](https://github.com/sean-lo/ElectricRouting.jl/blob/102c9c90f6021d2dea664f0d8ad6990661c7668a/src/read_evrptw_data.jl)
- [make_evrptw_multidepot_instances.jl](https://github.com/sean-lo/ElectricRouting.jl/blob/102c9c90f6021d2dea664f0d8ad6990661c7668a/scripts/make_evrptw_multidepot_instances.jl)
