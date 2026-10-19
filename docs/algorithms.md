# Algorithms and references

[Documentation index](README.md)

## Solver components

- Bucket-graph labeling: resource and NG preparation, bucket grids, strongly
  connected component ordering, jump arcs, preceding-bucket dominance,
  bidirectional search with a moving half-way point, and symmetric joins.
- Staged pricing with the native heuristic phase schedule.
- Reduced-cost fixing on bucket arcs and elementary route enumeration with
  inspection pricing at the root and descendant nodes.
- Limited-memory rank-one (subset-row) cuts: 1- and 3-row separation,
  local-search and tuple-based 4- and 5-row separation, cut memory, and
  limited- and full-memory cut state in pricing and enumeration labels.
- Dual-price smoothing, and a Clp-backed restricted master with cut rounds,
  basic-cut removal and artificial-column penalties; optional native IPM master solves.
- Rounded capacity and Kohl 2-path cuts, with time-window feasibility checks;
  strong k-path and route-load knapsack cuts integrated with pricing,
  enumeration, and inherited master rows.
- Solomon VRPTW and CVRPLIB CVRP input, HGS heuristic, initial-column seeding, and root solving.
- Dynamic NG strengthening from master cycles at configurable separator priority.

The core solver uses only the Rust standard library. The `clp` feature links
COIN-OR Clp through its C interface for the restricted master. The optional
`highs` feature adds a HiGHS LP backend and integer solving.

## Additional cut configuration

Strong k-path and route-load knapsack cuts are available in the Rust root API
through `with_kpath_cuts` and `with_knapsack_cuts`. The latter takes original
integer resource rows; `KnapsackCutConfig::with_resource_indices` maps original
resource IDs to solver coordinates. Load-dependent costs and that mapping pass
through pricing, enumeration, and inherited master rows.

## References

1. Ruslan Sadykov, Eduardo Uchoa, Artur Alves Pessoa. "A Bucket Graph Based Labeling Algorithm for Vehicle Routing." Transportation Science, 2021. https://doi.org/10.1287/trsc.2020.0985
2. Diego Pecin, Artur Pessoa, Marcus Poggi, Eduardo Uchoa, Haroldo Santos. "Limited memory rank-1 cuts for vehicle routing problems." Operations Research Letters 45.3 (2017): 206-209. https://doi.org/10.1016/j.orl.2017.02.006
3. Talal Vidal. "Hybrid genetic search for the CVRP: Open-source implementation and SWAP* neighborhood." Computers & Operations Research 140 (2022): 105643. https://doi.org/10.1016/j.cor.2021.105643
