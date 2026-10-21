# Subpath-based Column Generation for Electric Vehicle Routing Problems

This repository implements the algorithm and experiment suite for the paper: "Subpath-based Column Generation for Electric Vehicle Routing Problems" by Alexandre Jacquillat and Sean Lo. The paper can be found [here](https://arxiv.org/abs/2407.02640).

Please cite this paper as:
```
@misc{jacquillat2024subpath,
      title={Subpath-Based Column Generation for Electric Vehicle Routing Problems}, 
      author={Alexandre Jacquillat and Sean Lo},
      year={2024},
      eprint={2407.02640},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2407.02640}, 
}
```
<!-- TODO: add links to paper -->

## Usage

1) Create an instance `data::EVRPData` (found in `src/utils.jl`), either:
- by reading an instance from [Desaulniers et al (2016)](https://pubsonline.informs.org/doi/10.1287/opre.2016.1535) via `read_evrptw_instance()` in `src/read_evrptw_data.jl`,
- or via `generate_instance()` in `src/utils.jl`.
2) Create a graph `graph::EVRPGraph` from the problem instance using `generate_graph_from_data()`.
3) Implement column generation on the set-partitioning formulation, augmented with *ng*-routes and subset-row cuts, via the function `path_formulation_column_generation_with_adaptve_ngroute_SR3_cuts()` (found in `src/path_formulation.jl`)

Important arguments:
- `method::String = "ours"`: can be either `"ours"` or `"benchmark"`, where the benchmark implements the monodirectional path-based label-setting method by [Desaulniers et al (2016)](https://pubsonline.informs.org/doi/10.1287/opre.2016.1535) (nonlinear benchmark not yet supported)
- `use_time_windows::Bool = true`: whether to include time window constraints.
- `use_load::Bool = true`: whether to include vehicle capacity constraints.
- Nonlinear charging:
    - `use_nonlinear_charging::Bool = true`: whether to use nonlinear charging schedule $g$.
    - `charging_function::PiecewiseLinearIncreasingConcaveFunction`: definition of concave charging schedule $g$ to use (only piecewise linear functions currently supported)
- `use_pruned_graph::Bool = true`: whether to attempt heuristic pricing on a subset of arcs (number of arcs entering/leaving each node controlled in `prune_graph_outdegree_sequence`)
- `exploration_method::String = "bestfirst"`: in what order to retrieve unexplored, nondominated subpaths / paths from queue.
- Path elementarity: 
    - `use_ngroute::Bool = true`: whether to solve the *ng*-route relaxation [(Baldacci et al 2011)](https://pubsonline.informs.org/doi/10.1287/opre.1110.0975)
    - `neighborhoods::Union{Nothing, BitMatrix} = nothing` if solving the *ng*-route relaxation, the choice of *ng*-neighborhoods at each node: `neighborhoods[i][j] = 1` iff node `i` is in the *ng*-neighborhood of node `j` 
    - `ngroute_neighborhood_size::Int = Int(ceil(sqrt(graph.n_customers)))`: the number of customer nodes to include in the *ng*-set of each node
    - `ngroute_neighborhood_charging_size::String` determines the size of *ng*-sets at charging stations, can be one of `"small"`, `"medium"`, `"large"`. Recommended value `"small"`
    - `use_adaptive_ngroute::Bool = true`: whether to adaptively grow *ng*-relaxations according to [Martinelli et al (2014)](https://www.sciencedirect.com/science/article/abs/pii/S0377221714004123)
- Cutting planes
    - `use_SR3_cuts::Bool = true`: whether to use subset-row inequalities (includes *lm*-SRIs), first introduced in [Jepsen et al (2008)](https://pubsonline.informs.org/doi/10.1287/opre.1070.0449)
    - `use_lmSR3_cuts::Bool = true`: whether to use *lm*-SRIs instead of SRIs, according to [Pecin et al (2017)](https://link.springer.com/article/10.1007/s12532-016-0108-8)
    - `max_SR3_cuts::Int = 5`: how many cuts to add after each cut separation procedure; recommended to be a small number since cuts increase the pricing problem complexity
    - `randomize_cuts::Bool = false`: if `true`, selects which cuts to add in a random manner (weighted by violation of incumbent solution); if `false`, adds cuts greedily according to violation.
- `verbose::Bool = true`: whether to print output as the algorithm progresses
- `time_limit::Float64 = Inf`: time limit in seconds for the overall algorithm.
