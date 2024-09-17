# BALDES

BALDES, a Bucket Graph Labeling Algorithm for Vehicle Routing

This repository contains a C++ implementation of a Bucket Graph-based labeling algorithm designed to solve the Resource-Constrained Shortest Path Problem (RSCPP), which commonly arises as a subproblem in state-of-the-art Branch-Cut-and-Price algorithms for various Vehicle Routing Problems (VRPs).

The algorithm is based on the approach originally presented in the paper: **A Bucket Graph Based Labeling Algorithm for Vehicle Routing** by Sadykov et al.

## Overview

The Bucket Graph-based labeling algorithm implements the state-of-the-art way of organizing labels into "buckets" based on both vertex and resource consumption intervals. This structure significantly reduces the number of dominance checks needed, making the algorithm more efficient, especially for large VRP instances with extensive resource constraints.

### Key Features

- **Bucket Graph Organization:** Labels are grouped into buckets according to their associated vertex and resource consumption. This reduces unnecessary dominance checks by limiting them to labels within the same bucket.
- **Parallel Bi-Directional Labeling:** The algorithm supports both forward and backward search strategies, taking advantage of route symmetry in some VRP variants.
- **Dominance Rules:** Efficient dominance checks using resource-based comparisons and optional integration of additional dominance criteria from Limited-Memory Subset Row Cuts (SRCs).
- **Improvement Heuristics:** Fast improvement heuristics are optionally applied at the end of each labeling phase, in order to reach better labels.

## Disclaimer

Some features are in the highly experimental stages and will evolve based on community feedback. The following features, in particular, are subject to ongoing improvements:

- **[experimental]** Limited-Memory Subset Row Cuts
- **[experimental]** Knapsack Completion Bounds for Capacity Constraints
- **[experimental]** Bucket Arc Elimination

## Usage

### Prerequisites

- C++23 compliant compiler
- [nvidia/stdexec] for executing paralell tasks.
- [tbb] for concurrent maps.

### Compiling

```bash
cmake -S . -B build -DR_SIZE=1 -DSRC=OFF -DRIH=OFF
cd build
make -j$nprocs
```

### Input File Format

- The input file should specify the number of jobs, the time horizon, vehicle capacities, and any other relevant VRP constraints.
- The format and examples of input files can be found in the `examples/` directory.

### Running the Example Algorithm

We provide an example to run Solomon instances in the "example" folder. After building, one can run a sample instance as:

```bash
./vrptw C203.txt
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Cite

If you use this library please use the following citation:

```
@Misc{BucketGraphLabeling,
  author       = {Laio Oriel Seman and Pedro Munari and Teobaldo Bulhões and Eduardo Camponogara},
  title        = {BALDES: a modern C++ Bucket Graph Labeling Algorithm for Vehicle Routing},
  howpublished = {\url{https://github.com/lseman/baldes}},
  year         = {2024},
  note         = {GitHub repository},
  urldate      = {2024-09-17},
  month        = sep
}
```

A companion paper will be made available soon.

## Thank you notes

We would like to thank you [Vladislav Nepogodin](https://github.com/vnepogodin) for his insigths about C++.

## References

1. **A Bucket Graph Based Labeling Algorithm for Vehicle Routing.** Ruslan Sadykov, Eduardo Uchoa, Artur Alves Pessoa. Transportation Science, 2021. [DOI: 10.1287/trsc.2020.0985](https://doi.org/10.1287/trsc.2020.0985)
2. **Limited memory rank-1 cuts for vehicle routing problems.** Pecin, Diego, et al. Operations Research Letters 45.3 (2017): 206-209.