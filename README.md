<div align="center">

<img src="docs/top_logo.png" width="400"/>

<br>

  <img src="https://img.shields.io/badge/build-on_my_computer-green">
    <img src="https://img.shields.io/github/actions/workflow/status/lseman/baldes/docs.yaml?branch=main&label=docgen" alt="doc status" />
  <img src="https://img.shields.io/github/license/lseman/baldes.svg" alt="License" />

</div>
<br>

**BALDES**: A Bucket Graph Labeling Algorithm for Vehicle Routing

<table>
  <tr>
    <td>
      <img src="docs/logo.png" alt="BALDES" width="300"/>
    </td>
    <td>
      BALDES (pronounced /'baw-dɨs/) is a C++ implementation of a Bucket Graph-based labeling algorithm designed to solve the Resource-Constrained Shortest Path Problem (RCSPP), commonly used as a subproblem in state-of-the-art Branch-Cut-and-Price algorithms for various Vehicle Routing Problems (VRPs).
    </td>
  </tr>
</table>


The algorithm is based on state-of-the-art RCESPP techniques, including interior point stabilization, limited memory subset row cuts, clustering branching, and the so called bucket graph labelling.

## 📝 Overview

The Bucket Graph-based labeling algorithm organizes labels into **buckets** based on vertex and resource consumption intervals. This structure reduces the number of dominance checks, making the algorithm highly efficient, particularly in large VRP instances with extensive resource constraints.

### 🚀 Key Features

- **Bucket Graph Organization:** Grouping labels by vertex and resource consumption to minimize dominance checks, using n-dimensional Splay Trees to keep the most acessed buckets easier to reach.
- **Parallel Bi-Directional Labeling:** Supports paralell forward and backward search strategies.
- **Dominance Rules:** Efficient dominance checks using resource-based comparisons and integration of additional criteria from Limited-Memory Subset Row Cuts (SRCs) for enhanced speed.
- **Multi-phase solving:** Out-of-the-box multi-phase solving that begins with heuristics and dynamically guides the algorithm towards an exact solution.
- **Good initial solutions** Utilizes the state-of-the-art [HGS-VRPTW](https://github.com/ortec/euro-neurips-vrp-2022-quickstart) algorithm to generate high-quality initial bounds and routes, an extension of the [HGS-CVRP](https://github.com/vidalt/HGS-CVRP) method employed by the ORTEC team to win the VRPTW track at the DIMACS Implementation Challenge. We also enchanced the HGS-VRPTW with the concepts proposed in [MDM-HGS-CVRP](https://github.com/marcelorhmaia/MDM-HGS-CVRP/), resulting in what we call the **MDM-HGS-VRPTW**.
- **Multi-stage branching**: Perform a multi-stage branching, including the recent cluster branching.
- **Improvement Heuristics:** Optional fast improvement heuristics are applied at the end of each labeling phase to enhance label quality.

## ⚠️ Disclaimer

Some features are experimental and subject to ongoing improvements:

- **[experimental]** Knapsack Completion Bounds for Capacity Constraints
- **[experimental]** Branching (Branch-Cut-and-Price) for the VRPTW

## 📋 TODO

- [x] Consider branching constraints duals
- [x] Improve branching decisions
- [ ] Refactor high order SRC cuts

## 🛠️ Building

### 📋 Prerequisites

- C++23 compliant compiler (tested with GCC 14.* and Clang 19.*)
- [NVIDIA/stdexec](https://github.com/NVIDIA/stdexec) for parallel tasks
- [fmt](https://github.com/fmtlib/fmt) for console output formatting
- [jemalloc](https://jemalloc.net/) for better memory allocation (can be disabled, but it is highly recommended)
- [xxHash](https://github.com/Cyan4973/xxHash) for improved hash calculations
- [ankerl::unordered_dense](https://github.com/martinus/unordered_dense) for improved hash map and set
- [HiGHS](https://github.com/ERGO-Code/HiGHS/tree/master) for the MIP solver

*Optional*
- [pybind11](https://github.com/pybind/pybind11) for optional python wrapper
- [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) optional for using CHOLMOD as IPM solver instead of Eigen built-in solvers
- [Gurobi](https://www.gurobi.com/) for using Gurobi as the MIP solver

### ⚙️ Compiling

```bash
cmake -S . -B build
cd build
make -j$nprocs
```

Make sure the `GUROBI_HOME` environment variable is set.

###  macOS

For macOS, it is recommended to install latest llvm via homebrew.

```bash
brew install llvm
export CC=/opt/homebrew/opt/llvm/bin/clang
export CXX=/opt/homebrew/opt/llvm/bin/clang++
export PATH=/opt/homebrew/opt/llvm/bin:$PATH
```


### 🛠️ Compilation Options

---


**Boolean Options**

| Option                  | Description                            | Default |
| ----------------------- | -------------------------------------- | ------- |
| `RIH`                   | Enable improvement heuristics          | OFF     |
| `RCC`                   | Enable RCC cuts                        | OFF     |
| `SRC`                   | Enable limited memory SRC cuts         | ON      |
| `UNREACHABLE_DOMINANCE` | Enable unreachable dominance           | OFF     |
| `MCD`                   | Perform MCD on instance capacities     | OFF     |
| `LIMITED_BUCKETS`       | Limit the capacity of the buckets      | OFF     |
| `SORTED_LABELS`         | Sort labels on bucket insertion        | OFF     |
| `STAB`$^2$              | Use dynamic-alpha smooth stabilization | OFF     |
| `IPM`$^2$               | Use interior point stabilization       | ON      |
| `TR`                    | Use trust region stabilization         | OFF     |
| `WITH_PYTHON`           | Enable the python wrapper              | OFF     |
| `SCHRODINGER`           | Enable schrodinger pool                | OFF     |
| `PSTEP`                 | Enable PStep compilation               | OFF     |
| `FIXED_BUCKETS`         | Enable bucket arc fixing               | ON      |
| `JEMALLOC`              | Enable jemalloc                        | ON      |

**Numerical and Other Definitions**

| Option            | Description                                             | Default |
| ----------------- | ------------------------------------------------------- | ------- |
| `R_SIZE`          | Number of resources                                     | 1       |
| `N_SIZE`$^1$      | Number of customers                                     | 102     |
| `BUCKET_CAPACITY` | Maximum bucket capacity if `LIMITED_BUCKETS` is enabled | 50      |
| `N_ADD`           | Number of columns to be added for each pricing          | 10      |
| `HGS`             | Maximum HGS running time                                | 5       |

> **Note 1**: Including depot and depot copy (end node).

> **Note 2**: Only one stabilization can be selected.

**Resource Disposability Definition**

To control how each resource is treated (whether disposable, non-disposable, or binary), define the disposability type for each resource in the `BucketOptions` struct. Each position in the `resource_disposability` vector corresponds to a specific resource and should be assigned a value of `0`, `1`, or `2` to indicate the disposability type:

- **0**: Disposable resource – the resource can be consumed or reset during the routing process.
- **1**: Non-disposable resource – the resource is conserved and cannot be disposed of.
- **2**: Binary resource – the resource toggles between two states (e.g., `0` for off and `1` for on).

Example configuration:

```cpp
struct BucketOptions {
    int depot                   = 0;
    int end_depot               = N_SIZE - 1;
    int max_path_size           = N_SIZE / 2;
    int main_resources          = 1;
    std::vector<std::string> resources = {"time", "capacity"};
    std::vector<int> resource_disposability = {1, 1}; // 1=disposable, 0=non-disposable
    std::vector<int> or_resources = {1};
};
```

In this example:
- **"time"** is treated as a disposable resource (`0`).
- **"capacity"** is treated as a non-disposable resource (`1`).

**TUI**

If you prefer, you can run the configurer tool, which provides a TUI for configuring BALDES.

```sh
./configurer.sh
```

![configurer](docs/configure.png)
![configurer](docs/on_off.png)

## 📂 Input File Format

The input file should specify the number of nodes, time horizon, vehicle capacities, and other VRP constraints.  
See examples in the `examples/` directory.

### 🚀 Running the Example Algorithm

To run Solomon-like instances:

```bash
./baldes vrptw ../examples/C203.txt
```

To run CVRP-like instance like the 10k instance proposed by [Uchoa et al.](http://vrp.galgos.inf.puc-rio.br/media/com_vrp/instances/Vrp-Set-XML100.zip).


```bash
./baldes cvrp ../examples/XML100_1111_01.vrp
```


### 🐍 Python Wrapper

We also provide a Python wrapper, which can be used to instantiate the bucket graph labeling:

```python
import random

# Now you can import the BALDES module
import pybaldes as baldes

# Define nodes
nodes = [baldes.VRPNode() for _ in range(102)]
num_intervals = 1

# Set random bounds for each node
id = 0
for node in nodes:
    node.lb = [random.randint(0, 9000) for _ in range(num_intervals)]  # Set random lower bounds
    node.ub = [random.randint(lb + 1, 10000) for lb in node.lb]  # Set random upper bounds greater than lb
    node.duration = random.randint(1, 100)  # Set random duration
    node.cost = random.randint(1, 100)  # Set random cost
    node.start_time = random.randint(0, 10000)  # Set random start time
    node.end_time = random.randint(node.start_time, 10000)  # Set random end time greater than start time
    node.demand = random.randint(1, 100)  # Set random demand
    node.consumption = [random.randint(1, 100) for _ in range(num_intervals)]  # Set random consumption
    node.set_location(random.randint(0, 100), random.randint(0, 100))  # Set random location
    node.id = id
    id += 1

# Create fake distance matrix with size equal to the number of nodes
distances = [[random.randint(1, 100) for _ in range(len(nodes))] for _ in range(len(nodes))]

# Initialize BucketGraph using these nodes
bg = baldes.BucketGraph(nodes, 12000, 1)

# Create random duals with size equal to the number of nodes
duals = [random.random() for _ in range(len(nodes))]

# Set the distance matrix, adjacency list, and duals
bg.set_distance_matrix(distances)
bg.set_adjacency_list()
bg.set_duals(duals)
bg.setup()

# Call the solve method
labels = bg.solve()
```

## 📖 Documentation

You can access the doxygen documentation [here](https://lseman.github.io/baldes).

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🖊️ Cite

If you use this library, please cite it as follows:

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

## 🙏 Acknowledgements

We want to thank [Vladislav Nepogodin](https://github.com/vnepogodin) for his insights into C++.

## 📚 References

1. **A Bucket Graph Based Labeling Algorithm for Vehicle Routing.** Ruslan Sadykov, Eduardo Uchoa, Artur Alves Pessoa. Transportation Science, 2021. [DOI: 10.1287/trsc.2020.0985](https://doi.org/10.1287/trsc.2020.0985)
2. **Limited memory rank-1 cuts for vehicle routing problems.** Diego Pecin, Artur Pessoa, Marcus Poggi, Eduardo Uchoa, Haroldo Santos. Operations Research Letters 45.3 (2017): 206-209. [DOI: 10.1016/j.orl.2017.02.006](https://doi.org/10.1016/j.orl.2017.02.006)
3. **Hybrid Genetic Search for the Vehicle Routing Problem with Time Windows: a High-Performance Implementation.** Wouter Kool, Joep Olde Juninck, Ernst Roos, Kamiel Cornelissen, Pim Agterberg, Jelke van Hoorn, Thomas Visser. 12th DIMACS Implementation Challenge Workshop, 2022.
4. **Hybrid genetic search for the CVRP: Open-source implementation and SWAP\* neighborhood.** Thibaut Vidal. Computers & Operations Research, 140 (2022): 105643. [DOI: 10.1016/j.cor.2021.105643](https://doi.org/10.1016/j.cor.2021.105643)
