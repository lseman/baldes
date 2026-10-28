# BKS-seeded HGS intensification

The `hgs` command has optional BKS seeding and route-decomposition search.
Ordinary `baldes hgs INSTANCE` retains its existing behavior. Build the CLI with
`cargo build --release --features clp --bin baldes`.

```sh
# Fetch the published BKS, populate HGS with it, then run ordinary HGS.
target/release/baldes hgs instances/cvrplib/A-n32-k5.vrp --get-bks --time 30

# Download the BKS and repeatedly intensify bounded subsets of its routes.
target/release/baldes hgs instances/cvrplib/A-n32-k5.vrp \
  --get-bks --intensify --time 300 \
  --workers 4 \
  --subproblem-routes 8 --subproblem-customers 300 --subproblem-time 2 \
  --threshold 0.001 --elite-size 8 --restart-after 30 \
  --solution-output /tmp/A-n32-k5-best.sol

# Reuse a local VRPTW seed, without accessing the network.
target/release/baldes hgs instances/Solomon/C101.txt \
  --initial-solution /tmp/C101.sol --intensify --time 60 \
  --rounding tenth \
  --solution-output /tmp/C101-best.sol --seed 7
```

`--get-bks` uses the bundled standard-library Python downloader and requires
Python 3 plus internet access. It finds the instance name in the current CVRPLib
CVRP/VRPTW catalogs and downloads its public route file. Lookup and download
errors stop the command. `--initial-solution FILE` supplies the same route format
locally and conflicts with `--get-bks`. Seeds must cover every customer exactly
once and be feasible for the selected metric, capacity, windows, and fleet.
Consequently, a full-instance BKS cannot seed a `--customers` truncated instance.
Route indices are BALDES customer indices, starting at 1, with depot 0 omitted.

`--intensify` enables the outer search. Without an explicit seed, it first runs
ordinary HGS for up to 10% of the search time (at most five seconds); providing
a BKS avoids full-instance HGS setup on large instances. Its steps are:

`--std-similarity` optionally ranks route neighbors using the spatial–temporal–
demand formula in equations 11–14 of Kerscher and Minner, using the instance's
travel distance as spatial cost. Infinite time windows use a neutral temporal
factor. This changes route selection; it does not implement the paper's customer
clustering or its cross-cluster improvement phase.
The separate [`--dri` customer-clustering mode](hgs-dri.md) provides those phases.

1. Choose a random anchor route and nearby routes, measured by minimum distance
   from up to four anchor customers. Every fifth round chooses random neighbors.
   Neighborhoods expand from two routes during stagnation, bounded by
   `--subproblem-routes` and `--subproblem-customers`. Oversized routes are skipped.
2. Build a sub-instance with the same depot, capacity, windows, service times,
   rounding, and explicit matrix entries. Seed its HGS population with the
   selected routes before generating other individuals. On alternating rounds,
   try a feasible random customer swap before seeding to escape the current plan.
3. Solve with HGS, map customer indices back, and splice the new routes into the
   untouched routes. Independently replay the entire proposal before using it.
4. Accept proposals within `--threshold` times the current distance, cooling
   linearly to zero over `--restart-after` stalled rounds. Save strict improvements
   separately, so a temporary deterioration never replaces the best solution.
5. Maintain a bounded elite pool using differing directed arcs for diversity,
   ignoring route order. Restart from a randomly selected elite after stagnation.

`--workers N` enables CPU parallelism with one HGS population and random generator
per worker. Each batch selects disjoint groups of **complete routes** from the
same solution snapshot. Results are collected in job order, mapped back and
merged once, then the combined proposal is replayed against the full original
instance. A failed/infeasible subproblem leaves its original routes intact.
Arbitrary fragments of one route are not optimized independently: their boundary
loads and arrival times would couple the subproblems.

By default `--batch-size 0` selects up to `workers` jobs per snapshot. Set a fixed
positive batch size to keep decomposition and search decisions identical across
worker counts. For example, `--batch-size 4 --workers 1` solves those four jobs
serially, while `--batch-size 4 --workers 4` solves them concurrently. Fewer jobs
are dispatched when there are insufficient unassigned routes. `--intensify-rounds`
counts subproblem jobs; acceptance and best-plan updates happen once per batch.
Fixed seeds, batch size, and iteration budgets give reproducible results without
per-job wall-time limits. Changing automatic batch width changes the search
trajectory as well as throughput. Memory grows with simultaneous subproblems,
approximately `workers × subproblem_customers²` for their distance structures.

`--objective auto` uses distance for the current CVRPLib CVRP and VRPTW sets,
matching their published conventions. `--objective distance` and
`--objective vehicles-distance` select the objective explicitly.
The latter rejects increases in vehicle count regardless of threshold and tries
a reduced-fleet subproblem every fourth round. Inner HGS minimizes distance under
the selected fleet cap; it is not itself a lexicographic solver.

Search is bounded by `--time` (default 60 seconds) and `--intensify-rounds`
(default 1000). Each subproblem uses `--subproblem-time` and the ordinary HGS
`--nb-iter`, `--population-size`, and `--seed` controls. Time limits are cooperative:
a local-search operation or HGS setup can overrun a deadline. BKS downloading
precedes the search budget. The initial plan and each strict improvement are
atomically checkpointed when `--solution-output FILE` is supplied; this explicitly
replaces that file. Parent directories must already exist. `--json` retains the
existing HGS result schema; `generations` counts inner genetic generations, and
the intensification summary (including batch and worker counts) is written to stderr.
`--subproblem-time 0` uses the iteration budget within each job. The outer deadline
is checked before dispatch and before a queued job begins; an executing fixed-work
job can overrun it. Positive per-job limits are reduced to the remaining budget.

| Control | Default | Meaning |
| --- | --- | --- |
| `--threshold` | 0.001 | Initial allowed relative distance deterioration |
| `--workers` | 1 | Independent CPU HGS workers |
| `--batch-size` | 0 | Jobs selected per snapshot; 0 uses worker count |
| `--elite-size` | 8 | Maximum elite plans |
| `--elite-diversity` | 0.05 | Minimum directed-arc Jaccard distance |
| `--restart-after` | 30 | Stalled rounds before restart |
| `--subproblem-routes` | 8 | Maximum routes per subproblem |
| `--subproblem-customers` | 300 | Maximum customers per subproblem |
| `--subproblem-time` | 2 | Seconds per HGS solve |

Decomposition controls require `--intensify`; seeding and extension flags require
the `hgs` command. Seeded and intensified runs default to the instance's distance
convention (CVRPLib integers for EUC_2D; unrounded for local Solomon files).
With `--get-bks`, VRPTW instead uses truncated tenths, as specified by the current
CVRPLib Solomon/Homberger catalogs. `--rounding` is an explicit override. File
`Cost` metadata is ignored and totals are recomputed.
Current HGS capacity/demand storage requires integral demands and capacity fitting
32-bit integers; the intensifier rejects incompatible inputs.

The Rust APIs are `run_hgs_seeded`, `intensify`, and `intensify_with_callback` under
`baldes::extensions::hgs`; network access belongs to the CLI only. The callback
receives the initial plan and each strict improvement, and may fail to stop search
when a checkpoint cannot be saved. A zero API time limit selects fixed-work search.

The design is inspired by the route-subset HGS, threshold acceptance, elite-pool,
and decompose-and-intensify descriptions in the
[CVRPLib updates](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/updates/).
Those short reports do not specify complete algorithms or parameter values;
this implementation makes its own choices as documented above. It includes no
MIP subsolver, so it does not claim to reproduce the reported matheuristic or its
BKS improvements. Benchmark evidence is needed before making performance claims.

## Parallel benchmarks and GPU boundary

Use fixed batches and fixed-work inner solves to compare the same computation:

```sh
python3 scripts/benchmarks/benchmark_hgs_intensify.py \
  target/release/baldes instances/XL/XL-n2121-k186.vrp /tmp/XL-n2121-k186.sol \
  --workers 1 2 4 --batch-size 4 --rounds 24 --repeats 3 \
  --output results/hgs-workers.json
```

The runner alternates worker-count order and checks that costs, routes, and
generation counts match before reporting speedup. It records binary and seed
hashes and refuses comparisons that did not complete the round budget. It uses
an existing local seed, keeping downloading outside the measured runs.

On October 28, 2026, the command above with 300 non-improving HGS iterations,
population 12 and seed 7 completed 24 subproblem solves on `XL-n2121-k186` using
an AMD Ryzen 9 9950X (16 cores, 32 logical CPUs). Three repetitions per worker
count, using the release CLI, gave these median wall times:

| Workers | Seconds | Speedup over one worker |
| --- | --- | --- |
| 1 | 1.2601 | 1.00× |
| 2 | 0.7202 | 1.75× |
| 4 | 0.3838 | 3.28× |

All runs produced identical routes, cost (283,097), and generation counts. This
measures execution speed on a fixed search workload, not an improvement to the
published BKS or a claim about time-to-improvement. Local raw measurements and
binary/seed hashes are in `results/hgs-intensify-workers-20261028.json`.

No HGS GPU option is exposed. BALDES's existing `cuda` feature accelerates
p-step pricing, whose short-path enumeration is a different algorithm from HGS
local search and genetic population management. Enabling it does not accelerate
this extension. A useful HGS GPU backend would need batched candidate-move
evaluation, CPU/GPU agreement on capacity/time-window feasibility and rounding,
and a measured improvement in total solve time including transfers. No GPU
speedup is claimed. Independent CPU subproblems are the implemented parallel
backend; a GPU backend remains a separate, evidence-dependent extension.

## Validation

```sh
cargo fmt -p baldes --check
cargo clippy -p baldes --features clp --bin baldes --offline
cargo test -p baldes --features clp --offline --lib extensions::hgs::intensify
cargo test -p baldes --features clp --offline --test hgs_driver --test hgs_e2e --test cli hgs
python3 -m unittest discover -s scripts/tests -p test_get_bks.py
```

These checks passed: 12 algorithm unit tests, 12 focused CLI/HGS integration
tests and three downloader tests. Clippy reports existing repository warnings.
Live `--get-bks --intensify` smoke runs preserved feasible BKSs for both
`XL-n2121-k186` and `C101`. The default-feature library test build was also
attempted; unrelated coupled-kernel/robust-PD tests require unavailable
feature-gated APIs and serde_json, so the successful library validation used
`clp` as shown above.
