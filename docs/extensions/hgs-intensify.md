# BKS-seeded HGS intensification

[Documentation index](../README.md) · [Standalone AILS-II search](../guides/ails-ii.md)

The `hgs` command has optional BKS seeding and route-decomposition search.
Ordinary `baldes hgs INSTANCE` retains its existing behavior for other sets.
XTW selects vehicles-first route intensification automatically. Build the CLI with
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

## Compression and delayed seeds

`--compression-interval N` enables consensus-chain compression every N genetic
generations, including inside intensification subproblems. `--delay-seed` keeps
supplied plans outside the breeding pool during early exploration. Both are
opt-in and can be used independently:

```sh
target/release/baldes hgs instances/XTW/XTW-n856.vrp \
  --get-bks --intensify --time 120 --seed 7 \
  --compression-interval 75 --delay-seed \
  --solution-output /tmp/XTW-n856-best.sol
```

A delayed seed remains eligible as the returned incumbent immediately, including
when initialization exhausts the deadline. It enters the population when a
generated feasible plan reaches the best reserved seed's distance, half the
stagnation limit is reached, or half the HGS time budget has elapsed. Insertion
replaces one crossover step. The time trigger uses the current HGS subproblem's
budget; the outer search deadline still bounds all jobs.

Compression screens every pending seed as well as the protected population.
Pending seeds are projected through contractions together with incumbents and
population history, so compression cannot fix an arc incompatible with a pending
plan. Returned plans and collected route candidates use original customer IDs.
Verbose logs report seed insertions and contractions.

The library exposes the same controls as `HgsOptions::compression` and
`HgsOptions::delay_seed`. They work with `run_hgs`, `run_hgs_seeded`, and the HGS
options nested in intensification and route-decomposition settings. The defaults
are `None` and `false`. Existing explicit `HgsOptions` struct literals can use
`..HgsOptions::default()` to supply these new fields. Benchmark under equal time
budgets before selecting a policy; compression restricts the heuristic search
space and does not guarantee a better solution.

## Best-move education and reverse mode

`--best-move` evaluates candidates before applying the best move for each
customer. It immediately revisits that customer after a strict improvement.
The candidate set includes the ordinary relocate/exchange/2-opt operators,
complete intra-route positions, empty-route moves after the first pass, and
customer-focused RELOCATE*/SWAP*. SWAP* retains its own candidate lists and
periodic broad passes. Exact parent-plan records are keyed by the search policy
as well as penalties and granular size; an interrupted search supplies no new
inheritance certificate.

`--deterioration-credit X` requires `--best-move`. Credit starts at zero; improving
moves earn credit up to X in the original instance's distance units, and worsening
moves spend it. Every pruning bound uses the current allowance. Pass two and later
accept strict improvements only. Zero disables deterioration. The best feasible
witness found during education is retained even if the final penalized state is
infeasible or has a larger distance. The default policy remains the original
improvement scan with no deterioration.

```sh
target/release/baldes hgs instances/XTW/XTW-n856.vrp \
  --get-bks --reverse-mode --time 120 --seed 7 \
  --subproblem-customers 300 --exploration-level 4 --decomposition-phases 2 \
  --best-move --deterioration-credit 1 --compression-interval 75 --delay-seed \
  --solution-output /tmp/XTW-n856-reverse.sol
```

Reverse mode follows one master incumbent, without a master population. It groups
whole routes around spatial anchors and runs the specified number of regroupings
at each exploration level. Each subproblem has its own population and inherits a
feasible seed. Levels use these limits, additionally capped by explicit HGS
population, stagnation, and total-generation settings:

| Level | Population | Stagnation limit | Total genetic generations |
|---|---:|---:|---:|
| 1 | 2 | 1 | 0 (initialization only) |
| 2 | 3 | 40 | 150 |
| 3 | 7 | 100 | 400 |
| 4 | 15 | 200 | 2,000 |
| 5 | 25 | 2,000 | 20,000 |
| 6 | 25 | 5,000 | 50,000 |

`--exploration-level 0` uses constant user-supplied HGS settings. The default
reverse-mode level is 4 with two phases per level. `--max-generations N` caps
genetic generations across compression stages and restarts of each HGS call;
initial population education is separate. Delayed seeds also enter at half this
generation limit. This cap applies per subproblem, not to the whole outer run.

After a phase, full-instance local search can exchange customers across group
boundaries. It runs without a population, tries stronger repair penalties if
needed, and retains the pre-search plan if the proposal fails validation or
worsens the selected objective. Master education reserves 15% of each phase's
remaining budget. The shared outer deadline bounds all groups and master passes;
an indivisible operation may overrun slightly. `--no-master-local-search` disables
this step for comparisons. Reverse mode supports both distance and
vehicles-then-distance ranking; native XTW keeps vehicles-first ranking.

Reverse mode conflicts with `--intensify` and `--dri`, and processes its route
groups sequentially. It checkpoints the supplied seed before search and the final
best plan on completion. The library exposes these controls through
`HgsOptions::local_search`, `HgsOptions::generation_limit`, and
`RouteDecompositionOptions::{exploration_levels, master_local_search,
master_fraction, objective}`. The library decomposition defaults remain constant
depth with master education disabled. Enable each mechanism independently when
measuring quality and throughput.
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

`--barycenter` instead ranks neighbors by Euclidean distance between the mean
customer coordinates of each route. It cannot be combined with
`--std-similarity`. It uses coordinates even for explicit distance matrices, so
it is useful only when those coordinates carry meaningful spatial information.
Every fifth round still selects random neighbors, as in the default selector.

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

`--objective auto` uses vehicles then distance for named `XTW-n<dimension>`
instances with time windows, and distance for CVRP, Solomon and Homberger.
XTW automatically enables route intensification, including when `--intensify`
is omitted. Use `--intensify --objective distance` to opt into distance-only
search instead. XTW's free fleet is represented by one available vehicle per
customer; route elimination is a heuristic search, not a minimum-fleet proof. `--objective distance` and
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
With `--get-bks`, Solomon/Homberger instead use truncated tenths, as specified
by their catalogs. XTW retains its nearest-integer convention. `--rounding` is an explicit override. File
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

## Native XTW instances

Download all 100 original XTW `.vrp` files and their current published `.sol`
files, with source URLs and SHA-256 hashes in `instances/XTW/manifest.json`:

```sh
python3 scripts/setup/get_xtw.py
```

The downloader preserves existing files and reports a conflict if their content
differs from the current publication. It uses only the Python standard library.
The dataset directory is intentionally ignored by Git.

The reader accepts `TYPE: VRPTW`, `TIME_WINDOW_SECTION` and optional
`SERVICE_TIME_SECTION` directly. All sections use node IDs, so shuffled rows
and depot remapping preserve resources. Missing VRPTW windows, invalid IDs,
non-finite/reversed windows and negative service times are rejected. Service
times default to zero when their section is omitted. Explicit directed matrices
are supported as for CVRP. A provided `VEHICLES` header remains a hard fleet cap.

XTW uses TSPLIB nearest-integer distances and a vehicles-first objective, as
specified by [CVRPLib](https://galgos.inf.puc-rio.br/cvrplib/en/vrptw).
The CLI recognizes the catalog's `XTW-n<dimension>` names and applies those
defaults; `--get-bks` does not switch them to Solomon's truncated tenths.
Explicit `--rounding` still takes precedence. The library `run_hgs` API keeps
its distance objective; the automatic vehicles-first policy belongs to the CLI.

```sh
cargo build --release --features clp --bin baldes

# Original input; no conversion or manual metric/objective flags.
target/release/baldes hgs instances/XTW/XTW-n247.vrp \
  --get-bks --time 300 --solution-output /tmp/XTW-n247-best.sol

# Reuse the downloaded solution offline.
target/release/baldes hgs instances/XTW/XTW-n247.vrp \
  --initial-solution instances/XTW/XTW-n247.sol --time 300

# Parse every original instance and replay each published solution.
cargo run --release --example xtw_audit -- instances/XTW
```
