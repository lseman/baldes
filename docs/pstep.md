# P-step CVRP extension

`baldes::extensions::pstep` implements the compact continuous relaxation in
Proposition 1 of [A p-step Formulation for the Capacitated Vehicle Routing
Problem, Dollevoet, Munari and Spliet](https://repub.eur.nl/pub/123411/EI-2020-01.pdf).

The extension is available without Cargo features. Solving requires an
implementation of Baldes's `LpMaster`, such as Clp (`--features clp`).

```sh
cargo build --release --features clp --bin baldes
./target/release/baldes pstep instances/cvrplib/P-n16-k8.vrp --p 2 --threads 4
./target/release/baldes pstep instances/cvrplib/P-n16-k8.vrp --p 3 --threads 4 --enumerate --json pstep.json
```

Use `baldes pstep --help` for time, iteration and label limits. `--enumerate`
checks the bound against the complete column pool; use it on small cases.
The JSON separates a converged full-model `lp_bound` from a potentially
nonconverged `restricted_objective`. A work-limit error cannot certify a bound.
Neither output is an integer routing solution. The fleet is unlimited even if
the CVRPLIB instance name or header specifies a conventional truck count.

Rust callers can use `Problem::from_instance` on a parsed `inputs::Instance`.
It preserves the input's distance rounding. Then call
`Problem::solve_column_generation` with an empty `LpMaster` and
`ColumnGenerationOptions`; `pricing.p` and `pricing.threads` control the step
size and CPU parallelism. The older `examples/pstep.rs` illustrates full-pool
construction, and `examples/pstep_cvrp.rs` produces CSV benchmark records.

Create a `Problem` with customer demands, vehicle capacity and a directed cost
matrix. Matrix indices are start depot `0`, customers `1..=n`, end depot `n+1`.
For a conventional single-depot instance, copy depot-to-customer costs into
row 0 and customer-to-depot costs into column n+1. `None` means no arc.
Customer demands must be strictly positive; capacity and supplied costs must
be finite, with nonnegative costs. The model has an unlimited homogeneous fleet.

`Problem::enumerate` builds the complete column pool for a supplied positive
`p`. Every partial path is elementary and capacity feasible. Paths have exactly
`p` arcs, except those starting at the depot, which may be shorter. Empty depot
routes are omitted. At large `p`, the pool contains full routes and depot
prefixes; flow conservation forces incomplete prefixes to zero.

The compact prior-demand choices are zero for depot starts, maximum feasible
prior demand for depot ends, and both extremes for internal paths. Coincident
extremes are stored once. A junction customer's demand belongs to both adjacent
paths; the load coefficients account for this overlap.

`Formulation::entries` exposes sparse columns in this row order:

| Rows | Coefficient | Sense / RHS |
| --- | --- | --- |
| `0..n` | Customer degree (1 at endpoints, 2 internally) | `= 2` |
| `n..2n` | Start +1, end -1 | `= 0` |
| `2n..3n` | Start `d + q_i`, end `-(d + q(path))` | `>= 0` |

`Formulation::into_master` populates an empty LP backend with nonnegative
continuous columns. Solve through the backend's status and time-limit API.
An optimal solve gives the full p-step LP bound to solver tolerance. At `p=1`
this has the single-commodity flow bound; at `p>=n+1` it has the set-partitioning
bound. Bounds need not increase at every successive p; the paper guarantees
nondecrease when p is replaced by an integer multiple.

The reference `enumerate` implementation has combinatorial cost. Both column
count and search work are bounded by `EnumerationOptions`. Hitting a limit
returns an error instead of returning a restricted pool as a complete model.
It is intended as an exact column-pool reference for small instances and small p.

`price_steps` solves independent jobs for each start, end and prior-demand
extreme through the core bucket-graph labeling engine. Hop count is an equality
resource, preventing artificial waiting from creating extra arcs. Capacity is
disposable: every load window has lower bound zero, so waiting never helps and
a cheaper label using less capacity safely dominates. Reduced costs use actual
customer demands, not any resource waiting. Customer starts
require exactly p arcs; depot starts allow 1..p. The fixed endpoints and extreme
make the reduced cost additive along arcs, including degree, endpoint and load
duals. Each returned path is replayed against the master coefficients.

Workers share immutable instance costs and duals. Each owns its graph, label
storage and results. Results are merged in fixed job order and all threads
join before the main thread updates the LP. `threads=1` runs the same pricing
algorithm, making serial/parallel results reproducible. Parallel overhead may
outweigh the benefit on very small cases.

The solver starts with degree artificials, uses exact pricing to complete phase
I, and then switches to travel costs. Convergence requires every pricing job
to finish without a negative reduced-cost step at the requested tolerance.
Steps are deduplicated by node order and prior demand. LP columns are added in
batches. Restricted-master objectives are not full-model lower bounds until
pricing converges. Reported bounds are numerical, not independently certified.

For p<=3 the path topology itself enforces elementarity. For p=4, a relaxed
bucket solve is sufficient when its minimum is elementary. Otherwise, pricing
partitions by the first arc and forbids later returns to its head. Each resulting
four-arc path is elementary; every feasible elementary step belongs to one
partition. The relaxed minimum also bounds these partitions. Longer steps use
exact cycle exclusion around bucket solves: a cyclic relaxed minimum branches on
forbidding each arc of a detected cycle. Every elementary path lies in at least
one child; relaxed costs bound each child. This avoids the core's 15-element
local NG mask limit. It can still require exponential work for larger p.
`PricingOptions` bounds total labels and bucket subproblems per endpoint job,
as well as worker count and time. Exceeding a pricing limit returns an error.

Time windows, cuts, integer solving and integration with Baldes's
branch-and-price tree are not implemented here. For integer
optimization of this compact model, impose binary aggregate **arc flows**, as
in equations (7)–(9); making extreme p-step columns binary is not equivalent.
The exposed paths and coefficients provide the data needed for a separate
arc-flow integrality implementation.

Run `node scripts/benchmark_pstep.mjs ./target/release/baldes` from the repository
root to repeat the known-instance validation and serial/parallel timing checks.
It writes raw JSON and CSV records under `results/pstep-20261020/` and compares
every bound with full column enumeration.

Add `--grid` to run p=2,3,4 on all three bundled CVRPLIB instances, with four
workers, 120 seconds per solve and five million labels per endpoint job. Grid
records go under `results/pstep-20261020/p234/`. Reference enumeration is omitted
for the larger p=4 pools and A-n32-k5 at p=3, whose pool exceeds the CLI's column
limit. Pricing stops are recorded explicitly, without a claimed LP bound.

The validation runs gave these numerical bounds (four workers, development
profile). The normal CVRP column uses default root NG-route pricing; the integer
values were obtained separately with the normal tree solver.

| Instance | p=2 LP | p=3 LP | p=4 LP | Normal CVRP root LP | Integer optimum |
| --- | ---: | ---: | ---: | ---: | ---: |
| P-n16-k8 | 408.968169 | 434.947541 | 441.000000 | 441.000000 | 450 |
| E-n22-k4 | 336.229510 | 344.079301 | 351.662163 | 373.500000 | 375 |
| A-n32-k5 | 672.464631 | 678.388776 | Time limit | 757.983333 | 784 |

The A-n32-k5 p=4 run stopped at 120 seconds without a full LP bound. The other
eight cases converged; six also matched complete-column enumeration within
1e-5. The A-n32-k5 integer solve used capacity cuts, rank-one cuts and a
one-second HGS warm start before closing the gap at 784.
