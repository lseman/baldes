# Robust PDPTW formulation and timing comparison

Python/Gurobi implementations for the local
[`robust_pd.pdf`](../papers/robust_pd.pdf), Abreu, Battarra, Costa and Munari,
EJOR 335 (2026), 812–828, [DOI](https://doi.org/10.1016/j.ejor.2026.03.031).
Gurobi is used because the proposed separation runs at integer incumbents
and root LP nodes through native lazy-constraint/user-cut callbacks.

## Run

Use Python 3.9+ with `gurobipy` and a usable Gurobi license. The repository's
`.venv` already has a working installation in this workspace. For another
environment, install `pd_robust/requirements.txt` in that environment.

From the repository root:

```sh
python3 scripts/download_robust_pd.py --download
cargo build --release --features clp,multi-time-windows --example robust_pdptw

# One branch-and-cut solve, JSON output.
.venv/bin/python -m pd_robust.solve instances/RobustPDPTW/AA10.txt \
  --method branch-cut --gamma-q 1 --gamma-t 1 --sigma-q .1 --sigma-t .1 \
  --threads 1 --seconds 60

# Compare native BALDES, printed compact, separators, and exact compact.
.venv/bin/python -m pd_robust.benchmark \
  instances/RobustPDPTW/AA05.txt instances/RobustPDPTW/AA10.txt \
  --threads 1 --seconds 60 --repeats 3 --no-warm-start \
  --output internal/robust-pd/comparison

# Four classes × 5/10 requests × nominal/low/high uncertainty.
.venv/bin/python -m pd_robust.benchmark --cases pd_robust/cases/smoke.json \
  --threads 1 --seconds 5 --repeats 3 --no-warm-start \
  --output internal/robust-pd/python-comparison/smoke
```

`--methods` selects any subset of the following. Runs are sequential; method
order is deterministically shuffled within each repetition to reduce order bias.

| Method | Formulation | Intended use |
|---|---|---|
| `baldes` | Native physical-graph robust pricing and shared master | Existing implementation |
| `compact` | Paper (19)–(31), including printed W inequalities | Paper-formulation audit; see counterexample below |
| `branch-cut` | Nominal (8)–(18), dynamically adding (35) and (36) | Robust solver using proposed separators |
| `onboard-compact` | Exact on-board binaries + budgeted-load dual, robust time (28)–(31) | Independent exact compact baseline |
| `compact-guarded` | Printed compact plus RRCI/RIPEI callbacks | Audit whether separation repairs observed incumbents |

The default comparison includes the first four. The guarded model retains
the printed compact model's restrictions and is not asserted to be equivalent
to the exact model on every instance.

## Model fidelity

`models.py` implements degree, common-first-customer route tags, pairing and
precedence. It adds position variables to exclude zero-distance subtours
and enforce precedence at coincident nodes. Source resources are fixed to
their boundary conditions. A request that exceeds capacity by itself is
declared infeasible. The default travel graph uses the same structural and
single-arc robust time pruning as native BALDES. The wider preprocessing
pool attributed to Cordeau in the paper is not reproduced.

The printed time big-M in (11), (28), (29) is `l_i-e_j`. A valid inactive
constraint generally requires `max(0,l_i+d_i+t_ij-e_j)`, adding the deviation
for (29). **Safe big-M values are the default.** `--literal-big-m` reproduces
the printed values for audit. All result records identify the choice.

`compact` keeps the printed load constraints (19)–(26). Zero-budget W states
and source W values are zero; zero-deviation denominators are skipped. It
does not silently add on-board consistency or cardinality restrictions.

`onboard-compact` instead tracks binary `O[v,r]`, indicating whether request r
is on board after v, with conditional arc propagation and zero depot loads.
For each customer it applies Proposition 1 via Bertsimas–Sim dual variables:

```text
sum(q[r]*O[v,r]) + Γq*theta[v] + sum(z[v,r]) <= Q
z[v,r] >= deviation[r]*O[v,r] - theta[v]
theta[v], z[v,r] >= 0
```

This is a polynomial-size alternative, **not** printed formulation (19)–(26).
Zero budget and budgets covering all requests use simplified linear constraints.

## Separators

`separators.py` implements both proposed families:

- **RRCI (35), Algorithm 1:** grow customer sets using maximum incoming flow
  from the selected set. Recompute entering and leaving unpaired-request
  loads directly from (33), with the largest Γq deviations, rather than
  maintaining mutable top-Γ sets. Return at most 50 most violated cuts.
- **RIPEI (36), Algorithm 2:** grow maximum-outgoing-flow paths, evaluate
  worst service starts with (6), and generate tournament inequalities for
  minimal infeasible paths. Depot arcs use the restricted `L_i(u)` definition.
  Minimality is established by direct DP replay of the two subpaths, rather
  than relying on the paper's offset shortcut. Fractional seeds are not skipped.

Both are called at integer incumbents using `cbLazy`, and at optimal root LP
nodes using `cbCut`. Lazy cuts are resubmitted whenever violated; user cuts
are deduplicated. Callback exceptions terminate the solve and propagate to
the caller. `--no-root-cuts` disables fractional separation. This follows
Gurobi's [callback contract](https://docs.gurobi.com/projects/optimizer/en/current/reference/python/model.html#Model.cbLazy).

`solve` supplies the §6 adjacent-pair hard-worst-case insertion start unless
`--no-warm-start` is selected. The native example does not accept that start;
the controlled comparison above disables Python starts and records the setting.

## Printed compact counterexample

```sh
.venv/bin/python -m pd_robust.audit
```

The included three-request instance has capacity 10, demands `(2,6,3)`,
deviations `(1,2,1)` at σq=0.3, and Γq=1. Fix the route to:

```text
0 → pickup1 → pickup2 → delivery1 → pickup3 → delivery2 → delivery3 → depot
```

After pickup3, requests 2 and 3 are on board: nominal load is 9, and its
worst-case load is **11**. The printed compact load model accepts the fixed
route. Earlier marked requests can retain W=1 after a larger deviation
enters the worst-case state, allowing an incorrect subtraction at delivery1.
The guarded model, branch-and-cut and exact on-board compact all reject it.
This counterexample uses the paper's 30% setting and requires no demand
adjustment; its coordinates have positive travel times between customers.

Consequently, a compact incumbent is never considered valid merely because
Gurobi reports Optimal. The independent replay checker validates every
returned incumbent. Successful incumbent replay alone also does not prove
that all robust solutions are representable by the printed formulation.

## Timing and reproducibility

Each run uses a fresh process. Compilation is excluded from all timings.
The output directory contains:

- `results.jsonl`: parameters, input hashes, status, objective/bound, validation,
  routes, timings, node counts, cut counts, solver version, binary hash.
- `results.csv`: flat comparison data.
- `summary.md`: medians and solved counts, with objective disagreements flagged.
- `metadata.json`: machine/Python, Git revision and dirty state, exact command,
  source hashes, CPU affinity/load, solver threads, seeds, limits, start and
  separation settings. Shared-host load makes these local timings indicative.

**Wall time** includes process startup, imports, data reading, environment setup,
model construction, solve and validation. **Algorithm time** excludes startup
and includes model construction and optimization for Python, and graph/master
construction plus solving for native BALDES. Python **solver time** is Gurobi's
`Runtime`; Python build time is reported separately. Native internal time is
currently printed with millisecond precision. For tiny cases, compare repeated
wall-time medians and avoid treating rounded zero as an infinite speedup.

`--threads 1` sets Gurobi Threads and native `RAYON_NUM_THREADS=1`; zero allows
automatic thread selection. Time limits apply to optimization; construction
is timed separately and the subprocess watchdog allows additional overhead.
Gurobi MIPGap is zero with absolute tolerance 1e-6. Native BALDES retains its
default master tolerances. No Python model uses Gurobi parameter tuning beyond
the documented limits, seeds, tolerances, threads and callback requirements.
An Incomplete/TimeLimit native result remains such even if its printed bounds
coincide. Speed comparisons must use equal objectives and compatible termination.

The data use the same demand adjustment, nearest-hundredth (or explicit `none`)
Euclidean travel, ceiling deviations and departure cost as the native example.
The previously documented three workbook discrepancies under ceiling travel
deviations remain separate from formulation performance.

## Tests

```sh
.venv/bin/python -m pytest tests/pd_robust -q
```

Tests enumerate uncertainty-set extreme points and complete small-instance
partitions, check separator validity against every enumerated robust solution,
exercise fractional and integer separation, compare all formulations with the
enumerated optimum, and reproduce the compact counterexample. Solver tests
skip if gurobipy or a usable license is absent; pure data/separator tests run
without Gurobi. None of the unit tests requires the downloaded benchmark set.
