# Stochastic branch-and-cut versus branch-and-price

`scripts/benchmarks/stochastic_branch_cut.py` is a **directed Python
reimplementation of Approach 4 (BC-Theta)** from Tomazella, Munari and Morabito
(2026), not the authors' implementation:
https://optimization-online.org/wp-content/uploads/2026/04/Tomazella_Munari_Morabito_2026_Formulations-1.pdf

It uses gurobipy and supports P1 (classical) and P3 (optimal restocking). It
reads the same generated JSON consumed by the Rust stochastic extension. A
working Gurobi license is required. No extra failure penalty is added.

## Implemented mathematical components

- Directed binary routing arcs, exact customer degrees and upper fleet count,
  corresponding to equations (2)-(4),(6). The depot has source and sink copies.
- Expected-capacity inequalities (5). Every integer candidate is decomposed
  into depot routes and disconnected cycles. Violated capacity/subtour rows
  are added with `cbLazy`; this separation is complete on integer candidates.
  Zero-demand cycles also receive a subtour inequality.
- Nonnegative route-recourse variables and route Theta cuts, equation (31),
  generated at integer candidates. Each route is anchored at its minimum
  customer ID. Route recourse is computed by a separate Python Bellman evaluator
  using equations (9) or (11). Returned incumbents are independently replayed.
- Fractional capacity cuts through `cbCut`. Up to 12 customers all subsets are
  checked. Larger cases use threshold components and pairs. This separator is
  **not CVRPSEP**, and may miss fractional cuts; integer separation still
  establishes the intended integer model to Gurobi's tolerances.

The paper specializes BC-Theta to an undirected formulation for symmetric costs.
This implementation retains directed arcs, including both orientations and their
potentially distinct recourse costs. It has a different relaxation/model size;
it is not a reproduction of the reported C++/Gurobi 11.0.3/CVRPSEP run times.
Approaches 1-3 (full EECR MIP and EECR separation) are not implemented here.

Rounded Euclidean costs can violate triangle inequalities. For such inputs, a
safe nonnegative per-route shift `B = -n * min(0, min(c_i0+c_0j-c_ij))` (with a
small numerical margin) makes route recourse charges nonnegative for Theta cuts;
`B` is subtracted on every depot departure. This preserves the original
objective. The shift is zero for metric matrices. `recourse_shift_per_route`
records it; no metric closure silently changes either solver's costs.

## Running a baseline

```sh
.venv/bin/python scripts/benchmarks/stochastic_branch_cut.py \
  instances/stochastic/generated_salavati_style/synthetic-n20-m2-r01-fr0.90.json \
  --policy p3 --seconds 30 --threads 1
```

The output is JSON with status, incumbent, lower bound, gap, replayed objective,
node/cut counts, Gurobi version and build/solve times. `--log` enables Gurobi
console logging. The default clean JSON mode is used by the comparison runner.
`--no-fractional-cuts` retains complete integer separation with a weaker search.
Callback errors terminate the solve and are raised, never reported as optima.

## Paired timing runs

Build before timing. The full generated grid needs 63-entry NG memory:

```sh
cargo build --release --features clp,ng-memory-63 --example stochastic_instance
.venv/bin/python scripts/benchmarks/compare_stochastic.py \
  instances/stochastic/generated_salavati_style/synthetic-n20-m2-r01-fr0.90.json \
  --baldes-bin target/release/examples/stochastic_instance \
  --output results/stochastic-comparison-new \
  --policy p3 --seconds 30 --threads 1 --repetitions 3
```

The destination must be new/empty. Each solver runs as a separate process on
identical input bytes. Run order alternates. Gurobi threads and Rayon/OMP/BLAS
thread environment variables share the requested count. Reports contain input
and binary/source hashes, solver versions, commands, build description, bounds,
gaps, statuses and both **process wall time** and **internal solve time**.

Gurobi's TimeLimit covers optimization after model construction; BALDES's limit
also includes node master construction. Read build/process times alongside
internal solve times. Neither algorithm receives an external warm start; BALDES
internally seeds singleton columns and a greedy feasible route packing. BALDES's default runner work budget
is 1,000,000 labels per pricing call (`--max-labels` changes it). Work limits and
incomplete pricing remain visible and do not certify optimality. Completed
pricing columns/statistics, incumbents and relaxation bounds are retained.
Column generation defaults to NG prepricing with eight-customer memories;
`--ng-size 0` on the runner disables this stage. Every emitted column is replayed
with true stochastic recourse; cyclic walks are excluded from incumbents.
The reverse Bellman-vector fallback can be compared with the original
adapter using `stochastic_instance ... --bidirectional`.

The runner marks comparable completed solves only if BOTH statuses are Optimal,
and then checks objective agreement. An Error, timeout, work limit or absence
of an incumbent is not a speedup. `results.json` retains full records;
`summary.csv` is the flat timing table. Synthetic instances cannot be compared
to original published best-known costs. Shared-host smoke timings are not a
scalability study.

## Verification

```sh
.venv/bin/python -m pytest scripts/tests/test_stochastic_branch_cut.py -q
```

Tests compare against independent scenario-tree evaluation and exhaustive route
partitions, including both policies, asymmetric costs, negative recourse from
nonmetric distances, zero-demand subtours, fleet infeasibility and zero time.
