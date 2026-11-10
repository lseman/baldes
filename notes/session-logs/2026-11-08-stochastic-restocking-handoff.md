# Handoff: stochastic Bellman restocking and 30-second comparison

## User request and current stopping point

The user requested a second stochastic extension, distinct from the existing Theta-cut stochastic solver, carrying Bellman load states inside pricing, inspired by https://github.com/amflorio/vrpsd-optimal-restocking. They then requested 30-second solve comparisons on CVRPSD instances. Their latest instruction is **write the handoff now**. Do not assume the timed comparison is complete.

All changes below are UNCOMMITTED. Preserve them. The starting branch was `main`; the latest existing commit was `3d13d9e95c19f06a1b933a8bffbea28fa75c49c1` (`fix: honor master and pricing branch enforcement policies`). No new commit was requested for this work.

Repository instructions: use CodeGraph before structural code exploration because `.codegraph/` exists. Rust test bodies belong under `tests/`; integration targets are explicitly registered in Cargo.toml. Run formatting and Clippy. Do not spawn agents unless explicitly authorized. Keep native/reference evidence under `internal/`.

## Implemented extension

- `src/extensions/stochastic_restocking.rs`: separate public module, exported by `src/extensions/mod.rs`. Reexports shared stochastic `Demand`, `Problem`, `RouteEvaluation`.
- `RestockingResource` implements `CoupledPlugin`; physical arc positions map to `(model_tail, model_head)`, with both depot copies mapped to 0.
- `BellmanState` retains customer order, Q+1 continuation values, recourse objective, endpoint, closed flag and direction.
- Backward extension prepends a Bellman transition; forward extension recomputes the prefix continuation. Joins use canonical forward replay. Equality dominance intentionally preserves different customer orders.
- Travel stays in scalar arc original/reduced costs; plugin adds expected recourse. Repeats, discontinuity, extension after returning to depot and mean demand above Q are rejected.
- Shared model restrictions: independent finite integer demands with support <= Q; positive integer Q; planned route expected demand <= Q. No stochastic time-window recourse or support > Q.
- `examples/stochastic_restocking.rs`: runnable pricing example. Arc `.cost` is ORIGINAL cost and `.reduced_cost` is reduced cost; this distinction was corrected and tested.
- `docs/extensions/stochastic_restocking.md`: usage, scope and comparison instructions.
- `tests/extensions/stochastic_restocking.rs`, Cargo target `stochastic_restocking`: directional/replay, joins, capacity, elementarity, independent hand calculation, pricing original/reduced objectives, and new certification tests.

Initial validation before the comparison changes: 4 new tests + 56 main_pricing tests passed; example ran correctly; Clp example check, formatting and Clippy passed with existing warnings.

## Full solve adapter and benchmark harness

- `src/extensions/stochastic/solve.rs`: original `Problem::solve` delegates to shared `solve_formulation(..., bellman: bool)`.
- Existing Theta behavior remains the false branch. Both branches share HGS proposals, validated warm starts, seed routes and total time budget.
- Bellman graph starts from `theta_cut_graph`, restores literal planned arc travel, installs the plugin, creates the set-partitioning master and registers arc-flow branching, rank-one and mean-capacity cuts. It does NOT install Theta recourse cuts.
- Public Clp entry point: `stochastic_restocking::solve(&problem, vehicles, SolveOptions)`.
- `examples/stochastic_instance.rs`: adds `--restocking` (requires optimal-restocking policy p3), method identity and reported `bound_mode`.
- `scripts/benchmarks/compare_stochastic_restocking.py`: reads ORIGINAL local SGJR CVRPSD files, preserves literal cost matrices, uses documented five-point demand PMFs [.1,.2,.4,.2,.1] around nominal means 3/8/13, applies no extra failure penalty, converts to shared JSON, runs each method sequentially, saves raw results/stderr and CSV, independently checks coverage, fleet, mean load and expected route costs. Single-thread environment, HGS seed 0. Default solve budget 30 seconds including HGS 2 seconds. Outer process timeout is budget + 10 seconds.

Selected originals:
- `instances/CVRPSD/SGJR/20_2_0.94_1_0.05_2` (19 customers)
- `instances/CVRPSD/SGJR/30_2_0.94_1_0.05_2` (29 customers)
- `instances/CVRPSD/SGJR/40_2_0.94_1_0.05_2` (39 customers)

Do not silently call generated paper-style instances original benchmarks. The selected SGJR files are original local data. The comparison objective is travel + replenishment, without an additional failure penalty.

## Diagnostic comparison and discovered integration gaps

The FIRST six runs completed and are preserved at:
`internal/benchmarks/stochastic-restocking-20261108/`
with stdout summary `/tmp/baldes-stochastic-compare-runs.log`.

These are diagnostic results from BEFORE the certification fixes, not final comparative results:

| Customers | Theta objective | Theta numerical LB | Theta gap | Solve seconds | Bellman |
|---|---:|---:|---:|---:|---|
| 19 | 467.4112380791113 | 463.6903304629462 | 0.7961% | 30.0064 | Error |
| 29 | 503.84529711334875 | 495.4230700000001 | 1.6716% | 30.0821 | Error |
| 39 | 567.2916306841769 | 563.1213249712381 | 0.7351% | 30.5347 | Error |

All Theta route sets independently validated. Bound mode was Numerical, not Certified. Bellman failed with `initial tree solution is not feasible in the original model` because cost-bearing plugins had no objective certificate. After adding one, a second gap appeared: master incumbent validation required a singleton exact objective, rejecting legitimate outward intervals.

The user was told these failures explicitly and that they cannot support a speed comparison. They were also told the gaps are being fixed before rerunning.

## Certification fixes now in the working tree

1. `src/graph/coupled/plugin.rs`: optional `CoupledPlugin::certify_objective(arcs, work, deadline) -> Option<(bool, [f64;2])>`, default None. Feasibility-only proof never substitutes for a cost proof.
2. `src/graph/coupled/kernel/certificate/mod.rs`: obtains cost-plugin certificates, checks finite ordered bounds, work non-inflation and deadline, applies resource coefficients and cost scale, adds objective intervals. Existing non-cost callbacks remain supported. Unknown/malformed proofs stay unknown.
3. `src/extensions/stochastic_restocking/certificate.rs`: independent outward Bellman evaluation using directed add/multiply helpers. Checks physical model continuity, terminal depot, repeats, mean capacity; respects shared work and deadlines. Computes total Bellman cost and subtracts an outward planned-travel interval to match the resource objective. Complete-route certificates do NOT supply a global pricing lower-bound certificate.
4. `src/master/model/bounds/mod.rs`: renamed witness-facing `exact_contributions` to `witness_contributions`. Witness objectives may conservatively use the upper endpoint (lower endpoint for negative quantities). Internal `exact_contributions_in_domain` takes `enclose_objective`; fixed-route reduction calls keep it FALSE and still require singleton exact costs. Exact row/count arithmetic is unchanged.
5. `src/master/tree/adapters/model/mod.rs`: incumbent validation calls `witness_contributions`.
6. `src/graph/coupled/mod.rs`: documentation updated to describe optional cost certificates.

Important: do not weaken fixed-route reduction or lower-bound proof requirements to make the comparison run. An upper objective is valid for an incumbent, not a fixed cost used in both bound directions.

## Validation state at handoff

- Release build WITH both fixes succeeded:
  `cargo build --release -p baldes --features clp --example stochastic_instance`
  log `/tmp/baldes-stochastic-cert-build2.log` (about 1 minute).
- A post-fix 3-second Bellman smoke run on the 19-customer instance SUCCEEDED:
  stdout `/tmp/baldes-stochastic-cert-smoke2.json`, stderr `/tmp/baldes-stochastic-cert-smoke2.stderr`.
  Status TimeLimit, objective 471.6057201310807, lower_bound null, Numerical mode, solve_seconds 3.017561273, nodes 1, pricing_calls 9. HGS time was 0.2 seconds for this smoke test. This is NOT the final 30-second comparison.
- Two new generic plugin certificate unit tests PASSED:
  `cargo test -p baldes --features clp --lib graph::coupled::plugin::certificate_tests`
  log `/tmp/baldes-stochastic-cert-unit.log`.
  Test bodies: `tests/unit/graph/coupled/plugin_certificate_tests.rs`.
  They cover adding plugin costs to other signed objectives, phase-I suppression, missing/malformed proofs and work inflation.
  Caveat: this compile started before the later witness-contributions change; these tests cover the certificate interface, not that later witness change.
- Integration tests were RUNNING/COMPILING at latest inspection:
  `cargo test -p baldes --features clp --test stochastic_restocking`
  log `/tmp/baldes-stochastic-cert-integration.log`.
  Tool session 42682; cargo PID 4052710 (check whether still alive rather than assuming).
  New tests include `outward_certificate_encloses_replay_and_respects_budget_deadline_and_elementarity` and Clp `tree_accepts_an_outward_certified_stochastic_incumbent`.
  The latter constructs a two-customer problem with recourse 1.26, confirms a non-singleton certificate, and checks the tree accepts a conservative incumbent.
  It was appended while the integration command was blocked on the library test build; verify it actually ran, and rerun if needed.
- Earlier Clippy passed with existing warnings before the latest certification/master changes. FINAL Clippy and formatting checks are still required for the final state.
- One earlier paused debug compile was terminated; there should be NO intentionally paused processes now. Do not leave processes stopped. Avoid running benchmark timing while Rust compilation consumes resources.

## Remaining work / next steps

1. Read the pending integration log and finish/fix tests. In particular verify the outward-certified incumbent regression passes. Do not assume its outcome from the smoke run.
2. Run final Clippy and formatting/diff checks. Because generic certificate handling and master witness validation changed, run appropriate broader certificate/master regressions (or the compiled full Clp library test binary if current). Keep proof semantics strict.
3. Once compilation finishes, rerun all six 30-second cases with the FIXED release binary into a NEW output directory; preserve the diagnostic directory. Suggested command:

```sh
python3 scripts/benchmarks/compare_stochastic_restocking.py \
  instances/CVRPSD/SGJR/20_2_0.94_1_0.05_2 \
  instances/CVRPSD/SGJR/30_2_0.94_1_0.05_2 \
  instances/CVRPSD/SGJR/40_2_0.94_1_0.05_2 \
  --seconds 30 --hgs-seconds 2 \
  --output internal/benchmarks/stochastic-restocking-20261108-validated
```

4. Check binary/source freshness and metadata. Harness records binary hash, script hash and git HEAD; changes are uncommitted, so adding source hashes/diff hash and explicit build flags would improve provenance.
5. Verify every returned solution independently, compare warm-start objectives/routes as well as final objectives, retain null bounds and failed statuses honestly. Both use fixed seed/time but time-limited HGS can collect differing numbers of columns. Do not claim identical seed column pools unless verified.
6. Configuration nuance: Theta's `solve_theta_cuts` applies additional defaults (load-completion resource 1, subset size 4, dynamic NG and max NG size 15). Bellman wrapper currently shares command controls and enables rank-one/capacity cuts but does not duplicate all Theta defaults. Record or harmonize these if claiming a tightly controlled pricer comparison. Defaults-vs-defaults is acceptable only if labeled clearly.
7. Produce a concise table of validated costs, numerical bounds/gaps, statuses and actual elapsed times. Report that Bellman currently uses conservative equality dominance and forward recomputation; no premature performance or optimality claims. If Bellman only retains HGS, say so.
8. No commit or push unless the user asks.

## Main changed/new files

Tracked changes:
- Cargo.toml
- examples/stochastic_instance.rs
- src/extensions/mod.rs
- src/extensions/stochastic/solve.rs
- src/graph/coupled/kernel/certificate/mod.rs
- src/graph/coupled/mod.rs
- src/graph/coupled/plugin.rs
- src/master/model/bounds/mod.rs
- src/master/tree/adapters/model/mod.rs

New files:
- docs/extensions/stochastic_restocking.md
- examples/stochastic_restocking.rs
- scripts/benchmarks/compare_stochastic_restocking.py
- src/extensions/stochastic_restocking.rs
- src/extensions/stochastic_restocking/certificate.rs
- tests/extensions/stochastic_restocking.rs
- tests/unit/graph/coupled/plugin_certificate_tests.rs

This handoff deliberately records unfinished validation and the pending final benchmark rerun. The user's comparison request is not yet fully satisfied.
