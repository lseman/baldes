# Bucket Graph Implementation Review — Findings & Fix Plan

Review of the BALDES bucket-graph labeling implementation against:

- **Paper**: Sadykov, Pessoa, Uchoa, *A Bucket Graph Based Labelling Algorithm for Vehicle Routing*, Transportation Science 55(1), 2021 (`papers/BucketGraph.md`)
- **RouteOpt** (`RouteOpt/packages/application/cvrp/src/pricing/`)
- **bucket-graph-spprc** (`bucket-graph-spprc/include/bgspprc/bucket_graph.h`)

Files reviewed: `include/bucket/BucketSolve.h`, `BucketUtils.h`, `BucketJump.h`, `BucketGraph.h`, `include/Bucket.h`, `src/BucketGraph.cpp`, `include/cuts/Cut.h`, `include/bnb/BCP.h`.

---

## Summary

Most of the paper's machinery is implemented correctly: bucket definition and arc generation (§3.3), SCC-driven label-correcting loop with completion bounds c̄ᵇᵉˢᵗ (Alg. 1), the `DominatedInCompWiseSmallerBuckets` walk, bidirectional split at q\* with paper-correct concatenation pruning (Alg. 2), SRC-adjusted dominance per Eq. (12) with consistent dual sign convention (raw σ ≤ 0 stored; verified through `Extend`, `is_dominated`, `ConcatenateLabel`), the `UpdateBucketsSet`/`B̄ₐ,ᵦ` incremental construction for bucket-arc elimination (§4.2), and the symmetric-case arc redefinition (½wᵥ + ½wᵥ′, ½sᵥ + t + ½sᵥ′, §3.6).

Three findings are **correctness-critical**, all in the bucket-fixing path (Section 4 of the paper). Since `FIXED_BUCKETS`/`bucket_fixing` is ON by default, these can silently produce invalid lower bounds or missed improving columns.

---

## Critical findings (correctness)

### C1. Backward jump arcs are generated in the wrong direction

`include/bucket/BucketJump.h`, `ObtainJumpBucketArcs()` (~line 447):

```cpp
const int scan_begin = b + 1;
const int scan_end   = start_bucket + node_buckets;
const int scan_step  = 1;
```

The scan and the `is_jump_candidate` predicate (candidate_pos component-wise ≥ current_pos) are identical for both directions. With the physical bucket layout (indices increase with resource value in both senses, RouteOpt convention):

- **Forward**: jump bucket must satisfy κ_jump ≻ κ_base = *higher* physical index → scanning `b+1 …` is correct.
- **Backward**: κ measures consumption from the sink, so κ_jump ≻ κ_base = *lower* physical index → must scan `b-1 … start_bucket` and require candidate_pos component-wise ≤ current_pos (strict in one dim), keeping the component-wise *maximal* (closest) candidates.

Both references confirm this:
- RouteOpt `arc_elimination.hpp` `obtainJumpArcs`: `for (int b4_i = (dir ? b + 1 : b - 1); …)` — explicitly reversed for backward.
- bucket-graph-spprc `bucket_graph.h` (~line 983): *"Forward: b' ≻ b means k' component-wise > k (jump UP); Backward: b' ≻ b means k' component-wise < k (jump DOWN)"*.

Consequence: backward jump arcs point at buckets with *larger* upper bounds, so the jump boost `q ← min(q, ũ_jump)` is a no-op and the recovery mechanism of Proposition 1 does not exist in the backward sense. After `BucketArcElimination<Backward>`, the backward labeling may fail condition (14) → the Lagrangian LB used for fixing/enumeration can be invalid → wrong pruning, possibly wrong optima.

**Fix**: make the scan direction-aware (`scan_begin/end/step` from `D`), flip the candidate predicate and the "component-wise closest" filter for backward. The jump boost in `Extend` (`min(q, jump.ub)` backward) is already correct once the jump buckets are right.

### C2. `ObtainJumpBucketArcs` skips buckets whose bucket-arc list is empty

`BucketJump.h` ~line 373:

```cpp
if (bucket_arcs.empty()) { continue; }
```

If *all* bucket arcs out of bucket `b` were eliminated, the bucket has an empty arc list and gets **no jump arcs at all**, even though the paper's `ObtainJumpBucketArcs(Γ)` requires a jump arc for every (bucket, original-arc) pair lacking a γ in Γ. Labels in such buckets become dead ends; dominated-path recovery (Prop. 1) is again broken. RouteOpt loops over every bucket unconditionally.

**Fix**: drop the early `continue`; iterate `orig_arcs` for every bucket (the `have_path` check already handles the empty list naturally).

### C3. Fixing threshold θ omits the fleet-size multiplier (paper Eq. 15–16)

`BucketUtils.h`, `bucket_fixing()` / `heuristic_fixing()`:

```cpp
gap = std::ceil(incumbent - (relaxation + std::min(0.0, min_red_cost)));
```

Paper: LB = z_RSPF + **U·**min(0, z) and θ = UB − LB + min(0, z) = UB − z_RSPF − (U−1)·min(0,z). The implementation effectively uses U = 1. When the pricing still returns z < 0 and more than ~2 vehicles are available, the implemented θ is **smaller** than the valid one → arcs can be fixed that lie on improving paths → unsound.

RouteOpt sidesteps this by only running elimination after CG convergence (`opt_gap = ub − node.value`, with z ≈ 0). BALDES calls `bucket_fixing` on the stage-3→4 transition where `min_red_cost` can still be noticeably negative.

**Fix** (either):
1. multiply `min(0, min_red_cost)` by the number of available vehicles (e.g. `problem->num_vehicles` / U_m), or
2. only run fixing when `|min_red_cost| < ε` (post-convergence, RouteOpt-style) and document the invariant. Also reconsider `std::ceil` — only valid for integral cost data.

---

## High-impact findings (logic inverted / wrong adaptation)

### H1. `considerRegenerate()` halves the bucket count instead of doubling it

`BucketGraph.h` ~line 1038. The comment says it mirrors RouteOpt's `step_size /= 2` ("doubles the bucket count"), but `bucket_interval` in BALDES is the **number of buckets per full-range vertex** (ξ in the paper: `splits ≈ node_range·intervals[r].interval / full_range` in `define_buckets`), not a step size. So

```cpp
bucket_interval = bucket_interval / 2;   // makes bins COARSER
```

does the opposite of RouteOpt (`step_size /= 2; num_buckets_per_vertex *= 2;`) and of the paper's scheme (ξ ← 2ξ when dominance-checks/non-dominated > 500). Coarser bins → even more dominance checks → repeated halving → degenerates toward ξ = 1.

**Fix**: `bucket_interval *= 2` with a cap (paper: avg non-fixed bucket arcs per vertex < 10 000; RouteOpt: `BucketResizeFactorNumBucketArcsPerVertex`). The "even / halving" guard becomes unnecessary; replace the bucket-count cap with an arcs-per-vertex guard to match the references. Also mirror RouteOpt's `if_stop_arc_elimination` semantics (BALDES sets `fixed = false`, which re-enables fixing with stale `gap` — verify `gap` is recomputed before the next elimination).

---

## Medium findings (deviations from the paper — safe direction, but worth deciding deliberately)

### M1. Cost matrix used as travel time in feasibility checks

`UpdateBucketsSet` (BucketJump.h) and `check_feasibility` (BucketGraph.h) use `getcij()` both as reduced-cost component and as the *time* increment. Worse, in `UpdateBucketsSet` the RCC dual is subtracted from `cost` **before** it is used in the time-feasibility comparison, so duals perturb a resource-feasibility test. With ν ≥ 0 this direction is conservative (fewer arcs fixed), but it is conceptually wrong and breaks any instance where cost ≠ time. **Fix**: use the arc's actual time consumption (`node.duration + travel_time` from the distance matrix, untouched by duals).

### M2. θ-compatibility in elimination ignores the SRC term

Paper Prop. 2/4 define ω-compatibility with the −Σ_{S+S≥1} σ term; `UpdateBucketsSet` checks `label_cost + cost + L_opposite->cost < θ` without it. Since σ ≤ 0 the omission *under*-estimates path cost → more pairs counted compatible → fewer arcs fixed → **safe but weaker** fixing. Optionally add the SRC adjustment (both references handle R1C states here) for stronger elimination.

### M3. `DominatedInCompWiseSmallerBuckets` aborts the whole DFS on the first pruned branch

`BucketSolve.h` ~line 1299: the paper's pseudocode returns false only for the *current branch* (other Φ branches of ancestors must still be explored); the implementation `return false`s out of the entire walk. Safe (a missed dominance only means extra labels) but weakens the dominance rate the bucket graph exists to provide. **Fix**: `continue` instead of `return false` (don't push the pruned bucket's Φ neighbors; keep draining the stack).

### M4. Elimination uses dominated labels

`BucketArcElimination`/`UpdateBucketsSet` iterate `get_labels()` without filtering `is_dominated` (paper uses the non-dominated sets **L**). Extra compatible pairs → fewer arcs fixed (safe) + wasted work. spprc filters `L->dominated`. **Fix**: skip dominated labels in both loops.

### M5. Forward/backward bucket index spaces silently assumed identical

`Extend<…, Full::Reverse>` computes the arrival bucket with `get_bucket_number<Forward>` and `ConcatenateLabel` indexes `bw_buckets` with it. This only works because `define_buckets<Forward>` and `<Backward>` produce identical layouts. Add a debug assertion (same `num_buckets_index`, same bounds) or compute via `get_bucket_number<Backward>` to make it robust against future layout changes.

---

## Low / informational

- **L1. Active SRC cuts capped at 64** (`Cut.h::updateActiveCuts`, sorted by |dual|). Pricing ignores overflow cuts → computed reduced costs are below true values → LB stays valid (under-estimate), but master/pricer disagree on reduced costs (duplicate/zero-value columns possible). Document; consider raising the cap or warn when truncation triggers.
- **L2. Bucket boundary convention**: `Bucket::contains` is closed `[lb, ub]` on both ends while `get_bucket_number` assigns boundary values via `floor` (half-open semantics, paper-style). Consistent in practice; the closed `contains` is only used as a sanity check. Fine.
- **L3. q\*-adjustment scheme** (`updateSplit`, EMA on a work-weighted imbalance signal) differs from the paper (±20% count comparison → 5% move) and RouteOpt (decaying meet-point factor). Legitimate variant; document the departure and keep an eye on oscillation (the EMA static persists across instances within a process run).
- **L4. "Exact completion bounds" technique** (paper §5: during full runs keep a label beyond q\* only if a θ-compatible opposite label exists) is not implemented — full fixing/enumeration runs store everything. Memory/perf opportunity, not correctness.
- **L5. `get_best_label` ignores its `topological_order/c_bar/sccs` parameters** — dead parameters; minor cleanup.
- **L6. Heuristic stage thresholds** (`inner_obj >= -1`, `-10`, `-0.5`, iteration caps) are heuristics layered on the paper's 3-stage column generation; behavior matches the spirit (stage 1 = one label per bucket via cost-only dominance; stage 2 = no ng/SRC states in dominance; stage 4 = exact). OK.

---

## Verified-correct against paper (no action)

| Paper element | Implementation | Status |
|---|---|---|
| Bucket definition per vertex, step from ξ (§3.3) | `define_buckets`, `get_bucket_number` | ✓ (per-node rounded splits, acceptable variant) |
| Bucket arc set Γ + head bucket rule (§3.3) | `generate_arcs::try_add_arc` | ✓ (`lb + d ≤ u_v′`, head = `max(lb+d, l_v′)`) |
| Φ_b adjacency, extended graph, SCC topo order (§3.3) | `computePhi`, `SCC_handler` | ✓ (fw → b−1, bw → b+1 under physical layout) |
| Alg. 1 loop, c̄ᵇᵉˢᵗ update in κ-lex order | `labeling_algorithm` | ✓ (heap-driven within SCC; c̄ update sorted by lb/ub) |
| Dominance rule Eq. (12) incl. SRC duals | `is_dominated` | ✓ (raw σ ≤ 0 stored; `cost_diff ≤ Σσ` matches) |
| RC-bracketed single-pass bin dominance | port of RouteOpt `doDominance` | ✓ |
| rc2 whole-bin prune | port of RouteOpt `RC2TillThisBin` | ✓ (prefix min/max per direction) |
| Alg. 2 split at q\*, crossing-only concatenation | `Full::Partial` + `Extend<Full::Reverse>` | ✓ |
| `ConcatenateLabel` completion-bound prune + ω-compat (incl. SRC ≥ den) | `ConcatenateLabel` | ✓ |
| `UpdateBucketsSet` Φ-recursive B̄ construction, lex-order merge | `BucketArcElimination` | ✓ structure (see M1/M2/M4) |
| Fix test: no b̄ ⪯ b̄_arr in B̄ₐ,ᵦ | `has_compatible_predecessor` (Φ-reachability) | ✓ |
| Jump extension boost `max(q, l̃_jump)` / `min(q, ũ_jump)` | `Extend` jump path | ✓ (given correct jump buckets, see C1) |
| Symmetric case ½-consumption arcs, q\* = midpoint, fw-only run (§3.6) | `set_adjacency_list<Symmetric>`, `ConcatenateLabel<Symmetric>` | ✓ |

---

## Fix plan

### Phase 1 — correctness (do first, single PR)

1. **C1**: direction-aware jump-arc scan in `ObtainJumpBucketArcs` (`BucketJump.h`).
   - Backward: scan `b−1 → start_bucket`, candidate predicate `≤` with one strict, keep component-wise maximal candidates.
   - Verify: on a VRPTW instance (e.g. `examples/C203.txt`) with `FIX_BUCKETS` on, assert every backward jump bucket has `ub` component-wise ≤ base bucket `ub` (and at least one strict).
2. **C2**: remove the `bucket_arcs.empty()` early-continue in the same function.
   - Verify: after elimination, count buckets with eliminated arcs but no jump arcs — must be 0 whenever a κ-larger bucket still carries the arc.
3. **C3**: θ formula — multiply `min(0, min_red_cost)` by the fleet bound (or gate fixing on `min_red_cost > −ε`).
   - Verify: solve a set of instances with known optima (Solomon C2xx/R2xx) with fixing on/off; objective must match in both modes.

### Phase 2 — inverted heuristic

4. **H1**: `considerRegenerate` → `bucket_interval *= 2`, replace bucket-count cap with non-fixed-arcs-per-vertex cap; fix comment.
   - Verify: log dominance-check ratio across iterations; ratio must drop after a regeneration instead of rising.

### Phase 3 — fidelity / strength improvements

5. **M1**: separate travel-time lookup from reduced-cost in `UpdateBucketsSet` / `check_feasibility`; never apply duals to resource checks.
6. **M4**: skip `is_dominated` labels in elimination loops.
7. **M3**: branch-local pruning in `DominatedInCompWiseSmallerBuckets` (`continue`, not `return false`).
8. **M2** (optional): add SRC state term to θ-compatibility for stronger fixing.
9. **M5**: assert fw/bw bucket layout equality at `define_buckets` exit.

### Phase 4 — tests (gap: repo has none for this subsystem)

The bucket-graph-spprc repo ships unit tests/benchmarks for exactly these invariants — use it as a template:

- Unit: jump arc direction invariants (both senses), jump boost semantics, Φ adjacency, `get_bucket_number` boundary cases, dominance rule with synthetic SRC states (Eq. 12 truth table).
- Integration: root-node LB equality with fixing on vs. off on 3–4 Solomon instances; assert pricing z monotonicity per stage.

### Verification baseline

Before any change: record root LB + node counts + time on `C203`, `R202`, `RC207`, `XML100_1111_01.vrp` with current main. Re-run after each phase; Phase 1 may change results (that's the point — current fixing can be unsound); Phases 2–3 must keep LB identical and only shift time/label counts.
