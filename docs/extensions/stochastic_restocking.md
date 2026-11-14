# Bellman restocking pricing

`extensions::stochastic_restocking` is an alternative to the existing
`extensions::stochastic` Theta-cut formulation. It uses optimal preventive
restocking decisions inside pricing, with Q+1 continuation values in each label.
Do not add Theta recourse costs to a network using this resource: that would
charge recourse twice.

Run the complete example with `cargo run -p baldes --example stochastic_restocking`.

```rust
use baldes::extensions::stochastic_restocking::{Demand, Problem, RestockingResource};
use baldes::graph::coupled::{CoupledResource, PluginResource};
use baldes::Graph;

let demand = Demand::new(vec![(0, 0.5), (4, 0.5)])?;
let problem = Problem::new(4, vec![demand.clone(), demand], vec![
    vec![0., 3., 7.], vec![3., 0., 4.], vec![7., 4., 0.],
])?;
let mut graph = Graph::<2>::new(4, 0)?; // source 0, customers 1/2, sink 3
let mut endpoints = Vec::new();
for (tail, head, model_tail, model_head) in [
    (0,1,0,1), (0,2,0,2), (1,2,1,2), (2,1,2,1), (1,3,1,0), (2,3,2,0),
] {
    graph.add_arc(tail, head, problem.costs()[model_tail][model_head])?;
    endpoints.push((model_tail, model_head));
}
let resource = RestockingResource::new(problem, endpoints)?;
graph.set_coupled_resources(&[CoupledResource::Plugin(PluginResource::new(resource))])?;
```

The endpoint array uses physical arc **positions**, in graph arc-vector order.
Both depot copies map to model ID 0. Rebuild the resource after topology changes.
Keep original arc objectives equal to planned travel; apply master duals through
normal reduced-cost updates. The coupled kernel adds expected recourse to both
original and reduced objectives, including replay at joins and reconstruction.
It also suppresses this cost during phase-I feasibility pricing.

Backward extensions prepend one Bellman transition using contiguous load slices
for each demand outcome. A state holds recourse-only continuation costs after
its first customer's service for every residual load. Planned suffix travel is
tracked separately; literal probability-mass deviations receive a travel
correction. Closing the route uses the shared model's canonical evaluation.
The transition chooses the cheaper direct trip or preventive depot trip before
the next customer. Running out during service causes a depot round trip.
Forward extensions recompute the prefix continuation; this is more expensive
than backward extension. Joins replay the suffix from the prefix. Backward
dominance requires visited-set inclusion. An inexpensive recourse-vector proposal
uses a uniform forward-error enclosure rather than replaying interval Bellman
vectors for every comparison. Planned travel, literal probability mass and
roundoff in suffix/prefix evaluations are included in a compensation paid
against the scalar label cost. The shared kernel admits this
proof for the exclusive backward resource even when physical suffix histories
differ. Forward states retain equality dominance.

The solve adapter reverses the private pricing graph and runs a complete forward
sweep, matching the earlier Bellman pricer. Bellman transitions still prepend
customers in the original problem direction. HGS/initial routes are adapted on
entry, certificates reverse the physical arc sequence, and returned routes are
replayed in their original direction. This avoids generic backward suffix
programs while recovering load-walk and subset completion machinery from
`abd74267`. Each extension prepends one Bellman operator. Up to four influential
customers condition the prefix tables; best/second predecessor values exclude
immediate two-cycles. Completion checks run before and after Bellman evaluation.
Packing cut penalties may be relaxed away; unsupported cut families retain
unpruned labeling. Prepared completion resources are local to a pricing call,
so model replay and incumbent certification remain unrestricted. Pruned searches
report a minimum capped at the pricing threshold and cannot supply full-domain
fixing/enumeration labels. Subset completion currently supports up to 63
customers and includes an outward correction for literal probability mass;
unsupported cases keep backward labeling without this pruning. Knapsack completion is recovered
in the module but disabled pending timing evidence.

A separate directed load-indexed q-route relaxation supplies a reduced-cost
bound even before exact labeling finishes. It includes a conservative global
recourse envelope, covers branch-restricted route domains by relaxation, and
feeds the current master's dual correction. Work/deadline exhaustion, zero-load
negative cycles and unsupported row/resource combinations remain unknown.
An additional adaptive NG relaxation recovers the old scalar bound sweeps and
DSSR cycle refinement. It starts with two nearest neighbours, grows memory
along forgotten cycles up to 15 customers, and retains only completed sweeps.
Arc additions round downward. Rounded positive mean loads give an acyclic
sweep capped at 512 load levels; zero rounded loads and more than 63 customers
keep the q-route fallback. Each call is capped at 200 ms, 100,000 states and
the remaining shared proof-work budget. Separate scalar candidate stages use
16, 4 and 1 labels per bucket, preserve the current NG memory, cuts and branch
restrictions, and replay elementary candidates with the full Bellman objective.
Their surrogate cost is replaced while retaining the cut penalties. If these
heuristic stages find no useful route, a complete scalar sweep attempts to
establish a minimum over the relaxed domain. The candidate/proof sequence has a
two-second/100,000-label budget. Only a completed scalar sweep supplies a
numerical bound; it does not establish completion of Bellman pricing.

The Bellman solve adapter enables shared scalar mean-load completion bounds by
default (`BALDESloadCompletionResource = 1`) with six selected NG elements
(`BALDESloadCompletionSubsetSize = 6`). These bounds run in the scalar fork,
where the coupled Bellman resource has been removed; they do not replace the
Bellman continuation bounds above. Explicit pricing options take precedence:
resource `-1` disables shared load completion, subset size `0` keeps its
unconditional bounds, and sizes `1..=6` condition the scalar NG completion.
The example uses these Bellman defaults unless overridden by
`--no-load-completion-bound` or `--completion-subset-size N`.

Before exact labeling, a 250 ms local search uses the current master route pool.
Insertion, removal, replacement, relocation and reversal moves are ranked by
scalar reduced cost. The best moves receive full Bellman replay, explicit raw
arc-price adjustments and registered cut coefficients. Branch-incompatible moves
are rejected. Unsupported row types disable this search. Improving columns are
heuristic results and never supply a minimum. Public route replay evaluates
original objectives, so its result alone cannot identify improving reduced-cost
columns. If neither candidate stage finds a useful route, pricing continues
with Bellman labeling.
This tree bound supersedes the assignment fallback whenever stronger.
The exact graph uses an advancing hop-count coordinate, as in the historical
pricer; scalar NG candidates use mean load as their main coordinate. If scalar
prepricing finds no useful route, every requested CG phase falls back to staged
exact Bellman search. It does not spend the budget in truncated coupled-label
phases first. The functor uses raw arc and cut prices for that fallback even
when the master requested a heuristic phase, so completed minima never describe
rounded heuristic costs.

Exact Bellman pricing remembers targets from both replayed heuristic columns
and exact columns. It tries a remembered negative reduced-cost target, then a
target eight times closer to zero, before the full pricing threshold. The first
target can be seeded from the independent route relaxation. Targets include the
route-constant dual offset. Only completed stages contribute a numerical
minimum, which survives later interrupted stages. A single active network can
publish the associated multiplicity-weighted master correction even with no
new columns. It supplies neither a completion signal, a subgradient nor a
directed pricing certificate. New pricing points and forks clear these minima.

The solver returns batches of up to 20 improving Bellman columns without
waiting for pricing exhaustion. These batches and interrupted sweeps are
explicitly incomplete; their observed minima never prove convergence.

The shared model assumes independent finite integer demands, each supported in
0..=Q, a positive integer capacity, and a directed metric travel-cost matrix.
Routes must be elementary and their total expected demand cannot exceed Q.
The plugin enforces both conditions itself, independently of scalar resources.
Restocking changes expected cost; stochastic travel time, recourse time windows,
correlated demands and demand support above Q are not implemented here.
Complete routes have an independent outward Bellman cost certificate for master
incumbent validation. A separate global objective envelope covers unseen routes
and feeds the master's lower-bound relaxation even when Bellman pricing times
out. Depot-triangle costs give a nearly zero recourse lower bound, with an
outward correction for literal probability mass; nonmetric costs subtract the
maximum depot-triangle saving per edge. Unsupported arithmetic and
exhausted proof budgets remain unknown. Inspect the reported bound mode and
status before claiming certified tree bounds; this envelope does not establish
pricing completion or optimality.
The solve adapter also computes a deterministic assignment relaxation with depot
copies, then corrects it with the global recourse envelope. Hungarian potentials
are independently checked using directed arithmetic. This bound survives an
interrupted first pricing pass; the instance runner labels it
`RestockingAssignmentRelaxation` in `lower_bound_source`. It may be substantially
weaker than a completed stochastic pricing bound. Its bounded computation counts
against the solve budget; if unavailable, nonnegative total travel gives zero.
The existing Theta solve entry point remains a separate formulation. With Clp,
`stochastic_restocking::solve(&problem, vehicles, options)` uses the same tree and
HGS warm-start controls with Bellman pricing and no Theta recourse cuts.
The instance runner selects it with `--solve --restocking --policy p3`.

For a timed comparison on the repository's original SGJR files, build first:

```sh
cargo build --release -p baldes --features clp,ng-memory-63 --example stochastic_instance
python3 scripts/benchmarks/compare_stochastic_restocking.py \
  instances/CVRPSD/SGJR/20_2_0.94_1_0.05_2 \
  instances/CVRPSD/SGJR/30_2_0.94_1_0.05_2 \
  instances/CVRPSD/SGJR/40_2_0.94_1_0.05_2 \
  --seconds 30 --hgs-seconds 2 --output internal/benchmarks/stochastic-comparison
```

The `ng-memory-63` feature lets the exact engine retain all customers in its
NG word for these cases, matching the historical build. Larger instances still
use the plugin's visited set for elementarity.

The harness preserves the literal matrices, uses one thread and seed 0, counts
HGS inside each budget, independently validates solutions, and saves raw results,
input hashes and a CSV summary. It applies no additional failure penalty.

Algorithm reference: Florio et al.'s
[VRPSD optimal-restocking solver](https://github.com/amflorio/vrpsd-optimal-restocking/blob/main/src/Label.cpp).
The implementation shares model inputs with BALDES's existing stochastic route
evaluator; it does not depend on the reference solver's CPLEX or CVRPSEP code.

The local historical recovery audit
records the initial 6.18%/5.72% gaps on 29/39 customers. The subsequent
NG-refinement recovery audit
closed 29 at 503.826971. The
mean-load completion audit
records the current defaults, validated 39-customer bounds, actual elapsed times,
source provenance and remaining numerical/certified boundaries. These audits
are local investigation artifacts under `internal/benchmarks/` and are not
published with the documentation.

## Optional initial-plan refinement for Theta

The default Theta-cut solver can reuse true-recourse plan improvement without
moving Bellman recourse into pricing. Set
`SolveOptions::incumbent_search_time_limit` to a positive duration, or pass
`--incumbent-search-seconds 0.25` to `stochastic_instance`. The default is zero.
The search requires an existing complete initial plan, supplied explicitly or
by HGS. HGS, refinement, and the tree share `tree.time_limit`.

The search considers customer relocations, reversals, and swaps, keeps at most
48 travel-ranked candidates per step, and replays affected routes with the
selected stochastic policy. It accepts a move only after replaying the whole
plan, preserving exact coverage, mean-load feasibility, and the fleet limit.
It stops after 64 improving steps or its cooperative deadline. A route replay
already in progress may overrun that deadline.

Improved Theta routes retain their relaxed column objectives. Existing seed
validation supplies exact recourse through Theta variables, so recourse is
counted once. `SolveResult::incumbent_search` reports replayed before/after
objectives, evaluated and accepted moves, and elapsed time. This is a primal
heuristic; it provides no pricing certificate or lower bound.

Theta retains completion subset size 4. Controlled size-4/size-6 comparisons
on SGJR 19, 29, and 39 gave identical final bounds; see the
local Theta transfer audit.
