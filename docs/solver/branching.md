# Routing branching

[Documentation index](../README.md) · [Tree lifecycle](tree.md)

The tree branches on aggregate route quantities. Each child receives a typed
constraint, inherited master rows, and a compatible pricing domain. The
standard routing model uses customer vertices as packing sets and enforces
unit customer coverage with equality rows.

## Choosing families

| CLI mode | Candidate families, in priority order |
| --- | --- |
| `--branching native` (default) | Vehicle count, packing-set assignment, Ryan–Foster, arc flow |
| `--branching vehicles` | Vehicle count, then arc-flow fallback |
| `--branching ryan-foster` | Ryan–Foster, then arc-flow fallback |
| `--branching arc` | Arc flow |
| `--branching cluster` | Cluster boundary/pair variables, then arc-flow fallback |
| `--branching resource` | Accumulated-resource splits, then arc-flow fallback |
| `--branching hybrid` | Vehicle count, resource/cluster, assignment, Ryan–Foster, arc flow |

The default selector visits enabled families in descending model priority. Within
an applicable family it orders candidates by closeness of the fractional part
to 0.5, with stable ties. Phased strong branching then evaluates candidates
when enabled. These are adapter priorities, not a combined ranking across all
families. Families in the same integer priority group can contribute together.
Rust callers can set `BranchingOptions::priorities`; its defaults are vehicle
count 11, resource/cluster 10, assignment 3, Ryan–Foster 2 and arc flow 1.
These are Baldes adapter settings; reference model interfaces also register
priorities explicitly. Every supplied priority must be finite.

```sh
cargo run --release --features clp --bin baldes -- \
  vrptw instances/R101.txt --customers 10 --tree \
  --branching ryan-foster --time 60
```

Rust configures the family flags through `TreeOptions::branching`, whose type
is `BranchingOptions`. Arc flow remains the fallback when all flags are false.

## Number of vehicles

Let `v` be the sum of LP values of nonempty routes in the selected network.
For fractional `v`, the children enforce:

```text
down: number of nonempty routes <= floor(v)
up:   number of nonempty routes >= ceil(v)
```

Pricing receives this row through nonempty-route return arcs to the sink.
The split-depot source-to-sink arc contributes zero to this branch. The
original fleet upper bound remains in the master.

## Ryan–Foster

For customers `i` and `j`, let `w(i,j)` be the sum of LP route values for routes
visiting both customers. When `0 < w(i,j) < 1`, the children require:

| Child | Allowed routes |
| --- | --- |
| Apart | A route can visit at most one of the pair |
| Together | A route visits both customers or neither |

Both children work with **a single pricing network**. The together condition
applies to each route, rather than merely requiring both customers somewhere
in the final solution.

Binary resources enforce the rule during labeling and bidirectional joins.
Retained enumeration is filtered by the same rule. Existing NG routes that
repeat a pair endpoint are retired from the child master. A violation row also
keeps inherited and newly added master columns consistent with the branch.

An apart constraint consumes one binary resource; a together constraint
consumes three. The current representation holds 512 resources. Candidate
generation skips already-branched pairs and falls back to arc flow when it
cannot reserve resources for both Ryan–Foster children.

## Packing-set assignment

This family branches on assigning a customer packing set to a particular
pricing network. If its fractional assignment to network `k` is `y`, the
children enforce assignment to that network as zero or at least one.
For example, in a heterogeneous fleet this could decide whether customer 7
uses the small-vehicle network.

The current tree has **one network**. Its customer equality rows already fix
each assignment to one. Assignment constraints can be constructed and enforced,
but the normal single-network LP has no fractional assignment candidate. An
unassigned child for a mandatory customer is infeasible.

Useful assignment branching requires multiple route families/networks,
network-specific costs and resources, and master columns carrying the network
identity. Setting a graph ID alone does not add that modeling support.

## Directed arc flow

For a directed arc with fractional LP usage `f`, the children enforce flow at
most `floor(f)` or at least `ceil(f)`. Route coefficients count traversals of
the arc; branch duals also reach pricing through that arc.

`BranchConstraint::ArcFlow::arc` takes the arc's **position in
`problem.graph.arcs`**, while `Route::arcs` stores public arc IDs. Use
`RootProblem::arc_position(id)` to convert an ID to a position.

## Strong branching and diving

Strong branching is enabled by default. Its three phases retain up to 100
restricted-LP candidates, price three survivors heuristically, then evaluate
one with exact pricing. An intermediate restricted-LP or heuristic phase is
skipped when all remaining candidates fit in the next phase. Terminal phases
still run their trials, including explicitly heuristic-only configurations.
Each Rust `StrongPhase` also has a `subtree_ratio`: positive values limit
branching work using the estimated remaining subtree size; zero disables that
phase's adaptive limit. The defaults are 0.3, 0.1 and 1. Moving below the
parent's stored branch-family priority cutoff restores the full fresh quota.
These estimates schedule trials and never strengthen a pruning bound. The
`Reliable` comparison policy retains its fixed budgets.
Rust callers can enable `StrongBranchingOptions::use_history_subtree_size` to
prefer completed subtree sizes at the current depth. Until that depth has an
observation, the parent estimate remains the fallback. Partial or interrupted
subtrees do not enter this average. The default keeps parent estimates.
`StrongBranchingOptions` exposes the phase and time
budgets. Disable it with `--no-strong-branching`.

Strong round-up diving uses isolated route-fixing trials and depth,
discrepancy, node, and time limits. Disable it with `--no-diving`.
Heuristic trial objectives and dive bounds cannot certify the original node.
See [Tree lifecycle](tree.md) for bound and incumbent handling.

## Cluster branching

`--branching cluster` enables the cluster variables of Silva, Uchoa and
Subramanian, *Cluster Branching for Vehicle Routing Problems* (2024),
Sections 3 and 4.1. For a customer cluster `C`, the boundary variable counts
entries from outside `C`, equivalent to half the undirected cutset flow on a
complete depot-to-depot route. For disjoint clusters `C,D`, the pair variable
counts arc traversals in either direction between them. Both variables are
integer in an integer routing solution. Children impose floor/ceiling bounds
on their fractional LP values. The split source/sink form one depot cluster.

Coefficients are additive over physical arcs, including repeated crossings.
Branch duals therefore use the existing arc-cost pricing path, without extra
label resources. Arc-flow candidates remain available because cluster
variables alone do not ensure an integral directed solution.

`ClusterBranchingOptions::partition` accepts a partition of every customer
exactly once. Otherwise a deterministic customer MST is cut at edges strictly
above `mean + theta * standard_deviation`; `theta` defaults to 1. Distances use
the cheaper physical arc in either direction. This matches geometric distances
on symmetric routing graphs; for asymmetric/sparse graphs it is a heuristic
similarity measure. A disconnected customer graph supplies no automatic
customer clusters. The partition is constructed once per tree. Other clustering
methods and the paper's full computational experiment are not implemented.

## Accumulated-resource branching

`--branching resource` adds schedule candidates, retaining arc-flow fallback.
An individual constraint identifies a **public arc ID**, a resource coordinate, a threshold,
and a side. `threshold` is stored with an offset: the effective boundary is
`threshold - 1e-6`. It restricts the resource at that arc's head **after consumption
and waiting** to `<= boundary` or `>= boundary`. Equality belongs to both children;
there is no gap between their schedule domains. This is a continuous resource
split, not a floor/ceiling split of the resource value. Other routes need not
use the selected arc. An empty window intersection forbids that arc.

The implementation tightens the physical arc's resource window. Existing
forward/backward extensions, resource replay and enumeration enforce it.
Inherited columns also receive a nonnegative violation row, so an old schedule
cannot satisfy the new master solely because its arc path is still present.
Resource-domain changes discard inherited labels/enumeration before repricing;
parent and sibling graphs remain independent. Original resource limits remain
in effect. The existing requirement for backward equality resources to have a
fixed sink window still applies.

Automatic candidates group all arcs entering a covered customer, backed by its
unit packing row. For disposable resources, they combine forward head quantities
with backward feasible suffix upper bounds, including ancestor resource windows.
A tolerance-merged weighted event walk selects an endpoint threshold;
overlapping intervals can supply no useful split even when forward quantities
differ. The event kernel is checked against 839 native separator cases. Ordinary
fractional arcs retain arc-flow candidates; no speed improvement or complete
native child-workflow equivalence is claimed.

Rust master callers can construct `AccumulatedResourceGroupBranch` with a backing
packing-row index and unique physical arc IDs. The row must have upper RHS at
most one and nonnegative coefficients, with each member arc contributing at least
one. This proves at most one group occurrence in an integer solution; arbitrary
arc sets do not define a valid binary disjunction. Nonempty member windows are
tightened together. Empty member windows contribute zero-flow enforcement for
those arcs, while other members remain available. Parent and sibling states
stay independent.

Complete enumeration retains distinct physical paths even when their packing
support is identical. A costlier path can have a different resource profile
and remain necessary after fixing routes or applying a descendant restriction.

The Rust pricing API exposes `AccumulatedResourceBranch`, its `boundary()` and `satisfied_by()`
methods, and the add/clear methods on `Solver`. To request a split near resource value `t`, pass
`threshold = t + 1e-6`; `boundary()` reports the actual floating-point split.
Older builds used the supplied threshold directly. Existing callers that need
to preserve their split must account for the offset when upgrading.

`--branching hybrid` enables both new families alongside the native families.
Vehicle count keeps its first default priority. Resource/cluster candidates
share the next priority; assignment, Ryan–Foster and arc flow remain fallbacks.
`native` remains the default, with both new families disabled.

## Branch evaluation history

Tree search enables branch evaluation history by default, using the recovered
native ranking rules. No extra flag is required. `--no-branch-history` disables
recording and history-based ranking; `--branch-history` explicitly enables them
again. The last flag wins.

`TreeOutput::branch_history` records finite strong-trial and completed node
evaluations, exported as `branch_evaluations` in JSON. Each record identifies
the constraint, depth, evaluation kind, one-based evaluation stage (`stage`), parent bound, child estimate, fractional
distance and elapsed time. Restricted-master, heuristic-pricing, exact-pricing
and ordinary-node observations remain separately identified. Ordinary nodes have
`stage: null`; they contribute to the Rust `Reliable` comparison policy but not
the default strong-phase ranking. Unfinished exact trials and
sentinel/infinite estimates do not supply observations. Restricted-master
estimates can include artificial penalties.

For each constraint and child direction, history averages signed objective gain
divided by branching distance. It uses the latest local stage with either child
observed; an unobserved child has mean zero. Each predicted child gain is capped
by the incumbent-minus-parent gap, and their product supplies the score.
Candidates are ordered by stage, then decreasing score, then lexical generator
label. Resource groups use `ES_<packing_row>_rID_<resource>_<threshold>`, with
the stored threshold displayed to six significant digits. Other families use
canonical constraint names. Negative gains are retained and there is no positive score
floor.

At non-root nodes, half of the first strong-branching phase's budget is reserved
for history. Fresh generation stops before a lower integer priority group once
it has candidates, or when its quota fills. The stopping priority determines
which stored generators can enter history ranking. Enabled empty families still
affect this cutoff. Admission happens before stored activity is replayed; a high
historical score cannot bypass it. The remainder retains fresh candidate order. Without strong
trials, history can still rank candidates. History starts empty for each solve
and never supplies a pruning bound or marks a node infeasible.

With accumulated-resource branching enabled, the default policy also retains
evaluated group thresholds and refreshes their fractional activity at later
nodes. An older threshold can enter history selection even when fresh separation
chooses another threshold for that group. Evaluated arc, vehicle, assignment,
Ryan–Foster and cluster generators also refresh under their enabled family
priorities; numeric bounds are rebuilt around the current activity. Already applied splits, integral
activities and groups without a valid packing proof are excluded. The recorded
resource activity is the route mass strictly below the threshold boundary;
down/up history distances are that activity and its complement.

The default policy indexes observations by the number of evaluations actually
run on that candidate side at the current node, independently of whether a
trial uses restricted-master, heuristic or exact pricing. Skipped phases consume
no stage; an exact trial after skipped ranking phases is stage one. The `Reliable` policy keeps
semantic evaluation kinds separate and retains the earlier adapter selection
workflow for controlled comparisons. Custom priority scheduling applies to the
default policy. Exact resource split identities remain
distinct even when rounded display labels coincide. Rust adapters can register
original model names with `branching.generator_names.insert(&constraint, name)`.
Registering either side names both sides of that exact generator. Display names
only resolve equal-score ties and never merge histories; unnamed generators
retain canonical labels. Automatic name mapping across native model interfaces
remains unverified, so
identical native candidate selection is not guaranteed. No speed improvement
is claimed.

Rust defaults select `BranchHistoryPolicy::ObservedNative` in
`BranchHistoryOptions`. Per-network route-count branches remain fresh candidates;
this policy excludes them from stored-history ranking and generator replay.
Their trial measurements remain available in diagnostics. The earlier `Reliable` policy remains available through
Rust for controlled comparisons: it requires `minimum_observations` on both
sides, clamps negative gains, uses uncapped predictions with positive floors,
and preserves supplied order on ties. `minimum_observations` does not affect
the default observed-native policy. Set `TreeOptions::branch_history` to `None`
to disable history. The public `score_with_gap` method accepts the current
incumbent-minus-parent gap.

```sh
cargo run --release --features clp --bin baldes -- \
  vrptw instances/R101.txt --customers 10 --tree \
  --branching hybrid --time 60 --json /tmp/tree.json
```
