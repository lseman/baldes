# Routing branching

[Documentation index](README.md) · [Tree lifecycle](tree.md)

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
| `--branching cluster` | Cluster boundary/pair variables with arc-flow candidates |
| `--branching resource` | Accumulated-resource splits with arc-flow candidates |
| `--branching hybrid` | Vehicle count, then combined native/new candidates |

In native mode, the selector uses the first enabled family with fractional candidates. Within
an applicable family it orders candidates by closeness of the fractional part
to 0.5, with stable ties. Phased strong branching then evaluates candidates
when enabled. These are adapter priorities, not a combined ranking across all
families.

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
one with exact pricing. `StrongBranchingOptions` exposes the phase and time
budgets. Disable it with `--no-strong-branching`.

Strong round-up diving uses isolated route-fixing trials and depth,
discrepancy, node, and time limits. Disable it with `--no-diving`.
Heuristic trial objectives and dive bounds cannot certify the original node.
See [Tree lifecycle](tree.md) for bound and incumbent handling.

## Implementation and checks

- [Typed constraints and route coefficients](../src/master/branching/mod.rs)
- [Candidate generation and history checks](../src/master/tree/candidates.rs)
- [Ryan–Foster pricing resources](../src/pricing/restrictions/ryan_foster.rs)
- [Branching tests](../tests/master/branching.rs): both children, resource limits,
  fresh versus inherited models, and pricing with active cuts
- [Tree tests](../tests/master/branch_price.rs): exhaustive small instances, strong
  trials, diving, and incomplete-bound handling

```sh
cargo test --features clp --test branching --test branch_price
```

## Cluster branching

`--branching cluster` enables the cluster variables of Silva, Uchoa and
Subramanian, *Cluster Branching for Vehicle Routing Problems* (2024),
Sections 3 and 4.1 ([local paper](../papers/L_2024_2___Cluster_branching.pdf),
PDF pages 8–12). For a customer cluster `C`, the boundary variable counts
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
A constraint identifies a **public arc ID**, a resource coordinate, a threshold,
and a side. It restricts the resource at that arc's head **after consumption and
waiting** to `<= threshold` or `>= threshold`. Equality belongs to both children;
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

Candidates split distinct observed resource profiles on arcs with unit LP
flow. They use the midpoint of the observed minimum and maximum; ordinary
fractional arcs retain arc-flow candidates. This is an initial selection rule,
not a reproduction of VRPSolver's separator or a measured performance claim.
The C add/clear accumulated-resource APIs now install and restore actual window
restrictions instead of marking the feature unsupported. The Rust pricing API
exposes `AccumulatedResourceBranch` and the add/clear methods on `Solver`.

`--branching hybrid` enables both new families alongside the native families.
Vehicle count keeps its first priority; remaining enabled aggregate families
and arc-flow candidates compete in the existing phased strong branching.
`native` remains the default, with both new families disabled.

## Branch evaluation history

`--branch-history` records finite strong-trial and completed node evaluations
in `TreeOutput::branch_history`, and exports them in JSON as `branch_evaluations`.
Each record identifies the constraint, depth, evaluation kind, parent bound,
child estimate, fractional branching distance and elapsed time. Restricted-LP,
heuristic-pricing, exact-pricing and ordinary-node observations remain separate.
Timed-out/unfinished exact trials and sentinel/infinite values do not supply
pseudo-cost observations. Restricted-LP observations remain explicitly labeled
as estimates and may reflect artificial penalties.

For each variable and direction, history averages objective gain divided by
branching distance. With the default two observations per direction in one
stage, it can rank a candidate by the product of predicted child gains. At
non-root nodes, half of the first strong-branching phase's budget is reserved
for candidates ranked by history; the remainder follows fractional order.
Without strong branching, reliable history can order candidates directly.
Missing history retains the existing order. More complete observation stages
are preferred when both directions have enough samples. History is per solve,
starts empty, and never supplies a pruning bound or marks a node infeasible.
It is opt-in and does not change the default selection policy.

```sh
cargo run --release --features clp --bin baldes -- \
  vrptw instances/R101.txt --customers 10 --tree \
  --branching hybrid --branch-history --time 60 --json /tmp/tree.json
```

The implementation is organized by family under `src/master/branching/`:
`arc_flow`, `number_of_vehicles`, `packing_set_assignment`, `ryan_foster`,
`cluster`, and `accumulated_resource`, with shared configuration in `options`.
`mod.rs` retains the typed constraint enum, dispatch and public re-exports.
Pricing-specific constraints and binary labeling resources live separately in
`src/pricing/restrictions/{accumulated_resource,ryan_foster}.rs`.
`master::branching` defines LP branching decisions; `pricing::restrictions`
enforces route-level restrictions in pricing. The old `pricing::branching`
path remains available as a compatibility re-export.
