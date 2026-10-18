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

The selector uses the first enabled family with fractional candidates. Within
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

- [Typed constraints and route coefficients](../src/master/branching.rs)
- [Candidate generation and history checks](../src/master/tree/candidates.rs)
- [Ryan–Foster pricing resources](../src/pricing/branching.rs)
- [Branching tests](../tests/branching.rs): both children, resource limits,
  fresh versus inherited models, and pricing with active cuts
- [Tree tests](../tests/branch_price.rs): exhaustive small instances, strong
  trials, diving, and incomplete-bound handling

```sh
cargo test --features clp --test branching --test branch_price
```
