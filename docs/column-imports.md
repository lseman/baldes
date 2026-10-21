# Finite column imports

With `clp`, Rust callers can import finite column domains through
`master::multi_pricing::imported`. Routing networks are optional. Imported
columns are original-vector patterns, so this interface also supports finite
non-routing models without constructing synthetic routes.

## Model contract

Supply the ordinary shared `Problem<D>`, original `MasterVariable` definitions
and a list of `imported::Subproblem` definitions. Each subproblem declares:

- A nonnegative ID, distinct from other imported subproblems and routing IDs.
- The original variable IDs in its coordinate support.
- Nonnegative total multiplicity bounds, with an optional infinite upper bound.
- Its complete declared list of columns.

Each `imported::Column` has a local ID, additional objective cost, nonnegative
quantity bounds, an integrality flag and coalesced `(original_variable_id,
coefficient)` entries. IDs need only be unique within their subproblem. Entries
may be signed or fractional. An omitted entry has coefficient zero.

For every original variable declared by any imported subproblem, the interface
adds the equality `x = sum(coefficient * column_quantity)` over all participating
imports. Several subproblems may contribute to the same original variable.
Declared supports with no columns still imply zero contribution. Original
variables outside all imported supports remain independent master variables.
Original shared rows and original-variable bounds constrain these aggregate
values. Routing networks continue to contribute their own ordinary shared-row
coefficients.

The objective includes both original-variable costs and each column's additional
cost. To supply the whole pattern cost in `Column::cost`, set the corresponding
original-variable costs to zero. If an original variable represents a nonlinear
per-pattern feature, callers supply that feature's actual coefficient in each
column; this interface does not discover or separate nonlinear cuts.

The declared columns define this finite domain. Importing a subset of an
external pricing problem does not establish completeness for that larger
problem. Every declared column is loaded immediately; this is not a dynamic
pricing callback.

## Solving and restricting snapshots

```rust
use baldes::master::{
    multi_pricing::{self, imported, MasterVariable, Options, Problem},
    preprocessing::Bounds,
    root::Row,
    RowSense,
};

let nonnegative = Bounds { lower: 0.0, upper: f64::INFINITY };
let result = imported::solve(
    Problem::<2> {
        rows: vec![Row { sense: RowSense::GreaterEqual, rhs: 1.0 }],
        networks: vec![],
    },
    vec![MasterVariable {
        id: 10, cost: 0.0, bounds: nonnegative, integer: false,
        entries: vec![(0, 1.0)],
    }],
    vec![imported::Subproblem {
        id: 7, variables: vec![10], multiplicity: nonnegative,
        columns: vec![imported::Column {
            id: 0, cost: 2.0, bounds: nonnegative, integer: true,
            entries: vec![(10, 1.0)],
        }],
    }],
    Options::default(),
)?;
let node = result.snapshot()?;
let fixed = node.with_column_bounds(7, 0, Bounds { lower: 1.0, upper: 1.0 })?;
let child = fixed.solve(Options::default())?;
```

`ImportedOutput.output` contains ordinary solver status, bounds and routing
results. `ImportedOutput.values.original` contains the caller's original
variable IDs and values. `values.columns` identifies each quantity by subproblem
and local column ID. Internal master-variable IDs in low-level diagnostics are
not imported column IDs.

Snapshots offer `with_variable_bounds`, `with_column_bounds` and
`with_multiplicity`. Restrictions intersect the parent's domain. Fixed columns
keep their full objective, original-vector and multiplicity contributions;
they are not removed from the model. Parent and sibling snapshots remain
independent.

## Extending a declared domain

`node.solve_with_columns(subproblem_id, additional_columns, options)` solves a
new declared model containing the old and additional columns. IDs must be new
within that existing subproblem, and entries must use its declared original
coordinate support. Repeated extensions are supported; duplicate IDs and
invalid definitions are errors that leave the parent usable.

Explicit `with_variable_bounds`, `with_column_bounds`, `with_multiplicity`,
`with_routing_multiplicity`, `with_row`, `with_presence_row`,
`with_registered_cut_row`, `with_composite_cut_row`,
`with_resource_knapsack_row` and `fix_route`
restrictions survive the rebuild. They are replayed by original IDs and
qualified column IDs, so a changed internal allocation does not retarget a fix.
`with_multiplicity` addresses imported groups; `with_routing_multiplicity`
addresses total route use in a routing subproblem. `fix_route(route_index, quantity)`
fixes additional integer copies of a registered routing path. Rebuilds preserve
its physical arc path and quantity, subtract its contribution from every shared
row and remaining route multiplicity, and retain its objective contribution.
Fixed copies still count against group multiplicity and contribute their costs
and original vectors. Ordinary low-level child snapshots obtained through
`node.node()` are separate objects; only the import wrapper's explicit
restrictions are replayed by this extension operation.

Use `node.with_row(row, routing_coefficients, original_coefficients)` to add a
shared restriction. Original coefficients are `(original_variable_id, value)`
pairs; generated imported-column IDs are rejected. Routing coefficients use
`multi_pricing::RowCoefficients`, with physical arc order and an optional
once-per-route term. Omitted coordinates and routing families contribute zero.
Coefficients must be finite and coalesced. The caller supplies a valid cut or
branch restriction. Original-coordinate links apply the row to new imported
vectors automatically. Rebuilds replay the row and subtract any fixed routing
contribution, whether the row was added before or after fixing.

`node.with_presence_row(row, presence_coefficients, original_coefficients)`
couples nonlinear routing-event presence to original coordinates. It uses
`multi_pricing::PresenceCoefficients`, with event lists aligned to physical
arcs, and supports both pair-count and any-pair semantics. The original
presence descriptor survives extension. Fixed routes contribute their exact
presence coefficient before preprocessing the rebuilt model; imported vectors
contribute through the specified original-coordinate coefficients.

The import snapshot and multi-group session also expose
`with_registered_cut_row(row, routing, original)`,
`with_composite_cut_row(row, routing, original)` and
`with_resource_knapsack_row(row, routing, original)`. These accept the same
registered cut or resource-load descriptors as ordinary routing snapshots,
plus coefficients on original imported coordinates. Composite rows sum every
supplied kernel, including repeated descriptors on one routing group. Resource
rows retain the routing adapter's monotone binary-grid load and <= requirements.
Their descriptors survive source batches, phase-I recovery, domain extension
and tree descendants. Original-domain certificates may remain weak for these
kernels. A correct restricted LP does not establish certified optimality; complete
routing enumeration can supply the finite-domain proof after external sources
are complete. Rebuilds install all routing kernels before replaying
fixed-route contributions or running preprocessing. Callers remain responsible
for the validity of the supplied row.

Adding columns can invalidate bounds, zero fixings and infeasibility deductions
made from the previous finite domain. The implementation therefore retains the
original input definition, rebuilds the model, replays explicit restrictions and
reruns preprocessing and routing pricing. It starts with unknown lower bounds
and discards old certificates, smoothing, cached columns and derived bounds.
Even a zero-time solve of the expanded domain cannot inherit a narrower-domain
certificate. The supplied options configure this fresh solve. An empty addition
resumes the unchanged snapshot and preserves its valid bounds.

This operation rebuilds and reloads the declared model; it is not an incremental
pricing callback or a claim that an external pricing problem is exhausted. The
original definition is retained once and shared between snapshots for this
purpose. Callers obtain values through the returned output's new identity map.

`node.node()` exposes the ordinary immutable snapshot for preprocessing, trees,
additional shared rows and MIP. Use `node.values(&output.master_values)` to decode
values returned by those operations on this snapshot's descendants. Root solves
relax declared integrality; tree or MIP enforces it. Pure imports have a complete
column domain immediately, so `solve_enumerated_mip` needs no enumeration step.
### Incremental domain sources

`node.begin_import(subproblem_id)` creates an immutable collection session for
one existing group. Call `session.collect(&ImportOptions, callback)` to receive
batches from an external source. The callback receives the group definition,
including all currently known columns, and the remaining time. Return
`ColumnBatch { columns, complete }`; set `complete: true` only when the source
has supplied its entire intended domain. All other groups retain their declared
domains. This protocol enumerates a domain; it supplies no reduced-cost pricing
or external optimality certificate.

```rust
use baldes::master::multi_pricing::imported::{ColumnBatch, ImportOptions, ImportStatus};

let session = node.begin_import(7)?;
let progress = session.collect(&ImportOptions::default(), |known, remaining| {
    // Enumerate additional columns using known.columns as the authoritative
    // accepted set. Respect remaining when accessing the external source.
    Ok(ColumnBatch { columns: vec![], complete: true })
})?;
if progress.status == ImportStatus::Complete {
    let solved = progress.session.solve(Options::default())?;
}
```

A batch or time limit returns a resumable `progress.session`. Incomplete
sessions reject `solve` and inherit no finite-parent bounds or infeasibility
conclusion. Explicit envelope-covered certificate requests are described below.
Completion triggers a fresh solve with all explicit restrictions replayed.
Batches are validated before acceptance; duplicate IDs, invalid vectors or a
source error leave the input session usable. A failed collection call does not
commit its earlier batches. External callback side effects are not rolled back;
use the supplied known columns as authoritative when retrying. Time limits are
checked between synchronous callbacks, so sources must cooperate with the
remaining-time argument.

### Numerical pricing requests

`session.pricing(options)` optimizes the currently known restricted master and
returns `ImportPricing`. It disables finite-domain preprocessing, smoothing and
enumeration so deductions from the known columns cannot become assumptions
about future columns. A pricing point carries no bound or completion flag.

`point.costs()` exports an `ImportCosts` objective: additional column cost plus
`per_column` plus the sum of each original vector entry times its coordinate
price. The constant includes base and explicit multiplicity restrictions.
`point.reduced_cost(column)` validates a proposed new, zero-lower-bound column
and evaluates its numerical reduced cost with the same reconstructed row maps.
Original-coordinate objective costs enter through these duals; do not add them
a second time. These are numerical search signals, not certified lower bounds.

`point.reduced_cost_bounds(column)` returns an enclosure of that candidate's
exact-real reduced cost at the stored binary64 dual point. Exact expansions
preserve cancellation; directed intervals handle fractional underflow. `None`
means arithmetic could not establish a finite enclosure; invalid candidates
remain errors. An upper endpoint below zero establishes a negative reduced cost
for that candidate. A nonnegative lower endpoint establishes nonnegativity for
that candidate only. Neither result covers other columns or completes an
external domain. Phase-I requests automatically omit original objective costs.

Objective pricing fixes artificial quantities to zero before accepting duals.
If the objective pass did not retain necessary routing paths, recovery obtains
feasible paths from phase I and rebuilds their identities and original costs.
A small finite artificial penalty therefore cannot by itself block an otherwise
feasible restricted-domain pricing request.

A pricing request requires an optimal feasible restricted master. An empty or
infeasible restricted domain may need additional feasibility columns; the API
returns an error rather than concluding that the external domain is infeasible.
Numerical pricing requests publish no external-domain bound. Separate opt-in
certificate requests are described below. Use the collection protocol to accept columns;
only explicit enumeration completion permits the final declared-domain solve.

For a restricted domain missing feasibility columns, use
`session.feasibility_pricing(options)`. It minimizes artificial activity with
all real objective costs zero, including imported columns, original coordinates,
routing arcs and fixed route costs. The returned point has
`kind() == ImportPricingKind::Feasibility`; `restricted_violation()` reports
numerical artificial activity in the current restricted master. Positive
activity does not prove the external domain infeasible, and zero activity does
not complete that domain.

`ImportCosts.kind` identifies the pricing objective. In feasibility mode,
ignore additional column costs and minimize only `per_column` plus the vector
entries times coordinate prices. `reduced_cost` handles this distinction
automatically. After generating feasibility columns, request ordinary objective
prices to continue optimization. These phase-I signals are numerical; they do
not constitute an exact Farkas certificate for an unknown domain. Time-limited
or unsolved augmented masters return an error instead of a stale pricing point.

### Permitted column envelopes

`session.with_envelope(ColumnEnvelope { cost_lower, coordinates })` declares
restrictions on every column in that imported group, including future columns.
`cost_lower` bounds the additional `Column.cost`; each `(original_id, Bounds)`
entry bounds one vector coordinate. Declare every coordinate in the group's
support exactly once. Omitted column entries mean zero. Known columns must
already satisfy the declaration. Repeated envelopes intersect, preserving
previous restrictions.

Collection and direct extension reject violations, and envelopes persist through
completion, snapshots and later extensions. A declaration is model input; it
must describe the intended domain. It is not inferred from the known columns.

`point.envelope_reduced_cost_bound()` encloses the minimum reduced cost over the
permitted feature box and cost floor at the stored dual point. It relaxes
correlations, so it also covers unseen permitted columns. Objective pricing
uses the additional-cost floor; phase-I pricing uses zero real cost. An adverse
unbounded coordinate or missing cost floor can yield negative infinity. `None`
means no envelope was declared or arithmetic could not establish a bound.

Algebraic candidate queries may evaluate symbolic vectors outside the envelope;
model acceptance still enforces it. This is useful for constructing the affine
objective intercept. An envelope reduced-cost bound alone does not establish a
full-master bound, external feasibility or enumeration completion.

### Certificates for open domains

`session.objective_certificate(options)` requests a conservative objective lower
bound over the retained finite imports, original routing domains, and every
future column permitted by this open group's envelope.
`session.feasibility_certificate(options)` instead bounds the zero-real-cost
feasibility model. Its strict positivity proves infeasibility for that declared
model. `session.infeasibility_certificate(options)` returns `Some` only when
that strictly positive proof is available. Otherwise it returns `None`.

`ImportCertificate` exposes `kind()`, `lower_bound()` and `proves_infeasible()`.
These requests are opt-in; ordinary pricing remains numerical. They do not mark
a collection complete or enable a final finite-domain solve while it is open.

Publication requires coverage of future columns, valid dual signs and exact
fixed-route reductions. Adverse residuals at unbounded original-variable
endpoints, unsupported arithmetic or an adverse missing-column price without
a finite total multiplicity cap leave the bound unknown. Declared and explicit
group multiplicity limits both apply. A zero total cap needs no envelope; a
nonnegative missing-column price needs no finite cap. Corrections use the full
total cap, which is conservative even when known copies already consume it.

Bounds inferred by projecting only the known columns are excluded from these
certificates: future vectors can invalidate those bounds. `begin_import` opens one group; `begin_imports` below opens several together.
Unselected groups retain their declared finite domains.

### Several open imported groups

`node.begin_imports(&[group_ids])` creates `MultiImport` over one shared model.
Use `with_envelope(group_id, envelope)` for each selected group. Unknown or
duplicate IDs are rejected. Empty selections are already complete.

`collect` and `collect_priced` visit unfinished groups in the requested order,
round-robin, under a shared deadline and batch budget. Every pricing request
sees all accepted columns from every group. `ColumnBatch.complete` completes
only the requested group. `pending_groups()` reports the remaining groups;
`solve` is available only after all selected domains complete. Empty nonterminal
batches consume budget and cannot starve the other groups. Errors preserve the
input session, including batches accepted earlier in the failed call.

The objective/feasibility/infeasibility certificate methods cover **every**
unfinished group at the shared dual point. Missing coverage for any such group
keeps publication unknown. `ImportCertificate::open_groups()` identifies the
unfinished domains covered by that certificate. Completed groups use their
actual finite columns, even if their envelopes were coarse. Once all selected
groups complete, finite-domain projections may be used again. An empty selection
needs no pricing callback; use its ordinary solved output for certificates.

### Automatic pricing requests during collection

`session.collect_priced(&limits, options, callback)` runs a fresh restricted
phase-I request for each batch. Positive restricted artificial activity selects
feasibility prices; zero selects ordinary objective prices. The callback
receives `&ImportPricing` and the remaining time, with all previously accepted
columns visible through `point.subproblem()`.

Both LP requests share the collection deadline and respect stricter LP time
limits. A batch/time limit returns a resumable session; the next call rebuilds
prices for its accepted domain. Empty nonterminal batches consume the batch
budget. Invalid data or callback errors preserve the input session, with the
same external-side-effect caveat as ordinary collection.

The callback returns `ColumnBatch`. Its completion flag still declares full
enumeration of the intended domain. Finding no negative reduced cost does not
complete a session or authorize a bound. This driver automates feasibility and
objective pricing requests; it supplies no global pricing certificate. As with
ordinary collection, synchronous callbacks must cooperate with the remaining
time argument.

For mixed routing/import models, all routing families must also complete their
enumeration before the MIP can claim full-domain coverage. MIP requires `highs`.

This interface is currently Rust-only; CLI/Python input fields are unchanged.

### Trees with open external groups

`session.begin_tree()` creates an immutable `ImportTree`. Calling it declares
that every unseen column has quantity lower bound zero; register mandatory
columns before starting. Tree callbacks reject new positive-lower-bound
columns. This makes an exactly validated solution using known columns a valid
incumbent while external domains remain open.

Call `tree.run(&tree_options, &import_limits, callback)` with ordinary
`AssignmentTreeOptions` and `ImportOptions`. The callback receives fresh phase-I
or objective prices for the current node. Original-variable, qualified-column,
routing-multiplicity and physical-arc branches remain in the restriction journal
when new patterns arrive. Accepted patterns are shared with siblings. A source
completion flag must enumerate the group's entire intended domain, including
patterns excluded by a node's aggregate restrictions; completion is shared with
siblings too.

Open nodes use directed envelope certificates for objective pruning and
infeasibility. Missing or loose certificates keep the node open. An integral
restricted solution alone never closes it. Once every group completes, fresh
preprocessing and the existing finite-domain tree, enumeration and optional MIP
become available. No finite-parent proof survives opening a domain.

The result contains `tree`, `status`, `bound_mode`, `bound`, `incumbent`, `open_nodes` and
per-invocation node/batch/column counts. Resume with `result.tree.run(...)` after
a node, batch or time limit. Preserve the bound mode across resumptions.
`tree.solution()` returns the best validated objective, physical routes and
original/qualified column values from the incumbent's own lineage. Newly
registered optional columns can be absent from those stored values.

Empty nonterminal batches visit the other unfinished groups. A full round
without additions or completion returns `Incomplete` with a reusable frontier.
Callback errors or invalid batches preserve the input tree; external callback
side effects remain caller-owned. Both budgets share one deadline, and
synchronous callbacks must cooperate with the remaining-time argument.

Open sessions also provide `with_variable_bounds`, `with_column_bounds`,
`with_multiplicity`, `with_routing_multiplicity`, `with_row`,
`with_presence_row`, `with_registered_cut_row`, `with_composite_cut_row` and
`with_resource_knapsack_row`. These immutable restrictions apply during pricing and
survive future source batches. Original-coordinate links automatically carry
additive and nonlinear-row coupling to newly accepted vectors.

### Numerical oracle replies

Use `tree.run_with_oracles(&tree_options, &limits, callback)` when an external
optimizer can bound its entire missing-column domain without enumerating it.
The callback returns `ImportOracleReply` containing a `ColumnBatch` and an
optional `ImportPricingMinimum`:

```rust,ignore
let minimum = point.numerical_minimum(global_reduced_cost_lower_bound)?;
Ok(ImportOracleReply {
    batch: ColumnBatch { columns: generated_columns, complete: false },
    minimum: Some(minimum),
})
```

The supplied number must lower-bound the reduced cost of **every permitted
missing column** in that group's intended domain, at the supplied objective or
feasibility prices. A minimum over a heuristic candidate list does not satisfy
this contract. An optimizer may instead bound the entire domain, including known
patterns, which is conservative. Negative infinity means unknown; positive
infinity declares an empty missing-column domain. Neither marks enumeration
complete. The library checks request identity and direct contradictions with
returned columns; the external optimizer supplies the global-domain guarantee.

Minima are tied to the exact request, group and phase. Reusing one from an older
solve, another group or another phase is rejected. A round queries every
unfinished group at one stored dual point before branching. `for_subproblem`
can retarget a point to another unfinished group without solving another LP;
its clones retain the common request identity. Columns accepted during a round
are visible in the next round; the current round keeps its original prices and
known-column prefix.

In numerical bound mode, all required group minima are combined with the
retained-domain bound and total multiplicity caps. Existing declared envelopes
can cover a group whose reply lacks a minimum. Missing coverage or an adverse
minimum with unlimited quantity keeps the bound unknown. An unavailable
retained-domain bound also leaves publication unknown. A positive full-domain
phase-I numerical bound can close an infeasible node under that numerical
oracle contract. The result's `bound_mode` identifies its guarantee.

In certified mode, supplied numerical minima never contribute to publication or
pruning. Directed envelope certificates remain required. A resumed tree cannot
change bound mode. Plain-batch `run` requests no numerical minima; both methods
retain previously published bounds under the unchanged bound mode.

Callback, node and time limits remain resumable. A batch limit can stop partway
through a round; those accepted columns persist, but minima from different
request epochs are never mixed on resume. Allow enough callbacks to cover all
unfinished groups in one round when requesting a joint numerical bound.
A round that adds no columns and completes no domain either closes or branches
the node, or returns `Incomplete` with an open frontier.
Errors preserve the input tree, including batches accepted earlier in that
failed invocation; callback side effects remain caller-owned.
