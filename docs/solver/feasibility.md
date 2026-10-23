# Routing feasibility recovery

[Documentation index](../README.md) · [Tree lifecycle](tree.md)

The ordinary routing root and its descendants automatically attempt phase I
when an exact pricing round leaves positive artificial activity after the
configured penalty updates, or when the restricted master is primal infeasible.
The same recovery runs through Rust, CLI and Python routing solves; it requires
no new option. Generic CG controllers opt in through
`Controller::recover_feasibility`; the default declines this routing-specific
operation. External column imports retain their separate feasibility-pricing API.

Phase I copies the current master, keeps its physical rows and real-column
upper bounds, zeros real costs and the objective offset, and adds unit-cost
violation columns for every row. It retains fixed-route RHS reductions,
inherited cuts and branch rows. Independent pricing keeps logical exclusions
and resource restrictions, while discarding original-cost fixing, completion
caches and cutoff enumeration. Exact phases use unrounded arc and cut prices.

Recovery does not require enumerating all physical paths. It generates columns
under the phase-I duals and re-solves the auxiliary master. A zero-artificial
point is checked against the real matrix and upper bounds before recovery is
accepted. Recovered columns enter the original master with their original
costs and current coefficients. Original artificials are then fixed to zero
so an expensive feasible route cannot lose to a finite artificial penalty.
The objective offset, incumbent and original route costs remain unchanged.
Stabilization and objective convergence bookkeeping restart before ordinary
pricing resumes.

## Outcomes and proof scope

`Output::feasibility` records auxiliary rounds separately from original-objective
iteration history. CLI root JSON exposes a `feasibility` list; Python returns
a tuple of typed `FeasibilityReport` records. Each record includes status, iteration/pricing counts,
LP work, elapsed seconds, artificial activity, an optional feasibility lower
bound and `certified_infeasible`.

| Phase-I status | Meaning |
| --- | --- |
| `feasible` | A real feasible restricted point was recovered within the configured numerical tolerance; original-objective optimization resumes |
| `infeasible` | An original-domain directed proof excludes zero artificial activity |
| `stalled` | Pricing or proof information did not justify further recovery or infeasibility |
| `time_limit` / `iteration_limit` | The shared solve budget stopped recovery; accepted original-cost columns remain available for resumption |

Feasibility estimates concern artificial activity under the original fleet bound;
they never become finite bounds for the original objective. Exact pricing at one
projected dual point and a total-route bound from the original fleet row supply
the numerical estimate. Backend infinity sentinels are not route-count proofs.
A positive estimate schedules a directed original-domain check in both bound
modes. Floating resource extension can reject boundary paths admitted by the
conservative relaxation, so the numerical estimate alone cannot prove original
infeasibility. The conservative path proof can decline unsupported models,
arithmetic or exhausted budgets. Declining that proof leaves the result unknown.
Positive restricted artificial activity, an LP failure, or a finite penalty
alone never proves original infeasibility.

The private operation uses the remaining node deadline and CG iteration budget.
Limit outcomes retain valid generated columns at original costs, with no phase-I
dual/minimum reused on a later request. Original-objective state is retained on
failure; model installation stages edits on a private copy. Parents and siblings
retain their own snapshots. A subsequent solve can reconstruct phase I from the
retained original model and columns; retaining the auxiliary basis is unnecessary.

A proof-qualified infeasible result closes ordinary tree nodes and strong
branching sides. A stalled or interrupted phase I does not close a node.
