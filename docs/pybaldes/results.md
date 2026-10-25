# Results, bounds, and exactness

[pyBaldes documentation](README.md) · [Routing](routing.md) · [Networks](networks.md)

## Identify the quantity before using it

For a minimization model, an independently justified lower bound L and a valid
integer incumbent U imply `L <= optimum <= U`. An LP column vector may be
fractional. Its objective is not automatically an integer incumbent, and a
restricted-master LP objective is not automatically a bound on all possible
columns before pricing has justified it.

| Quantity | Interpretation |
| --- | --- |
| `lp` | Current reported master LP objective; check status and pricing completion |
| `bound` | Selected lower bound under `bound_mode` |
| `numerical_bound` | Floating-point pricing estimate under solver tolerances |
| `certified_bound` | Independently validated bound, when available |
| `incumbent` | Best reported integer objective, when one is known |
| Low-level pricing `minimum` | Reduced-cost minimum; use as a pricing bound only with returned `exact=True` |
| Low-level `Route.is_certified` | Pricing-phase exactness marker; not an independent master certificate |

`bound_mode="numerical"` selects the numerical bound and normally skips the
independent certificate search. `bound_mode="certified"` requests independent
original-domain validation when pricing or the incumbent gap can close. An
unsupported operator, exhausted work budget, or unknown arithmetic can leave
the certificate unknown even when the numerical estimate is finite.

For root results an unknown certificate is `-math.inf` in Python and `null` in
JSON. Network result unknown quantities are `None`, serialized as `null`.
Do not replace an unknown bound with `lp`, zero, or the incumbent.

## Status and solve mode

`solve_root` returns a routing root relaxation. `Optimal` means the root LP
converged. `NetworkResult.mode` distinguishes root, tree, and enumerated-MIP
workflows. In a tree, `Optimal` describes integer gap closure under the selected
bound policy and tolerances. Read `bound_mode` alongside status.

A time, iteration, or node limit can leave a useful incumbent without proving
optimality. A network tree can report `Incomplete` when its available branching
families cannot resolve the remaining fractionality. Infeasibility conclusions
must come from the solver's relevant feasibility/proof status, not an empty
route list. The [feasibility guide](../solver/feasibility.md) explains phase I.

## RootResult reference

The frozen result objects contain tuple-valued collections.

| Fields | Meaning |
| --- | --- |
| `status`, `bound_mode` | Termination and selected bound guarantee |
| `lp`, `bound`, `numerical_bound`, `certified_bound` | Objective and lower-bound quantities above |
| `lp_before_cuts` | LP objective before cut strengthening |
| `incumbent`, `pruned_by_bound` | Integer upper bound and bound-pruning outcome |
| `cuts`, `cut_rounds` | Generated rows and separation rounds; counts include retired rows |
| `iterations`, `pricing_calls` | Column-generation work counters |
| `columns` | Route columns added over the run |
| `master_columns` | Final master size, including artificial columns |
| `fixing_calls`, `enumerated_routes` | Fixing/enumeration counters |
| `duals` | Exported master dual vector |
| `seconds`, `pricing_seconds`, `solver_seconds` | Reported elapsed/component timers |
| `simplex_iterations`, `ipm_iterations` | LP backend iteration totals |
| `routes` | Positive LP `RootRoute` columns |
| `history` | `Iteration` records |
| `feasibility` | Auxiliary `FeasibilityReport` records |

`RootRoute(value, cost, nodes, arcs)` contains the column weight, original route
cost, node sequence, and arc sequence. For N routing customers, the source is
0 and the separate terminal depot is N+1; internal customer IDs are 1 through N.
The node sequence may repeat under NG relaxation. RootRoute uses `.nodes` and
`.arcs` attributes; low-level Route uses `.nodes()` and `.arcs()` methods.

`Iteration` fields are `iteration`, `lp`, `minimum_reduced_cost`, `new_columns`,
`columns`, `exact`, `phase`, `pricing_calls`, `alpha`, and `misprices`.
They expose the per-iteration objective, pricing output, phase exactness, and
stabilization diagnostics. Do not infer a global certificate from `exact`.

`FeasibilityReport` fields are `status`, `iterations`, `pricing_calls`,
`simplex_iterations`, `ipm_iterations`, `seconds`, `artificial_activity`,
`lower_bound`, and `certified_infeasible`. Its bounds belong to the auxiliary
phase-I objective, not the original routing objective.

## NetworkResult reference

| Fields | Meaning |
| --- | --- |
| `mode`, `status`, `bound_mode` | Solve workflow, termination, and bound policy |
| `lp`, `bound`, `numerical_bound`, `certified_bound`, `incumbent` | Optional objective/bound quantities |
| `routes`, `fixed_routes` | Active route solution and separately retained fixed contributions |
| `duals` | Shared-master dual values |
| `iterations`, `pricing_calls`, `oracle_calls` | Total work and `(network ID, calls)` pairs |
| `nodes`, `open_nodes` | Tree evaluation and remaining-node counts |
| `seconds` | Reported elapsed time |
| `enumeration_reports` | Per-attempt route coverage reports |
| `enumerated_mip_calls`, `enumerated_mip_nodes`, `enumerated_mip_failures` | Integer transfer work |
| `snapshot` | Thread-bound resumable node for root results; otherwise may be `None` |

`NetworkRoute` fields are `network`, `route_index`, `value`, `cost`, `nodes`,
`arcs`, `resources`, and `resource_profile`. Network and arc IDs are local to the
model's declared network; `route_index` is a snapshot route handle, not an arc
or a position in the returned positive-route list. It can be absent.
`resources` gives final scalar values; `resource_profile` records states along
the route. Include `fixed_routes` when interpreting child-node contributions.
Synchronization rewards can be master contributions outside the route costs.

`NetworkEnumerationReport` fields are `attempts`, `complete`, `networks`,
`routes`, `transferred`, `failure`, and `coverage_cutoff`. A route limit is a
budget, not evidence of complete enumeration. A finite `coverage_cutoff`
restricts the covered improvement domain; consult it before interpreting a MIP
result as a proof. See [network enumeration](networks.md#enumeration-and-integer-transfer).

## Serialization

```python
import json
import math

# With a RootResult named result:
data = result.to_dict()  # ordinary dataclass data; may retain infinities
text = result.to_json(indent=2)  # strict JSON; non-finite floats become null
assert isinstance(json.loads(text), dict)
if math.isfinite(result.bound):
    print("Selected finite bound:", result.bound)
```

`NetworkResult.to_json(indent=2)` omits the native `snapshot` and uses strict
JSON. There is no public NetworkResult `to_dict()` method. Exported result JSON
is a record of the result, not a resumable node checkpoint. Model JSON round trips
through `NetworkProblem`; result JSON is a different schema.
