# Solver concepts and interface choices

[Documentation index](../README.md) · [First solve](first-solve.md)

## From paths to a routing solution

Pricing finds resource-feasible source-to-sink paths with low reduced cost.
A master chooses quantities of route columns to satisfy customer or other shared
rows. Column generation alternates master LP solves with pricing. Cuts add valid
inequalities; branch-and-price searches integer decisions across a tree.

These layers return different objects. A cheap feasible path is not a full fleet
plan. A converged root LP can be fractional. A feasible integer incumbent is an
upper bound in minimization; pricing supplies lower-bound information under its
configured guarantee. Compare the two only when both concern the same model,
distance convention, and inherited node restrictions.

## Select the workflow

| Need | CLI | Python | Reading |
| --- | --- | --- | --- |
| Feasible heuristic routing plan | `hgs` | HGS seeding within root solve | [CLI](cli.md) |
| Routing root LP | `solve` | `solve_root` | [pyBaldes routing](../pybaldes/routing.md) |
| Single-network integer routing tree | `solve --tree` | Not exposed by `solve_root` | [Routing tree](../solver/tree.md) |
| Custom RCSP pricing | Through a model | `Solver` | [Pricing](../pybaldes/pricing.md) |
| Shared-master root / integer search | `multi-pricing` / `multi-pricing --tree` | `NetworkProblem.solve(tree=...)` | [Multi-pricing](../extensions/mp.md) |

A network model can represent several route families. Its branching capabilities
and integer-domain contract differ from the specialized routing tree. A shared
master is part of the core framework, while specialized routing formulations
are organized under extensions. See [source layout](source-layout.md).

## Three meanings to keep separate

**Route feasibility** checks a path against resource and node restrictions.
**Pricing exactness** establishes a minimum over the configured pricing domain,
which may be an NG relaxation. **Independent certification** validates bounds
or witnesses using its supported arithmetic/model contract. A pricing exactness
flag alone does not establish an independent master certificate.

The selected `bound_mode` is numerical by default. Certified mode keeps missing
certificates unknown rather than substituting numerical estimates. JSON uses
`null` for unknown bounds. See [Python results](../pybaldes/results.md) and the
[CLI bound fields](cli.md).

## Resources and route domains

Scalar resources combine arc consumption, initial states, and vertex windows.
Disposable coordinates may wait; equality coordinates do not. Coupled resources
express additional domains and state-dependent costs on the same physical graph.
Their definitions are distinct from scalar coordinate IDs. See
[scalar resources](../reference/resources.md), [coupled resources](../extensions/coupled-resources.md),
and [multiple windows](../extensions/multi-time-windows.md).

NG memory is a relaxation of elementary routing: a priced path can revisit a
customer when the memory permits it. Complete supported enumeration can transfer
an elementary route universe into a master. A bounded collection attempt is not
complete just because it returned routes. Inspect enumeration coverage before
using a finite-pool MIP as a proof. See [enumerated MIP](../solver/enumerated-mip.md)
and [network transfer](../extensions/mp.md#complete-enumeration-and-integer-transfer).

## Reproduce an objective

Match instance data, customer subset, rounding, explicit matrix, NG construction,
cuts, backend, bound mode, and limits. CLI coordinate distances default to
truncated tenths; Python follows the instance's convention. Root benchmark
reproduction requires `--rounding none`. Compare objective values only after
those choices match, and report limits and proof scope with every result.
