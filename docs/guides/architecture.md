# Solver architecture and ownership

[Documentation index](../README.md) · [Solve interface](solve.md) · [Source layout](source-layout.md)

BALDES has one master framework. A single pricing network is a collection of
length one. Input shape does not select a separate column-generation or tree
algorithm. A master containing only static variables can have no pricing networks.

## From input to a solution

1. The CLI, Python wrapper or Rust caller supplies routing data or an explicit
   network model. Routing registration produces the same `model::Problem`
   representation: shared rows and a collection of networks.
2. `master/formulation` builds the restricted LP, including network-quantity
   rows and shared rows. `master/preprocessing` owns propagation and flow rows.
3. `master/cg` sequences LP solves, pricing, stabilization, cut installation,
   feasibility recovery and enumeration handoffs.
4. `master/pricing` schedules enabled networks, collects their columns and
   combines their bound contributions. Each network uses `pricing::Solver`
   for its resource-constrained labeling search or retained route universe.
5. For integer search, `master/tree` selects and evaluates nodes through one
   lifecycle driver. Model adapters supply branch candidates, original-model
   witnesses and certificates. The selected LP backend owns the native LP state.

| Concern | One implementation owner |
| --- | --- |
| Canonical rows and network collection | `master/model` |
| Restricted-master construction and row batches | `master/formulation` |
| LP/pricing/cut iteration and restart bookkeeping | `master/cg` |
| Network scheduling and aggregate pricing feedback | `master/pricing` |
| One network's labels, resource transitions and cost preparation | `pricing` with `labeling`, `graph` and `cuts` |
| Propagation and conservation rows | `master/preprocessing` |
| Frontier, node limits, dispatch and tree status | `master/tree` |
| LP backend and integer-copy controls | `master/solvers` |

The two pricing directories describe two levels. `master/pricing` coordinates
network oracles; `pricing` implements an oracle. A scheduling change belongs in
the coordinator, and a change to route extension or dominance belongs in the
oracle/labeling layer. Both benefit one-network and many-network solves.

## Why routing and model adapters remain

`master::model::Problem` stores the model. `master::routing::RoutingContext`
registers customer-cover rows, fleet semantics and routing metadata around that
same model. It does not define another graph/rows representation.

Routing and explicit network inputs can supply different cuts, branch families,
preprocessing rules and proof contracts. For example, routing hooks configure
capacity/rank-one separation; explicit models register their row coefficients
and original-variable decisions. These differences are properties of the
registration. They do not depend on whether a collection contains one network
or several. Encoding a routing problem manually as a network model does not
implicitly install all routing-specific hooks.

The routing, model and imported-column tree adapters share selection, budgets,
frontier storage and evaluation dispatch. Imported sources have their own
column transport and completeness contracts; `model/imported/multiple` groups
external sources and is not another tree or multi-pricing engine.

## Public entry points

| Input / operation | Entry point |
| --- | --- |
| CLI routing file or pricing JSON | `baldes solve INPUT` |
| CLI integer search | `baldes solve INPUT --tree` |
| Python instance, path, pricing network, array or model | `pybaldes.solve(problem, **options)` |
| Python model constructors | `Instance`, `PricingNetwork`, `NetworkProblem` |
| Rust explicit network root | `master::model::solve` |
| Rust routing registration/root | `master::routing::RoutingContext`, `master::routing::solve` |
| Rust integer search for either registration | `master::tree::solve` |
| Low-level resource-constrained path pricing | `pricing::Solver` / Python `Solver` |

Python routing input currently exposes root relaxation solves; explicit network
models additionally expose `tree=True`. These are interface capabilities, not
separate implementations selected by network count. The Python `_solve`
transport compiles model-specific input into the shared Rust framework.

`RootResult` and `NetworkResult` preserve registration-specific reports.
A shared engine does not promise identical LP bases or branching trajectories
across different registrations or policies. See [results and guarantees](../pybaldes/results.md).

## Extending the framework

Keep a new generic scheduling, construction, restart or frontier rule in its
owner above. Put new model coefficients and witnesses in an adapter. Keep cut
mathematics in `cuts`, with installation through the shared CG lifecycle.

Tests belong in `tests/`, with private unit tests attached through `#[path]`.
Local investigations belong under ignored `internal/`; local experimental work
belongs under ignored `working/`. Neither is a public documentation source.
