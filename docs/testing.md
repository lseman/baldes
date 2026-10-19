# Build and validation

[Documentation index](README.md) · [Project setup](../README.md#build)

Run commands from the repository root. Keep feature selection explicit when
checking behavior that depends on an LP or integer backend.

## Build matrix

| Features | Dependencies | Available components |
| --- | --- | --- |
| Default | Rust toolchain | Core pricing, cuts, enumeration, instance utilities, and HGS library |
| `clp` | C++14 compiler; initialized Clp/CoinUtils submodules | Clp master, routing root/tree, and `baldes` CLI |
| `highs` | Linkable HiGHS shared library with 32-bit `HighsInt` | HiGHS LP backend and generic integer solve |
| `clp,highs` | Both dependencies above | Routing tree plus optional enumerated HiGHS MIP |

```sh
git submodule update --init
cargo build --release --features clp,highs
```

HiGHS discovery checks `HIGHS_DIR/lib`, `vendor/highs/lib`, `pkg-config`, and
the system linker path. The shared library must also be available to the
runtime loader. See [build.rs](../build.rs) for native linking and Clp source
configuration.

## Full Rust checks

```sh
cargo test --workspace --all-features
cargo check --workspace --all-targets --all-features
cargo check --workspace --all-targets
```

The all-feature suite requires both backends. A build without `highs` rejects
requests for enumerated MIP; a passing default-feature build alone does not
exercise that path.

## Test organization

Rust integration suites are grouped by area under `tests/`:

| Directory | Coverage |
| --- | --- |
| `api/` | C API, CLI, and options |
| `pricing/` | Labeling, enumeration, fixing, and heuristic scheduling |
| `cuts/` | Cut separation and root integration |
| `master/` | Branching, tree/node lifecycle, column generation, preprocessing, and MIP |
| `hgs/` | Genetic-search driver and end-to-end feasibility |
| `instances/` | Instance parsing and distance conventions |
| `pricing/porting_regressions.rs` | Independent pricing and enumeration checks |

Each file remains a separate test target, explicitly registered in the root
`Cargo.toml`. For example,
`cargo test --test main_pricing` runs `tests/pricing/main_pricing.rs`.
When adding a suite, place it in the appropriate directory and add a `[[test]]`
entry with its name and path. Small inline unit tests remain beside their
implementation; Python integration tests live under `python/tests/`. Standalone
network, root checkpoint, labeling profile, and arena test modules live outside
this checkout in `../baldes-verification/overlays/unit/`. Run them with:

```sh
python ../baldes-verification/run.py -- test --lib --all-features
```

The workbench runner attaches these private modules to its source snapshot.
Normal Cargo builds and integration tests do not require the workbench.

## Focused checks

The native interior-point solver is tested without optional LP backends:

```sh
cargo test -p baldes --test ipm
cargo test -p baldes --lib ipm::
cargo run -p baldes --example ipm
```

| Area | Integration test targets |
| --- | --- |
| Branch candidates, pricing constraints, tree bounds | `branching`, `branch_price` |
| Forks, labels, enumeration, siblings, descendants | `node_lifecycle` |
| Capacity/Kohl and strong k-path/knapsack integration | `capacity_root`, `kpath_knapsack` |
| Supported option discovery, validation, atomic setters | `options` |
| Model transfer, integer outcomes, complete-node proofs | `highs_mip` |
| Fixed partial routes, residual rows, cumulative fixing and objective offsets | `partial_solution` |
| Argument handling and JSON output | `cli` |
| Input parsing and distance conventions | `cvrplib` |

```sh
cargo test --features clp --test branching --test branch_price --test node_lifecycle
cargo test --features clp --test capacity_root --test kpath_knapsack --test options
cargo test --features clp,highs --test highs_mip --test cli
```

Check backend boundaries as well as the combined build:

```sh
cargo test --features clp --test options --test cli
cargo test --no-default-features --features highs --test highs_mip --test options
```

Feature-gated tests run only when their backend is present. These commands
exercise different subsets, so compare coverage as well as pass counts.

## Python

```sh
python -m pip install '.[test]'
python -m pytest python/tests
```

The default extension build includes Clp. Rebuild/reinstall the extension after
changing Rust or PyO3 code; tests against an old installed wheel do not validate
the current sources. `pybaldes.has_clp` reports availability. Python exposes
root relaxations and low-level pricing, but no routing-tree entry point.

## What the checks establish

Small integer tests compare against exhaustive route or variable enumeration.
Lifecycle tests verify parent/sibling isolation and inherited branch/cut rows.
MIP tests distinguish feasible witnesses from proof outcomes and check that
limits do not certify an unfinished node.

Column-generation tests also check artificial-penalty exhaustion. Exact pricing
with positive artificials proves optimality only for the penalized master;
exhausting the penalty-update budget returns `Stalled`. It does not prove the
original model infeasible. Increasing that budget can recover a feasible real
solution, but callers must still inspect status and solution feasibility.

Root incumbent checks round approximately integral route quantities, validate
the resulting logical row activities, and evaluate an upper bound from original
arc costs. The numerical LP objective alone is not accepted as an integer
incumbent. Arithmetic overflow leaves the candidate unaccepted.
Scalar resource checks accumulate original consumptions exactly, apply disposable
waiting or equality semantics, and compare exact states with vertex/arc windows.
They also recheck fixed routes and match recorded nodes to the physical arc
chain. Stored floating resource profiles do not prove witness feasibility.

CG passes its remaining wall-time budget to both Clp and HiGHS through
`LpMaster::solve_with_time_limit`. Root preparation, CG and pricing share the
same root deadline. Auxiliary Clp callers can also use `solve_limited` with an
explicit iteration budget. Limits are cooperative and scoped to the call;
previous backend limits are restored. Typed stop outcomes are distinct from LP
infeasibility, and cached arrays from a stopped call do not authorize a new
solution or certificate. Custom LP backends must override the compatibility
method to enforce their own in-solve deadline.

Finite-matrix certificate tests cover fractional quantities, invalid dual signs,
original cost cancellation, fixed objective terms and unproved multiplicity
rows. Unsupported or overflowing arithmetic returns an unknown bound. These
checks do not establish completeness of a routing column universe.

In explicit certified mode (`RootOptions::bound_mode = BoundMode::Certified`),
the root controller combines conservative disposable-resource path coverage,
residual cover-row proofs and finite-matrix numerical bounds before publishing
a lower certificate. The transaction shares the root deadline, with an
additional 250 ms cap, ten million logical work units, 10,000 paths and one million
stored arc/coefficient positions. Row DP has at most 65,536 states and the
auxiliary LP has at most 10,000 iterations. Limits are cooperative rather than
hard heap/CPU caps. Failure leaves the bound unknown; the ordinary pricing
estimate is not substituted. Node tests cover interruption, omission of
unproved rows and parent-model isolation. General resource kinds and arbitrary
fractional fixed-row shifts remain unsupported by this certificate.

Numerical mode is the default and skips that certificate transaction. It keeps
exact pricing and route-feasibility checks, and labels its bounds as numerical.
Tests verify that numerical mode never invokes certification, unknown
certificates cannot authorize strict pruning, and warm starts preserve bound
provenance when changing modes. Certified mode attempts proof at closing exact
passes or when the current master objective suggests the incumbent gap can
close, rather than on every improving exact batch.

Weighted packing rows can use an exact additive envelope instead of partition
DP: every path must satisfy `denominator * coefficient <= customer weight sum`.
An integer residual cover then has row activity at most the floor of its total
remaining weight divided by the denominator. The proof checks actual path
coefficients, signed weights, fixed residual covers and interruption. Other
rows retain the general partition audit. Sparse row extraction avoids scanning
the entire matrix once per cut.

CG keeps its LP pricing convergence estimate separate from the published
integer-domain certificate, including for warm starts. A bound on integer
solutions can exceed the pricing LP optimum. The independent fractional-cover
regression checks that such a bound does not skip improving LP columns.

If exact pricing exhausts the allowed artificial penalties, the routing
controller can restore bounded original columns before returning `Stalled`.
Each candidate passes physical, exact-resource and vertex-elementarity replay;
relaxation resource bounds are not route witnesses. Recovery adds columns and
re-solves the master before accepting its point. Incomplete recovery leaves the
node unknown. Arc-element descriptors currently decline this optional recovery.
Normal regressions cover a large-cost routing case, exhausted penalties,
ineligible columns and deadline interruption without route-cache mutation.

A finite caller-supplied `RootOptions::incumbent` is an assumed upper bound.
Its scalar value is not a validated route witness; callers must supply a
conservative upper bound on the original objective. Automatically discovered
incumbents use the exact checks described above.

Canonical path collector unit tests cover cycles, parallel arcs, waiting,
equality resources, direct terminal arcs, malformed input and exact budget
boundaries. A compile-fail example checks that a completed domain cannot outlive
mutation of its borrowed graph. The collector covers canonical profiles only;
its completion type alone does not certify an original routing master.

Cover-relaxation eligibility checks additionally exercise cyclic graphs,
fractional cover coefficients, unsupported resource kinds, fixed-route residual
mismatches and a shared work/deadline budget. Extra paths excluded by other
constraints may remain in this relaxation. Cut validity and schedule-dependent
row domination still require separate checks before using a matrix certificate.

Cover-row audit tests verify exact partition extrema, rejection of a row that
excludes a cover, empty versus impossible residual covers, fractional RHS
comparisons and state/work/deadline limits. Integer row sums use `i128`.
Failure returns no partial range; a failed relaxed-cover test alone does not
prove invalidity under additional fleet or branch restrictions.

Resource-boundary regressions cover backward subtraction accepting a path whose
forward additions exceed a window by one ulp. Pricing retries with a fresh
forward-only engine using the remaining call deadline, recovering valid paths
displaced in a bounded candidate pool. Checks cover physical exclusions,
topology changes, zero-time stops and an unresolved root without an incumbent.
Forward binary64 replay can also reject a path that satisfies exact-real sums
of its input consumptions. These checks establish forward consistency; original
resource-domain certificates need conservative coverage of arithmetic boundaries.
The conservative collector has a separate exact-real coverage test, including
an exactly representable sum that nearest/even accumulation rejects. A retained
boundary path also passes the integer row audit and a finite-matrix bound check.

Passing these checks establishes the cases exercised by the suite. Benchmark
timings depend on the instance, graph conventions, enabled cuts, backend, and
hardware. Preserve that configuration when reporting a performance change.
For documentation-only edits, verify local links, API examples, command flags,
and option tables without treating a full solver rerun as necessary.

Compiled-reference parity tests, binary adapters, fixtures, and tracing tools are
maintained in a separate local workbench outside this Git checkout. They are not
required by Cargo builds or the regular test suite.
