# BALDES IPM in Rust

`baldes::ipm` is a native Rust port of the active solver in
`/home/seman/dev/solvers/ipm` and its optimization loop in
`solvers/src/ipm/ip_solver.cpp`. It is part of the main `baldes` crate, lives
under `src/ipm`, requires Rust 1.70+, and has no external dependencies,
C++ bindings, BLAS requirement, or unsafe code.

From the BALDES repository root:

```sh
cargo run -p baldes --example ipm --offline
cargo test -p baldes --test ipm --offline
cargo test -p baldes --lib ipm:: --offline
cargo fmt -p baldes --check
```

The example minimizes `-x - 2y`, with `x + y <= 3` and `x >= 1`, and
returns objective `-5` at approximately `[1, 2]`.

## API

```rust
use baldes::ipm::{solve, ModelData, Options, Sense, SparseMatrix, Status};

let model = ModelData {
    a: SparseMatrix::from_triplets(1, 2, &[(0, 0, 1.), (0, 1, 1.)])?,
    b: vec![3.],
    c: vec![1., 2.],
    lower: vec![0., 0.],
    upper: vec![f64::INFINITY; 2],
    sense: vec![Sense::Equal],
};
let solution = solve(&model, &Options::default())?;
assert_eq!(solution.status, Status::Optimal);
assert!((solution.objective - 3.).abs() < 1e-6);
# Ok::<(), baldes::ipm::Error>(())
```

Objectives are minimized. Matrices accept triplets or CSR input; duplicate
entries are summed, columns are sorted, and explicit zeros are removed.
`ModelData::validate` checks dimensions, finite coefficients and valid bounds.
Row senses use an enum instead of numerical codes. Both original primal values
and original row multipliers are returned. Multipliers use the convention
`c - Aᵀy` for reduced costs: `<=` row multipliers are nonpositive, and `>=`
row multipliers are nonnegative.

`Options` controls tolerance, iteration limit, backend, Ruiz scaling passes, and
an optional cooperative wall-time budget. `Status::TimeLimit` preserves the
distinction from iteration exhaustion; budgets are checked between Newton
iterations, so a factorization can finish after the deadline.
`gap_tolerance: Some(epsilon)` independently relaxes optimality using
`abs(primal - dual) / (1 + abs(primal))`; primal and dual feasibility still use
`tolerance`. The default `None` retains the original stopping rule. The routing
IPM master chooses epsilon from LP bounds during column generation, following
Munari and Gondzio (2013), Algorithm 1; see [master options](../../docs/options.md).
`SolverType::{HiPO, NormalEquations, Auto}` selects the backend; HiPO is the
default. `SparseSolver`, `HiPOLDLT`, and `NormalEqLDLT` also expose the linear
systems independently. Construct a new solver when matrix values or its pattern
change; numerical `factorize` calls reuse symbolic analysis for a fixed A.
Failed refactorization invalidates the previous factor.

`Optimal` requires primal feasibility (also checked in original coordinates),
dual feasibility and relative gap below tolerance. Primal residuals include
both rows and bounds in original coordinates; dual residuals use the unscaled
standard-form stationarity equations. `IterationLimit` returns the last iterate
without claiming convergence. `Infeasible` and `Unbounded` require a checked
approximate Farkas certificate or primal ray; the certificate vector refers to
the equilibrated standard form, as documented on `Solution::certificate`.
For these statuses, primal/dual values and objective describe the last iterate,
not an optimum. Invalid inputs and numerical breakdowns return `Error`.
Certificate detection is conservative: some difficult infeasible/unbounded
problems may instead reach the iteration limit or return a numerical error.

## Source mapping and implementation

| C++ source | Rust module | Behavior |
| --- | --- | --- |
| `ip_solver.h`, `src/ipm/ip_solver.cpp` | `model.rs`, `solver.rs` | Standard form, homogeneous self-dual Mehrotra loop, upper-bound elimination, adaptive regularization, predictor/corrector and centrality corrections, recovery |
| `hipo_ldlt.h` | `factor.rs`, `sparse_solver.rs::HiPOLDLT` | Block ordering, symbolic fill, fundamental supernodes, elimination forest, frontal assembly, child Schur updates, restricted 1x1/2x2 Bunch–Kaufman pivots, sign-preserving small-pivot regularization, refinement |
| `normal_eq_ldlt.h` | `sparse_solver.rs::NormalEqLDLT` | SPD normal equations `A D⁻¹ Aᵀ + Rd`, followed by primal recovery |
| `sparse_solver.h` | `sparse_solver.rs::SparseSolver` | Backend dispatch and structural fill estimate |
| Eigen sparse/dense plumbing | `matrix.rs`, `factor.rs` | Owned sparse CSC input and factors stored as sparse columns; dense arrays only for individual fronts and Schur updates |

The augmented matrix is `[-(theta + Rp), Aᵀ; A, Rd]`. The ordering preserves
the primal/dual boundary, and fundamental supernodes cannot cross it. Fronts
are processed in increasing elimination order, a valid child-before-parent
order, and child Schur buffers are released after parent assembly.
Numeric pivot swaps are composed with the symbolic permutation and applied to
rows of previously assembled factor columns. Iterative refinement uses
`|b - Kx| / (|b| + |K||x|)` with at most two correction solves.

Deliberate differences from C++:

- Ordering uses deterministic **exact minimum degree with explicit graph
  updates**, rather than the shared C++ approximate minimum degree quotient
  graph. Complete remaining graphs are emitted directly; dense primal
  intersection graphs use packed adjacency (at most 32 MiB), while sparse
  or larger graphs retain tree sets. These representations produce the same
  ordering. Ordering can still take more time and memory on large graphs.
- The normal-equations path uses this crate's multifrontal factorization,
  rather than the external C++ supernodal implementation.
- The comparison backends `LDLT`, `SUPERNOODAL`, and `FRONTAL` from the sibling
  `linear_system` directory are not separately ported or aliased to HiPO.
- Fixed variables are eliminated. Free variables, upper-only bounds, shifted
  bounds, zero columns, mixed senses, objective offsets and row dual signs
  are tracked explicitly through recovery.
- Complementarity includes the upper-bound slack/multiplier product. Success
  checks primal feasibility as well as dual feasibility and gap.
- Centrality corrections are added to the current Newton direction instead
  of overwriting it. The homogeneous Schur denominator uses its equivalent
  positive quadratic form to avoid subtractive cancellation.
- A singular 2x2 pivot returns a numerical error instead of replacing only
  its determinant, which would be inconsistent with the stored D block.
- Ruiz equilibration is implemented and enabled by default (five passes).
  The C++ header declares scaling routines, but its optimization loop does
  not invoke them. Set `scaling_iterations: 0` to disable scaling.
- The internal objective is normalized after equilibration. Dual recovery and
  dual/gap checks undo objective scaling, retaining tolerances in original
  units. Wide upper bounds start with feasible slacks and unit complementarity
  products, avoiding a large artificial residual from bounds such as `1e12`.

The restricted master selects this solver with `--master ipm`; the API is also
usable independently. Performance parity with the
C++ solver and large-instance robustness have not been established; there is
no parallel factorization, relaxed supernode amalgamation or nested dissection.

## Validation

The tests cover mixed row senses, shifted/one-sided/free/fixed bounds, multiple
free variables, zero columns, empty dimensions, malformed input, repeated
factorization, failure invalidation, pivot interchanges, 2x2 pivots,
infeasibility/unboundedness certificates, and iteration limits.

Thirty seeded LPs are constructed from known KKT witnesses and solved with
each backend, both with and without equilibration. Forty additional seeded
sparse KKT systems check both numerical backends against known solution
vectors. The three LP fixtures from `solvers/tests/test_ip_solver.cpp` are
included as Rust regressions. The original C++ test executable was also built
with C++20/Eigen/BLAS and run successfully during the port. These tests verify
numerical results and residuals, not identical iteration histories or factors.

A captured R201 master checks convergence at `1e-9` feasibility with both
relaxed and strict gap tolerances, against an independent simplex objective.
Objective-scale tests check original-unit dual recovery with both backends.
