# BALDES IPM in Rust

`baldes::extensions::ipm` is a native Rust port of the active solver in
`/home/seman/dev/solvers/ipm` and its optimization loop in
`solvers/src/ipm/ip_solver.cpp`. It is part of the main `baldes` crate, lives
under `src/extensions/ipm`, requires Rust 1.70+, and has no external dependencies,
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
use baldes::extensions::ipm::{solve, ModelData, Options, Sense, SparseMatrix, Status};

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
# Ok::<(), baldes::extensions::ipm::Error>(())
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
Munari and Gondzio (2013), Algorithm 1; see [master options](../reference/options.md).
`SolverType::{HiPO, NormalEquations, Auto}` selects the backend; HiPO is the
default. `SparseSolver`, `HiPOLDLT`, and `NormalEqLDLT` also expose the linear
systems independently. Construct a new solver when matrix values or its pattern
change; numerical `factorize` calls reuse symbolic analysis for a fixed A.
Failed refactorization invalidates the previous factor.

`Solver::default().solve(&model, &options)` provides a persistent alternative
to the one-shot `solve` function. Retain the `Solver` between related LPs:

```rust
# use baldes::extensions::ipm::{ModelData, Options, Solver};
# fn example(model: &ModelData) -> baldes::extensions::ipm::Result<()> {
let mut solver = Solver::default();
let first = solver.solve(model, &Options::default())?;
let repeated = solver.solve(model, &Options::default())?;
println!("{:?}", solver.statistics());
# Ok(())
# }
```

An unchanged LP can retain its interior iterate; all stopping tests still run,
including when a later call tightens the requested accuracy. Cost/RHS edits
reuse standard-form storage, scaling and symbolic analysis, with a cold iterate.
Matrix, bounds, senses, backend or scaling-option changes rebuild preparation
and discard the old iterate. For the explicit HiPO backend, appended columns
can extend the existing symbolic factorization when their row cliques fit its
filled dual graph. Existing inequality-slack indices are retained across these
appends. New fill falls back to full symbolic analysis. Numerical factors are
always recomputed before solving a changed LP.

Warm attempts are bounded and can retry cold within the caller's remaining
iteration and time budgets. Only optimal interior iterates are retained.
`SolverStatistics` counts fresh analyses, symbolic extensions, warm attempts,
cold retries, and completed Newton iterations. Cloning a solver creates an
independent mutable state with its existing statistics and immutable shared
factor patterns.

The persistent solver also retains scratch for KKT right-hand sides,
permutations, solutions, residuals and refinement corrections. Initialization
and Newton directions use the same workspace, including after model growth;
buffers are resized and overwritten before use. Refinement computes `K*x` and
`abs(K)*abs(x)` in one CSC scan with the same per-row accumulation order as
the separate products. Its `1e-10` backward-error acceptance test and numerical
recovery remain unchanged. The public allocating linear-solve APIs remain
available, and solver clones own independent mutable scratch.

The IPM master adapter retains this solver and a CSC model mirror. Column
appends copy only the new columns; coefficient edits and row/column removals
invalidate the matrix mirror. Metadata is synchronized on every solve. Master
forks copy their IPM state independently.

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
| `ip_solver.h`, `src/ipm/ip_solver.cpp` | `model.rs`, `solver/` | Standard form, homogeneous self-dual Mehrotra loop, upper-bound elimination, adaptive regularization, predictor/corrector and centrality corrections, recovery |
| `hipo_ldlt.h` | `factor/mod.rs`, `sparse_solver.rs::HiPOLDLT` | Block ordering, symbolic fill, fundamental supernodes, elimination forest, frontal assembly, child Schur updates, restricted 1x1/2x2 Bunch–Kaufman pivots, sign-preserving small-pivot regularization, refinement |
| `normal_eq_ldlt.h` | `sparse_solver.rs::NormalEqLDLT` | SPD normal equations `A D⁻¹ Aᵀ + Rd`, followed by primal recovery |
| `sparse_solver.h` | `sparse_solver.rs::SparseSolver` | Backend dispatch and structural fill estimate |
| Eigen sparse/dense plumbing | `matrix.rs`, `factor/mod.rs` | Owned sparse CSC input and factors stored as sparse columns; dense arrays only for individual fronts and Schur updates |

The augmented matrix is `[-(theta + Rp), Aᵀ; A, Rd]`. The ordering preserves
the primal/dual boundary, and fundamental supernodes cannot cross it. Fronts
are processed in increasing elimination order, a valid child-before-parent
order, and child Schur buffers are released after parent assembly. Child updates
store only the lower triangle, packed by row; parents assemble that triangle
without copying an unused upper half.
Sparse augmented factors may use an up-looking row-update path when the
primal block has at least twice as many variables as rows and symbolic L has
at most 32 off-diagonal entries per system variable on average. This path
expects negative primal and positive dual pivots; small, non-finite or wrongly
signed pivots retry the existing pivoted multifrontal factorization. Original
KKT refinement remains enabled. Sparse factor columns use contiguous CSC values
and share an immutable pattern across numerical refactors and clones.

Numeric pivot swaps are composed with the symbolic permutation and applied to
rows of previously assembled factor columns. Iterative refinement uses
`|b - Kx| / (|b| + |K||x|)` with at most two correction solves.

Deliberate differences from C++:

- Ordering keeps the independent diagonal primal block in natural order.
  Eliminating these vertices cannot connect other primal vertices, so building
  a primal column-intersection graph cannot reduce fill. The dual Schur graph
  uses deterministic **exact minimum degree with explicit graph updates**,
  rather than the shared C++ approximate minimum degree quotient graph.
  Complete remaining graphs are emitted directly; dense intersection graphs
  use packed adjacency (at most 32 MiB), while sparse or larger graphs retain
  tree sets. Those graph representations produce the same ordering. Primal
  reordering preserves structural fill but can change floating-point sums.
- The normal-equations path uses this crate's multifrontal factorization,
  rather than the external C++ supernodal implementation. Its fixed CSC pattern
  and row-to-column source positions are cached; refactors accumulate values
  directly without duplicate triplets or repeated sorting.
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
- A regularized least-squares starting estimate reduces initial row and dual
  residuals before positive shifts for the HSD variables. Finite upper bounds
  retain unit complementarity. Unsafe estimates, empty systems and very tight
  bounds retain the unit initializer. This adds one factorization and two solves
  per eligible LP; it is not a warm start from the previous CG solve.

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

## Performance investigation

`cargo run --offline --release --example ipm_sparse_bench -- 20` measures setup,
numeric refactorization and triangular solves on synthetic matrices and the
captured R201 master sparsity pattern. Every solve is checked against a known
KKT witness; factorization diagonals change between repetitions.

`cargo run --offline --release --example ipm_master_bench -- 3 hipo r201-sequence`
compares fresh solves with a persistent solver on eight growing prefixes of
the captured R201 master. Use `c202-sequence` for C202. Matrix construction is
outside the timed region; every result is checked for optimality, `1e-9`
primal/dual feasibility and gap, and agreement with fresh-solve objectives.
One warmup precedes the requested number of measured sequences.

Root CLI JSON and Python `RootResult.ipm_iterations` report completed Newton
iterations separately from CG `iterations` and Clp `simplex_iterations`.
The master counter resets on a fork, like the simplex counter. Centering now
adapts to the affine step, bounded between `1e-4` and `0.1`; the previous
expression was always `0.1` for valid step lengths. Stopping tolerances and
original-coordinate feasibility checks are unchanged.

### Parallel factorization

`Options` exposes `parallel` (`Parallel::Off`, `On`, or `Choose`),
`parallel_type` (`ParallelType::Tree`, `Node`, or `Both`), and `threads`.
The defaults are `Choose`, `Both`, and zero (available CPU parallelism).
Set `threads: 1` to run serially. `Off` creates no worker pool and retains
serial ordering selection and the up-looking augmented-system fast path.

```rust
use baldes::extensions::ipm::{Options, Parallel, ParallelType};
let options = Options {
    parallel: Parallel::On,
    parallel_type: ParallelType::Both,
    threads: 4,
    ..Options::default()
};
```

`On` uses the requested factorization level. Tree mode processes independent
fronts at each elimination-tree level. Node mode parallelizes trailing dense
updates within a front; pivot selection remains serial. Small dense updates
stay serial to avoid scheduling overhead. Both levels share one private worker
pool, bounded by `threads`, including nested work. Solver clones share the pool
but own independent factorization data.

`Choose` ignores `parallel_type`: tree work is enabled when a level has at
least four fronts whose individual work estimates reach 4,096 (front size
squared times pivot count). Node work is enabled for fronts of at least 512
rows with at least 32 pivots. These conservative thresholds keep small sparse
systems on the serial fast path. `Off` also ignores `parallel_type`.
Parallel setup compares minimum-degree and natural orderings for an explicitly
selected Newton approach. With `SolverType::Auto`, it prepares both orderings for both the
augmented and normal-equation approaches concurrently. Selection minimizes a
deterministic estimate of symbolic factorization work, rather than wall time.
Explicit `HiPO` and `NormalEquations` choices retain their requested approach.
With one worker, setup retains the serial fill heuristic.

Changing thread or effective parallel options on a persistent `Solver` rebuilds its
preparation. Numeric failures invalidate the old factor, as in serial mode.
Parallel tree levels can retain more frontal workspace than serial traversal.
These controls currently belong to the native IPM Rust API.
