# DeLuxing column fixing

BALDES provides an opt-in Rust implementation of deep Lagrangian underestimate
fixing, following the algorithm in RouteOpt's `packages/deluxing` reference.
It filters a **complete enumerated integer column pool** before a MIP solve.
Do not use these fixes to certify an incomplete restricted master: later
pricing could introduce columns that invalidate the bounds.

The procedure first solves the full LP and fixes columns by reduced cost.
It divides surviving columns at half the incumbent gap, clusters the vectors
`rhs - column`, and solves auxiliary LPs over the low reduced-cost columns.
Each cluster supplies a new RHS; successful fixing rounds supply subsequent
RHS vectors. Small pools use deterministic Lloyd clustering; larger pools use
norm quantiles. Seeds and deletion trajectories can differ from RouteOpt.

The Rust implementation accounts for row senses, finite upper bounds, integer
multiplicities and objective offsets. It projects row multipliers onto their
valid sign cone and corrects negative reduced costs using column bounds. An
improving integer solution can use at most one column above the half-gap
threshold, which supplies the stronger bound in the auxiliary rounds. Fixing
requires a bound strictly greater than the cutoff plus a safety tolerance.

## Rust API

With `highs` enabled, call
`master::solvers::highs::solve_integer_with_deluxing` on a complete matrix in
which **every column is integer**:

```rust,ignore
use baldes::master::deluxing::DeluxingOptions;
use baldes::master::solvers::{highs, mip::MipOptions};

let options = MipOptions {
    objective_cutoff: incumbent,
    ..Default::default()
};
let (mip, fixing) = highs::solve_integer_with_deluxing(
    &complete_master, &options, &DeluxingOptions::default(),
)?;
println!("fixed {} columns", fixing.fixed_columns.len());
```

Routing snapshots expose
`node.solve_enumerated_mip_with_deluxing(&options, &fixing_options)` with
`clp,highs` enabled. This reconstructs the complete enumerated matrix, including
branch rows and full-memory cut coefficients, before filtering. Existing
`solve_enumerated_mip` calls retain their behavior. In Rust tree search, set
`TreeOptions.deluxing` and `TreeOptions.enumerated_mip` to enable filtering.
The CLI enables both with `--deluxing`:

```sh
cargo run --release --features clp,highs -- vrptw instances/R101.txt --tree --deluxing
```

`--deluxing` requires `--tree` and the `highs` build feature, and conflicts with
`--no-enumeration`. Filtering runs when a finite incumbent/cutoff is available;
otherwise the enumerated MIP runs directly to find a first witness. Python and
shared masters with continuous static variables do not expose this option.

The cutoff must be finite and includes fixed-route objective contributions.
It is an objective threshold; the API does not presume a feasible incumbent.
After filtering, infeasibility means `ObjectiveCutoff` and the reported bound
is capped at the cutoff. It does not prove infeasibility above the cutoff.
Solutions retain the source column order. The live source model, solution,
basis and enumeration pool are untouched.

For another backend, `master::deluxing::filter` accepts an `LpMaster` source and
an empty-model factory. It returns sorted original column indices without
applying changes. Its caller must guarantee integrality and completeness.
A time limit or nonoptimal LP stops filtering and retains only fixes already
certified. The MIP wrapper subtracts preprocessing time from the MIP budget.
Clustering checks the budget between iterations and points. Allocations,
matrix construction and native model edits can overshoot the time limit.

Defaults are 10 clusters, Lloyd clustering up to 10,000 columns, deep rounds
while more than 10 columns are fixed per round, 30 seconds, and an absolute
fixing margin of `1e-6`. No speedup or reduction ratio is assumed; benchmark on
the intended enumerated route pools before enabling it in a search policy.
