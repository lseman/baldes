# Robust pickup and delivery with time windows

`baldes::extensions::robust_pd` models the one-to-one RPDPTW in
Abreu, Battarra, Costa and Munari, *The robust pickup and delivery problem
with time windows*, EJOR 335 (2026), 812–828
([paper](https://doi.org/10.1016/j.ejor.2026.03.031),
[local PDF](../../papers/robust_pd.pdf)).

The [60 original benchmark files](../../instances/RobustPDPTW/) contain
5–75 requests, in classes AA, BB, CC and DD. Their pinned source and hashes
are recorded in [`scripts/data/robust_pd_manifest.json`](../../scripts/data/robust_pd_manifest.json).
Instance files follow the repository's existing policy for local downloaded data.
Download or verify them with:

```sh
python3 scripts/download_robust_pd.py --download
```

Run native branch-and-price with Clp:

```sh
# FILE Γq Γt σq σt SECONDS
cargo run --release --features clp,multi-time-windows --example robust_pdptw -- \
  instances/RobustPDPTW/AA05.txt 1 1 0.1 0.1 60

# Deterministic baseline, retaining the paper's nominal-demand adjustment.
cargo run --release --features clp,multi-time-windows --example robust_pdptw -- \
  instances/RobustPDPTW/AA05.txt 0 0 0 0 60
```

The example prints termination status, numerical lower bound, incumbent,
routes, worst load, and worst return time. It independently replays every
incumbent route and verifies request coverage and total objective.
An incomplete or time-limited result is not an optimality claim.

For Python/Gurobi compact formulations, the paper's RRCI/RIPEI separators,
and a reproducible timing comparison with this native implementation, see
[`pd_robust`](../../pd_robust/README.md). That comparison also includes a
counterexample to the compact load equations as printed and an exact
on-board compact baseline.

## Model

Each request i has pickup i and delivery i+n on the same route, in that
order. Vehicles start at depot 0 and finish at the distinct depot 2n+1.
Every request is served exactly once. The fleet is homogeneous and has
up to n vehicles; departures cost 10,000 by default, as in paper §6.1.

At each route position, let O be the set of requests currently on board.
Robust capacity is checked using Proposition 1 (paper p. 815):

```text
load(O) = sum(q[i] for i in O)
        + sum(the Γq largest demand deviations among requests in O)
```

A delivery removes its entire request, including its uncertainty. Already
delivered requests do not consume load or demand-uncertainty budget at a
later position. This preserves the common demand realization of each
pickup/delivery pair.

Travel-time robustness uses the existing `BudgetedTime` resource and the
paper's equation (6). For each γ from zero through Γt, it retains the
worst service start, including waiting and the preceding node's service.
Every customer window and the return-depot window must hold. Waiting can
absorb delays, so summing the largest travel deviations alone is insufficient.

The physical graph has one vertex per original node. A coupled plugin
tracks visited and open requests, enforces elementarity, precedence and
pairing, and checks capacity in both directions. Joins replay the suffix
against the forward state. Conservative equality dominance compares
both sets. The shared master has one set-partitioning row per request;
the pickup contributes its coefficient exactly once.

## Input conventions

`Instance::read` validates consecutive node ids, signed demand pairs,
finite data, time windows, and zero depot demand/service.
`Problem::new(instance, Parameters)` preserves original file data and
derives effective demands and deviations separately.

Defaults apply the §6.1 adjustment before generating demand deviations:
for original request demand q and `a=ceil(0.3*q)`, if `q+a > Q`, replace
the nominal demand by `Q-a`. This adjustment also applies in deterministic
runs so comparisons share the same nominal demands. Set
`Parameters::adjust_demands=false` to use the raw demands.
Deviations are `ceil(σq*q)` and `ceil(σt*t)`. Budgets are nonnegative integers.

The coordinate-only files do not explicitly supply a travel matrix. The
default uses **Euclidean distance rounded to the nearest 0.01** for nominal
cost and travel time. This convention matches the upstream workbook's
nominal objectives for all four five-request instances; the PDF itself
does not specify a coordinate rounding rule. `Parameters::distance_rounding`
can select `DistanceRounding::FullPrecision`. The example accepts a final
`none` argument for that convention (default: `hundredth`).

## Validation and boundaries

Regression tests compare replay against exhaustive demand and arc-delay
scenarios, compare every forward/backward join split, compare native
pricing against exhaustive routes, and compare the integer master against
an independently enumerated set partition on two requests. All 60 files
are parsed by an opt-in test after downloading:

```sh
cargo test -p baldes --lib extensions::robust_pd
cargo test -p baldes --lib all_downloaded_instances_parse -- --ignored
cargo test -p baldes --features clp,multi-time-windows --lib extensions::robust_pd
```

All 76 five-request configurations listed in the upstream workbook were
also solved to optimality and matched an independent exhaustive route and
set-partitioning reference. Of those, 73 match the published objectives.
Three differ under the paper's stated ceiling rule for travel deviations:
`AA05_5_30_5_30`, `BB05_0_0_1_10`, and `BB05_1_10_1_10`.
Those differences disappear in the reference when replacing ceiling by
unrounded or nearest-rounded travel deviations. The implementation retains
the paper's ceiling rule. Details are in the internal evidence directory.

This is native robust pricing with a general branch-and-price master.
It does not implement the paper's compact MIP, RRCI/RIPEI separators,
or its complete experimental benchmark campaign. Exact request histories
can produce exponentially many labels; large-instance scalability is
not established. The pairing/load plugin supplies an independent bounded-work certificate
using outward-rounded load sums. The example uses the master's numerical
bound mode; no large-instance bound-certification campaign has been run.
In a twelve-case ten-request smoke test, all incumbents replayed successfully;
five runs returned Optimal and seven returned Incomplete. Equal values at
printed precision do not override that status. Evidence and remaining work are tracked in
[`internal/robust-pd`](../../internal/robust-pd/).
