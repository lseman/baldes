# Background AILS-II

For nominal CVRP/VRPTW input, enable the existing AILS-II heuristic on an
independent worker with:

```sh
baldes solve INSTANCE --background-ils
```

The flag also applies to `solve --tree`; `--seed N` sets the search seed. The
CLI supplies the instance with the same rounding and time semantics as the
pricing graph, including `--ignore-time-windows`.

Rust callers supply the nominal instance matching the selected network:

```rust
use baldes::master::model::{BackgroundIlsOptions, Options};

let heuristic_instance = instance.heuristic_instance(&graph, rounding);
let mut background = BackgroundIlsOptions::new(heuristic_instance);
background.network = 0;
background.max_routes = 16;
background.search.time_limit = 0.1;
background.search.iterations = 20;
background.search.seed = 1;
let mut options = Options::default();
options.background_ils = Some(background);
```

`search` is the same `AilsIiOptions` type used by the standalone `ails-ii`
command. Both call one AILS-II engine with the same perturbation, adaptive
acceptance, elite selection and shared local-search operators. The background
adapter contains no separate local-search implementation.

The worker ranks pricing routes by current reduced cost and selects up to
`max_routes` resource-feasible routes without overlapping customers. Their union
forms a complete feasible seed for a smaller instance. It uses the subproblem
builder shared with HGS intensification, preserving explicit distance matrices,
capacity, windows, service times and rounding. AILS-II can move customers between
these routes. Improved routes are mapped back to original customer IDs and
replayed through the physical route converter shared with HGS seeding.

One persistent OS thread handles at most one job at a time and sleeps on a
bounded channel when idle. The master hands off shared immutable references;
selection, seed replay, subproblem construction and search run on the worker.
Completed results enter a later pricing batch without waiting. The master checks
current branch constraints, arc fixings, multiplicity and complete-domain
restrictions, then rebuilds costs and coefficients at the current dual point.
Ordinary column admission and deduplication still apply; exact pricing alone
supplies its bounds. The returned neighborhood is not a full-model incumbent.

Shutdown closes the job channel, requests cancellation and joins the worker.
AILS-II observes cancellation between passes and uses the earlier of its job
budget and the node deadline within operators. The background time budget must
be positive. Complete enumerated domains and retained gap pools decline jobs.
Only the configured instance's nominal split-depot network is supported;
coupled-resource graphs require a model-specific heuristic.

The feature is disabled by default. The OS schedules its thread without reserving
or pinning a core. It can compete with pricing for CPU/cache bandwidth, and extra
columns can increase LP solve time. There is no guaranteed speedup or deterministic
arrival order. Previous timing results for the removed route-level ILS do not
measure this AILS-II integration.
