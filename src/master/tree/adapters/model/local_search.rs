//! Incumbent partial-column fixing followed by child diving.
use super::*;
use crate::rng::GnuRand;

/// Random incumbent neighborhoods, followed by the ordinary diving heuristic.
/// `depth_limit = -1` disables the heuristic, matching the control default.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LocalSearchOptions {
    pub depth_limit: i32,
    /// Fraction of incumbent variables whose contributions are fixed.
    pub fixed_column_ratio: f64,
    /// Number of partial-incumbent dives after the initial feasibility dive.
    pub max_iterations: usize,
    /// Seed of this tree's isolated GNU libc-compatible random stream.
    pub seed: u32,
    /// Resource limits and candidate evaluation controls of each child dive.
    pub dive: DivingOptions,
}

impl Default for LocalSearchOptions {
    fn default() -> Self {
        Self {
            depth_limit: -1,
            fixed_column_ratio: 0.8,
            max_iterations: 3,
            seed: 1,
            dive: DivingOptions::default(),
        }
    }
}

impl LocalSearchOptions {
    pub(super) fn validate(&self) -> Result<()> {
        if self.depth_limit < -1
            || !self.fixed_column_ratio.is_finite()
            || !(0. ..=1.).contains(&self.fixed_column_ratio)
        {
            return Err(invalid(
                "local search requires depth >= -1 and a fixing ratio in [0, 1]",
            ));
        }
        if self.dive.depth_limit != 0
            || self.dive.frequency != 0
            || self.dive.lds_depth != 0
            || self.dive.max_discrepancy != 0
            || self.dive.stop_after_first_solution
        {
            return Err(invalid(
                "local search controls its own dive schedule and discrepancy settings",
            ));
        }
        self.dive.validate()
    }

    pub(super) fn due(&self, depth: usize) -> bool {
        self.depth_limit >= 0 && depth <= self.depth_limit as usize
    }
}

/// `ceil(N * ratio)` with the native Double absolute/relative comparison band.
fn selected_count(entries: usize, ratio: f64) -> Option<usize> {
    let product = entries as f64 * ratio;
    let count = (product - product * 1e-10 - 1e-6).ceil().max(0.) as usize;
    (count < entries).then_some(count)
}

fn selection(entries: usize, ratio: f64, random: &mut GnuRand) -> Option<Vec<usize>> {
    let count = selected_count(entries, ratio)?;
    let mut used = vec![false; entries];
    let mut selected = Vec::with_capacity(count);
    while selected.len() < count {
        let index = random.next() as usize % entries;
        if !used[index] {
            used[index] = true;
            selected.push(index);
        }
    }
    Some(selected)
}

#[derive(Clone)]
enum IncumbentEntry {
    Master(usize, f64),
    Route(NetworkRoute, f64),
}

fn incumbent_entries<const D: usize>(
    node: &NodeSnapshot<D>,
    witness: &IntegerWitness,
) -> Vec<IncumbentEntry> {
    // Original static variables precede generated columns. Generated-variable
    // identities retain their registry insertion order across each node fork.
    let mut entries = Vec::new();
    for (index, variable) in node.state.master_variables.iter().enumerate() {
        if let Some(&(_, value)) = witness.2.iter().find(|&&(id, _)| id == variable.id) {
            if value != 0. {
                entries.push(IncumbentEntry::Master(index, value));
            }
        }
    }
    let mut routes = Vec::<(NetworkRoute, f64)>::new();
    for fixed in &witness.1 {
        if fixed.quantity == 0. {
            continue;
        }
        let identity = fixed.route.route.identity();
        if let Some((_, quantity)) = routes.iter_mut().find(|(route, _)| {
            route.network == fixed.route.network && route.route.identity() == identity
        }) {
            *quantity += fixed.quantity;
        } else {
            routes.push((fixed.route.clone(), fixed.quantity));
        }
    }
    routes.sort_by_key(|(route, _)| {
        node.state
            .tags
            .get(&(route.network, route.route.identity()))
            .copied()
            .unwrap_or(i32::MAX)
    });
    entries.extend(
        routes
            .into_iter()
            .map(|(route, quantity)| IncumbentEntry::Route(route, quantity)),
    );
    entries
}

fn fix_entry<const D: usize>(
    node: &NodeSnapshot<D>,
    entry: &IncumbentEntry,
) -> Result<NodeSnapshot<D>> {
    match entry {
        IncumbentEntry::Master(index, value) => {
            let variable = &node.state.master_variables[*index];
            let mut bounds = variable.bounds;
            // Residual fixing preserves additional contribution in the
            // variable's sign direction; it does not impose an equality.
            if bounds.lower >= 0. {
                bounds.lower = bounds.lower.max(*value);
            } else if bounds.upper <= 0. {
                bounds.upper = bounds.upper.min(*value);
            } else if *value != 0. {
                return Err(invalid(
                    "local search does not support partial fixing of free signed master variables",
                ));
            }
            node.with_master_variable_bounds(variable.id, bounds)
        }
        IncumbentEntry::Route(route, quantity) => {
            let identity = route.route.identity();
            let already: f64 = node
                .state
                .fixed_routes
                .iter()
                .filter(|fixed| {
                    fixed.route.network == route.network && fixed.route.route.identity() == identity
                })
                .map(|fixed| fixed.quantity)
                .sum();
            let quantity = quantity - already;
            if quantity <= 0. {
                return node.fork();
            }
            let key = (route.network, identity);
            if let Some(&tag) = node.state.tags.get(&key) {
                return node.fix_route(tag as usize, quantity);
            }
            // An incumbent from another sibling can contain an unloaded
            // variable. Register its original path before residual fixing.
            let mut child = node.fork()?;
            let oracle = child
                .state
                .oracles
                .iter()
                .find(|o| o.network.id == route.network)
                .ok_or_else(|| invalid("local-search incumbent network"))?;
            if !lifecycle::integer_route_feasible(oracle, &route.route)? {
                return Err(invalid("local-search incumbent route violates this node"));
            }
            let mut route = route.clone();
            route.route.original_cost = original_costs::route_cost(oracle, &route.route)?;
            let tag = i32::try_from(child.state.routes.len())
                .map_err(|_| invalid("local-search route tag overflow"))?;
            child.state.tags.insert(key, tag);
            child.state.routes.push(Arc::new(route));
            child.fix_route(tag as usize, quantity)
        }
    }
}

fn merge(total: &mut diving::Dive, next: diving::Dive, best: &mut Option<IntegerWitness>) {
    total.nodes += next.nodes;
    total.iterations += next.iterations;
    total.pricing_calls += next.pricing_calls;
    for (id, calls) in next.oracle_calls {
        match total
            .oracle_calls
            .iter_mut()
            .find(|(network, _)| *network == id)
        {
            Some((_, total)) => *total += calls,
            None => total.oracle_calls.push((id, calls)),
        }
    }
    if let Some(witness) = next.incumbent {
        if best.as_ref().map_or(true, |old| witness.0 < old.0) {
            *best = Some(witness.clone());
            total.incumbent = Some(witness);
        }
    }
}

/// Every local-search stage starts by evaluating a new child, even when its
/// domain is unchanged. The subsequent dive reuses that evaluated root and
/// counts only its descendants, so the initial child is charged exactly once.
fn child_dive<const D: usize>(
    parent: &NodeSnapshot<D>,
    options: &ModelTreeOptions,
    config: &DivingOptions,
    incumbent: f64,
    deadline: Instant,
) -> Result<diving::Dive> {
    if config.node_limit == 0 || config.time_limit == 0. || Instant::now() >= deadline {
        return Ok(diving::Dive::default());
    }
    let duration = Duration::try_from_secs_f64(config.time_limit)
        .map_err(|_| invalid("local-search dive time limit"))?;
    let end = deadline.min(Instant::now().checked_add(duration).unwrap_or(deadline));
    let mut child = parent.fork()?;
    child.state.node_depth = parent.state.node_depth.saturating_add(1);
    let mut node_options = options.node.clone();
    node_options.fixing_upper_bound = incumbent.is_finite().then_some(incumbent);
    node_options.cg.max_iterations = node_options.cg.max_iterations.min(config.max_iterations);
    node_options.cg.time_limit = node_options
        .cg
        .time_limit
        .min(end.saturating_duration_since(Instant::now()));
    let solved = match child.solve(node_options) {
        Ok(output) => output,
        Err(error)
            if error.message.contains("time limit") || error.message.contains("label limit") =>
        {
            return Ok(diving::Dive::default());
        }
        Err(error) => return Err(error),
    };
    let mut total = diving::Dive {
        incumbent: solved
            .snapshot
            .integer_witness(options.tolerance)?
            .filter(|candidate| candidate.0 < incumbent),
        nodes: 1,
        iterations: solved.master.iterations,
        pricing_calls: solved.master.pricing_calls,
        oracle_calls: solved.oracle_calls.clone(),
    };
    if config.node_limit == 1
        || Instant::now() >= end
        || (config.stop_after_first_solution && total.incumbent.is_some())
    {
        return Ok(total);
    }
    let mut remaining = config.clone();
    remaining.node_limit -= 1;
    let cutoff = total.incumbent.as_ref().map_or(incumbent, |w| w.0);
    let next = diving::run(&solved, options, &remaining, cutoff, end)?;
    let mut best = total.incumbent.clone();
    merge(&mut total, next, &mut best);
    Ok(total)
}

/// Heuristic bounds never escape this function. Only replayed integer witnesses
/// may update the tree incumbent; every neighborhood is a private snapshot.
pub(super) fn run<const D: usize>(
    root: &Output<D>,
    options: &ModelTreeOptions,
    config: &LocalSearchOptions,
    incumbent: Option<IntegerWitness>,
    random: &mut GnuRand,
    deadline: Instant,
) -> Result<diving::Dive> {
    let mut total = diving::Dive::default();
    let mut best = incumbent;
    if Instant::now() >= deadline || config.dive.node_limit == 0 || config.dive.time_limit == 0. {
        return Ok(total);
    }
    if root
        .snapshot
        .state
        .master_variables
        .iter()
        .any(|v| v.bounds.lower < 0. && v.bounds.upper > 0.)
    {
        return Err(invalid(
            "local search does not support partial fixing of free signed master variables",
        ));
    }
    let mut initial = config.dive.clone();
    initial.lds_depth = 10_000;
    initial.max_discrepancy = 1;
    initial.stop_after_first_solution = true;
    let first = child_dive(
        &root.snapshot,
        options,
        &initial,
        best.as_ref().map_or(f64::INFINITY, |w| w.0),
        deadline,
    )?;
    merge(&mut total, first, &mut best);
    for _ in 0..config.max_iterations {
        if Instant::now() >= deadline {
            break;
        }
        let Some(witness) = &best else {
            break;
        };
        let entries = incumbent_entries(&root.snapshot, witness);
        let Some(selected) = selection(entries.len(), config.fixed_column_ratio, random) else {
            break;
        };
        let mut node = root.snapshot.fork()?;
        let mut valid = true;
        for index in selected {
            match fix_entry(&node, &entries[index]) {
                Ok(child) => node = child,
                Err(_) => {
                    valid = false;
                    break;
                }
            }
        }
        if !valid {
            continue;
        }
        let mut partial = config.dive.clone();
        partial.lds_depth = 0;
        partial.max_discrepancy = 0;
        partial.stop_after_first_solution = false;
        let next = child_dive(&node, options, &partial, witness.0, deadline)?;
        merge(&mut total, next, &mut best);
    }
    Ok(total)
}

#[cfg(test)]
#[path = "../../../../../tests/unit/master/tree/local_search_tests.rs"]
mod tests;
