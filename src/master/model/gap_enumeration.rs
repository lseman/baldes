//! Reduced-cost arc fixing and enumeration of the routes within the gap to
//! the incumbent, using each network's labeling engine and a shared cutoff plan.
//!
//! An attempt runs after exact pricing converged when the gap
//! `upper - bound` (upper is the incumbent, or 1e12 without one) has shrunk
//! below 0.9 times the gap of the previous attempt. A complete enumeration
//! replaces the master columns by the enumerated routes and the admissible
//! cached ones; pricing then scans the retained pool. Enumerated routes use
//! full-memory rank-one coefficients, as the enumerated pool prices them.
use super::*;

/// Live pricing state saved before a tentative fixing/enumeration pass.
pub(super) struct GapCheckpoint<const D: usize> {
    oracles: Vec<(Solver<D>, Vec<f64>, Arc<HashSet<Vec<i32>>>)>,
    route_count: usize,
    previous_fixing_gap: f64,
    enumerated_routes: usize,
    objective_fixed: bool,
    gap_pool_cutoff: Option<f64>,
}

const FIXING_RATIO: f32 = 0.9;

fn upper<const D: usize>(state: &State<D>) -> f64 {
    let upper = state.tree_incumbent.min(
        state
            .pricing_witness
            .as_ref()
            .map_or(f64::INFINITY, |w| w.0),
    );
    if upper.is_finite() {
        upper
    } else {
        1e12
    }
}

fn eligible_gap<const D: usize>(state: &State<D>, bound: f64) -> Option<f64> {
    if !state.gap_enumeration
        || !state.integer_primal_termination
        || !bound.is_finite()
        || !state.original_variable_costs.is_empty()
        || state.oracles.is_empty()
        || ((state.oracles.len() != 1
            || !state.master_variables.is_empty()
            || state
                .oracles
                .iter()
                .any(|oracle| oracle.network.multiplicity.lower != 0.))
            && state.fixing_cutoffs.is_none())
        || state.universe.is_some()
    {
        return None;
    }
    let gap = upper(state) - bound;
    let ratio = gap / state.previous_fixing_gap;
    let threshold = f64::from(FIXING_RATIO);
    (gap > 0.
        && gap.is_finite()
        && ratio < threshold - (threshold.abs().max(ratio.abs()) * 1e-10 + 1e-6))
        .then_some(gap)
}

fn sorted(route: &Route) -> Vec<i32> {
    let mut arcs = route.arcs.clone();
    arcs.sort_unstable();
    arcs
}

impl<const D: usize> Oracle<D> {
    /// Whether `route` belongs to the last successful enumeration of this
    /// network; its registry cuts then use full-memory coefficients.
    pub(super) fn is_enumerated(&self, route: &Route) -> bool {
        !self.enumerated.is_empty() && self.enumerated.contains(&sorted(route))
    }
}

/// Register `route` of network `network`, returning its stable tag.
fn register<const D: usize>(state: &mut State<D>, network: i32, mut route: Route) -> Result<i32> {
    let key = (network, route.identity());
    if let Some(&tag) = state.tags.get(&key) {
        return Ok(tag);
    }
    let oracle = state
        .oracles
        .iter()
        .find(|o| o.network.id == network)
        .ok_or_else(|| invalid("unknown enumerated route network"))?;
    let cost = original_costs::route_cost(oracle, &route)?;
    route.cost = cost;
    route.original_cost = cost;
    let tag =
        i32::try_from(state.routes.len()).map_err(|_| invalid("network route tag overflow"))?;
    state.tags.insert(key, tag);
    state.routes.push(Arc::new(NetworkRoute { network, route }));
    Ok(tag)
}

fn column<const D: usize>(state: &State<D>, tag: i32, duals: &[f64]) -> Result<cg::Column> {
    let route = &state.routes[tag as usize];
    let oracle = state
        .oracles
        .iter()
        .find(|o| o.network.id == route.network)
        .ok_or_else(|| invalid("unknown enumerated route network"))?;
    let entries = retirement::active_entries(state, oracle, &route.route)?;
    let cost = route.route.original_cost;
    let reduced_cost = entries.iter().fold(cost, |value, &(row, a)| {
        value - duals.get(row).copied().unwrap_or(0.) * a
    });
    Ok(cg::Column {
        cost,
        reduced_cost,
        entries,
        acceptable: true,
        tag,
        identity: format!("{}:{}", route.network, route.route.identity()),
    })
}

/// Save the live pricing state and fix on a fork; false declines.
pub(super) fn begin<const D: usize>(state: &mut State<D>, bound: f64) -> Result<bool> {
    if eligible_gap(state, bound).is_none() {
        return Ok(false);
    }
    if state.gap_checkpoint.is_some() {
        return Err(invalid("nested enumeration transaction"));
    }
    let oracles = state
        .oracles
        .iter_mut()
        .map(|oracle| {
            let mut tentative = oracle.solver.fork_for_node();
            tentative.enumeration_preserve_arc_paths = true;
            (
                std::mem::replace(&mut oracle.solver, tentative),
                oracle.costs.clone(),
                oracle.enumerated.clone(),
            )
        })
        .collect();
    state.gap_checkpoint = Some(Box::new(GapCheckpoint {
        oracles,
        route_count: state.routes.len(),
        previous_fixing_gap: state.previous_fixing_gap,
        enumerated_routes: state.enumerated_routes,
        objective_fixed: state.objective_fixed,
        gap_pool_cutoff: state.gap_pool_cutoff,
    }));
    Ok(true)
}

pub(super) fn finish<const D: usize>(state: &mut State<D>, commit: bool) {
    let Some(saved) = state.gap_checkpoint.take() else {
        return;
    };
    if commit {
        return;
    }
    for (oracle, (solver, costs, enumerated)) in state.oracles.iter_mut().zip(saved.oracles) {
        oracle.solver = solver;
        oracle.costs = costs;
        oracle.enumerated = enumerated;
    }
    state.routes.truncate(saved.route_count);
    let count = saved.route_count;
    state
        .tags
        .retain(|_, tag| *tag >= 0 && (*tag as usize) < count);
    state.previous_fixing_gap = saved.previous_fixing_gap;
    state.enumerated_routes = saved.enumerated_routes;
    state.objective_fixed = saved.objective_fixed;
    state.gap_pool_cutoff = saved.gap_pool_cutoff;
}

/// Fix arcs and enumerate every route with reduced cost below the gap.
/// None when the pass did not produce a complete enumeration.
pub(super) fn universe<const D: usize>(
    state: &mut State<D>,
    duals: &[f64],
    bound: f64,
) -> Result<Option<Vec<cg::Column>>> {
    let Some(gap) = eligible_gap(state, bound) else {
        return Ok(None);
    };
    let upper = upper(state);
    state.previous_fixing_gap = gap;
    let relative = gap / upper.abs();
    let mode = if relative - (1f64.max(relative.abs()) * 1e-10 + 1e-6) > 1. {
        -1
    } else {
        0
    };
    let mut columns = vec![];
    let mut total_routes = 0;
    let mut coverage = upper;
    for index in 0..state.oracles.len() {
        let ceiling = match state.fixing_cutoffs.as_ref().map(|cutoffs| cutoffs[index]) {
            Some(Some(ceiling)) => ceiling,
            Some(None) => {
                state.oracles[index].solver.restore_arcs();
                state.oracles[index].enumerated = Arc::new(HashSet::new());
                continue;
            }
            None => gap,
        };
        let offset = adapter::cache_duals(state, index, duals)?;
        let cutoff = ceiling - offset;
        let remaining = state.deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(incomplete("node time limit"));
        }
        let solver = &mut state.oracles[index].solver;
        // Native false-gap controls may deliberately enumerate a smaller
        // domain. Record that smaller coverage instead of claiming the full
        // incumbent cutoff for the retained MIP.
        let false_gap = solver.options.get("BALDESredCostFixingFalseGap")?;
        if mode == 0 && (false_gap > 1. || (false_gap < -1. && solver.has_cuts())) {
            let other = crate::core::rounding::add_down(upper, -ceiling)
                .ok_or_else(|| invalid("gap coverage overflow"))?;
            let reduced =
                crate::core::rounding::add_down((cutoff / false_gap.abs()).next_down(), offset)
                    .ok_or_else(|| invalid("gap coverage overflow"))?;
            coverage = coverage.min(
                crate::core::rounding::add_down(other, reduced)
                    .ok_or_else(|| invalid("gap coverage overflow"))?,
            );
        }
        let saved_time = solver.controls.time_limit;
        let saved_modes = (
            solver.cuts.round_duals,
            solver.exact_cut_duals,
            solver.knapsack_exact_coefficients,
        );
        solver.cuts.round_duals = false;
        solver.exact_cut_duals = true;
        solver.knapsack_exact_coefficients = true;
        let saved_fixing_time = solver.options.get("BALDESredCostFixingTimeThreshold")?;
        solver.controls.time_limit = saved_time.min(remaining);
        solver.options.set(
            "BALDESredCostFixingTimeThreshold",
            saved_fixing_time.min(remaining.as_secs_f64()),
        )?;
        let fixed = solver.run_fixing_enumeration(mode, 0., cutoff);
        solver.controls.time_limit = saved_time;
        solver.cuts.round_duals = saved_modes.0;
        solver.exact_cut_duals = saved_modes.1;
        solver.knapsack_exact_coefficients = saved_modes.2;
        solver
            .options
            .set("BALDESredCostFixingTimeThreshold", saved_fixing_time)?;
        let fixed = fixed?;
        state.fixing_calls += 1;
        state.objective_fixed |= fixed.eliminated > 0 || fixed.state == 2 || fixed.state == 3;
        if fixed.state != 2 && fixed.state != 3 {
            if mode == -1 {
                continue;
            }
            return Err(incomplete("network gap enumeration did not complete"));
        }
        let solver = &state.oracles[index].solver;
        let routes = solver
            .enumeration
            .as_ref()
            .ok_or_else(|| invalid("complete enumeration without a route pool"))?
            .paths
            .routes(&solver.graph)?;
        total_routes += routes.len();
        state.oracles[index].enumerated = Arc::new(routes.iter().map(sorted).collect());
        let network = state.oracles[index].network.id;
        for route in routes {
            let tag = register(state, network, route)?;
            columns.push(column(state, tag, duals)?);
        }
    }
    if mode == -1 {
        return Ok(None);
    }
    state.enumerated_routes = total_routes;
    state.gap_pool_cutoff = Some(coverage);
    state.objective_fixed = true;
    Ok(Some(columns))
}

/// Cached generated columns still admissible after enumeration: NG-feasible
/// paths on unfixed arcs whose reduced cost is below the gap.
pub(super) fn regenerate<const D: usize>(
    state: &mut State<D>,
    pool: &[cg::Column],
    duals: &[f64],
    bound: f64,
) -> Result<Vec<cg::Column>> {
    let gap = upper(state) - bound;
    let mut retained = vec![];
    for cached in pool {
        let Some(route) = usize::try_from(cached.tag)
            .ok()
            .and_then(|tag| state.routes.get(tag))
        else {
            continue;
        };
        let Some(index) = state
            .oracles
            .iter()
            .position(|oracle| oracle.network.id == route.network)
        else {
            continue;
        };
        if state
            .fixing_cutoffs
            .as_ref()
            .is_some_and(|cutoffs| cutoffs[index].is_none())
        {
            continue;
        }
        let oracle = &state.oracles[index];
        let graph = &oracle.solver.graph;
        let ng = crate::graph::ng::PreparedNg::new(graph)?;
        let route = &route.route;
        if oracle.solver.ryan_foster.iter().any(|rule| {
            !rule.satisfied_by(route.nodes.iter().map(|&v| v as usize))
                || !rule.elementary_on_pair(route.nodes.iter().map(|&v| v as usize))
        }) {
            continue;
        }
        let mut memory = ng.vertex_bits[graph.source];
        let mut vertex = graph.source;
        let mut proper = true;
        for &id in &route.arcs {
            let Some(&physical) = oracle.index.get(&id) else {
                proper = false;
                break;
            };
            let arc = &graph.arcs[physical];
            if oracle
                .solver
                .forbidden_arcs
                .get(physical)
                .copied()
                .unwrap_or(false)
            {
                proper = false;
                break;
            }
            let next = ng.forward[physical].extend(memory, ng.vertex_bits[arc.head]);
            if arc.tail != vertex || next.is_none() {
                proper = false;
                break;
            }
            memory = next.unwrap();
            vertex = arc.head;
        }
        if proper && vertex == graph.sink {
            retained.push((index, cached.tag, sorted(route)));
        }
    }
    // Admissible cached paths also use the enumeration coefficient contract.
    for (index, _, arcs) in &retained {
        Arc::make_mut(&mut state.oracles[*index].enumerated).insert(arcs.clone());
    }
    let mut columns = vec![];
    for (index, tag, _) in retained {
        let refreshed = column(state, tag, duals)?;
        let ceiling = state
            .fixing_cutoffs
            .as_ref()
            .and_then(|cutoffs| cutoffs[index])
            .unwrap_or(gap);
        let rc = refreshed.reduced_cost;
        if rc < ceiling - (ceiling.abs().max(rc.abs()) * 1e-10 + 1e-6) {
            columns.push(refreshed);
        }
    }
    Ok(columns)
}

/// After the enumerated pool is dropped (for example by phase I), loaded
/// enumerated routes return to the limited-memory coefficients of labeling.
pub(super) fn restore_limited_memory<const D: usize>(
    state: &mut State<D>,
    master: &mut ClpMaster,
    loaded: &mut [Option<cg::Column>],
) -> Result<bool> {
    let previous: Vec<_> = state
        .oracles
        .iter_mut()
        .enumerate()
        .filter_map(|(index, oracle)| {
            (oracle.solver.enumeration.is_none() && !oracle.enumerated.is_empty()).then(|| {
                (
                    index,
                    std::mem::replace(&mut oracle.enumerated, Arc::new(HashSet::new())),
                )
            })
        })
        .collect();
    if previous.is_empty() {
        return Ok(false);
    }
    let mut destination = match master.try_clone() {
        Ok(destination) => destination,
        Err(error) => {
            for (index, enumerated) in previous {
                state.oracles[index].enumerated = enumerated;
            }
            return Err(error);
        }
    };
    let mut refreshed_loaded = loaded.to_vec();
    let result = (|| -> Result<()> {
        for (index, slot) in refreshed_loaded.iter_mut().enumerate() {
            let Some(column) = slot else { continue };
            let Some(route) = usize::try_from(column.tag)
                .ok()
                .and_then(|tag| state.routes.get(tag))
            else {
                continue;
            };
            let oracle = state
                .oracles
                .iter()
                .find(|o| o.network.id == route.network)
                .ok_or_else(|| invalid("cached route network"))?;
            let refreshed = retirement::active_entries(state, oracle, &route.route)?;
            let old: HashMap<_, _> = column.entries.iter().copied().collect();
            let new: HashMap<_, _> = refreshed.iter().copied().collect();
            for row in old
                .keys()
                .chain(new.keys())
                .copied()
                .collect::<HashSet<_>>()
            {
                let value = new.get(&row).copied().unwrap_or(0.);
                if old.get(&row).copied().unwrap_or(0.) != value {
                    destination.set_coefficient(row, index, value)?;
                }
            }
            column.entries = refreshed;
        }
        Ok(())
    })();
    if let Err(error) = result {
        for (index, enumerated) in previous {
            state.oracles[index].enumerated = enumerated;
        }
        return Err(error);
    }
    state.gap_pool_cutoff = None;
    *master = destination;
    loaded.clone_from_slice(&refreshed_loaded);
    Ok(true)
}

#[cfg(test)]
#[path = "../../../tests/unit/master/model/gap_enumeration_tests.rs"]
mod tests;
