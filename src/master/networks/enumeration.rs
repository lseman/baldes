//! Atomic all-network full-domain transfer. No reduced-cost cutoff is used.
use super::*;
use crate::{
    enumeration::{CompleteResourceRelaxation, PathDomainLimits},
    master::solvers::mip::{MipOptions, MipOutput},
};
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct EnumerationOptions {
    pub max_routes: usize,
    pub max_work: usize,
    pub max_stored_arcs: usize,
    pub time_limit: Duration,
}
impl Default for EnumerationOptions {
    fn default() -> Self {
        Self {
            max_routes: 10000,
            max_work: 10000000,
            max_stored_arcs: 1000000,
            time_limit: Duration::from_secs(1),
        }
    }
}
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct EnumerationReport {
    pub attempts: usize,
    pub complete: bool,
    pub networks: Vec<(i32, usize)>,
    pub routes: usize,
    pub transferred: usize,
    pub failure: Option<String>,
}
#[derive(Debug)]
pub struct EnumeratedMipOutput {
    pub mip: MipOutput,
    pub route_values: Vec<FixedRoute>,
    /// Full original elementary domain: no objective cutoff excluded routes.
    pub coverage_cutoff: f64,
}

fn collect<const D: usize>(
    state: &State<D>,
    options: &EnumerationOptions,
) -> std::result::Result<(Vec<NetworkRoute>, Vec<(i32, usize)>), String> {
    let deadline = Instant::now()
        .checked_add(options.time_limit)
        .ok_or("enumeration deadline overflow")?
        .min(state.deadline);
    let mut paths_left = options.max_routes;
    let mut work_left = options.max_work;
    let mut arcs_left = options.max_stored_arcs;
    let mut routes = vec![];
    let mut counts = vec![];
    for oracle in &state.oracles {
        if Instant::now() >= deadline {
            return Err("TimeLimit".into());
        }
        if oracle.network.multiplicity.upper == 0. {
            counts.push((oracle.network.id, 0));
            continue;
        }
        let graph = &oracle.network.graph;
        // The current integer witness contract requires an interior visit.
        // A direct source/sink route may be meaningful in a generic model;
        // decline rather than silently exclude it from a completeness claim.
        if graph
            .arcs
            .iter()
            .any(|a| a.tail == graph.source && a.head == graph.sink)
        {
            return Err("UnsupportedDomain: direct source/sink route domain".into());
        }
        // The downward disposable-resource relaxation cannot omit a physically
        // simple exact-real feasible path. Other kernels remain unsupported.
        let domain = CompleteResourceRelaxation::collect(
            graph,
            PathDomainLimits {
                max_paths: paths_left,
                max_work: work_left,
                max_stored_arcs: arcs_left,
                deadline: Some(deadline),
            },
        )
        .map_err(|e| e.to_string())?;
        paths_left -= domain.paths().len();
        work_left -= domain.work_used();
        arcs_left -= domain.stored_arcs();
        let problem = crate::master::root::RootProblem::set_partitioning(
            graph.clone(),
            oracle.total_multiplicity.upper.max(1.),
        )
        .map_err(|e| e.to_string())?;
        let before = routes.len();
        for path in domain.paths() {
            if Instant::now() >= deadline {
                return Err("TimeLimit".into());
            }
            work_left = work_left
                .checked_sub(path.arc_positions().len() + 1)
                .ok_or("WorkLimit")?;
            let ids: Vec<_> = path
                .arc_positions()
                .iter()
                .map(|&a| graph.arcs[a].id)
                .collect();
            let mut probe = Route {
                nodes: std::iter::once(graph.source as i32)
                    .chain(
                        path.arc_positions()
                            .iter()
                            .map(|&a| graph.arcs[a].head as i32),
                    )
                    .collect(),
                arcs: ids,
                cost: 0.,
                original_cost: 0.,
                resources: vec![],
                resource_profile: vec![],
            };
            if crate::master::root::exact_route_feasible(&problem, &probe).is_none() {
                continue;
            }
            if !oracle
                .solver
                .route_satisfies_relaxation(&probe.nodes, &probe.arcs)
                .map_err(|e| e.to_string())?
            {
                continue;
            }
            // Never silently omit an exact-feasible path whose rounded forward
            // schedule cannot be reconstructed by the current scalar kernel.
            let canonical = problem
                .elementary_route(&probe.arcs)
                .ok_or("Arithmetic: canonical resource replay")?;
            probe.resources = canonical.resources;
            probe.resource_profile = canonical.resource_profile;
            let mut ids = probe.arcs.clone();
            ids.sort_unstable();
            probe.original_cost = ids.iter().map(|id| graph.arcs[oracle.index[id]].cost).sum();
            probe.cost = probe.original_cost;
            if !probe.cost.is_finite() {
                return Err("Arithmetic: route cost".into());
            }
            routes.push(NetworkRoute {
                network: oracle.network.id,
                route: probe,
            });
        }
        counts.push((oracle.network.id, routes.len() - before));
    }
    if Instant::now() >= deadline {
        return Err("TimeLimit".into());
    }
    Ok((routes, counts))
}
fn path_key(route: &NetworkRoute) -> (i32, Vec<i32>) {
    (route.network, route.route.arcs.clone())
}

pub(super) fn transfer<const D: usize>(
    master: &mut ClpMaster,
    state: &mut State<D>,
    cg: &mut cg::Options,
    options: &EnumerationOptions,
) -> Result<()> {
    if state.universe.is_some() {
        return Ok(());
    }
    state.enumeration_report.attempts += 1;
    let deadline = Instant::now()
        .checked_add(options.time_limit)
        .ok_or_else(|| invalid("enumeration time limit"))?
        .min(state.deadline);
    let (routes, counts) = match collect(state, options) {
        Ok(result) => result,
        Err(reason) => {
            state.enumeration_report.failure = Some(reason);
            return Ok(());
        }
    };
    // All budgets passed before modifying the destination. Keep a private copy
    // so a failed model insertion cannot leave half an all-network transfer.
    let mut destination = master.try_clone()?;
    let mut warm = cg.warm_start.clone();
    let mut known = cg.known.clone();
    let mut registry = state.routes.clone();
    let mut path_tags = HashMap::new();
    for (index, route) in registry.iter().enumerate() {
        path_tags.entry(path_key(route)).or_insert(index);
    }
    let mut tags = state.tags.clone();
    let mut universe = vec![];
    let mut loaded: HashSet<usize> = warm.as_ref().map_or_else(
        || known.iter().map(|c| c.tag as usize).collect(),
        |w| w.columns.iter().flatten().map(|c| c.tag as usize).collect(),
    );
    let allowed: HashSet<_> = routes.iter().map(path_key).collect();
    if let Some(warm) = &warm {
        for (column, metadata) in warm.columns.iter().enumerate() {
            if let Some(metadata) = metadata {
                if !allowed.contains(&path_key(&registry[metadata.tag as usize])) {
                    destination.set_upper(column, 0.)?;
                }
            }
        }
    }
    let mut transferred = 0;
    for route in routes {
        if Instant::now() >= deadline {
            state.enumeration_report.failure = Some("TimeLimit".into());
            return Ok(());
        }
        let key = path_key(&route);
        let tag = if let Some(&index) = path_tags.get(&key) {
            index
        } else {
            let index = registry.len();
            let tag = i32::try_from(index).map_err(|_| invalid("enumeration tag overflow"))?;
            tags.insert((route.network, route.route.identity()), tag);
            registry.push(route);
            path_tags.insert(key, index);
            index
        };
        universe.push(tag);
        if !loaded.insert(tag) {
            continue;
        }
        let route = &registry[tag];
        let oracle = state
            .oracles
            .iter()
            .find(|o| o.network.id == route.network)
            .unwrap();
        let column = cg::Column {
            cost: route.route.original_cost,
            reduced_cost: 0.,
            entries: entries(oracle, &route.route, state.shared_offset)?,
            acceptable: true,
            tag: tag as i32,
            identity: format!("{}:{}", route.network, route.route.identity()),
        };
        destination.add_column(column.cost, &column.entries)?;
        if let Some(warm) = &mut warm {
            warm.columns.push(Some(column.clone()));
            warm.generated_pool.push(column);
        } else {
            known.push(column);
        }
        transferred += 1;
    }
    if Instant::now() >= deadline {
        state.enumeration_report.failure = Some("TimeLimit".into());
        return Ok(());
    }
    *master = destination;
    cg.warm_start = warm;
    cg.known = known;
    state.routes = registry;
    state.tags = tags;
    state.universe = Some(universe.clone());
    state.correction = None;
    state.gradient = None;
    state.enumeration_report.complete = true;
    state.enumeration_report.failure = None;
    state.enumeration_report.networks = counts;
    state.enumeration_report.routes = universe.len();
    state.enumeration_report.transferred = transferred;
    // A stronger elementary master needs a new smoothing point, not the old
    // NG relaxation's accepted dual center. Inherited lower bounds stay valid.
    if let Some(warm) = &mut cg.warm_start {
        warm.stabilization = None;
    }
    Ok(())
}

pub(super) fn price<const D: usize>(state: &mut State<D>, duals: &[f64]) -> Result<cg::Answer> {
    let mut columns = vec![];
    let mut minimum = f64::INFINITY;
    let mut correction = 0.;
    let mut gradient = state.rhs.clone();
    for oracle in &mut state.oracles {
        if oracle.network.multiplicity.upper == 0. {
            continue;
        }
        oracle.calls += 1;
        let mut best: Option<(f64, Vec<(usize, f64)>)> = None;
        for &tag in state.universe.as_ref().unwrap() {
            let route = &state.routes[tag];
            if route.network != oracle.network.id {
                continue;
            }
            let coefficients = entries(oracle, &route.route, state.shared_offset)?;
            let reduced = coefficients
                .iter()
                .fold(route.route.original_cost, |v, &(r, a)| v - duals[r] * a);
            if !reduced.is_finite() {
                return Err(invalid("enumerated reduced cost overflow"));
            }
            if best.as_ref().map_or(true, |(v, _)| reduced < *v) {
                best = Some((reduced, coefficients.clone()));
            }
            columns.push(cg::Column {
                cost: route.route.original_cost,
                reduced_cost: reduced,
                entries: coefficients,
                acceptable: true,
                tag: tag as i32,
                identity: format!("{}:{}", route.network, route.route.identity()),
            });
        }
        if let Some((reduced, coefficients)) = best {
            minimum = minimum.min(reduced);
            let quantity = if reduced > 0. {
                oracle.network.multiplicity.lower
            } else {
                oracle.network.multiplicity.upper
            };
            correction += quantity * reduced;
            for (r, a) in coefficients {
                gradient[r] -= quantity * a;
            }
        }
    }
    state.correction = Some(correction);
    state.gradient = Some(gradient);
    Ok(cg::Answer {
        columns,
        minimum: if minimum == f64::INFINITY {
            0.
        } else {
            minimum
        },
        exact: true,
        bound_minimum: None,
    })
}

pub(super) fn minimum<const D: usize>(
    state: &State<D>,
    oracle: &Oracle<D>,
    universe: &[usize],
    duals: &[f64],
) -> Option<Option<f64>> {
    use crate::master::rounding::{add_down, multiply_down};
    let mut minimum: Option<f64> = None;
    for &tag in universe {
        let route = &state.routes[tag];
        if route.network != oracle.network.id {
            continue;
        }
        let mut value = 0.;
        for id in &route.route.arcs {
            let index = oracle.index[id];
            value = add_down(value, oracle.network.graph.arcs[index].cost)?;
            for &(r, a) in &oracle.network.arc_entries[index] {
                value = add_down(value, multiply_down(-duals[state.shared_offset + r], a)?)?;
            }
        }
        for &(r, a) in &oracle.route_entries {
            value = add_down(value, multiply_down(-duals[r], a)?)?;
        }
        for &(r, a) in &oracle.network.route_entries {
            value = add_down(value, multiply_down(-duals[state.shared_offset + r], a)?)?;
        }
        minimum = Some(minimum.map_or(value, |old| old.min(value)));
    }
    Some(minimum)
}

impl<const D: usize> NodeSnapshot<D> {
    pub fn enumeration_report(&self) -> &EnumerationReport {
        &self.state.enumeration_report
    }
    /// Prepare a private complete master. Failure keeps all original columns and
    /// pricing state, with an explicit unavailable report; the parent is untouched.
    pub fn enumerate(&self, options: &EnumerationOptions) -> Result<Self> {
        let mut node = self.fork()?;
        node.state.deadline = Instant::now()
            .checked_add(options.time_limit)
            .ok_or_else(|| invalid("enumeration time limit"))?;
        let mut cg = cg::Options::default();
        cg.warm_start = Some(node.warm);
        transfer(&mut node.master, &mut node.state, &mut cg, options)?;
        node.warm = cg.warm_start.unwrap();
        Ok(node)
    }
    /// HiGHS solves only fully transferred copies. No source model is mutated.
    pub fn solve_enumerated_mip(
        &self,
        options: &MipOptions,
    ) -> Result<Option<EnumeratedMipOutput>> {
        if self.state.universe.is_none() {
            return Ok(None);
        }
        #[cfg(not(feature = "highs"))]
        {
            let _ = options;
            Err(invalid("enumerated MIP requires the highs feature"))
        }
        #[cfg(feature = "highs")]
        {
            use crate::master::solvers::mip::MipStatus;
            let started = Instant::now();
            let mut master = self.master.try_clone()?;
            for &column in &self.warm.artificials {
                master.set_upper(column, 0.)?;
            }
            let integers = self
                .warm
                .columns
                .iter()
                .enumerate()
                .filter_map(|(c, meta)| meta.as_ref().map(|_| c))
                .collect::<Vec<_>>();
            let mut options = options.clone();
            options.time_limit = options.time_limit.saturating_sub(started.elapsed());
            let mut mip =
                crate::master::solvers::highs::solve_integer(&master, &integers, &options)?;
            let mut routes = vec![];
            if let Some(solution) = mip.solution.clone() {
                for (column, metadata) in self.warm.columns.iter().enumerate() {
                    let value = solution.values[column];
                    if value == 0. {
                        continue;
                    }
                    let Some(metadata) = metadata else {
                        return Err(invalid("MIP artificial witness"));
                    };
                    if value < 0.
                        || value.fract() != 0.
                        || value > master.upper[column]
                        || value > 9_007_199_254_740_992.
                    {
                        mip.solution = None;
                        mip.status = MipStatus::Incomplete;
                        mip.bound = f64::NEG_INFINITY;
                        break;
                    }
                    routes.push(FixedRoute {
                        route: self.state.routes[metadata.tag as usize].clone(),
                        quantity: value,
                    });
                }
                routes.extend(self.state.fixed_routes.clone());
                if mip.solution.is_some() {
                    if let Some(upper) = super::tree::validate_routes(&self.state, &routes)? {
                        let objective = mip.solution.as_ref().unwrap().objective;
                        if (upper - objective).abs()
                            > 1e-6 + 1e-10 * upper.abs().max(objective.abs())
                        {
                            // A rounded matrix objective is not a proof about
                            // a materially different original arc-cost sum.
                            mip.status = MipStatus::Incomplete;
                            mip.bound = f64::NEG_INFINITY;
                        }
                        if upper < options.objective_cutoff {
                            mip.solution.as_mut().unwrap().objective = upper;
                        } else {
                            // Original-unit replay could not establish a strict
                            // improvement, even though the backend rounded down.
                            mip.solution = None;
                            mip.status = MipStatus::Incomplete;
                            mip.bound = f64::NEG_INFINITY;
                        }
                    } else {
                        mip.solution = None;
                        mip.status = MipStatus::Incomplete;
                        mip.bound = f64::NEG_INFINITY;
                    }
                }
                if mip.solution.is_none() {
                    routes.clear();
                }
            }
            Ok(Some(EnumeratedMipOutput {
                mip,
                route_values: routes,
                coverage_cutoff: f64::INFINITY,
            }))
        }
    }
}

#[cfg(all(test, feature = "highs"))]
mod tests {
    use super::*;
    use crate::master::solvers::mip::MipStatus;
    #[test]
    fn transfer_preserves_loaded_column_upper_bounds_and_artificials_stay_zero() {
        let problem = super::super::format::Model::from_json(include_str!(
            "../../../examples/networks/triangle.json"
        ))
        .unwrap()
        .problem::<2>()
        .unwrap();
        let root = solve(problem, Options::default()).unwrap();
        let mut node = root
            .snapshot
            .enumerate(&EnumerationOptions::default())
            .unwrap();
        for (column, metadata) in node.warm.columns.iter().enumerate() {
            if metadata.is_some() {
                node.master.set_upper(column, 0.).unwrap();
            }
        }
        // A duplicate transfer cannot re-add a bounded route as a fresh column.
        let repeated = node.enumerate(&EnumerationOptions::default()).unwrap();
        assert_eq!(repeated.master.costs.len(), node.master.costs.len());
        let mip = repeated
            .solve_enumerated_mip(&MipOptions::default())
            .unwrap()
            .unwrap();
        assert_eq!(mip.mip.status, MipStatus::Infeasible);
        assert!(mip.mip.solution.is_none());
        assert!(root.snapshot.solve(Options::default()).unwrap().master.lp < 2.);
    }
}
