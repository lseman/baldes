//! Atomic all-network full-domain transfer. No reduced-cost cutoff is used.
pub(super) use super::route_bounds::upper as route_upper;
use super::*;
use crate::{
    enumeration::{CompleteResourceRelaxation, CostBoundedResourceRelaxation, PathDomainLimits},
    master::solvers::mip::{MipOptions, MipOutput},
};
use std::time::Duration;
mod cutoff;
pub use cutoff::{CutoffCostBound, CutoffEnumeration};

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
    /// None covers the full domain; Some covers only objectives below this cutoff.
    pub coverage_cutoff: Option<f64>,
    pub networks: Vec<(i32, usize)>,
    pub routes: usize,
    pub transferred: usize,
    pub failure: Option<String>,
}
#[derive(Debug)]
pub struct EnumeratedMipOutput {
    pub mip: MipOutput,
    pub route_values: Vec<FixedRoute>,
    pub master_values: Vec<(i32, f64)>,
    /// Full original elementary domain: no objective cutoff excluded routes.
    pub coverage_cutoff: f64,
}

fn collect<const D: usize>(
    state: &State<D>,
    options: &EnumerationOptions,
) -> std::result::Result<(Vec<NetworkRoute>, Vec<(i32, usize)>), String> {
    collect_with_plan(state, options, None)
}
fn collect_with_plan<const D: usize>(
    state: &State<D>,
    options: &EnumerationOptions,
    plan: Option<&[super::bounds::NetworkCutoff]>,
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
    for (network_index, oracle) in state.oracles.iter().enumerate() {
        if Instant::now() >= deadline {
            return Err("TimeLimit".into());
        }
        if oracle.multiplicity_upper() == 0. {
            counts.push((oracle.network.id, 0));
            continue;
        }
        let cutoff = plan.map(|p| &p[network_index]);
        if let Some(super::bounds::NetworkCutoff::Empty { network }) = cutoff {
            if *network != oracle.network.id {
                return Err("cutoff network mismatch".into());
            }
            counts.push((oracle.network.id, 0));
            continue;
        }
        let domain =
            super::resource_groups::residual_domain_graph(oracle).map_err(|e| e.to_string())?;
        let graph = domain.as_ref();
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
        // Disposable transitions relax scalar equality transitions: clamping a
        // too-small load adds paths, never removes an equality-feasible path.
        // Collect that superset, then replay every path on the original graph.
        // Coupled restrictions are also relaxed only in this outer path
        // superset. Every candidate is certified and repriced against `graph`
        // below before the atomic transfer. The scalar collectors themselves
        // retain their explicit unsupported-coupled proof contract.
        let relaxed = (graph.has_coupled_resources()
            || graph
                .resources
                .iter()
                .any(|r| *r == crate::graph::ResourceKind::Equality))
        .then(|| -> Result<_> {
            let mut relaxed = graph.clone();
            relaxed
                .resources
                .fill(crate::graph::ResourceKind::Disposable);
            relaxed.set_coupled_resources(&[])?;
            Ok(relaxed)
        })
        .transpose()
        .map_err(|e| e.to_string())?;
        if matches!(cutoff, Some(super::bounds::NetworkCutoff::Cost { .. }))
            && graph
                .coupled_kernel()
                .is_some_and(|kernel| kernel.has_costs())
        {
            return Err("UnsupportedDomain: additive cutoff omits coupled costs".into());
        }
        let limits = PathDomainLimits {
            max_paths: paths_left,
            max_work: work_left,
            max_stored_arcs: arcs_left,
            deadline: Some(deadline),
        };
        let full_domain;
        let pruned_domain;
        let costs;
        let (paths, work_used, stored_arcs) = match cutoff {
            Some(super::bounds::NetworkCutoff::Cost {
                network,
                arc_cost_lower_bounds,
                arc_ceiling,
                ..
            }) => {
                if *network != oracle.network.id {
                    return Err("cutoff network mismatch".into());
                }
                costs = graph
                    .arcs
                    .iter()
                    .map(|a| arc_cost_lower_bounds[oracle.index[&a.id]])
                    .collect::<Vec<_>>();
                pruned_domain = CostBoundedResourceRelaxation::collect(
                    relaxed.as_ref().unwrap_or(graph),
                    &costs,
                    *arc_ceiling,
                    limits,
                )
                .map_err(|e| e.to_string())?;
                (
                    pruned_domain.paths(),
                    pruned_domain.work_used(),
                    pruned_domain.stored_arcs(),
                )
            }
            None => {
                full_domain =
                    CompleteResourceRelaxation::collect(relaxed.as_ref().unwrap_or(graph), limits)
                        .map_err(|e| e.to_string())?;
                (
                    full_domain.paths(),
                    full_domain.work_used(),
                    full_domain.stored_arcs(),
                )
            }
            Some(super::bounds::NetworkCutoff::Empty { .. }) => unreachable!(),
        };
        paths_left -= paths.len();
        work_left -= work_used;
        arcs_left -= stored_arcs;
        let problem = crate::master::root::RootProblem::set_partitioning(graph.clone(), 1.)
            .map_err(|e| e.to_string())?;
        let before = routes.len();
        for path in paths {
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
            match crate::master::root::exact_route_feasibility(&problem, &probe) {
                Some(false) => continue,
                Some(true) => {}
                None => return Err("Arithmetic: exact resource replay is unknown".into()),
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
            probe.original_cost = ids
                .iter()
                .map(|id| oracle.network.graph.arcs[oracle.index[id]].cost)
                .sum::<f64>()
                + oracle
                    .network
                    .graph
                    .coupled_path_cost(
                        &probe
                            .arcs
                            .iter()
                            .map(|id| oracle.index[id])
                            .collect::<Vec<_>>(),
                    )
                    .map_err(|e| e.to_string())?
                    .ok_or("infeasible coupled path")?;
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
    transfer_with(
        master,
        state,
        cg,
        options,
        |destination, column| {
            destination
                .add_column(column.cost, &column.entries)
                .map(|_| ())
        },
        Instant::now,
    )
}

fn transfer_with<const D: usize>(
    master: &mut ClpMaster,
    state: &mut State<D>,
    cg: &mut cg::Options,
    options: &EnumerationOptions,
    insert: impl FnMut(&mut ClpMaster, &cg::Column) -> Result<()>,
    mut now: impl FnMut() -> Instant,
) -> Result<()> {
    if state.universe.is_some() {
        return Ok(());
    }
    state.enumeration_report.attempts += 1;
    let deadline = now()
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
    install_collected_with(master, state, cg, routes, counts, deadline, insert, now)
}
fn install_collected_with<const D: usize>(
    master: &mut ClpMaster,
    state: &mut State<D>,
    cg: &mut cg::Options,
    routes: Vec<NetworkRoute>,
    counts: Vec<(i32, usize)>,
    deadline: Instant,
    mut insert: impl FnMut(&mut ClpMaster, &cg::Column) -> Result<()>,
    mut now: impl FnMut() -> Instant,
) -> Result<()> {
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
    let mut route_upper_bounds = state.route_upper_bounds.clone();
    let mut universe = vec![];
    let mut loaded: HashSet<usize> = warm.as_ref().map_or_else(
        || {
            known
                .iter()
                .filter(|c| c.tag >= 0)
                .map(|c| c.tag as usize)
                .collect()
        },
        |w| {
            w.columns
                .iter()
                .flatten()
                .filter(|c| c.tag >= 0)
                .map(|c| c.tag as usize)
                .collect()
        },
    );
    let allowed: HashSet<_> = routes.iter().map(path_key).collect();
    if let Some(warm) = &warm {
        for (column, metadata) in warm.columns.iter().enumerate() {
            if let Some(metadata) = metadata {
                if metadata.tag < 0 {
                    continue;
                }
                let tag = metadata.tag as usize;
                let key = path_key(&registry[tag]);
                // Keep one canonical variable per physical path. Resource
                // profile aliases must not bypass a path's proved cap.
                if !allowed.contains(&key) || path_tags.get(&key) != Some(&tag) {
                    destination.set_upper(column, 0.)?;
                    route_upper_bounds.insert(tag, 0.);
                } else {
                    let route = &registry[tag];
                    let network = state
                        .oracles
                        .iter()
                        .find(|o| o.network.id == route.network)
                        .unwrap();
                    route_upper_bounds.insert(
                        tag,
                        (if destination.upper[column] == f64::MAX {
                            f64::INFINITY
                        } else {
                            destination.upper[column]
                        })
                        .min(network.multiplicity_upper()),
                    );
                }
            }
        }
    }
    let mut transferred = 0;
    for route in routes {
        if now() >= deadline {
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
        insert(&mut destination, &column)?;
        if let Some(warm) = &mut warm {
            warm.columns.push(Some(column.clone()));
            warm.generated_pool.push(column);
        } else {
            known.push(column);
        }
        transferred += 1;
    }
    if now() >= deadline {
        state.enumeration_report.failure = Some("TimeLimit".into());
        return Ok(());
    }
    *master = destination;
    cg.warm_start = warm;
    cg.known = known;
    state.routes = registry;
    state.tags = tags;
    state.route_upper_bounds = route_upper_bounds;
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
        if oracle.multiplicity_upper() == 0. {
            continue;
        }
        oracle.calls += 1;

        for &tag in state.universe.as_ref().unwrap() {
            let route = &state.routes[tag];
            if route.network != oracle.network.id
                || route.route.arcs.iter().any(|id| {
                    oracle
                        .proved_zero_arcs
                        .get(oracle.index[id])
                        .copied()
                        .unwrap_or(false)
                })
            {
                continue;
            }
            let cap = state
                .route_upper_bounds
                .get(&tag)
                .copied()
                .unwrap_or(oracle.multiplicity_upper())
                .min(oracle.multiplicity_upper());
            if cap == 0. {
                continue;
            }
            let coefficients = entries(oracle, &route.route, state.shared_offset)?;
            let reduced = coefficients
                .iter()
                .fold(route.route.original_cost, |v, &(r, a)| v - duals[r] * a);
            if !reduced.is_finite() {
                return Err(invalid("enumerated reduced cost overflow"));
            }
            minimum = minimum.min(reduced);
            if reduced < 0. {
                correction += reduced * cap;
                for &(r, a) in &coefficients {
                    gradient[r] -= cap * a;
                }
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
    }
    if let Some((static_correction, static_gradient)) =
        super::master_variables::correction(state, duals)
    {
        correction += static_correction;
        for (row, value) in static_gradient {
            gradient[row] -= value;
        }
        if correction.is_finite() && gradient.iter().all(|g| g.is_finite()) {
            state.correction = Some(correction);
            state.gradient = Some(gradient);
        } else {
            state.correction = None;
            state.gradient = None;
        }
    } else {
        state.correction = None;
        state.gradient = None;
    }
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

// Proof outcomes require the finite LP encoding to preserve original costs,
// coefficients and fixed reductions. Witness replay cannot detect a route
// which the rounded model excluded before returning any solution.
#[cfg(feature = "highs")]
fn exact_integer_encoding<const D: usize>(
    state: &State<D>,
    master: &ClpMaster,
    columns: &[Option<cg::Column>],
    deadline: Instant,
) -> Result<bool> {
    use crate::master::root::ExactSum;
    if !super::bounds::exact_fixed_reduction(state, master)? {
        return Ok(false);
    }
    for (column, metadata) in columns.iter().enumerate() {
        if Instant::now() >= deadline {
            return Ok(false);
        }
        if master.upper[column] == 0. {
            continue;
        }
        let Some(metadata) = metadata else {
            return Ok(false);
        };
        let mut cost = ExactSum::default();
        let mut rows = BTreeMap::<usize, ExactSum>::new();
        if let Some((index, sign)) = super::master_variables::decode(metadata.tag) {
            let variable = &state.master_variables[index];
            if cost.product(sign, variable.cost).is_none() {
                return Ok(false);
            }
            for &(row, a) in &variable.entries {
                if rows
                    .entry(state.shared_offset + row)
                    .or_default()
                    .product(sign, a)
                    .is_none()
                {
                    return Ok(false);
                }
            }
        } else {
            let Some(route) = usize::try_from(metadata.tag)
                .ok()
                .and_then(|tag| state.routes.get(tag))
            else {
                return Ok(false);
            };
            let Some(oracle) = state.oracles.iter().find(|o| o.network.id == route.network) else {
                return Ok(false);
            };
            if let Some(kernel) = oracle.network.graph.coupled_kernel() {
                let arcs: Vec<_> = route.route.arcs.iter().map(|id| oracle.index[id]).collect();
                let Some((true, bounds)) = kernel.certify(&arcs) else {
                    return Ok(false);
                };
                if bounds[0] != bounds[1] || cost.add(bounds[0]).is_none() {
                    return Ok(false);
                }
            }
            for &arc in &route.route.arcs {
                let position = oracle.index[&arc];
                if cost.add(oracle.network.graph.arcs[position].cost).is_none() {
                    return Ok(false);
                }
                for &(row, a) in &oracle.network.arc_entries[position] {
                    if rows
                        .entry(state.shared_offset + row)
                        .or_default()
                        .add(a)
                        .is_none()
                    {
                        return Ok(false);
                    }
                }
            }
            for &(row, a) in &oracle.route_entries {
                if rows.entry(row).or_default().add(a).is_none() {
                    return Ok(false);
                }
            }
            for &(row, a) in &oracle.network.route_entries {
                if rows
                    .entry(state.shared_offset + row)
                    .or_default()
                    .add(a)
                    .is_none()
                {
                    return Ok(false);
                }
            }
            for (row, a) in super::nonlinear::entries(oracle, &route.route)? {
                if rows
                    .entry(state.shared_offset + row)
                    .or_default()
                    .add(a)
                    .is_none()
                {
                    return Ok(false);
                }
            }
        }
        if cost.add(-master.costs[column]).is_none() || cost.sign() != 0. {
            return Ok(false);
        }
        for (row, a) in master.column_entries(column)? {
            if rows.entry(row).or_default().add(-a).is_none() {
                return Ok(false);
            }
        }
        if rows.values().any(|value| value.sign() != 0.) {
            return Ok(false);
        }
    }
    Ok(true)
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
            let mut columns = self.warm.columns.clone();
            let mut present: HashSet<_> = columns
                .iter()
                .flatten()
                .filter(|c| c.tag >= 0)
                .map(|c| c.tag as usize)
                .collect();
            // Cleanup may retire a route from the restricted LP without
            // changing complete-domain coverage. Restore every missing path
            // before claiming an integer optimum for that complete domain.
            for &tag in self.state.universe.as_ref().unwrap() {
                if !present.insert(tag) {
                    continue;
                }
                let route = &self.state.routes[tag];
                let oracle = self
                    .state
                    .oracles
                    .iter()
                    .find(|o| o.network.id == route.network)
                    .unwrap();
                let metadata = cg::Column {
                    cost: route.route.original_cost,
                    reduced_cost: 0.,
                    entries: entries(oracle, &route.route, self.state.shared_offset)?,
                    acceptable: true,
                    tag: tag as i32,
                    identity: format!("{}:{}", route.network, route.route.identity()),
                };
                master.add_column(metadata.cost, &metadata.entries)?;
                columns.push(Some(metadata));
            }
            for (column, metadata) in columns.iter().enumerate() {
                if let Some(metadata) = metadata {
                    let cap = if metadata.tag >= 0 {
                        route_upper(&self.state, metadata.tag as usize)
                    } else {
                        super::master_variables::decode(metadata.tag).map(|(index, sign)| {
                            super::master_variables::upper(
                                &self.state.master_variables[index],
                                sign,
                            )
                        })
                    };
                    if let Some(cap) = cap {
                        if cap < master.upper[column] {
                            master.set_upper(column, cap)?;
                        }
                    }
                }
            }
            for &column in &self.warm.artificials {
                master.set_upper(column, 0.)?;
            }
            let integers = columns
                .iter()
                .enumerate()
                .filter_map(|(c, meta)| {
                    meta.as_ref().and_then(|m| {
                        let integer = super::master_variables::decode(m.tag)
                            .map_or(true, |(index, _)| {
                                self.state.master_variables[index].integer
                            });
                        integer.then_some(c)
                    })
                })
                .collect::<Vec<_>>();
            let encoding_exact = exact_integer_encoding(
                &self.state,
                &master,
                &columns,
                started
                    .checked_add(options.time_limit)
                    .ok_or_else(|| invalid("MIP deadline"))?,
            )?;
            let mut options = options.clone();
            options.time_limit = options.time_limit.saturating_sub(started.elapsed());
            let mut mip =
                crate::master::solvers::highs::solve_integer(&master, &integers, &options)?;
            if !encoding_exact {
                mip.bound = f64::NEG_INFINITY;
                if matches!(
                    mip.status,
                    MipStatus::Optimal | MipStatus::Infeasible | MipStatus::ObjectiveCutoff
                ) {
                    mip.status = MipStatus::Incomplete;
                }
            }
            let mut routes = vec![];
            let mut master_values: Vec<_> = self
                .state
                .master_variables
                .iter()
                .map(|v| (v.id, 0.))
                .collect();
            if let Some(solution) = mip.solution.clone() {
                for (column, metadata) in columns.iter().enumerate() {
                    let value = solution.values[column];
                    if value == 0. {
                        continue;
                    }
                    let Some(metadata) = metadata else {
                        return Err(invalid("MIP artificial witness"));
                    };
                    let static_variable = super::master_variables::decode(metadata.tag);
                    let integer = static_variable.map_or(true, |(index, _)| {
                        self.state.master_variables[index].integer
                    });
                    if value < 0.
                        || integer && value.fract() != 0.
                        || value > master.upper[column]
                        || value > 9_007_199_254_740_992.
                    {
                        mip.solution = None;
                        mip.status = MipStatus::Incomplete;
                        mip.bound = f64::NEG_INFINITY;
                        break;
                    }
                    if let Some((index, sign)) = static_variable {
                        master_values[index].1 += sign * value;
                        continue;
                    }
                    routes.push(FixedRoute {
                        route: self.state.routes[metadata.tag as usize].clone(),
                        quantity: value,
                    });
                }
                routes.extend(self.state.fixed_routes.clone());
                if mip.solution.is_some() {
                    if let Some(upper) =
                        super::tree::validate_routes(&self.state, &routes, &master_values)?
                    {
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
                    master_values.clear();
                }
            }
            if mip.solution.is_none() {
                routes.clear();
                master_values.clear();
            }
            Ok(Some(EnumeratedMipOutput {
                mip,
                route_values: routes,
                master_values,
                coverage_cutoff: f64::INFINITY,
            }))
        }
    }
}

#[cfg(all(test, feature = "highs"))]
#[path = "../../../../tests/unit/master/multi_pricing/enumeration/tests.rs"]
mod tests;
