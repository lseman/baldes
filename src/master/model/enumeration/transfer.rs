//! Install collected route universes atomically on a private master.
use super::*;

fn path_key(route: &NetworkRoute) -> (i32, Vec<i32>) {
    (route.network, route.route.arcs.clone())
}

pub(in crate::master::model) fn transfer<const D: usize>(
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

pub(super) fn transfer_with<const D: usize>(
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
    let (routes, counts) = match collect_routes(state, options) {
        Ok(result) => result,
        Err(reason) => {
            state.enumeration_report.failure = Some(reason);
            return Ok(());
        }
    };
    install_collected_with(master, state, cg, routes, counts, deadline, insert, now)
}
pub(super) fn install_collected_with<const D: usize>(
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
    // Controlled networks must retain the certified representative, not an
    // arbitrary earlier physical-path alias with different selected metadata.
    for route in &routes {
        let oracle = state
            .oracles
            .iter()
            .find(|o| o.network.id == route.network)
            .unwrap();
        if !super::super::control_pricing::needed(oracle) {
            continue;
        }
        let key = path_key(route);
        path_tags.remove(&key);
        if let Some(&tag) = state.tags.get(&(route.network, route.route.identity())) {
            let index = usize::try_from(tag).map_err(|_| invalid("controlled enumeration tag"))?;
            if registry
                .get(index)
                .is_some_and(|old| old.route.original_cost == route.route.original_cost)
            {
                path_tags.insert(key, index);
                registry[index] = Arc::new(route.clone());
            }
        }
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
            registry.push(Arc::new(route));
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
