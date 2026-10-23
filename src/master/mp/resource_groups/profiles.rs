//! Refresh schedules in place, preserving physical-path caps and stable tags.
use super::*;

pub(super) fn refresh<const D: usize>(
    node: &mut NodeSnapshot<D>,
    touched: &HashSet<i32>,
) -> Result<bool> {
    let mut domains = HashMap::new();
    for oracle in &node.state.oracles {
        if touched.contains(&oracle.network.id) {
            domains.insert(
                oracle.network.id,
                crate::master::root::RootProblem::set_partitioning(
                    domain_graph(oracle)?.into_owned(),
                    1.,
                )?,
            );
        }
    }
    let fixed_impossible = node.state.fixed_routes.iter().any(|fixed| {
        domains.get(&fixed.route.network).is_some_and(|domain| {
            crate::master::root::exact_route_feasibility(domain, &fixed.route.route) == Some(false)
        })
    });
    for (tag, route) in node.state.routes.iter_mut().enumerate() {
        let Some(domain) = domains.get(&route.network) else {
            continue;
        };
        let oracle = node
            .state
            .oracles
            .iter()
            .find(|o| o.network.id == route.network)
            .unwrap();
        let compatible = entries(oracle, &route.route).iter().all(|&(_, a)| a == 0.);
        // Integer validation is exact-real; a rounded reconstruction failure
        // cannot silently remove an original-feasible path from the universe.
        match crate::master::root::exact_route_feasibility(domain, &route.route) {
            Some(false) => {
                node.state.route_upper_bounds.insert(tag, 0.);
                continue;
            }
            Some(true) => {}
            None if compatible => continue,
            None => {
                return Err(incomplete(
                    "resource branch exact schedule replay is unsupported",
                ))
            }
        }
        if compatible {
            continue;
        }
        let canonical = domain
            .elementary_route(&route.route.arcs)
            .ok_or_else(|| incomplete("resource branch canonical schedule reconstruction"))?;
        let route = Arc::make_mut(route);
        route.route.resources = canonical.resources;
        route.route.resource_profile = canonical.resource_profile;
    }
    node.state.tags.clear();
    for (tag, route) in node.state.routes.iter().enumerate() {
        node.state
            .tags
            .entry((route.network, route.route.identity()))
            .or_insert(tag as i32);
    }
    let state = &node.state;
    for (column, metadata) in node.warm.columns.iter_mut().enumerate() {
        let Some(metadata) = metadata else {
            continue;
        };
        if metadata.tag < 0 {
            continue;
        }
        let tag = metadata.tag as usize;
        let route = &state.routes[tag];
        if !touched.contains(&route.network) {
            continue;
        }
        let oracle = state
            .oracles
            .iter()
            .find(|o| o.network.id == route.network)
            .unwrap();
        let updated = super::super::entries(oracle, &route.route, state.shared_offset)?;
        let mut coefficients: BTreeMap<_, _> =
            metadata.entries.iter().map(|&(r, _)| (r, 0.)).collect();
        coefficients.extend(updated.iter().copied());
        for (row, value) in coefficients {
            node.master.set_coefficient(row, column, value)?;
        }
        if let Some(&upper) = state.route_upper_bounds.get(&tag) {
            if upper < node.master.upper[column] {
                node.master.set_upper(column, upper)?;
            }
        }
        metadata.entries = updated;
        metadata.identity = format!("{}:{}", route.network, route.route.identity());
    }
    for metadata in &mut node.warm.generated_pool {
        if metadata.tag < 0 {
            continue;
        }
        let route = &state.routes[metadata.tag as usize];
        if !touched.contains(&route.network) {
            continue;
        }
        let oracle = state
            .oracles
            .iter()
            .find(|o| o.network.id == route.network)
            .unwrap();
        metadata.entries = super::super::entries(oracle, &route.route, state.shared_offset)?;
        metadata.identity = format!("{}:{}", route.network, route.route.identity());
    }
    Ok(fixed_impossible)
}
