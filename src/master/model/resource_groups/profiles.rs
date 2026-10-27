//! Refresh schedules in place, preserving physical-path caps and stable tags.
use super::*;

pub(in crate::master::model) fn refresh<const D: usize>(
    node: &mut NodeSnapshot<D>,
    previous: &State<D>,
    touched: &HashSet<i32>,
) -> Result<bool> {
    let mut domains = HashMap::new();
    for oracle in &node.state.oracles {
        if touched.contains(&oracle.network.id) {
            domains.insert(
                oracle.network.id,
                crate::master::routing::RoutingContext::set_partitioning(
                    domain_graph(oracle)?.into_owned(),
                    1.,
                )?,
            );
        }
    }
    let fixed_impossible = node.state.fixed_routes.iter().any(|fixed| {
        domains.get(&fixed.route.network).is_some_and(|domain| {
            crate::master::routing::exact_route_feasibility(domain, &fixed.route.route)
                == Some(false)
        })
    });
    if !fixed_impossible {
        let mut rhs = node.state.rhs.clone();
        for fixed in &mut node.state.fixed_routes {
            let Some(domain) = domains.get(&fixed.route.network) else {
                continue;
            };
            if crate::master::routing::exact_route_feasibility(domain, &fixed.route.route)
                != Some(true)
            {
                return Err(incomplete("fixed route schedule refresh is unsupported"));
            }
            let old_oracle = previous
                .oracles
                .iter()
                .find(|o| o.network.id == fixed.route.network)
                .ok_or_else(|| invalid("fixed route network disappeared during refresh"))?;
            let new_oracle = node
                .state
                .oracles
                .iter()
                .find(|o| o.network.id == fixed.route.network)
                .ok_or_else(|| invalid("fixed route network disappeared during refresh"))?;
            let old_entries =
                super::super::entries(old_oracle, &fixed.route.route, previous.shared_offset)?;
            let canonical = domain
                .elementary_route(&fixed.route.route.arcs)
                .ok_or_else(|| incomplete("fixed route canonical schedule reconstruction"))?;
            let canonical = if fixed.route.route.coupled_controls.is_empty() {
                canonical
            } else {
                canonical.with_paired_controls(
                    &new_oracle.network.graph,
                    fixed.route.route.coupled_controls.clone(),
                )?
            };
            fixed.route.route.resources = canonical.resources;
            fixed.route.route.resource_profile = canonical.resource_profile;
            fixed.route.route.coupled_controls = canonical.coupled_controls;
            fixed.route.route.coupled_values = canonical.coupled_values;
            let new_entries =
                super::super::entries(new_oracle, &fixed.route.route, node.state.shared_offset)?;
            let old: BTreeMap<_, _> = old_entries.into_iter().collect();
            let new: BTreeMap<_, _> = new_entries.into_iter().collect();
            for row in old
                .keys()
                .chain(new.keys())
                .copied()
                .collect::<HashSet<_>>()
            {
                let before = old.get(&row).copied().unwrap_or(0.);
                let after = new.get(&row).copied().unwrap_or(0.);
                let value = rhs
                    .get_mut(row)
                    .ok_or_else(|| invalid("fixed route row refresh index"))?;
                *value -= fixed.quantity * (after - before);
            }
        }
        if rhs.iter().any(|value| !value.is_finite()) {
            return Err(invalid("fixed route row refresh overflow"));
        }
        if rhs != node.state.rhs {
            node.master.set_rhs(&rhs)?;
            node.state.rhs = rhs;
        }
    }
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
        match crate::master::routing::exact_route_feasibility(domain, &route.route) {
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
        if compatible && oracle.synchronized_windows.is_empty() {
            continue;
        }
        let canonical = domain
            .elementary_route(&route.route.arcs)
            .ok_or_else(|| incomplete("resource branch canonical schedule reconstruction"))?;
        let canonical = if route.route.coupled_controls.is_empty() {
            canonical
        } else {
            canonical
                .with_paired_controls(&oracle.network.graph, route.route.coupled_controls.clone())?
        };
        let route = Arc::make_mut(route);
        route.route.resources = canonical.resources;
        route.route.resource_profile = canonical.resource_profile;
        route.route.coupled_controls = canonical.coupled_controls;
        route.route.coupled_values = canonical.coupled_values;
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
