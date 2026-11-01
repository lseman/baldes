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
            domains.insert(oracle.network.id, domain_graph(oracle)?.into_owned());
        }
    }
    let fixed_impossible = node.state.fixed_routes.iter().any(|fixed| {
        domains
            .get(&fixed.route.network)
            .is_some_and(|domain| domain.exact_route_feasibility(&fixed.route.route) == Some(false))
    });
    if !fixed_impossible {
        let mut rhs = node.state.rhs.clone();
        for fixed in &mut node.state.fixed_routes {
            let Some(domain) = domains.get(&fixed.route.network) else {
                continue;
            };
            if domain.exact_route_feasibility(&fixed.route.route) != Some(true) {
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
                .ok()
                .flatten()
                .ok_or_else(|| incomplete("fixed route canonical schedule reconstruction"))?;
            let positions = fixed
                .route
                .route
                .arcs
                .iter()
                .map(|id| domain.arc_position(*id))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| invalid("unknown fixed route arc"))?;
            if !fixed
                .route
                .route
                .selected_values_match(domain, &positions)?
            {
                return Err(incomplete(
                    "fixed route paired selection is unavailable in the refreshed domain",
                ));
            }
            fixed.route.route.resources = canonical.resources;
            fixed.route.route.resource_profile = canonical.resource_profile;
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
            node.state.rhs = rhs;
            node.master
                .set_rhs(&retirement::physical_rhs(&node.state))?;
        }
    }
    let complete_universe = node.state.universe.is_some();
    let mut schedules_changed = false;
    for tag in 0..node.state.routes.len() {
        let route = &node.state.routes[tag];
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
        match domain.exact_route_feasibility(&route.route) {
            // During ordinary pricing, an existing column is not capped: its
            // Boolean violation coefficient on the
            // rhs-zero branch row blocks it, and that row's dual reaches
            // pricing. A zero cap would hide the column from the row while
            // pricing can still return it, so pricing could never certify the
            // LP. A complete retained universe can exclude a proved-impossible
            // path without competing with an open pricing oracle. Synchronized
            // windows have no such row and also keep the cap.
            Some(false) if !complete_universe && oracle.synchronized_windows.is_empty() => continue,
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
            .ok()
            .flatten()
            .ok_or_else(|| incomplete("resource branch canonical schedule reconstruction"))?;
        let positions = route
            .route
            .arcs
            .iter()
            .map(|id| domain.arc_position(*id))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| invalid("unknown route arc"))?;
        if !route.route.selected_values_match(domain, &positions)? {
            return Err(incomplete(
                "route paired selection is unavailable in the refreshed domain",
            ));
        }
        if route.route.resources != canonical.resources
            || route.route.resource_profile != canonical.resource_profile
        {
            let route = Arc::make_mut(&mut Arc::make_mut(&mut node.state.routes)[tag]);
            route.route.resources = canonical.resources;
            route.route.resource_profile = canonical.resource_profile;
            schedules_changed = true;
        }
    }
    if schedules_changed {
        // A shared map will be rebuilt; copying its old strings first is wasted work.
        if Arc::get_mut(&mut node.state.tags).is_none() {
            node.state.tags = Arc::new(HashMap::with_capacity(node.state.routes.len()));
        }
        let tags = Arc::get_mut(&mut node.state.tags).unwrap();
        tags.clear();
        for (tag, route) in node.state.routes.iter().enumerate() {
            tags.entry((route.network, route.route.identity()))
                .or_insert(tag as i32);
        }
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
        let updated = super::super::retirement::active_entries(state, oracle, &route.route)?;
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
        metadata.entries = super::super::retirement::active_entries(state, oracle, &route.route)?;
        metadata.identity = format!("{}:{}", route.network, route.route.identity());
    }
    Ok(fixed_impossible)
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/model/resource_profile_tests.rs"]
mod tests;
