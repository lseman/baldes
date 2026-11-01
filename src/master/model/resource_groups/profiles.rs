//! Refresh schedules in place, preserving physical-path caps and stable tags.
use super::*;
use std::borrow::Cow;

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
            for (row, before, after) in merged_coefficients(&old_entries, &new_entries) {
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
        let previous = ordered_coefficients(&metadata.entries);
        let ordered = matches!(previous, Cow::Borrowed(_));
        for (row, before, value) in merged_coefficients(&previous, &updated) {
            // Unordered/duplicate metadata gets the original full-write behavior;
            // otherwise unchanged coefficients already have the desired value.
            if !ordered || before != value || row >= node.master.rhs.len() {
                node.master.set_coefficient(row, column, value)?;
            }
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

/// Preserve last-entry semantics for legacy unordered/duplicate metadata.
/// Ordinary route replay already produces sorted, unique rows and borrows here.
fn ordered_coefficients(entries: &[(usize, f64)]) -> Cow<'_, [(usize, f64)]> {
    if entries.windows(2).all(|pair| pair[0].0 < pair[1].0) {
        return Cow::Borrowed(entries);
    }
    let mut ordered = entries.to_vec();
    ordered.sort_by_key(|&(row, _)| row);
    ordered.dedup_by(|later, earlier| {
        if later.0 == earlier.0 {
            earlier.1 = later.1;
            true
        } else {
            false
        }
    });
    Cow::Owned(ordered)
}

/// Merge sorted, unique sparse rows without allocating temporary trees or sets.
fn merged_coefficients<'a>(
    before: &'a [(usize, f64)],
    after: &'a [(usize, f64)],
) -> impl Iterator<Item = (usize, f64, f64)> + 'a {
    let (mut left, mut right) = (0, 0);
    std::iter::from_fn(move || match (before.get(left), after.get(right)) {
        (Some(&(row, value)), Some(&(next_row, _))) if row < next_row => {
            left += 1;
            Some((row, value, 0.))
        }
        (Some(&(row, _)), Some(&(next_row, value))) if row > next_row => {
            right += 1;
            Some((next_row, 0., value))
        }
        (Some(&(row, previous)), Some(&(_, value))) => {
            left += 1;
            right += 1;
            Some((row, previous, value))
        }
        (Some(&(row, value)), None) => {
            left += 1;
            Some((row, value, 0.))
        }
        (None, Some(&(row, value))) => {
            right += 1;
            Some((row, 0., value))
        }
        (None, None) => None,
    })
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/model/resource_profile_tests.rs"]
mod tests;
