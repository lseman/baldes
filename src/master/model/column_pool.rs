//! Solver-owned physical paths; no donor LP rows, tags or proof state transfer.
use super::*;
use std::borrow::Cow;

#[derive(Default)]
pub(super) struct Pool {
    identities: HashSet<(i32, String)>,
    routes: Vec<Arc<NetworkRoute>>,
}

impl Pool {
    pub fn absorb<const D: usize>(&mut self, node: &NodeSnapshot<D>) {
        // Only retain paths represented by an actual donor column. The route
        // registry can also contain paths that were enumerated or registered
        // speculatively; native inactive-pool search observes columns, not the
        // whole pricing-domain registry. A donor-active column is inactive
        // when absent from another candidate's master, so the recipient's
        // seed check still classifies it against its own loaded tags.
        let columns = node
            .warm
            .columns
            .iter()
            .flatten()
            .chain(&node.warm.generated_pool);
        for column in columns {
            let Ok(tag) = usize::try_from(column.tag) else {
                continue;
            };
            let Some(route) = node.state.routes.get(tag) else {
                continue;
            };
            if self
                .identities
                .insert((route.network, route.route.identity()))
            {
                self.routes.push(route.clone());
            }
        }
    }

    /// Publish a validated subset atomically, leaving the LP and proofs intact.
    /// Cache misses never exclude paths from subsequent ordinary pricing.
    pub fn seed<const D: usize>(
        &self,
        node: &mut NodeSnapshot<D>,
        deadline: Instant,
    ) -> Result<usize> {
        if self.routes.is_empty() || Instant::now() >= deadline {
            return Ok(0);
        }
        // Keep a stable owner for borrowed path keys while the staged registry grows.
        let original_registry = node.state.routes.clone();
        let mut registry = original_registry.clone();
        let mut tags = node.state.tags.clone();
        // Existing columns stay in place; publish only validated additions.
        let mut pooled = Vec::new();
        let mut cap_updates = Vec::new();
        let mut present: HashSet<_> = node
            .warm
            .columns
            .iter()
            .flatten()
            .chain(&node.warm.generated_pool)
            .filter_map(|c| usize::try_from(c.tag).ok())
            .collect();
        // `tags` is the maintained identity index, including the first tag
        // when schedule refresh coalesces registry entries. Only physical-path
        // caps need aggregation here; do not format every registered schedule.
        let mut path_caps = HashMap::new();
        for (tag, route) in original_registry.iter().enumerate() {
            let path = (route.network, route.route.arcs.as_slice());
            let cap = enumeration::route_upper(&node.state, tag).unwrap_or(f64::INFINITY);
            path_caps
                .entry(path)
                .and_modify(|previous: &mut f64| *previous = previous.min(cap))
                .or_insert(cap);
        }
        let mut domains = HashMap::new();
        let mut added = 0;
        for seed in &self.routes {
            if Instant::now() >= deadline {
                break;
            }
            let Some(oracle) = node
                .state
                .oracles
                .iter()
                .find(|o| o.network.id == seed.network)
            else {
                continue;
            };
            if oracle.multiplicity_upper() == 0.
                || seed.route.arcs.iter().any(|id| {
                    oracle.index.get(id).map_or(true, |&a| {
                        oracle.proved_zero_arcs.get(a).copied().unwrap_or(false)
                    })
                })
            {
                continue;
            }
            let path = (seed.network, seed.route.arcs.as_slice());
            let identity_key = (seed.network, seed.route.identity());
            let existing = tags
                .get(&identity_key)
                .and_then(|&tag| usize::try_from(tag).ok());
            let path_cap = path_caps.get(&path).copied().unwrap_or(f64::INFINITY);
            if path_cap == 0. {
                continue;
            }
            if let Some(tag) = existing {
                if present.contains(&tag)
                    || enumeration::route_upper(&node.state, tag) == Some(0.)
                    || node
                        .state
                        .universe
                        .as_ref()
                        .is_some_and(|u| !u.contains(&tag))
                {
                    continue;
                }
            } else if node.state.universe.is_some() {
                // Never widen a complete/cutoff universe or alias a path cap.
                continue;
            }
            let domain = match domains.entry(seed.network) {
                std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(resource_groups::domain_graph(oracle)?)
                }
            };
            let route = if let Some(tag) = existing {
                Cow::Borrowed(&registry[tag].route)
            } else if compatible_schedule(oracle, &seed.route) {
                Cow::Borrowed(&seed.route)
            } else if let Some(route) = domain.elementary_route(&seed.route.arcs).ok().flatten() {
                Cow::Owned(route)
            } else {
                continue;
            };
            if domain.exact_route_feasible(&route).is_none()
                || !oracle
                    .solver
                    .route_satisfies_relaxation(&route.nodes, &route.arcs)?
            {
                continue;
            }
            let coefficients = retirement::active_entries(&node.state, oracle, &route)?;
            let cost = original_costs::route_cost(oracle, &route)?;
            if !cost.is_finite() {
                return Err(invalid("inactive shared route cost overflow"));
            }
            let (identity, resolved) = match &route {
                // Neither borrowed route changed its schedule during replay.
                Cow::Borrowed(_) => (identity_key.1, existing),
                Cow::Owned(route) => {
                    let identity = route.identity();
                    let resolved = tags
                        .get(&(seed.network, identity.clone()))
                        .and_then(|&tag| usize::try_from(tag).ok());
                    (identity, resolved)
                }
            };
            if resolved.is_some_and(|tag| present.contains(&tag)) {
                continue;
            }
            let tag = match resolved {
                Some(tag) => {
                    if path_cap.is_finite() && path_cap < oracle.multiplicity_upper() {
                        cap_updates.push((tag, path_cap));
                    }
                    tag
                }
                None => {
                    let tag = registry.len();
                    let signed =
                        i32::try_from(tag).map_err(|_| invalid("shared route tag overflow"))?;
                    let mut route = route.into_owned();
                    route.cost = cost;
                    route.original_cost = cost;
                    Arc::make_mut(&mut registry).push(Arc::new(NetworkRoute {
                        network: seed.network,
                        route,
                    }));
                    Arc::make_mut(&mut tags).insert((seed.network, identity.clone()), signed);
                    path_caps
                        .entry(path)
                        .and_modify(|cap| *cap = cap.min(path_cap))
                        .or_insert(path_cap);
                    if path_cap.is_finite() && path_cap < oracle.multiplicity_upper() {
                        cap_updates.push((tag, path_cap));
                    }
                    tag
                }
            };
            pooled.push(
                cg::Column {
                    cost,
                    reduced_cost: 0.,
                    entries: coefficients,
                    acceptable: true,
                    tag: i32::try_from(tag).map_err(|_| invalid("shared route tag overflow"))?,
                    identity: format!("{}:{identity}", seed.network),
                }
                .into(),
            );
            present.insert(tag);
            added += 1;
        }
        node.state.routes = registry;
        node.state.tags = tags;
        // No fallible replay remains. Publish caps together with the routes
        // they describe, so a failed transfer cannot leave a cap on a reused tag.
        for (tag, upper) in cap_updates {
            node.state
                .route_upper_bounds
                .entry(tag)
                .and_modify(|cap| *cap = cap.min(upper))
                .or_insert(upper);
        }
        node.warm.generated_pool.extend(pooled);
        Ok(added)
    }
}

/// Validate a donor's exact schedule against the recipient graph and any
/// candidate-specific synchronized service windows. A failed check falls back
/// to reconstructing the canonical recipient schedule.
fn compatible_schedule<const D: usize>(oracle: &Oracle<D>, route: &Route) -> bool {
    let graph = &oracle.network.graph;
    if graph.has_coupled_resources() || !route.coupled_values.is_empty() {
        let positions = route
            .arcs
            .iter()
            .map(|id| oracle.index.get(id).copied())
            .collect::<Option<Vec<_>>>();
        if positions.and_then(|positions| route.selected_values_match(graph, &positions).ok())
            != Some(true)
        {
            return false;
        }
    }
    if route.nodes.len() != route.arcs.len().saturating_add(1)
        || route.resource_profile.len() != route.nodes.len()
        || route
            .resource_profile
            .iter()
            .any(|state| state.len() < graph.resources.len())
    {
        return false;
    }
    for resource in 0..graph.resources.len() {
        let Some(initial) = graph.root_resource(resource, true) else {
            return false;
        };
        if route.resource_profile[0][resource] != initial {
            return false;
        }
        for (position, &arc_id) in route.arcs.iter().enumerate() {
            let Some(&arc_index) = oracle.index.get(&arc_id) else {
                return false;
            };
            let arc = &graph.arcs[arc_index];
            let next = route.resource_profile[position + 1][resource];
            let previous = route.resource_profile[position][resource];
            let required = previous + arc.consumption[resource];
            if !next.is_finite()
                || !previous.is_finite()
                || !required.is_finite()
                || (graph.resources[resource] == crate::graph::ResourceKind::Equality
                    && next != required)
                || (graph.resources[resource] == crate::graph::ResourceKind::Disposable
                    && next < required)
            {
                return false;
            }
            let head = arc.head;
            let Some(vertex_bounds) = graph.vertices[head].bounds.get(resource) else {
                return false;
            };
            let Some(arc_bounds) = arc.bounds.get(resource) else {
                return false;
            };
            let mut lower = vertex_bounds.lower;
            let mut upper = vertex_bounds.upper;
            if let Some(bounds) = arc_bounds {
                lower = lower.max(bounds.lower);
                upper = upper.min(bounds.upper);
            }
            if next < lower || next > upper {
                return false;
            }
            #[cfg(feature = "multi-time-windows")]
            {
                let windows = graph.time_windows(head, resource);
                if !windows.is_empty()
                    && !windows.iter().any(|w| w.lower <= next && next <= w.upper)
                {
                    return false;
                }
            }
        }
    }
    for &(vertex, resource, window) in &oracle.synchronized_windows {
        if resource >= graph.resources.len() {
            return false;
        }
        let mut visited = false;
        for (position, node) in route.nodes.iter().enumerate() {
            if *node == vertex as i32 {
                visited = true;
                let Some(&value) = route.resource_profile[position].get(resource) else {
                    return false;
                };
                if value < window.lower || value > window.upper {
                    return false;
                }
            }
        }
        if !visited {
            return false;
        }
    }
    true
}

#[cfg(test)]
#[path = "../../../tests/unit/master/model/column_pool_tests.rs"]
mod tests;
