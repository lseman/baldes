//! Exact replay of a physical arc path.
use super::*;
use crate::labeling::extend_resources;
use crate::Route;

impl<const D: usize> Graph<D> {
    /// Replay an arc-ID path from source to sink: resources and windows are
    /// extended exactly, vertices are visited once (unless routes are bounded
    /// walks) and explicit elementarity groups are respected. None means the
    /// path is infeasible in this graph.
    pub fn elementary_route(&self, ids: &[i32]) -> Result<Option<Route>> {
        let g = self;
        let lookup = g.sparse_arc_lookup();
        let position = |id: i32| {
            lookup
                .as_ref()
                .and_then(|lookup| lookup.get(id))
                .or_else(|| g.arc_by_id(id))
                .map(|(position, _)| position)
        };
        let Some((nodes, positions, cost, resources, resource_profile)) = (|| {
            if ids.len() > g.original_max_arcs() {
                return None;
            }
            let mut at = g.source;
            let mut nodes = vec![at as i32];
            let mut visited = HashSet::new();
            let mut groups = HashSet::new();
            if g.has_explicit_elementarity_groups() {
                groups.extend(g.vertices[g.source].elements.iter().copied());
            }
            let mut q = crate::graph::ResourceValues::filled(g.resources.len(), 0.);
            for (r, value) in q.iter_mut().enumerate().take(g.resources.len()) {
                *value = g.root_resource(r, true)?;
                if *value < g.vertices[at].bounds[r].lower
                    || *value > g.vertices[at].bounds[r].upper
                {
                    return None;
                }
            }
            let mut resource_profile = vec![q[..g.resources.len()].to_vec()];
            let mut cost = 0.;
            for &id in ids {
                let arc = &g.arcs[position(id)?];
                if arc.tail != at || at == g.sink {
                    return None;
                }
                if g.has_explicit_elementarity_groups() {
                    let mut transition_elements = HashSet::new();
                    for &element in arc.elements.iter().chain(&g.vertices[arc.head].elements) {
                        if transition_elements.insert(element) && !groups.insert(element) {
                            return None;
                        }
                    }
                }
                q = extend_resources(g, arc, &q, true)?;
                at = arc.head;
                if at == g.source
                    || (at != g.sink && !visited.insert(at) && g.max_route_arcs().is_none())
                {
                    return None;
                }
                cost += arc.cost;
                nodes.push(at as i32);
                resource_profile.push(q[..g.resources.len()].to_vec());
            }
            if at != g.sink {
                return None;
            }
            let positions: Vec<_> = ids.iter().map(|id| position(*id)).collect::<Option<_>>()?;
            Some((
                nodes,
                positions,
                cost,
                q[..g.resources.len()].to_vec(),
                resource_profile,
            ))
        })() else {
            return Ok(None);
        };
        let Some(coupled) = g.coupled_route_cost(&positions)? else {
            return Ok(None);
        };
        let cost = cost + coupled;
        Ok(Some(Route {
            coupled_values: g.coupled_paired_values(&positions)?,
            nodes,
            arcs: ids.to_vec(),
            cost,
            original_cost: cost,
            resources,
            resource_profile,
        }))
    }

    /// Index of the arc with this ID: the arc at index `id` when its ID
    /// matches, else the first arc carrying the ID.
    pub fn arc_position(&self, id: i32) -> Option<usize> {
        usize::try_from(id)
            .ok()
            .filter(|&index| self.arcs.get(index).is_some_and(|arc| arc.id == id))
            .or_else(|| self.arcs.iter().position(|arc| arc.id == id))
    }

    #[cfg(feature = "clp")]
    /// Exact-real scalar feasibility of a nonempty physically elementary path.
    /// Stored floating resource profiles are not used as feasibility proofs.
    /// Selected paired values match retained split joins. Other resource and
    /// physical-path proofs remain separate.
    pub fn exact_route_feasible(&self, route: &Route) -> Option<()> {
        let graph = self;
        if graph.has_coupled_resources() || !route.coupled_values.is_empty() {
            let positions = route
                .arcs
                .iter()
                .map(|id| self.arc_position(*id))
                .collect::<Option<Vec<_>>>()?;
            if !route.selected_values_match(graph, &positions).ok()? {
                return None;
            }
        }
        self.exact_route_feasibility_using(route, |kernel, arcs| {
            kernel
                .certify_with_budget(arcs, &mut 1_000_000, None)
                .map(|result| result.0)
        })?
        .then_some(())
    }

    #[cfg(feature = "clp")]
    /// Distinguish a proved invalid path from unsupported arithmetic/resource rules.
    /// `None` must not be used to discard paths from an original-domain proof.
    pub fn exact_route_feasibility(&self, route: &Route) -> Option<bool> {
        self.exact_route_feasibility_using(route, |kernel, arcs| {
            kernel.certify(arcs).map(|result| result.0)
        })
    }

    #[cfg(feature = "clp")]
    /// The same original resource check under a caller-owned proof budget.
    pub fn exact_route_feasibility_with_budget(
        &self,
        route: &Route,
        work: &mut usize,
        deadline: std::time::Instant,
    ) -> Option<bool> {
        if std::time::Instant::now() >= deadline {
            return None;
        }
        let result = self.exact_route_feasibility_using(route, |kernel, arcs| {
            kernel
                .certify_with_budget(arcs, work, Some(deadline))
                .map(|result| result.0)
        });
        if std::time::Instant::now() >= deadline {
            None
        } else {
            result
        }
    }

    #[cfg(feature = "clp")]
    fn exact_route_feasibility_using(
        &self,
        route: &Route,
        certify: impl FnOnce(&crate::graph::coupled::CoupledKernel, &[usize]) -> Option<bool>,
    ) -> Option<bool> {
        use crate::graph::ResourceKind;
        use std::cmp::Ordering::{Greater, Less};
        let g = self;
        if g.resources.len() > D {
            return None;
        }
        if g.source == g.sink
            || g.sink >= g.vertices.len()
            || route.arcs.len() > g.original_max_arcs()
            || route.nodes.len() != route.arcs.len().checked_add(1)?
            || route.nodes.first().copied() != Some(i32::try_from(g.source).ok()?)
        {
            return Some(false);
        }
        let source = g.vertices.get(g.source)?;
        let mut resources = crate::graph::ScalarValues::<_, D>::filled(
            g.resources.len(),
            crate::master::numerics::ExactSum::default(),
        );
        let valid_window = |w: crate::graph::Window| {
            w.lower.is_finite() && w.upper.is_finite() && w.lower <= w.upper
        };
        for (r, value) in resources.iter_mut().enumerate().take(g.resources.len()) {
            let bounds = *source.bounds.get(r)?;
            if !valid_window(bounds) {
                return None;
            }
            let Some(initial) = g.root_resource(r, true) else {
                return Some(false);
            };
            value.add(initial)?;
            if value.compare(bounds.lower)? == Less || value.compare(bounds.upper)? == Greater {
                return Some(false);
            }
        }
        let mut at = g.source;
        let mut visited = HashSet::new();
        let mut groups = HashSet::new();
        if g.has_explicit_elementarity_groups() {
            groups.extend(g.vertices[g.source].elements.iter().copied());
        }
        for (position, &id) in route.arcs.iter().enumerate() {
            let Some(index) = self.arc_position(id) else {
                return Some(false);
            };
            let arc = g.arcs.get(index)?;
            if arc.id != id
                || arc.tail != at
                || at == g.sink
                || arc.head == g.source
                || route.nodes[position + 1] != i32::try_from(arc.head).ok()?
            {
                return Some(false);
            }
            if g.has_explicit_elementarity_groups() {
                let mut transition_elements = HashSet::new();
                for &element in arc.elements.iter().chain(&g.vertices[arc.head].elements) {
                    if transition_elements.insert(element) && !groups.insert(element) {
                        return Some(false);
                    }
                }
            }
            let head = g.vertices.get(arc.head)?;
            if arc.head != g.sink && !visited.insert(arc.head) && g.max_route_arcs().is_none() {
                return Some(false);
            }
            for (r, value) in resources.iter_mut().enumerate().take(g.resources.len()) {
                let mut bounds = *head.bounds.get(r)?;
                if !valid_window(bounds) {
                    return None;
                }
                if let Some(window) = *arc.bounds.get(r)? {
                    if !valid_window(window) {
                        return None;
                    }
                    bounds.lower = bounds.lower.max(window.lower);
                    bounds.upper = bounds.upper.min(window.upper);
                }
                let consumption = *arc.consumption.get(r)?;
                // Signed increments are valid on a supplied finite path. Exact
                // addition and the lower-window projection below preserve the
                // earliest feasible state for either sign of the increment.
                value.add(consumption)?;
                #[cfg(feature = "multi-time-windows")]
                if !g.time_windows(arc.head, r).is_empty() {
                    if g.resources[r] != ResourceKind::Disposable {
                        return None;
                    }
                    let mut feasible = false;
                    for window in g.time_windows(arc.head, r) {
                        let lower = window.lower.max(bounds.lower);
                        let upper = window.upper.min(bounds.upper);
                        if lower <= upper && value.compare(upper)? != Greater {
                            if value.compare(lower)? == Less {
                                *value = crate::master::numerics::ExactSum::default();
                                value.add(lower)?;
                            }
                            feasible = true;
                            break;
                        }
                    }
                    if !feasible {
                        return Some(false);
                    }
                    continue;
                }
                if value.compare(bounds.lower)? == Less {
                    if g.resources[r] == ResourceKind::Equality {
                        return Some(false);
                    }
                    *value = crate::master::numerics::ExactSum::default();
                    value.add(bounds.lower)?;
                }
                if value.compare(bounds.upper)? == Greater {
                    return Some(false);
                }
            }
            at = arc.head;
        }
        if let Some(kernel) = g.coupled_kernel() {
            let arcs = route
                .arcs
                .iter()
                .map(|id| self.arc_position(*id))
                .collect::<Option<Vec<_>>>()?;
            if !certify(kernel, &arcs)? {
                return Some(false);
            }
        }
        Some(at == g.sink)
    }
}
