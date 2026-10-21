//! Canonical earliest terminal usage for independent binary-grid resources.
use super::*;

pub(super) fn load<const D: usize>(oracle: &Oracle<D>, route: &Route, r: usize) -> Result<f64> {
    let g = &oracle.network.graph;
    let mut q = g
        .root_resource(r, true)
        .ok_or_else(|| invalid("resource-row initial state outside source windows"))?;
    for id in &route.arcs {
        let a = &g.arcs[*oracle
            .index
            .get(id)
            .ok_or_else(|| invalid("unknown resource-row arc"))?];
        q = crate::labeling::extend_resource(g, a, r, q, true)
            .ok_or_else(|| invalid("resource-row load replay outside windows"))?;
    }
    Ok(q)
}

pub(in crate::master::multi_pricing) fn requires_forward<const D: usize>(
    oracle: &Oracle<D>,
) -> bool {
    let g = &oracle.network.graph;
    oracle.knapsack_rows.iter().any(|&(index, _)| {
        let r = oracle.solver.cuts.knapsack_cuts[index].graph_resources[&oracle.network.id].0;
        let capacity = g.vertices[g.sink].bounds[r].upper;
        g.resource_has_time_windows(r)
            || capacity.fract() != 0.
            || g.arcs.iter().any(|a| a.consumption[r].fract() != 0.)
            || g.resources[r] == crate::graph::ResourceKind::Equality
            || g.initial[r].unwrap_or(g.vertices[g.source].bounds[r].lower) != 0.
            || g.vertices
                .iter()
                .any(|v| v.bounds[r].lower != 0. || v.bounds[r].upper != capacity)
            || g.arcs
                .iter()
                .any(|a| a.bounds[r].is_some_and(|w| w.lower > 0. || w.upper < capacity))
    })
}
