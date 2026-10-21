//! Canonical earliest terminal usage for independent binary-grid resources.
use super::*;

pub(super) fn load<const D: usize>(oracle: &Oracle<D>, route: &Route, r: usize) -> Result<f64> {
    let g = &oracle.network.graph;
    let mut q = g.initial[r].unwrap_or(g.vertices[g.source].bounds[r].lower);
    for id in &route.arcs {
        let a = &g.arcs[*oracle
            .index
            .get(id)
            .ok_or_else(|| invalid("unknown resource-row arc"))?];
        let mut window = g.vertices[a.head].bounds[r];
        if let Some(w) = a.bounds[r] {
            window.lower = window.lower.max(w.lower);
            window.upper = window.upper.min(w.upper);
        }
        q += a.consumption[r];
        if q < window.lower {
            if g.resources[r] == crate::graph::ResourceKind::Equality {
                return Err(invalid("resource-row equality replay"));
            }
            q = window.lower;
        }
        if q > window.upper || !q.is_finite() {
            return Err(invalid("resource-row load replay outside window"));
        }
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
        capacity.fract() != 0.
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
