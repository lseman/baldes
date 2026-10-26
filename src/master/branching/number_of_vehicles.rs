//! Nonempty route-count branching for one pricing network.
use crate::{invalid, master::routing::RoutingContext, Result};

pub(super) fn validate(graph_id: i32, rhs: f64) -> Result<()> {
    if rhs < 0. || graph_id < 0 {
        return Err(invalid("vehicle branch graph/count"));
    }
    Ok(())
}

pub(super) fn coefficient<const D: usize>(
    problem: &RoutingContext<D>,
    arc: usize,
    graph_id: i32,
) -> f64 {
    let a = &problem.model.networks[0].graph.arcs[arc];
    (graph_id == problem.model.networks[0].graph.graph_id.max(0)
        && a.head == problem.model.networks[0].graph.sink
        && a.tail != problem.model.networks[0].graph.source) as u8 as f64
}
