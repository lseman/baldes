//! Nonempty route-count branching for one pricing network.
use crate::{invalid, master::root::RootProblem, Result};

pub(super) fn validate(graph_id: i32, rhs: f64) -> Result<()> {
    if rhs < 0. || graph_id < 0 {
        return Err(invalid("vehicle branch graph/count"));
    }
    Ok(())
}

pub(super) fn coefficient(problem: &RootProblem, arc: usize, graph_id: i32) -> f64 {
    let a = &problem.graph.arcs[arc];
    (graph_id == problem.graph.graph_id.max(0)
        && a.head == problem.graph.sink
        && a.tail != problem.graph.source) as u8 as f64
}
