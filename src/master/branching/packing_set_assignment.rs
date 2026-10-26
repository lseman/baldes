//! Assignment of a customer packing set to a pricing network.
use crate::{
    invalid,
    master::{routing::RoutingContext, RowSense},
    Result,
};

pub(super) fn row(assigned: bool) -> (RowSense, f64) {
    if assigned {
        (RowSense::GreaterEqual, 1.)
    } else {
        (RowSense::LessEqual, 0.)
    }
}

pub(super) fn validate<const D: usize>(
    problem: &RoutingContext<D>,
    vertex: usize,
    graph_id: i32,
) -> Result<()> {
    if !problem.customers.contains(&vertex) || graph_id < 0 {
        return Err(invalid(
            "assignment requires a customer packing set and nonnegative graph ID",
        ));
    }
    Ok(())
}

pub(super) fn coefficient<const D: usize>(
    problem: &RoutingContext<D>,
    arc: usize,
    vertex: usize,
    graph_id: i32,
) -> f64 {
    (graph_id == problem.model.networks[0].graph.graph_id.max(0)
        && problem.model.networks[0].graph.arcs[arc].head == vertex) as u8 as f64
}
