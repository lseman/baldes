//! Branching on the number of times each time window of a vertex is selected
//! (multi-time-window builds). Window copies of the incoming arcs make each
//! count an aggregate arc flow, so the usual floor/ceil children apply and
//! pricing charges their duals per arc.
use super::AggregateVariable;
use crate::{graph::WindowArcs, Result};

/// One aggregate per (vertex, window) with at least one window copy, named
/// `time_window_<vertex>_<window>`.
pub fn selection_variables(split: &WindowArcs) -> Result<Vec<AggregateVariable>> {
    split
        .selections()
        .into_iter()
        .map(|((vertex, window), arcs)| {
            let arcs: Vec<_> = arcs.into_iter().map(|id| (id, 1.)).collect();
            AggregateVariable::new(format!("time_window_{vertex}_{window}"), &arcs)
        })
        .collect()
}
