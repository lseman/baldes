//! Directed physical arc-flow branching.
use crate::{invalid, master::root::RootProblem, Result};

pub(super) fn validate<const D: usize>(problem: &RootProblem<D>, arc: usize) -> Result<()> {
    if arc >= problem.graph.arcs.len() {
        return Err(invalid("branch arc index"));
    }
    Ok(())
}

pub(super) fn coefficient(arc: usize, target: usize) -> f64 {
    (arc == target) as u8 as f64
}
