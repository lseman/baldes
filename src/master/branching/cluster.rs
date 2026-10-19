//! Cluster boundary and pair validation, coefficients and configuration.
use crate::{invalid, master::root::RootProblem, Result};

pub(super) fn boundary_coefficient(vertices: &[usize], tail: usize, head: usize) -> f64 {
    (!vertices.contains(&tail) && vertices.contains(&head)) as u8 as f64
}

pub(super) fn pair_coefficient(first: &[usize], second: &[usize], tail: usize, head: usize) -> f64 {
    ((first.contains(&tail) && second.contains(&head))
        || (second.contains(&tail) && first.contains(&head))) as u8 as f64
}

pub(super) fn validate_pair(p: &RootProblem, first: &[usize], second: &[usize]) -> Result<()> {
    validate(p, first)?;
    validate(p, second)?;
    if first.iter().any(|v| second.contains(v)) {
        return Err(invalid("cluster pair must be disjoint"));
    }
    Ok(())
}

pub(super) fn validate(p: &RootProblem, vertices: &[usize]) -> Result<()> {
    let depot = vertices.len() == 2
        && vertices.contains(&p.graph.source)
        && vertices.contains(&p.graph.sink);
    if vertices.is_empty()
        || vertices
            .iter()
            .enumerate()
            .any(|(i, v)| vertices[..i].contains(v))
        || (!depot && vertices.iter().any(|v| !p.customers.contains(v)))
    {
        return Err(invalid(
            "cluster must contain distinct customers or the split depot",
        ));
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct ClusterBranchingOptions {
    /// Explicit customer partition, or an automatically constructed MST partition.
    pub partition: Option<Vec<Vec<usize>>>,
    /// Remove MST edges above mean + theta * standard deviation.
    pub theta: f64,
}
impl Default for ClusterBranchingOptions {
    fn default() -> Self {
        Self {
            partition: None,
            theta: 1.,
        }
    }
}
