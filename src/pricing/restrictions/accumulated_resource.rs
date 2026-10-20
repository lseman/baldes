//! Arc-head accumulated-resource window restrictions.
use crate::{invalid, Graph, Result};

/// Restrict the resource at an arc's head, after consumption and waiting.
/// Both sides include the threshold, so their union covers every schedule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AccumulatedResourceBranch {
    pub arc: i32,
    pub resource: usize,
    pub threshold: f64,
    /// Upper side: resource >= threshold; otherwise resource <= threshold.
    pub upper: bool,
}
impl AccumulatedResourceBranch {
    pub(crate) fn window<const D: usize>(
        &self,
        graph: &Graph<D>,
    ) -> Result<Option<crate::graph::Window>> {
        if self.resource >= graph.resources.len() || !self.threshold.is_finite() {
            return Err(invalid("accumulated-resource branch coordinate/threshold"));
        }
        let arc = graph
            .arcs
            .iter()
            .find(|a| a.id == self.arc)
            .ok_or_else(|| invalid("accumulated-resource branch arc ID"))?;
        let mut window = graph.vertices[arc.head].bounds[self.resource];
        if let Some(w) = arc.bounds[self.resource] {
            window.lower = window.lower.max(w.lower);
            window.upper = window.upper.min(w.upper);
        }
        if self.upper {
            window.lower = window.lower.max(self.threshold);
        } else {
            window.upper = window.upper.min(self.threshold);
        }
        Ok((window.lower <= window.upper).then_some(window))
    }
}
