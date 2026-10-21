//! Arc-head accumulated-resource window restrictions.
use crate::{invalid, Graph, Result};

/// Restrict the resource at an arc's head, after consumption and waiting.
/// Both sides include `boundary()`, so their union covers every finite schedule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AccumulatedResourceBranch {
    pub arc: i32,
    pub resource: usize,
    /// Stored threshold. The effective split is `threshold - THRESHOLD_OFFSET`.
    pub threshold: f64,
    /// Upper side: resource >= boundary(); otherwise resource <= boundary().
    pub upper: bool,
}
impl AccumulatedResourceBranch {
    /// Offset of a stored threshold from its effective resource boundary.
    pub const THRESHOLD_OFFSET: f64 = 1e-6;

    pub fn boundary(&self) -> f64 {
        self.threshold - Self::THRESHOLD_OFFSET
    }

    /// Membership at the arc head, after consumption and waiting.
    pub fn satisfied_by(&self, quantity: f64) -> bool {
        quantity.is_finite()
            && self.threshold.is_finite()
            && if self.upper {
                quantity >= self.boundary()
            } else {
                quantity <= self.boundary()
            }
    }

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
            window.lower = window.lower.max(self.boundary());
        } else {
            window.upper = window.upper.min(self.boundary());
        }
        Ok((window.lower <= window.upper).then_some(window))
    }
}
