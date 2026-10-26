use super::*;
use crate::Graph;

pub(crate) struct Additive<'a, const D: usize> {
    pub graph: &'a Graph<D>,
    pub costs: &'a [f64],
    pub lower: f64,
    pub upper: f64,
    pub supported: bool,
}
impl<const D: usize> Additive<'_, D> {
    pub fn nonnegative(&self, offset: f64) -> bool {
        self.lower == 0.
            && offset.is_finite()
            && offset >= 0.
            && self.supported
            && !self.graph.has_coupled_resources()
            && self.costs.iter().all(|v| v.is_finite() && *v >= 0.)
    }
    pub fn required(&self, offset: f64) -> Option<Estimate> {
        if self.lower <= 0.
            || !offset.is_finite()
            || !self.supported
            || self.graph.has_coupled_resources()
            || self.costs.iter().any(|v| !v.is_finite() || *v < 0.)
        {
            return None;
        }
        Estimate::contribution(offset, self.lower, self.upper)
    }
    pub fn near_zero(&self, offset: f64, tolerance: f64) -> Option<Estimate> {
        if self.lower != 0.
            || !offset.is_finite()
            || offset < 0.
            || !self.supported
            || self.graph.has_coupled_resources()
            || self.costs.iter().any(|v| !v.is_finite())
        {
            return None;
        }
        let minimum = self.costs.iter().copied().fold(0_f64, f64::min);
        if !(-1e-6..0.).contains(&minimum) {
            return None;
        }
        let hops = self.graph.max_route_arcs().or_else(|| {
            self.graph
                .is_acyclic()
                .then_some(self.graph.vertices.len().saturating_sub(1))
        })?;
        if (hops as u128) > 9_007_199_254_740_992
            || !self.upper.is_finite()
            || self.upper < 0.
            || self.upper.fract() != 0.
            || self.upper > 9_007_199_254_740_992.
        {
            return None;
        }
        let mut magnitude = crate::master::numerics::ExactSum::default();
        magnitude.product(-minimum, hops as f64)?;
        magnitude.add(-offset)?;
        let lower = (-magnitude.upper()?).min(0.);
        if lower < -tolerance {
            return None;
        }
        Estimate::contribution(lower, self.lower, self.upper)
    }
}
