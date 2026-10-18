//! Variable-cost caching and route bound correction.
use super::*;
impl Solver {
    pub fn cache_variable_costs(&mut self, costs: &[f64]) -> Result<()> {
        if costs.iter().any(|v| !v.is_finite())
            || self.maximum_variable_id.is_some_and(|id| id >= costs.len())
        {
            return Err(invalid("variable cost domain"));
        }
        let mut next = vec![];
        for arc in &self.graph.arcs {
            let mut value = arc.cost;
            for &(id, a) in &arc.variables {
                let raw = *costs
                    .get(id)
                    .ok_or_else(|| invalid("variable cost array is too short"))?;
                value += (raw * 1e8 + 0.5).floor() / 1e8 * a;
            }
            if !value.is_finite() {
                return Err(invalid("cached arc cost overflow"));
            }
            next.push(value);
        }
        for (arc, cost) in self.graph.arcs.iter_mut().zip(next) {
            arc.reduced_cost = cost;
        }
        self.variable_costs = costs.to_vec();
        Ok(())
    }
    pub fn corrected_bound(&self, mut bound: f64, route: &crate::Route) -> Result<f64> {
        if self.variable_costs.is_empty() {
            return Ok(bound);
        }
        let mut values = std::collections::BTreeMap::new();
        for &id in &route.arcs {
            let arc = self
                .graph
                .arcs
                .iter()
                .find(|a| a.id == id)
                .ok_or_else(|| invalid("unknown functor route arc"))?;
            for &(id, a) in &arc.variables {
                *values.entry(id).or_insert(0.) += a;
            }
        }
        for (id, value) in values {
            let raw = *self
                .variable_costs
                .get(id)
                .ok_or_else(|| invalid("cached variable cost domain"))?;
            bound += (raw - (raw * 1e8 + 0.5).floor() / 1e8) * value;
        }
        Ok(bound)
    }
}
