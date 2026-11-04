//! Validated graph edits. Failed edits leave the graph unchanged.
use super::*;

impl<const D: usize> Graph<D> {
    pub fn arcs(&self) -> &[Arc<D>] {
        &self.arcs
    }
    pub fn vertices(&self) -> &[Vertex<D>] {
        &self.vertices
    }
    pub fn resources(&self) -> &[ResourceKind] {
        &self.resources
    }

    /// Edit one physical arc transactionally. Validation includes IDs, resource
    /// domains and graph-wide invariants. A failed edit preserves the original.
    pub fn edit_arc(&mut self, id: i32, edit: impl FnOnce(&mut Arc<D>)) -> Result<()> {
        let position = self
            .arcs
            .iter()
            .position(|arc| arc.id == id)
            .ok_or_else(|| invalid("unknown physical arc ID"))?;
        let original = self.arcs[position].clone();
        edit(&mut self.edit().arcs[position]);
        if let Err(error) = self.validate() {
            self.edit().arcs[position] = original;
            return Err(error);
        }
        Ok(())
    }

    /// Change the original and current arc objective together.
    pub fn set_arc_cost(&mut self, id: i32, cost: f64) -> Result<()> {
        if !cost.is_finite() {
            return Err(invalid("finite arc cost required"));
        }
        self.edit_arc(id, |arc| {
            arc.cost = cost;
            arc.reduced_cost = cost;
        })
    }
}

#[cfg(test)]
#[path = "../../tests/unit/graph/mutation.rs"]
mod tests;

impl<const D: usize> Graph<D> {
    /// Refresh only pricing costs; prepared topology remains reusable. Invalid
    /// inputs leave all costs unchanged. Original arc costs are unaffected.
    pub fn set_reduced_costs(&mut self, costs: &[f64]) -> Result<()> {
        if costs.len() != self.arcs.len() || costs.iter().any(|c| !c.is_finite()) {
            return Err(invalid(
                "finite reduced costs with matching arc count required",
            ));
        }
        for (arc, &cost) in self.data.arcs.iter_mut().zip(costs) {
            arc.reduced_cost = cost;
        }
        Ok(())
    }

    pub(crate) fn clear_reduced_costs(&mut self) {
        for arc in &mut self.data.arcs {
            arc.reduced_cost = 0.;
        }
    }
}
