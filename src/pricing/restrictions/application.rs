//! Apply route restrictions to the graph and retained pricing state.
use super::super::*;

impl<const D: usize> Solver<D> {
    pub fn accumulated_resource_branches(&self) -> &[restrictions::AccumulatedResourceBranch] {
        &self.accumulated_branches
    }
    /// Install a head-resource window in the physical graph. The existing
    /// forward/backward extension, joins, replay and enumeration enforce it.
    pub fn add_accumulated_resource_branch(
        &mut self,
        constraint: restrictions::AccumulatedResourceBranch,
    ) -> Result<usize> {
        let window = constraint.window(&self.graph)?;
        let original = self.graph.arc_mut(constraint.arc)?.bounds[constraint.resource];
        let position = self
            .graph
            .arcs
            .iter()
            .position(|a| a.id == constraint.arc)
            .unwrap();
        let forbidden = self.forbidden_arcs.get(position).copied().unwrap_or(false);
        self.accumulated_original_bounds.push((
            constraint.arc,
            constraint.resource,
            original,
            forbidden,
        ));
        if let Some(window) = window {
            self.graph.arc_mut(constraint.arc)?.bounds[constraint.resource] = Some(window);
        } else {
            self.forbidden_arcs.resize(self.graph.arcs.len(), false);
            self.forbidden_arcs[position] = true;
        }
        let id = self.accumulated_branches.len();
        self.accumulated_branches.push(constraint);
        self.restore_arcs();
        Ok(id)
    }
    pub fn clear_accumulated_resource_branches(&mut self) -> Result<()> {
        for &(arc, resource, window, forbidden) in self.accumulated_original_bounds.iter().rev() {
            self.graph.arc_mut(arc)?.bounds[resource] = window;
            if let Some(position) = self.graph.arcs.iter().position(|a| a.id == arc) {
                if let Some(flag) = self.forbidden_arcs.get_mut(position) {
                    *flag = forbidden;
                }
            }
        }
        self.accumulated_original_bounds.clear();
        self.accumulated_branches.clear();
        self.restore_arcs();
        Ok(())
    }
    pub fn ryan_foster_constraints(&self) -> &[restrictions::RyanFosterConstraint] {
        &self.ryan_foster
    }
    /// Install original arcs fixed to zero at this node. Indices address
    /// `graph.arcs`, not physical IDs. Descendants inherit the restriction.
    /// Replacing it with a weaker domain discards the restricted enumeration.
    pub fn set_forbidden_arcs(&mut self, arcs: Vec<usize>) -> Result<()> {
        let mut next = vec![false; self.graph.arcs.len()];
        for a in arcs {
            *next
                .get_mut(a)
                .ok_or_else(|| invalid("forbidden arc index"))? = true;
        }
        if next == self.forbidden_arcs {
            return Ok(());
        }
        let weaker = self
            .forbidden_arcs
            .iter()
            .enumerate()
            .any(|(a, &b)| b && !next.get(a).copied().unwrap_or(false));
        if weaker {
            self.restore_arcs();
        } else {
            // New labels must be built in the restricted domain. Existing
            // reduced-cost bucket fixing remains valid for descendants.
            if let Some(engine) = &mut self.engine {
                if engine.consumption_joins {
                    self.engine = None;
                    self.prepared = None;
                } else {
                    engine.forbidden_arcs = next.clone();
                    *engine = engine.fork_for_node();
                }
            }
            if let Some(universe) = &mut self.enumeration {
                universe.paths.retain(|p| p.arcs.iter().all(|&a| !next[a]));
            }
        }
        self.forbidden_arcs = next;
        Ok(())
    }
    /// Install the current node's complete branch list. Fixed bucket arcs are
    /// valid for descendants; dual-dependent labels and enumeration filters refresh.
    pub fn set_ryan_foster(
        &mut self,
        constraints: Vec<restrictions::RyanFosterConstraint>,
    ) -> Result<()> {
        let prepared = restrictions::RyanFosterResources::prepare(&self.graph, &constraints)?;
        if constraints != self.ryan_foster {
            if !self.ryan_foster.iter().all(|c| constraints.contains(c)) {
                self.restore_arcs();
            }
            if let Some(engine) = &mut self.engine {
                engine.branching = prepared;
                engine.half_ready = false;
                engine.full_ready = false;
            }
            self.ryan_foster = constraints;
            if let Some(e) = &mut self.enumeration {
                let g = &self.graph;
                let rules = &self.ryan_foster;
                e.paths.retain(|p| {
                    rules.iter().all(|c| {
                        c.satisfied_by(
                            std::iter::once(g.source).chain(p.arcs.iter().map(|&a| g.arcs[a].head)),
                        )
                    })
                });
            }
        }
        Ok(())
    }
}
