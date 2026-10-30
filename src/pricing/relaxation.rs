//! Dynamic NG relaxation updates and route validation.
use super::*;
impl<const D: usize> Solver<D> {
    pub fn strengthen_ng(
        &mut self,
        paths: &[crate::graph::relaxation::FractionalPath],
        controls: &crate::graph::relaxation::Controls,
    ) -> Result<crate::graph::relaxation::Counts> {
        // Updating NG tables may mark the graph as prepared below. Discard a
        // grid built before unrelated resource/topology edits first. In a
        // bounded-walk model, NG may only add exclusions when hard vertex
        // groups already make every feasible route physically simple.
        if self.graph.max_route_arcs().is_some() && !self.graph.bounded_walks_are_vertex_simple() {
            return Err(invalid(
                "NG strengthening is not a bounded-walk model restriction",
            ));
        }
        self.invalidate_changed_graph();
        let baseline = (
            self.graph
                .vertices
                .iter()
                .map(|v| v.memory.clone())
                .collect(),
            self.graph.arcs.iter().map(|a| a.memory.clone()).collect(),
        );
        let counts = crate::graph::relaxation::strengthen(&mut self.graph, paths, controls)?;
        if counts.columns_forbidden > 0 {
            if self.ng_baseline.is_none() {
                self.ng_baseline = Some(baseline);
            }
            // Grow the live elementarity resource: the bucket graph,
            // fixed arcs, half-way point and adaptive depths stay. Only the NG
            // tables, the labels and enumerated paths are invalidated.
            match &mut self.engine {
                Some(engine) => {
                    engine.ng = crate::graph::ng::PreparedNg::new(&self.graph)?;
                    engine.half_ready = false;
                    engine.full_ready = false;
                    self.prepared = Some(self.topology());
                }
                None => self.prepared = None,
            }
            self.enumeration = None;
            self.enumeration_snapshot = None;
        }
        Ok(counts)
    }
    /// Whether a route (vertices and arc ids in path order) is feasible for
    /// the current NG relaxation.
    pub fn route_satisfies_relaxation(&self, vertices: &[i32], arcs: &[i32]) -> Result<bool> {
        if self
            .graph
            .max_route_arcs()
            .is_some_and(|limit| arcs.len() > limit)
        {
            return Ok(false);
        }
        if vertices.len() != arcs.len() + 1 {
            return Err(invalid("route node/arc dimensions"));
        }
        if self.ryan_foster.iter().any(|c| {
            !c.satisfied_by(vertices.iter().map(|&v| v as usize))
                || !c.elementary_on_pair(vertices.iter().map(|&v| v as usize))
        }) {
            return Ok(false);
        }

        let ng = crate::graph::ng::PreparedNg::new(&self.graph)?;
        let Some(&first) = vertices.first() else {
            return Ok(false);
        };
        let mut bits = *usize::try_from(first)
            .ok()
            .and_then(|v| ng.vertex_bits.get(v))
            .ok_or_else(|| invalid("route vertex out of range"))?;
        for (i, &id) in arcs.iter().enumerate() {
            let a = self
                .graph
                .arcs
                .iter()
                .position(|a| a.id == id)
                .ok_or_else(|| invalid("route arc out of range"))?;
            let arc = &self.graph.arcs[a];
            if vertices[i] != arc.tail as i32 || vertices[i + 1] != arc.head as i32 {
                return Ok(false);
            }
            let Some(next) = ng.forward[a].extend(bits, ng.vertex_bits[arc.head]) else {
                return Ok(false);
            };
            bits = next;
        }
        if self.graph.selection_relaxed() {
            let positions = arcs
                .iter()
                .map(|&id| self.graph.arcs.iter().position(|a| a.id == id))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| invalid("route arc out of range"))?;
            let complete = vertices.first() == Some(&(self.graph.source as i32))
                && vertices.last() == Some(&(self.graph.sink as i32));
            return Ok(if complete {
                self.graph.coupled_route_cost(&positions)?
            } else {
                self.graph.coupled_path_cost(&positions)?
            }
            .is_some());
        }
        Ok(true)
    }
    /// Relaxed selection (augmentNGneighbourhoods' selection step): narrow
    /// the selection memory at the vertices of improper paths' violating
    /// windows. Returns the number of paths the stronger relaxation now
    /// excludes.
    ///
    /// `BALDESselectionTighteningSchedule` 1 follows the reference: a negative
    /// BALDESmaxNGneighbourhoodSize disables tightening, and
    /// BALDESmaxNGaverNeighbourhoodSize caps the window length, as both share
    /// the reference's relaxation parameters. The reference first sorts paths
    /// by decreasing value; windows come from the fixed masks and memory only
    /// grows, so that order cannot change the result and is not reproduced.
    /// Schedule 0 remembers every minimal violating window of every path.
    pub fn strengthen_selection(
        &mut self,
        paths: &[crate::graph::relaxation::FractionalPath],
    ) -> Result<usize> {
        if !self.graph.selection_relaxed() {
            return Ok(0);
        }
        let first_windows = self.options.get("BALDESselectionTighteningSchedule")? != 0.;
        if first_windows && self.options.get("BALDESmaxNGneighbourhoodSize")? < 0. {
            return Ok(0);
        }
        let max_length = self.options.get("BALDESmaxNGaverNeighbourhoodSize")? as i64;
        self.invalidate_changed_graph();
        let mut forbidden = 0;
        for path in paths {
            let arcs = path
                .arcs
                .iter()
                .map(|&id| {
                    self.graph
                        .arcs
                        .iter()
                        .position(|a| a.id == id)
                        .ok_or_else(|| invalid("route arc out of range"))
                })
                .collect::<Result<Vec<_>>>()?;
            forbidden += usize::from(if first_windows {
                self.graph
                    .strengthen_selection_first_windows(&arcs, max_length)
            } else {
                self.graph.strengthen_selection(&arcs)
            });
        }
        if forbidden > 0 {
            // As for NG: labels and enumerated paths no longer hold; the
            // bucket graph, fixed arcs and half-way point stay.
            match &mut self.engine {
                Some(engine) => {
                    engine.half_ready = false;
                    engine.full_ready = false;
                    self.prepared = Some(self.topology());
                }
                None => self.prepared = None,
            }
            self.enumeration = None;
            self.enumeration_snapshot = None;
        }
        Ok(forbidden)
    }
    pub fn reset_ng(&mut self) -> bool {
        let Some((vertices, arcs)) = self.ng_baseline.take() else {
            return false;
        };
        for (v, memory) in self.graph.vertices.iter_mut().zip(vertices) {
            v.memory = memory;
        }
        for (a, memory) in self.graph.arcs.iter_mut().zip(arcs) {
            a.memory = memory;
        }
        self.restore_arcs();
        true
    }
}
