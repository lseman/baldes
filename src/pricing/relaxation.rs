//! Dynamic NG relaxation updates and route validation.
use super::*;
impl Solver {
    pub fn strengthen_ng(
        &mut self,
        paths: &[crate::graph::relaxation::FractionalPath],
        controls: &crate::graph::relaxation::Controls,
    ) -> Result<crate::graph::relaxation::Counts> {
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
            // the solver grows the live elementarity resource: the bucket graph,
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
    /// the current NG relaxation (the solver satisfiesCurrentRelaxation).
    pub fn route_satisfies_relaxation(&self, vertices: &[i32], arcs: &[i32]) -> Result<bool> {
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
        Ok(true)
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
