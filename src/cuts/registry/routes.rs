use super::*;

impl CutRegistry {
    /// Physical (tail, head, arc ID) route of a customer sequence; source and
    /// sink are added. Parallel arcs resolve to the smallest ID.
    pub fn route(&self, g: &Graph, vertices: &[usize]) -> Result<Vec<(usize, usize, i32)>> {
        let mut ends = self.ends.borrow_mut();
        if ends.0 != g.arcs.len() || ends.1.is_empty() {
            ends.1.clear();
            for a in &g.arcs {
                let slot = ends
                    .1
                    .entry((a.tail, a.head))
                    .or_insert((a.tail, a.head, a.id));
                if a.id < slot.2 {
                    slot.2 = a.id;
                }
            }
            ends.0 = g.arcs.len();
        }
        if vertices.is_empty() {
            return Ok(vec![]);
        }
        let mut route = Vec::with_capacity(vertices.len() + 2);
        if vertices[0] != g.source {
            route.push(g.source);
        }
        route.extend_from_slice(vertices);
        if *route.last().unwrap() != g.sink {
            route.push(g.sink);
        }
        route
            .windows(2)
            .map(|w| {
                ends.1
                    .get(&(w[0], w[1]))
                    .copied()
                    .ok_or_else(|| invalid("cut route has no physical arc"))
            })
            .collect()
    }

    /// or_rcsp_cut_coefficient_ex for a customer sequence.
    pub fn coefficient(
        &self,
        g: &Graph,
        cut: usize,
        vertices: &[usize],
        enumerated: bool,
    ) -> Result<i32> {
        let route = self.route(g, vertices)?;
        coefficient(&self.cuts[cut], g.graph_id.max(0), &route, enumerated)
    }

    pub fn route_coefficient(
        &self,
        g: &Graph,
        cut: usize,
        route: &[(usize, usize, i32)],
        enumerated: bool,
    ) -> Result<i32> {
        coefficient(&self.cuts[cut], g.graph_id.max(0), route, enumerated)
    }
}
