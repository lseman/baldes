use super::*;

/// Per-call lookup of physical arcs in one pricing graph.
pub(super) struct FractionalRouteIndex {
    by_id: BTreeMap<i32, (usize, usize)>,
    by_ends: BTreeMap<(usize, usize), i32>,
}

impl FractionalRouteIndex {
    pub(super) fn new<const D: usize>(graph: &Graph<D>) -> Self {
        let mut by_id = BTreeMap::new();
        let mut by_ends = BTreeMap::new();
        for arc in &graph.arcs {
            by_id.insert(arc.id, (arc.tail, arc.head));
            by_ends
                .entry((arc.tail, arc.head))
                .and_modify(|id: &mut i32| *id = (*id).min(arc.id))
                .or_insert(arc.id);
        }
        Self { by_id, by_ends }
    }
}

impl CutRegistry {
    /// Physical (tail, head, arc ID) route of a customer sequence; source and
    /// sink are added. Parallel arcs resolve to the smallest ID.
    pub fn route<const D: usize>(
        &self,
        g: &Graph<D>,
        vertices: &[usize],
    ) -> Result<Vec<(usize, usize, i32)>> {
        let mut ends = self.ends.borrow_mut();
        if ends.0 != g.arcs.len()
            || ends.1.is_empty()
            || ends
                .2
                .iter()
                .zip(&g.arcs)
                .any(|(&(id, tail, head), arc)| (id, tail, head) != (arc.id, arc.tail, arc.head))
        {
            ends.1.clear();
            ends.2.clear();
            for a in &g.arcs {
                let slot = ends
                    .1
                    .entry((a.tail, a.head))
                    .or_insert((a.tail, a.head, a.id));
                if a.id < slot.2 {
                    slot.2 = a.id;
                }
            }
            ends.2
                .extend(g.arcs.iter().map(|arc| (arc.id, arc.tail, arc.head)));
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

    /// Resolve a fractional column while retaining the actual parallel arc IDs.
    pub(super) fn fractional_route<const D: usize>(
        &self,
        graph: &Graph<D>,
        column: &FractionalColumn,
        index: &FractionalRouteIndex,
    ) -> Result<Vec<(usize, usize, i32)>> {
        let Some(ids) = &column.arc_ids else {
            if column.vertices.is_empty() {
                return Ok(vec![]);
            }
            let mut vertices = Vec::with_capacity(column.vertices.len() + 2);
            if column.vertices[0] != graph.source {
                vertices.push(graph.source);
            }
            vertices.extend_from_slice(&column.vertices);
            if vertices.last() != Some(&graph.sink) {
                vertices.push(graph.sink);
            }
            return vertices
                .windows(2)
                .map(|pair| {
                    index
                        .by_ends
                        .get(&(pair[0], pair[1]))
                        .copied()
                        .map(|id| (pair[0], pair[1], id))
                        .ok_or_else(|| invalid("cut route has no physical arc"))
                })
                .collect();
        };
        let mut vertices = Vec::with_capacity(column.vertices.len() + 2);
        vertices.push(graph.source);
        vertices.extend_from_slice(&column.vertices);
        vertices.push(graph.sink);
        if ids.len() + 1 != vertices.len() {
            return Err(invalid("fractional column arc count"));
        }
        vertices
            .windows(2)
            .zip(ids)
            .map(|(ends, &id)| {
                let &(tail, head) = index
                    .by_id
                    .get(&id)
                    .ok_or_else(|| invalid("unknown fractional column arc"))?;
                if tail != ends[0] || head != ends[1] {
                    return Err(invalid("fractional column arc endpoints"));
                }
                Ok((tail, head, id))
            })
            .collect()
    }

    /// or_rcsp_cut_coefficient_ex for a customer sequence.
    pub fn coefficient<const D: usize>(
        &self,
        g: &Graph<D>,
        cut: usize,
        vertices: &[usize],
        enumerated: bool,
    ) -> Result<i32> {
        let route = self.route(g, vertices)?;
        coefficient(&self.cuts[cut], g.graph_id.max(0), &route, enumerated)
    }

    pub fn route_coefficient<const D: usize>(
        &self,
        g: &Graph<D>,
        cut: usize,
        route: &[(usize, usize, i32)],
        enumerated: bool,
    ) -> Result<i32> {
        coefficient(&self.cuts[cut], g.graph_id.max(0), route, enumerated)
    }
}
