//! Resource-constrained routing graph construction.
use super::*;
impl Instance {
    pub fn graph(&self, ng_size: usize, rounding: Rounding) -> Result<Graph> {
        if !self.has_time_windows() {
            return Err(invalid(
                "the instance has no time windows (CVRPLIB input); solve it as CVRP",
            ));
        }
        self.build_graph(ng_size, rounding, true)
    }
    /// Capacity-only routing: time windows and service times impose no constraints.
    pub fn capacity_graph(&self, ng_size: usize, rounding: Rounding) -> Result<Graph> {
        self.build_graph(ng_size, rounding, false)
    }
    /// Distance in graph time units: `distance * rounding.scale()`, with
    /// `Tenth` computed as HGS does, from the scaled coordinates.
    pub fn graph_distance(&self, i: usize, j: usize, rounding: Rounding) -> f64 {
        let scale = rounding.scale();
        if rounding != Rounding::Tenth || self.distances.is_some() {
            return self.distance(i, j, rounding) * scale;
        }
        let (a, b) = (&self.customers[i], &self.customers[j]);
        let (dx, dy) = (a.x * scale - b.x * scale, a.y * scale - b.y * scale);
        (dx * dx + dy * dy).sqrt().floor()
    }
    fn build_graph(&self, ng_size: usize, rounding: Rounding, time_windows: bool) -> Result<Graph> {
        let scale = rounding.scale();
        let n = self.customers.len() - 1;
        let sink = n + 1;
        let mut graph = Graph::new(n + 2, 2)?;
        let mut add = |u: usize, v: usize| -> Result<()> {
            let i = if u == sink { 0 } else { u };
            let j = if v == sink { 0 } else { v };
            let from = &self.customers[i];
            let to = &self.customers[j];
            let d = self.graph_distance(i, j, rounding);
            let travel = if time_windows {
                from.service * scale + d
            } else {
                0.
            };
            if (time_windows && from.ready * scale + travel > to.due * scale + 1e-6)
                || (u != 0 && v != sink && from.demand + to.demand > self.capacity + 1e-6)
            {
                return Ok(());
            }
            let id = graph.add_arc(u, v, d / scale)?;
            graph.arc_mut(id)?.consumption = [travel, if v == sink { 0. } else { to.demand }];
            Ok(())
        };
        for j in 1..=n {
            add(0, j)?;
            add(j, sink)?;
            for k in 1..=n {
                if j != k {
                    add(j, k)?;
                }
            }
        }
        for v in 0..=sink {
            let i = if v == sink { 0 } else { v };
            let (ready, due) = if time_windows {
                (
                    self.customers[i].ready * scale,
                    self.customers[i].due * scale,
                )
            } else {
                (0., 1.)
            };
            graph.set_bounds(v, 0, ready, due)?;
            graph.set_bounds(v, 1, 0., self.capacity)?;
            let mut order: Vec<_> = (0..=sink).collect();
            order.sort_by(|&a, &b| {
                let ai = if a == sink { 0 } else { a };
                let bi = if b == sink { 0 } else { b };
                self.distance(i, ai, rounding)
                    .total_cmp(&self.distance(i, bi, rounding))
                    .then(a.cmp(&b))
            });
            order.truncate(ng_size.min(n + 2));
            graph.set_ng(v, &order)?;
        }
        graph.initial[0] = Some(if time_windows {
            self.customers[0].ready * scale
        } else {
            0.
        });
        Ok(graph)
    }
}
