//! Enumeration and regeneration of admissible cached columns.
use super::*;

impl RootController {
    pub(super) fn enumerate_columns(
        &mut self,
        duals: &[f64],
        bound: f64,
    ) -> Result<Option<Vec<Column>>> {
        let mut state = self.state.borrow_mut();
        if !state.enumerate || !bound.is_finite() {
            return Ok(None);
        }
        let upper = if state.incumbent.is_finite() {
            state.incumbent
        } else {
            1e12
        };
        let gap = upper - bound;
        let ratio = gap / state.previous_fixing_gap;
        let fixing_ratio = 0.9f32 as f64;
        if !(ratio < fixing_ratio - (fixing_ratio.abs().max(ratio.abs()) * 1e-10 + 1e-6)) {
            return Ok(None);
        }
        state.previous_fixing_gap = gap;
        let relative = gap / upper.abs();
        let mode = if relative - (1f64.max(relative.abs()) * 1e-10 + 1e-6) > 1. {
            -1
        } else {
            0
        };
        let offset = state.reprice(&self.problem, duals)?;
        let cutoff = gap.max(0.) - offset;
        let fixing_started = std::time::Instant::now();
        let remaining = state.remaining_time()?;
        let saved_time = state.solver.controls.time_limit;
        let saved_fixing_time = state
            .solver
            .options
            .get("BALDESredCostFixingTimeThreshold")?;
        state.solver.controls.time_limit = saved_time.min(remaining);
        state.solver.options.set(
            "BALDESredCostFixingTimeThreshold",
            saved_fixing_time.min(remaining.as_secs_f64()),
        )?;
        let fixed = state.solver.run_fixing_enumeration(mode, 0., cutoff);
        state.solver.controls.time_limit = saved_time;
        state
            .solver
            .options
            .set("BALDESredCostFixingTimeThreshold", saved_fixing_time)?;
        let fixed = fixed?;
        state.pricing_seconds += fixing_started.elapsed().as_secs_f64();
        state.fixing_calls += 1;
        if fixed.state != 2 && fixed.state != 3 {
            return Ok(None);
        }
        let routes = state
            .solver
            .enumeration
            .as_ref()
            .unwrap()
            .paths
            .routes(&state.solver.graph)?;
        state.enumerated_routes = routes.len();
        state.cut.enumerated = routes
            .iter()
            .map(|r| {
                let mut arcs = r.arcs.clone();
                arcs.sort_unstable();
                arcs
            })
            .collect();
        let mut columns = vec![];
        for route in routes {
            let tag = register(&mut state, &route)?;
            columns.push(state.column(&self.problem, &route, duals, tag)?);
        }
        Ok(Some(columns))
    }

    pub(super) fn regenerate_pool(
        &mut self,
        pool: &[Column],
        duals: &[f64],
        bound: f64,
        _universe: &[Column],
    ) -> Result<Vec<Column>> {
        let mut state = self.state.borrow_mut();
        let graph = &state.solver.graph;
        let ng = crate::graph::ng::PreparedNg::new(graph)?;
        let upper = if state.incumbent.is_finite() {
            state.incumbent
        } else {
            1e12
        };
        let gap = upper - bound;
        let mut retained = vec![];
        for column in pool {
            let Some(route) = usize::try_from(column.tag)
                .ok()
                .and_then(|tag| state.routes.get(tag))
            else {
                continue;
            };
            if state.solver.ryan_foster.iter().any(|rule| {
                !rule.satisfied_by(route.nodes.iter().map(|&v| v as usize))
                    || !rule.elementary_on_pair(route.nodes.iter().map(|&v| v as usize))
            }) {
                continue;
            }
            let mut memory = ng.vertex_bits[graph.source];
            let mut vertex = graph.source;
            let mut proper = true;
            for &id in &route.arcs {
                let Some(physical) = graph.arcs.iter().position(|a| a.id == id) else {
                    proper = false;
                    break;
                };
                let arc = &graph.arcs[physical];
                if state
                    .solver
                    .forbidden_arcs
                    .get(physical)
                    .copied()
                    .unwrap_or(false)
                {
                    proper = false;
                    break;
                }
                let next = ng.forward[physical].extend(memory, ng.vertex_bits[arc.head]);
                if arc.tail != vertex || next.is_none() {
                    proper = false;
                    break;
                }
                memory = next.unwrap();
                vertex = arc.head;
            }
            if proper && vertex == graph.sink {
                let refreshed = state.column(&self.problem, route, duals, column.tag)?;
                let rc = refreshed.reduced_cost;
                if rc < gap - (gap.abs().max(rc.abs()) * 1e-10 + 1e-6) {
                    retained.push((column.tag, route.clone()));
                }
            }
        }
        // The regenerated columns use the enumeration coefficient contract,
        // including admissible cached paths other than the representatives.
        for (_, route) in &retained {
            let mut arcs = route.arcs.clone();
            arcs.sort_unstable();
            state.cut.enumerated.insert(arcs);
        }
        retained
            .iter()
            .map(|(tag, route)| state.column(&self.problem, route, duals, *tag))
            .collect()
    }
}
