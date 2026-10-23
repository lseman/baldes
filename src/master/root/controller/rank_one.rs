//! Rank-one cut separation and fixed-route contributions.
use super::*;

impl<const D: usize> RootController<D> {
    pub(super) fn fixed_rank_one(&self, state: &State<D>, index: usize) -> Result<f64> {
        self.problem.fixed_sum(|route| {
            let path = state
                .solver
                .cuts
                .route(&state.solver.graph, &State::<D>::customers(route))?;
            Ok(state
                .solver
                .cuts
                .route_coefficient(&state.solver.graph, index, &path, true)? as f64)
        })
    }

    /// New rank-one cuts, and whether existing cut memories changed.
    pub(super) fn separate_rank_one<M: LpMaster>(
        &self,
        state: &mut State<D>,
        master: &mut M,
        rows: &mut PendingRows,
        tagged: &[(usize, usize)],
        fractional: &[FractionalColumn],
    ) -> Result<(usize, bool)> {
        let before = state.solver.cuts.cuts.len();
        let options = &state.solver.options;
        let graph = &state.solver.graph;
        state
            .solver
            .cuts
            .separate_default(graph, options, &self.problem.customers, fractional)?;
        let after = state.solver.cuts.cuts.len();
        let mut paths = vec![];
        for &(_, tag) in tagged {
            let route = &state.routes[tag];
            paths.push((
                state
                    .solver
                    .cuts
                    .route(graph, &State::<D>::customers(route))?,
                state.is_enumerated(route),
            ));
        }
        for index in before..after {
            let mut entries = vec![];
            for (&(column, _), (path, enumerated)) in tagged.iter().zip(&paths) {
                let c = state
                    .solver
                    .cuts
                    .route_coefficient(graph, index, path, *enumerated)?;
                if c != 0 {
                    entries.push((column, c as f64));
                }
            }
            let cut = &state.solver.cuts.cuts[index];
            let sense = if state.solver.cuts.is_covering(index) {
                RowSense::GreaterEqual
            } else {
                RowSense::LessEqual
            };
            let rhs = cut.rhs as f64 - self.fixed_rank_one(state, index)?;
            let row = rows.push(sense, rhs, entries);
            state.cut.rows.push(Some(row));
            state
                .cut
                .memories
                .push(cut.memory_for_graph(graph.graph_id.max(0)));
        }
        let mut changed = false;
        for index in 0..before {
            let Some(row) = state.cut.rows[index] else {
                continue;
            };
            let memory = state.solver.cuts.cuts[index].memory_for_graph(graph.graph_id.max(0));
            if memory == state.cut.memories[index] {
                continue;
            }
            changed = true;
            state.cut.memories[index] = memory;
            for (&(column, _), (path, enumerated)) in tagged.iter().zip(&paths) {
                let c = state
                    .solver
                    .cuts
                    .route_coefficient(graph, index, path, *enumerated)?;
                master.set_coefficient(row, column, c as f64)?;
            }
        }
        Ok((after - before, changed))
    }
}
