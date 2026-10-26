//! Strong k-path cut separation.
use super::*;

impl<const D: usize> RootController<D> {
    pub(super) fn separate_kpath<M: LpMaster>(
        &self,
        state: &mut State<D>,
        master: &mut M,
        rows: &mut PendingRows,
        tagged: &[(usize, usize)],
        fractional: &[FractionalColumn],
    ) -> Result<(usize, bool)> {
        let graph_id = state.solver.graph.graph_id.max(0);
        let mut solutions = vec![];
        let mut values = vec![];
        if matches!(
            state.cut.kpath_separator,
            Some(RootKPathSeparator::General { .. })
        ) {
            for &(column, tag) in tagged {
                if master.values()[column] <= 1e-9 {
                    continue;
                }
                let route = &state.routes[tag];
                solutions.push(crate::cuts::separation::Solution {
                    graph_id,
                    enumerated: state.is_enumerated(route),
                    arc_ids: route.arcs.clone(),
                });
                values.push(master.values()[column]);
            }
        }
        if matches!(
            state.cut.kpath_separator,
            Some(RootKPathSeparator::General { .. })
        ) {
            for (route, quantity) in &self.problem.fixed_routes {
                solutions.push(crate::cuts::separation::Solution {
                    graph_id,
                    enumerated: true,
                    arc_ids: route.arcs.clone(),
                });
                values.push(f64::from(*quantity));
            }
        }
        let Some(separator) = state.cut.kpath_separator.as_mut() else {
            return Ok((0, false));
        };
        let before = state.solver.cuts.cuts.len();
        match separator {
            RootKPathSeparator::Standalone(separator) => {
                let vertices = state.solver.graph.vertices.len();
                for cut in separator.separate_fractional(fractional)? {
                    let index = state
                        .solver
                        .cuts
                        .register_strong_k_path_cut(vertices, graph_id, &cut, 0.)?;
                    state.cut.kpath_indices.push(index);
                }
            }
            RootKPathSeparator::General { separator, graphs } => {
                for cut in separator.separate(&solutions, &values)? {
                    let index = state.solver.cuts.register_general_k_path_cut(
                        graphs,
                        separator.preparation.use_packing_sets,
                        &cut,
                        0.,
                    )?;
                    state.cut.kpath_indices.push(index);
                }
            }
        }
        let after = state.solver.cuts.cuts.len();
        let graph = &state.solver.graph;
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
            let rhs = cut.rhs as f64 - self.fixed_rank_one(state, index)?;
            let row = rows.push(RowSense::GreaterEqual, rhs, entries);
            state.cut.rows.push(Some(row));
            state.cut.memories.push(cut.memory_for_graph(graph_id));
        }
        let mut changed = false;
        for index in 0..before {
            let Some(row) = state.cut.rows[index] else {
                continue;
            };
            let memory = state.solver.cuts.cuts[index].memory_for_graph(graph_id);
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
