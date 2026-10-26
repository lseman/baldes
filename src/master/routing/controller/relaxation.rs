//! Dynamic NG relaxation improvement.
use super::*;

impl<const D: usize> RootController<D> {
    /// Dynamic NG (runColGenSpRelaxationImprovement and
    /// BcRCSPFunctor::improveCurrentSpRelaxation): strengthen the NG memories
    /// from the cycles of the routes in the LP solution, then mark the master
    /// columns the stronger relaxation forbids for removal. A negative
    /// BALDESdynamicNGmode only strengthens while no rank-one cut is in the
    /// master (cuts of the current round do not count yet).
    pub(super) fn improve_relaxation(
        &self,
        state: &mut State<D>,
        rank_one_in_master: bool,
        tagged: &[(usize, usize)],
        values: &[f64],
    ) -> Result<bool> {
        let paths: Vec<_> = tagged
            .iter()
            .filter(|&&(column, _)| values[column] > 1e-9)
            .map(|&(column, tag)| crate::graph::relaxation::FractionalPath {
                arcs: state.routes[tag].arcs.clone(),
                value: values[column],
            })
            .collect();
        // Relaxed selection is tightened whenever the solution uses an
        // improper route, independently of the NG limits and mode.
        let selection = state.solver.strengthen_selection(&paths)?;
        let ng = self.improve_ng(state, rank_one_in_master, &paths, values.len())?;
        if selection == 0 && ng == 0 {
            return Ok(false);
        }
        let mut removed = vec![];
        for &(column, tag) in tagged {
            let route = &state.routes[tag];
            if !state
                .solver
                .route_satisfies_relaxation(&route.nodes, &route.arcs)?
            {
                removed.push(column);
            }
        }
        state.cut.removed_columns = removed;
        Ok(true)
    }

    /// Whether the master solution uses a route that breaks a relaxed
    /// selection rule. Such a solution must not end the node.
    pub(super) fn uses_improper_selection(
        &self,
        state: &State<D>,
        tagged: &[(usize, usize)],
        values: &[f64],
    ) -> bool {
        let graph = &self.problem.model.networks[0].graph;
        let Some(kernel) = graph.coupled_kernel().filter(|k| k.selection_relaxed()) else {
            return false;
        };
        tagged.iter().any(|&(column, tag)| {
            values[column] > 1e-9
                && state.routes[tag]
                    .arcs
                    .iter()
                    .map(|&id| self.problem.arc_position(id))
                    .collect::<Option<Vec<_>>>()
                    .is_some_and(|arcs| !kernel.selection_proper(&arcs))
        })
    }

    /// The NG part: returns the number of paths the stronger memories forbid.
    fn improve_ng(
        &self,
        state: &mut State<D>,
        rank_one_in_master: bool,
        paths: &[crate::graph::relaxation::FractionalPath],
        columns: usize,
    ) -> Result<usize> {
        let option = |name: &str| state.solver.options.get(name);
        let mode = option("BALDESdynamicNGmode")? as i32;
        if mode == 0 || (mode < 0 && rank_one_in_master) {
            return Ok(0);
        }
        let controls = crate::graph::relaxation::Controls {
            mode: mode.abs(),
            max_vertex_size: option("BALDESmaxNGneighbourhoodSize")?.max(0.) as usize,
            max_average_size: option("BALDESmaxNGaverNeighbourhoodSize")?.max(0.) as usize,
            max_columns: option("BALDESmaxNumOfColumnsInDynamicNG")?.max(0.) as usize,
            max_cycle_size: option("BALDESmaxCycleSizeInDynamicNG")? as i32,
            mirror: false,
        };
        let counts = state.solver.strengthen_ng(paths, &controls)?;
        if std::env::var_os("BALDES_DEBUG_NG").is_some() {
            eprintln!(
                "dynamic NG: columns={} paths={} forbidden={} steps={:?}",
                columns,
                paths.len(),
                counts.columns_forbidden,
                state.solver.pricing_engine().map(|e| e.grid.step)
            );
        }
        Ok(counts.columns_forbidden)
    }
}
