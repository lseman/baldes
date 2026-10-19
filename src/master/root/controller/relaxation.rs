//! Dynamic NG relaxation improvement.
use super::*;

impl RootController {
    /// Dynamic NG (runColGenSpRelaxationImprovement and
    /// BcRCSPFunctor::improveCurrentSpRelaxation): strengthen the NG memories
    /// from the cycles of the routes in the LP solution, then mark the master
    /// columns the stronger relaxation forbids for removal. A negative
    /// BALDESdynamicNGmode only strengthens while no rank-one cut is in the
    /// master (cuts of the current round do not count yet).
    pub(super) fn improve_relaxation(
        &self,
        state: &mut State,
        rank_one_in_master: bool,
        tagged: &[(usize, usize)],
        values: &[f64],
    ) -> Result<bool> {
        let option = |name: &str| state.solver.options.get(name);
        let mode = option("BALDESdynamicNGmode")? as i32;
        if mode == 0 || (mode < 0 && rank_one_in_master) {
            return Ok(false);
        }
        let controls = crate::graph::relaxation::Controls {
            mode: mode.abs(),
            max_vertex_size: option("BALDESmaxNGneighbourhoodSize")?.max(0.) as usize,
            max_average_size: option("BALDESmaxNGaverNeighbourhoodSize")?.max(0.) as usize,
            max_columns: option("BALDESmaxNumOfColumnsInDynamicNG")?.max(0.) as usize,
            max_cycle_size: option("BALDESmaxCycleSizeInDynamicNG")? as i32,
            mirror: false,
        };
        let paths: Vec<_> = tagged
            .iter()
            .filter(|&&(column, _)| values[column] > 1e-9)
            .map(|&(column, tag)| crate::graph::relaxation::FractionalPath {
                arcs: state.routes[tag].arcs.clone(),
                value: values[column],
            })
            .collect();
        let counts = state.solver.strengthen_ng(&paths, &controls)?;
        if std::env::var_os("BALDES_DEBUG_NG").is_some() {
            eprintln!(
                "dynamic NG: columns={} paths={} forbidden={} steps={:?}",
                values.len(),
                paths.len(),
                counts.columns_forbidden,
                state.solver.engine.as_ref().map(|e| e.grid.step)
            );
        }
        if counts.columns_forbidden == 0 {
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
}
