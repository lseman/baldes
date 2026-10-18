//! Column-generation callbacks and cut separation scheduling.
use super::*;
pub(super) struct RootController {
    pub(super) problem: Rc<RootProblem>,
    pub(super) state: Rc<RefCell<State>>,
}
impl<M: LpMaster> Controller<M> for RootController {
    fn protected_rows(&self) -> Vec<usize> {
        self.problem.row_map.borrow().clone()
    }
    fn master_solved(&mut self, master: &M, tags: &[i32], artificials: &[usize]) -> Result<()> {
        if artificials.iter().any(|&i| master.values()[i] > 1e-6) {
            return Ok(());
        }
        if master
            .values()
            .iter()
            .all(|x| (*x - x.round()).abs() <= 1e-6)
            && tags
                .iter()
                .zip(master.values())
                .all(|(&tag, &x)| tag >= 0 || x <= 1e-6)
        {
            let mut state = self.state.borrow_mut();
            for (&tag, &value) in tags.iter().zip(master.values()) {
                if tag < 0 || value <= 1e-6 {
                    continue;
                }
                let route = &state.routes[tag as usize];
                if self.problem.branch_constraints.iter().any(|(_, c)| {
                    matches!(c, BranchConstraint::RyanFoster(_))
                        && c.coefficient(&self.problem, route) != 0.
                }) {
                    return Ok(());
                }
                let customers = &route.nodes[1..route.nodes.len() - 1];
                if customers
                    .iter()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    != customers.len()
                {
                    return Ok(());
                }
            }
            state.incumbent = state.incumbent.min(master.objective());
        }
        Ok(())
    }
    fn incumbent(&self) -> f64 {
        self.state.borrow().incumbent
    }
    fn enumerated_universe(&mut self, duals: &[f64], bound: f64) -> Result<Option<Vec<Column>>> {
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
        let saved_fixing_time = state.solver.options.get("RCSPredCostFixingTimeThreshold")?;
        state.solver.controls.time_limit = saved_time.min(remaining);
        state.solver.options.set(
            "RCSPredCostFixingTimeThreshold",
            saved_fixing_time.min(remaining.as_secs_f64()),
        )?;
        let fixed = state.solver.run_fixing_enumeration(mode, 0., cutoff);
        state.solver.controls.time_limit = saved_time;
        state
            .solver
            .options
            .set("RCSPredCostFixingTimeThreshold", saved_fixing_time)?;
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
    fn remap_rows(&mut self, rows: &[Option<usize>]) -> Result<()> {
        for row in self.problem.row_map.borrow_mut().iter_mut() {
            *row = rows[*row].ok_or_else(|| invalid("attempt to remove a base or branch row"))?;
        }
        let mut state = self.state.borrow_mut();
        for index in 0..state.cut.knapsack_rows.len() {
            if let Some(row) = state.cut.knapsack_rows[index] {
                state.cut.knapsack_rows[index] = rows[row];
                if rows[row].is_none() {
                    state.solver.cuts.knapsack_cuts[index].dual = 0.;
                }
            }
        }
        for row in &mut state.cut.capacity_rows {
            *row = row.and_then(|r| rows[r]);
        }
        for entries in &mut state.cut.capacity_arc_entries {
            *entries = entries
                .iter()
                .filter_map(|&(r, a)| rows[r].map(|row| (row, a)))
                .collect();
        }
        for index in 0..state.cut.rows.len() {
            let Some(row) = state.cut.rows[index] else {
                continue;
            };
            state.cut.rows[index] = rows[row];
            if rows[row].is_none() {
                state.solver.cuts.retire(index);
            }
        }
        Ok(())
    }
    fn separates(&self) -> bool {
        !self.state.borrow().cut.separators().is_empty()
    }
    fn separate(&mut self, master: &mut M, tags: &[i32]) -> Result<usize> {
        let mut state = self.state.borrow_mut();
        let state = &mut *state;
        let value = master.objective();
        // Every converged column generation, round 0 included, reports to the
        let stop = state.solver.column_generation_terminated(0, 0)?;
        if state.cut.rounds == 0 {
            state.cut.lp_before_cuts = value;
        } else {
            // Alg4EvalByColAndCutGen::eval: the gain is relative to the gap to
            // the incumbent (an integral master solution) when that gap is at
            // most a tenth of the bound, else to a tenth of the bound.
            let previous = state.cut.separation_lp;
            let upper = if state.incumbent.is_finite() {
                state.incumbent
            } else {
                1e12
            };
            let gap = upper - previous;
            let tenth = (0.1 * previous.abs()).max(1e-9);
            let denominator = if gap - (gap.abs().max(tenth) * 1e-10 + 1e-6) > tenth {
                tenth
            } else {
                gap
            };
            let gain = (value - previous) / denominator;
            let rank_active = state.cut.rank_one_active();
            let min_rounds = state.cut.options.min_rounds_before_stop_by_sp;
            if rank_active && stop && state.cut.rounds >= min_rounds {
                return Ok(0);
            }
            // Store the threshold as a float.
            let threshold = state.cut.options.tailing_off_threshold as f32 as f64;
            if gain < threshold - (threshold.abs().max(gain.abs()) * 1e-10 + 1e-6) {
                if state.cut.options.enabled
                    && state.cut.level <= RANK_ONE_PRIORITY
                    && !state.solver.cuts.separation_phase(&state.solver.options)?.1
                {
                    let options = state.solver.options.clone();
                    state.solver.cuts.increase_phase(&options)?;
                } else {
                    state.cut.tailing += 1;
                }
                // Tailing off at a level moves to the next lower priority, and
                // stops cut generation at the lowest one.
                if state.cut.tailing >= state.cut.options.tailing_off_counter
                    && state.cut.rounds >= min_rounds
                {
                    match state.cut.next_level() {
                        Some(level) => {
                            state.cut.level = level;
                            state.cut.tailing = 0;
                        }
                        None => return Ok(0),
                    }
                }
            }
        }
        state.cut.separation_lp = value;
        // Each cut round first checks gapSmallerThanTol: separation stops once
        // the bound reaches the incumbent (absolute 1e-6, relative 1e-9).
        if state.incumbent.is_finite() {
            let gap = state.incumbent - value;
            if gap <= value.abs().max(state.incumbent.abs()) * 1e-10 + 1e-6
                || gap / state.incumbent.abs() <= 1e-9
            {
                return Ok(0);
            }
        }
        let values = master.values().to_vec();
        let tagged: Vec<(usize, usize)> = tags
            .iter()
            .enumerate()
            .filter(|(_, &t)| t >= 0)
            .map(|(i, &t)| (i, t as usize))
            .collect();
        let mut fractional = vec![];
        for &(index, tag) in &tagged {
            if values[index] > 1e-9 {
                let route = &state.routes[tag];
                fractional.push(FractionalColumn {
                    vertices: State::customers(route),
                    value: values[index],
                    graph_id: -1,
                    enumerated: state.is_enumerated(route),
                });
            }
        }
        // Alg4EvalOfNode::addCutToMaster: separators by decreasing priority.
        // Those at or above the level run; a lower one runs, and becomes the
        // level, only while nothing has been added this round.
        let mut level = state.cut.level;
        let mut rows = PendingRows {
            base: master.rhs().len(),
            rows: vec![],
        };
        // Cuts separated this round reach the formulation after the loop.
        let rank_one_in_master = state.cut.rows.iter().any(Option::is_some);
        let mut added = 0;
        let mut success = false;
        for (priority, separator) in state.cut.separators() {
            if priority < level {
                if success {
                    break;
                }
                level = priority;
            }
            let (count, changed) = match separator {
                Separator::Capacity => (
                    self.separate_capacity(state, &mut rows, &tagged, &fractional)?,
                    false,
                ),
                Separator::KPath => {
                    self.separate_kpath(state, master, &mut rows, &tagged, &fractional)?
                }
                Separator::Knapsack => (
                    self.separate_knapsack(state, &mut rows, &tagged, &values)?,
                    false,
                ),
                Separator::RankOne => {
                    self.separate_rank_one(state, master, &mut rows, &tagged, &fractional)?
                }
                Separator::Relaxation => (
                    0,
                    self.improve_relaxation(state, rank_one_in_master, &tagged, &values)?,
                ),
            };
            added += count;
            success |= count > 0 || changed;
        }
        state.cut.level = level;
        // Add the round's cuts to Clp in one call.
        let batch: Vec<_> = rows
            .rows
            .iter()
            .map(|(sense, rhs, entries)| (*sense, *rhs, entries.as_slice()))
            .collect();
        master.add_rows(&batch)?;
        if success {
            state.cut.rounds += 1;
        }
        Ok(if added > 0 { added } else { success as usize })
    }
    fn take_removed_columns(&mut self) -> Vec<usize> {
        std::mem::take(&mut self.state.borrow_mut().cut.removed_columns)
    }
}
impl RootController {
    fn separate_capacity(
        &self,
        state: &mut State,
        rows: &mut PendingRows,
        tagged: &[(usize, usize)],
        fractional: &[FractionalColumn],
    ) -> Result<usize> {
        let Some(separator) = state.cut.capacity_separator.as_mut() else {
            return Ok(0);
        };
        let columns: Vec<_> = fractional
            .iter()
            .map(|c| (c.vertices.clone(), c.value))
            .collect();
        let mut added = 0;
        for cut in separator.separate(&columns)? {
            let entries: Vec<_> = tagged
                .iter()
                .filter_map(|&(column, tag)| {
                    let coefficient = cut.coefficient(&State::customers(&state.routes[tag]));
                    (coefficient != 0).then_some((column, coefficient as f64))
                })
                .collect();
            let sense = if cut.greater_equal() {
                RowSense::GreaterEqual
            } else {
                RowSense::LessEqual
            };
            let row = rows.push(sense, cut.rhs as f64, entries);
            for (index, arc) in self.problem.graph.arcs.iter().enumerate() {
                let u = if arc.tail == self.problem.graph.sink {
                    0
                } else {
                    arc.tail
                };
                let v = if arc.head == self.problem.graph.sink {
                    0
                } else {
                    arc.head
                };
                let coefficient = cut.edge_coefficient(u, v);
                if coefficient != 0. {
                    state.cut.capacity_arc_entries[index].push((row, coefficient));
                }
            }
            state.cut.capacity_rows.push(Some(row));
            state.cut.capacity_cuts.push(cut);
            added += 1;
        }
        Ok(added)
    }
    fn separate_knapsack(
        &self,
        state: &mut State,
        rows: &mut PendingRows,
        tagged: &[(usize, usize)],
        values: &[f64],
    ) -> Result<usize> {
        let graph = state.solver.graph.graph_id.max(0);
        let columns: Vec<_> = tagged
            .iter()
            .filter(|&&(column, _)| values[column] > 1e-9)
            .map(|&(column, tag)| KnapsackColumn {
                graph_id: graph,
                resources: state.routes[tag].resources.clone(),
                value: values[column],
            })
            .collect();
        let Some(separator) = &mut state.cut.knapsack_separator else {
            return Ok(0);
        };
        let indices = std::collections::BTreeMap::from([(graph, state.solver.knapsack_indices())]);
        let cuts = separator.separate_mapped(&columns, &indices)?;
        let count = cuts.len();
        for cut in cuts {
            let entries = tagged
                .iter()
                .filter_map(|&(column, tag)| {
                    let coefficient = state
                        .solver
                        .knapsack_coefficient(&cut, &state.routes[tag].resources);
                    (coefficient != 0).then_some((column, coefficient as f64))
                })
                .collect();
            let row = rows.push(RowSense::LessEqual, cut.cut.rhs as f64, entries);
            state.cut.knapsack_rows.push(Some(row));
            state.solver.cuts.knapsack_cuts.push(cut);
        }
        Ok(count)
    }
    fn separate_kpath<M: LpMaster>(
        &self,
        state: &mut State,
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
            Some(RootKPathSeparator::Native { .. })
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
            RootKPathSeparator::Native { separator, graphs } => {
                for cut in separator.separate(&solutions, &values)? {
                    let index = state.solver.cuts.register_native_k_path_cut(
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
        let graph = state.solver.graph.clone();
        let mut paths = vec![];
        for &(_, tag) in tagged {
            let route = &state.routes[tag];
            paths.push((
                state.solver.cuts.route(&graph, &State::customers(route))?,
                state.is_enumerated(route),
            ));
        }
        for index in before..after {
            let mut entries = vec![];
            for (&(column, _), (path, enumerated)) in tagged.iter().zip(&paths) {
                let c = state
                    .solver
                    .cuts
                    .route_coefficient(&graph, index, path, *enumerated)?;
                if c != 0 {
                    entries.push((column, c as f64));
                }
            }
            let cut = &state.solver.cuts.cuts[index];
            let row = rows.push(RowSense::GreaterEqual, cut.rhs as f64, entries);
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
                    .route_coefficient(&graph, index, path, *enumerated)?;
                master.set_coefficient(row, column, c as f64)?;
            }
        }
        Ok((after - before, changed))
    }
    /// New rank-one cuts, and whether existing cut memories changed.
    fn separate_rank_one<M: LpMaster>(
        &self,
        state: &mut State,
        master: &mut M,
        rows: &mut PendingRows,
        tagged: &[(usize, usize)],
        fractional: &[FractionalColumn],
    ) -> Result<(usize, bool)> {
        let before = state.solver.cuts.cuts.len();
        let options = state.solver.options.clone();
        let graph = state.solver.graph.clone();
        state.solver.cuts.separate_default(
            &graph,
            &options,
            &self.problem.customers,
            fractional,
        )?;
        let after = state.solver.cuts.cuts.len();
        let mut paths = vec![];
        for &(_, tag) in tagged {
            let route = &state.routes[tag];
            paths.push((
                state.solver.cuts.route(&graph, &State::customers(route))?,
                state.is_enumerated(route),
            ));
        }
        for index in before..after {
            let mut entries = vec![];
            for (&(column, _), (path, enumerated)) in tagged.iter().zip(&paths) {
                let c = state
                    .solver
                    .cuts
                    .route_coefficient(&graph, index, path, *enumerated)?;
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
            let row = rows.push(sense, cut.rhs as f64, entries);
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
                    .route_coefficient(&graph, index, path, *enumerated)?;
                master.set_coefficient(row, column, c as f64)?;
            }
        }
        Ok((after - before, changed))
    }
    /// Dynamic NG (runColGenSpRelaxationImprovement and
    /// BcRCSPFunctor::improveCurrentSpRelaxation): strengthen the NG memories
    /// from the cycles of the routes in the LP solution, then mark the master
    /// columns the stronger relaxation forbids for removal. A negative
    /// RCSPdynamicNGmode only strengthens while no rank-one cut is in the
    /// master (cuts of the current round do not count yet).
    fn improve_relaxation(
        &self,
        state: &mut State,
        rank_one_in_master: bool,
        tagged: &[(usize, usize)],
        values: &[f64],
    ) -> Result<bool> {
        let option = |name: &str| state.solver.options.get(name);
        let mode = option("RCSPdynamicNGmode")? as i32;
        if mode == 0 || (mode < 0 && rank_one_in_master) {
            return Ok(false);
        }
        let controls = crate::graph::relaxation::Controls {
            mode: mode.abs(),
            max_vertex_size: option("RCSPmaxNGneighbourhoodSize")?.max(0.) as usize,
            max_average_size: option("RCSPmaxNGaverNeighbourhoodSize")?.max(0.) as usize,
            max_columns: option("RCSPmaxNumOfColumnsInDynamicNG")?.max(0.) as usize,
            max_cycle_size: option("RCSPmaxCycleSizeInDynamicNG")? as i32,
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
