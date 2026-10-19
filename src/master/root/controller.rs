//! Column-generation callbacks and cut separation scheduling.
use super::*;
pub(super) struct PricingCheckpoint {
    master: ClpMaster,
    row_map: Vec<usize>,
    cut: CutState,
    pricing: Solver,
    incumbent: f64,
    route_count: usize,
    output: Output,
    previous_fixing_gap: f64,
    enumerated_routes: usize,
}
pub(super) struct RootController {
    pub(super) problem: Rc<RootProblem>,
    pub(super) state: Rc<RefCell<State>>,
    pub(super) checkpoint: Option<PricingCheckpoint>,
    pub(super) certificate_search: super::certificate::SearchCache,
}

fn separation_gap_closed(value: f64, incumbent: f64) -> bool {
    if !incumbent.is_finite() { return false; }
    let gap = incumbent - value;
    gap <= value.abs().max(incumbent.abs()) * 1e-10 + 1e-6
        || gap / incumbent.abs() <= 1e-9
}

/// Reuse the refreshed integer cut incidence of loaded routes. Base/branch
/// coefficients stay in their original representation, including tiny values
/// the LP backend may have dropped. Routing tags are local to this node.
fn synchronize_pool(
    pool: &mut [Column],
    loaded: &[Option<Column>],
    route_count: usize,
    base_rows: &[bool],
    mut refresh: impl FnMut(&mut Column) -> Result<()>,
) -> Result<()> {
    let mut by_tag = vec![None; route_count];
    for current in loaded.iter().flatten() {
        if let Some(slot) = usize::try_from(current.tag)
            .ok()
            .and_then(|tag| by_tag.get_mut(tag))
        {
            *slot = Some(current);
        }
    }
    let is_base = |row: usize| base_rows.get(row).copied().unwrap_or(false);
    for column in pool {
        if let Some(current) = usize::try_from(column.tag)
            .ok()
            .and_then(|tag| by_tag.get(tag))
            .and_then(|c| *c)
            .filter(|c| c.identity == column.identity && c.cost == column.cost)
        {
            column.entries.retain(|&(row, _)| is_base(row));
            column.entries.extend(
                current
                    .entries
                    .iter()
                    .copied()
                    .filter(|&(row, value)| !is_base(row) && value != 0.),
            );
            column.entries.sort_by_key(|&(row, _)| row);
        } else {
            refresh(column)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod synchronization_tests {
    use super::*;

    fn column(tag: i32) -> Column {
        Column {
            tag,
            cost: 10.,
            reduced_cost: -1.,
            acceptable: true,
            identity: format!("route-{tag}"),
            entries: vec![(0, 1.), (1, 9.), (2, 1e-30)],
        }
    }

    #[test]
    fn loaded_cut_incidence_preserves_original_base_rows_and_refreshes_unloaded_routes() {
        let mut current = column(0);
        current.entries = vec![(0, 1.), (1, 0.), (3, 4.)];
        let loaded = vec![Some(current.clone()), None];
        let mut stale_identity = column(0);
        stale_identity.identity = "different route".into();
        let mut stale_cost = column(0);
        stale_cost.cost = 11.;
        let mut pool = vec![column(0), column(1), stale_identity, stale_cost];
        let mut refreshed = vec![];
        synchronize_pool(&mut pool, &loaded, 2, &[true, false, true], |c| {
            refreshed.push(c.identity.clone());
            c.entries = vec![(7, 2.)];
            Ok(())
        })
        .unwrap();
        // Row 2 is a branch row added after cuts; keep its original tiny
        // coefficient even if the LP dropped it. Row 1's zero is omitted.
        assert_eq!(pool[0].entries, vec![(0, 1.), (2, 1e-30), (3, 4.)]);
        assert_eq!(pool[0].reduced_cost, -1.);
        assert_eq!(refreshed, vec!["route-1", "different route", "route-0"]);
        assert!(pool[1..].iter().all(|c| c.entries == vec![(7, 2.)]));
        pool[0].entries[0].1 = 8.;
        assert_eq!(loaded[0].as_ref().unwrap().entries, current.entries);
    }

    #[test]
    fn invalid_route_tags_do_not_bypass_refresh_validation() {
        for tag in [-1, 10] {
            let mut pool = vec![column(tag)];
            let loaded = vec![Some(column(tag))];
            let result = synchronize_pool(&mut pool, &loaded, 2, &[], |_| {
                Err(invalid("cached route tag"))
            });
            assert!(result.is_err());
        }
    }

    #[test]
    fn owned_checkpoint_restores_warm_pool_rows_and_routes_after_pricing_timeout() {
        let mut graph = Graph::new(3, 1).unwrap();
        for v in 0..3 {
            graph.set_bounds(v, 0, 0., 10.).unwrap();
        }
        graph.add_arc(0, 1, 1.).unwrap();
        graph.add_arc(1, 2, 1.).unwrap();
        let problem = RootProblem::set_partitioning(graph, 1.).unwrap();
        let mut options = RootOptions::default();
        options.capture_state = true;
        options.enumerate = false;
        let result = super::super::solve(problem, options).unwrap();
        let snapshot = result.snapshot.as_ref().unwrap();
        let state = Rc::new(RefCell::new(State {
            solver: snapshot.pricing.fork_for_node(),
            routes: snapshot.routes.clone(),
            route_tags: snapshot
                .routes
                .iter()
                .enumerate()
                .map(|(i, route)| (route.identity(), i as i32))
                .collect(),
            incumbent: snapshot.incumbent,
            enumerate: false,
            rounded: true,
            previous_fixing_gap: 42.,
            deadline: None,
            pricing_seconds: 0.,
            cut: snapshot.cut.clone(),
            fixing_calls: 0,
            enumerated_routes: 7,
        }));
        let problem = Rc::new(snapshot.problem.clone());
        let mut controller = RootController {
            problem: problem.clone(),
            state: state.clone(),
            checkpoint: None,
            certificate_search: Default::default(),
        };
        let expected_warm = format!("{:?}", result.master.warm_start);
        struct Legacy { refreshed: bool, saved: bool }
        impl Controller<ClpMaster> for Legacy {
            fn refresh_generated_columns(&mut self, _: &mut [Column]) -> Result<()> {
                self.refreshed = true;
                Ok(())
            }
            fn save_pricing_checkpoint(&mut self, _: &ClpMaster, _: &Output) -> Result<()> {
                self.saved = true;
                Ok(())
            }
        }
        let mut legacy = Legacy { refreshed: false, saved: false };
        legacy.synchronize_generated_columns(&mut [], &[]).unwrap();
        legacy.store_pricing_checkpoint(&snapshot.master, result.master.clone()).unwrap();
        assert!(legacy.refreshed && legacy.saved);
        controller
            .store_pricing_checkpoint(&snapshot.master, result.master)
            .unwrap();
        assert!(state.borrow().solver.rollback_point_saved);
        let previous_calls = controller.checkpoint.as_ref().unwrap().output.pricing_calls;
        let mut terminal = controller.checkpoint.as_ref().unwrap().output.clone();
        terminal.pricing_calls += 1;
        state.borrow_mut().incumbent = snapshot.master.objective();
        controller.store_pricing_checkpoint(&snapshot.master, terminal).unwrap();
        assert_eq!(controller.checkpoint.as_ref().unwrap().output.pricing_calls, previous_calls);
        {
            let mut changed = state.borrow_mut();
            let mut abandoned = changed.routes[0].clone();
            for profile in abandoned.resource_profile.iter_mut().skip(1) {
                for value in profile { *value += 0.25; }
            }
            for value in &mut abandoned.resources { *value += 0.25; }
            register(&mut changed, &abandoned).unwrap();
            changed.cut.rows.push(Some(999));
            changed.solver.requires_rollback = true;
            changed.previous_fixing_gap = 99.;
            changed.enumerated_routes = 0;
        }
        problem.row_map.borrow_mut().reverse();
        let (master, output) = controller
            .recover_pricing_timeout(crate::incomplete("pricing time limit"))
            .unwrap();
        assert_eq!(master.costs, snapshot.master.costs);
        assert_eq!(master.rhs, snapshot.master.rhs);
        assert_eq!(master.values, snapshot.master.values);
        assert_eq!(format!("{:?}", output.warm_start), expected_warm);
        assert_eq!(
            *problem.row_map.borrow(),
            *snapshot.problem.row_map.borrow()
        );
        let restored = state.borrow();
        assert_eq!(
            format!("{:?}", restored.routes),
            format!("{:?}", snapshot.routes)
        );
        assert_eq!(restored.cut.rows, snapshot.cut.rows);
        assert_eq!(restored.previous_fixing_gap, 42.);
        assert_eq!(restored.enumerated_routes, 7);
        assert!(!restored.solver.requires_rollback);
        drop(restored);
        // The restored registry must still distinguish schedules before
        // assigning their short column identities.
        let mut restored = state.borrow_mut();
        let original = restored.routes[0].clone();
        let original_tag = register(&mut restored, &original).unwrap();
        let mut waited = original.clone();
        for profile in waited.resource_profile.iter_mut().skip(1) {
            for value in profile { *value += 0.25; }
        }
        for value in &mut waited.resources { *value += 0.25; }
        let waited_tag = register(&mut restored, &waited).unwrap();
        assert_ne!(original_tag, waited_tag);
        assert_eq!(waited_tag as usize + 1, restored.routes.len());
        assert_eq!(register(&mut restored, &waited).unwrap(), waited_tag);
        let first = restored.column(&problem, &original, &master.duals, original_tag).unwrap();
        let second = restored.column(&problem, &waited, &master.duals, waited_tag).unwrap();
        assert_ne!(first.identity, second.identity);
        assert_eq!(first.identity, format!("route:{original_tag}"));
        assert_eq!(first.entries, second.entries);
        let mut signed_zero = original.clone();
        signed_zero.resource_profile[0][0] = -0.;
        assert_eq!(register(&mut restored, &signed_zero).unwrap(), original_tag);
    }
}

impl Controller<ClpMaster> for RootController {
    fn restore_feasible_columns(&mut self, duals: &[f64]) -> Result<Vec<Column>> {
        super::restoration::columns(&self.problem, &mut self.state.borrow_mut(), duals)
    }
    fn certified_pricing_bound(&self, master: &ClpMaster, _duals: &[f64]) -> Result<Option<f64>> {
        let bound = super::certificate::lower_bound(
            &self.problem, &self.state.borrow(), master, &self.certificate_search,
        );
        Ok(Some(bound.unwrap_or(f64::NEG_INFINITY)))
    }
    fn save_pricing_checkpoint(&mut self, master: &ClpMaster, output: &Output) -> Result<()> {
        self.store_pricing_checkpoint(master, output.clone())
    }
    fn store_pricing_checkpoint(&mut self, master: &ClpMaster, output: Output) -> Result<()> {
        let mut state = self.state.borrow_mut();
        // Separation returns without changing the formulation at a closed
        // incumbent gap. Keep the previous rollback point; no new pricing
        // transaction can start from this terminal point.
        if self.checkpoint.is_some() && separation_gap_closed(master.objective(), state.incumbent) {
            return Ok(());
        }
        let mut pricing = state.solver.fork_for_node();
        pricing.rollback_point_saved = false;
        pricing.requires_rollback = false;
        self.checkpoint = Some(PricingCheckpoint {
            master: master.try_clone()?,
            row_map: self.problem.row_map.borrow().clone(),
            cut: state.cut.clone(),
            pricing,
            incumbent: state.incumbent,
            // Routes are immutable after registration and only appended.
            // Rollback therefore needs a prefix length, not a deep copy of
            // every resource schedule in the route store.
            route_count: state.routes.len(),
            output,
            previous_fixing_gap: state.previous_fixing_gap,
            enumerated_routes: state.enumerated_routes,
        });
        state.solver.rollback_point_saved = true;
        Ok(())
    }

    fn unstabilized_rows(&self) -> Vec<usize> {
        let map = self.problem.row_map.borrow();
        std::iter::once(map[0])
            .chain(
                self.problem
                    .branch_constraints
                    .iter()
                    .filter_map(|(row, constraint)| {
                        matches!(constraint, BranchConstraint::NumberOfVehicles { .. })
                            .then_some(map[*row])
                    }),
            )
            .collect()
    }
    fn protected_rows(&self) -> Vec<usize> {
        self.problem.row_map.borrow().clone()
    }
    fn master_solved(&mut self, master: &ClpMaster, tags: &[i32], artificials: &[usize]) -> Result<()> {
        if artificials.iter().any(|&i| master.values()[i] > 1e-6) {
            return Ok(());
        }
        let mut state = self.state.borrow_mut();
        if let Some(upper) = super::incumbent::validated_upper(
            &self.problem, &state.routes, tags, &master.values(),
        ) {
            state.incumbent = state.incumbent.min(upper);
        }
        Ok(())
    }
    fn incumbent(&self) -> f64 {
        self.state.borrow().incumbent
    }
    fn refresh_generated_columns(&mut self, pool: &mut [Column]) -> Result<()> {
        let state = self.state.borrow();
        for column in pool {
            let route = state
                .routes
                .get(usize::try_from(column.tag).map_err(|_| invalid("cached route tag"))?)
                .ok_or_else(|| invalid("cached route tag"))?;
            let mut entries = self.problem.base_entries(route)?;
            entries.extend(state.cut_entries(route)?);
            entries.sort_by_key(|&(row, _)| row);
            column.entries = entries;
        }
        Ok(())
    }
    fn synchronize_generated_columns(
        &mut self,
        pool: &mut [Column],
        loaded: &[Option<Column>],
    ) -> Result<()> {
        let state = self.state.borrow();
        // Scaled knapsack coefficients need not survive the backend's small
        // element threshold. Reconstruct them from their original scales.
        if state.cut.knapsack_rows.iter().any(Option::is_some) {
            drop(state);
            return self.refresh_generated_columns(pool);
        }
        let route_count = state.routes.len();
        drop(state);
        let map = self.problem.row_map.borrow();
        let mut base_rows = vec![false; map.iter().max().map_or(0, |r| r + 1)];
        for &row in map.iter() {
            base_rows[row] = true;
        }
        drop(map);
        synchronize_pool(pool, loaded, route_count, &base_rows, |column| {
            self.refresh_generated_columns(std::slice::from_mut(column))
        })
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
    fn regenerate_enumerated_columns(
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
    fn separate(&mut self, master: &mut ClpMaster, tags: &[i32]) -> Result<usize> {
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
        if separation_gap_closed(value, state.incumbent) {
            return Ok(0);
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
        // Separation sees the complete solution, including fixed routes.
        // New rows are then shifted once for the residual master below.
        for (route, quantity) in &self.problem.fixed_routes {
            fractional.push(FractionalColumn {
                vertices: State::customers(route),
                value: f64::from(*quantity),
                graph_id: -1,
                enumerated: true,
            });
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
    /// Recover only an engine timeout that explicitly requests saved-point rollback.
    pub(super) fn recover_pricing_timeout(
        &mut self, error: crate::core::error::OrError,
    ) -> Result<(ClpMaster, Output)> {
        if error.kind != crate::core::error::ErrorKind::Runtime
            || !error.message.contains("time limit")
            || !self.state.borrow().solver.requires_rollback
        {
            return Err(error);
        }
        let Some(checkpoint) = &self.checkpoint else {
            return Err(error);
        };
        if self.state.borrow().routes.len() < checkpoint.route_count {
            return Err(error);
        }
        // Clone before replacing any state so a failed allocation cannot partially restore it.
        let master = checkpoint.master.try_clone()?;
        let pricing = checkpoint.pricing.fork_for_node();
        let mut state = self.state.borrow_mut();
        state.incumbent = state.incumbent.min(checkpoint.incumbent);
        state.cut = checkpoint.cut.clone();
        state.solver = pricing;
        state.routes.truncate(checkpoint.route_count);
        state.previous_fixing_gap = checkpoint.previous_fixing_gap;
        state.enumerated_routes = checkpoint.enumerated_routes;
        state.route_tags.retain(|_, tag| *tag >= 0 && (*tag as usize) < checkpoint.route_count);
        *self.problem.row_map.borrow_mut() = checkpoint.row_map.clone();
        Ok((master, checkpoint.output.clone()))
    }
    fn fixed_rank_one(&self, state: &State, index: usize) -> Result<f64> {
        self.problem.fixed_sum(|route| {
            let path = state
                .solver
                .cuts
                .route(&state.solver.graph, &State::customers(route))?;
            Ok(state
                .solver
                .cuts
                .route_coefficient(&state.solver.graph, index, &path, true)? as f64)
        })
    }
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
            let rhs = cut.rhs as f64
                - self
                    .problem
                    .fixed_sum(|route| Ok(cut.coefficient(&State::customers(route)) as f64))?;
            let row = rows.push(sense, rhs, entries);
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
        let mut columns: Vec<_> = tagged
            .iter()
            .filter(|&&(column, _)| values[column] > 1e-9)
            .map(|&(column, tag)| KnapsackColumn {
                graph_id: graph,
                resources: state.routes[tag].resources.clone(),
                value: values[column],
            })
            .collect();
        columns.extend(
            self.problem
                .fixed_routes
                .iter()
                .map(|(route, quantity)| KnapsackColumn {
                    graph_id: graph,
                    resources: route.resources.clone(),
                    value: f64::from(*quantity),
                }),
        );
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
            let rhs = cut.cut.rhs as f64
                - self.problem.fixed_sum(|route| {
                    Ok(state.solver.knapsack_coefficient(&cut, &route.resources) as f64)
                })?;
            let row = rows.push(RowSense::LessEqual, rhs, entries);
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
        if matches!(
            state.cut.kpath_separator,
            Some(RootKPathSeparator::Native { .. })
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
    /// BALDESdynamicNGmode only strengthens while no rank-one cut is in the
    /// master (cuts of the current round do not count yet).
    fn improve_relaxation(
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
