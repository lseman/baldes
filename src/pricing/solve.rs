//! Staged pricing and retained-universe repricing.
use super::*;
impl Solver {
    pub fn price(&mut self, maximum: usize, negative_only: bool) -> Result<PricingResult> {
        if !negative_only {
            return self.price_phase(0, maximum, false);
        }
        let saved = self.phases.clone();
        let saved_options = self.options.clone();
        let outcome = (|| {
            for (labels, columns) in [(32, 20usize), (128, 50), (i64::MAX, 100)] {
                self.options
                    .set("BALDESmaxNumOfColsPerIteration", columns as f64)?;
                self.phases = vec![phases::Phase {
                    check_ng: true,
                    jump_arcs: true,
                    labels_per_bucket: labels,
                }];
                let result = self.price_phase(0, maximum.min(columns), true)?;
                if !result.columns.is_empty() || result.exact {
                    return Ok(result);
                }
            }
            Err(invalid("empty pricing stage schedule"))
        })();
        self.phases = saved;
        self.options = saved_options;
        outcome
    }
    pub(super) fn enumerated_kpath_inputs(&self) -> Vec<LimitedKPathInput> {
        self.applied_cuts
            .3
            .iter()
            .map(|cut| LimitedKPathInput {
                dual: cut.dual,
                arc_memory: false,
                support: cut.vertex_support.clone(),
                memory_vertices: vec![],
            })
            .collect()
    }
    pub fn price_phase(
        &mut self,
        index: usize,
        maximum: usize,
        negative_only: bool,
    ) -> Result<PricingResult> {
        if self.accumulated_branches > 0 {
            return Err(invalid(
                "native pricing has no accumulated resource branches",
            ));
        }
        let phase = *self
            .phases
            .get(index)
            .ok_or_else(|| invalid("pricing phase out of range"))?;
        self.refresh_cuts()?;
        let enumerated_kpath = self.enumerated_kpath_inputs();
        if let Some(universe) = &mut self.enumeration {
            if !phase.exact() {
                return Ok(PricingResult {
                    columns: vec![],
                    minimum_reduced_cost: 1e12,
                    exact: false,
                    statistics: Default::default(),
                });
            }
            universe.paths.inspect_with_knapsack(
                &self.graph,
                f64::INFINITY,
                &self.applied_cuts.0,
                &enumerated_kpath,
                &self.knapsack_pricing,
            )?;
            let mut routes = universe.paths.routes(&self.graph)?;
            routes.sort_by(|a, b| enumerated_cost_order(a.cost, b.cost));
            routes.truncate(maximum.min(self.controls.max_columns));
            let minimum = routes.first().map_or(1e12, |r| r.cost);
            return Ok(PricingResult {
                columns: routes,
                minimum_reduced_cost: minimum,
                exact: true,
                statistics: Default::default(),
            });
        }
        let mut controls = self.controls.clone();
        let limit = if phase.exact() {
            controls.max_columns
        } else {
            self.options
                .get("BALDESmaxNumOfColsPerIteration")
                .unwrap_or(30.)
                .max(1.) as usize
        };
        controls.max_columns = maximum.min(limit);
        controls.negative_only = negative_only;
        let call_time_limit = self.pricing_time_limit()?;
        let started = std::time::Instant::now();
        let engine_controls = self.engine_controls(phase.exact());
        let reuse = self
            .prepared
            .as_ref()
            .is_some_and(|prepared| prepared.same_topology(&self.graph))
            && self
                .engine
                .as_ref()
                .is_some_and(|e| e.controls.bidirectional == engine_controls.bidirectional);
        let mut engine = if reuse {
            self.engine.take().unwrap()
        } else {
            self.new_engine(engine_controls.clone(), phase)?
        };
        engine.phase = phase;
        engine.controls = engine_controls;
        self.requires_rollback = false;
        controls.time_limit = call_time_limit.saturating_sub(started.elapsed());
        let mut answer = engine.solve(&self.graph, &controls);
        if engine.controls.bidirectional
            && answer.as_ref().is_err_and(|error| {
                error.kind == crate::core::error::ErrorKind::Runtime
                    && error.message == crate::labeling::engine::RESOURCE_REPLAY_MISMATCH
            })
        {
            // Reverse subtraction can accept a boundary path whose forward
            // additions exceed a window by one ulp. The candidate may also
            // have displaced valid paths in the bounded pool: discarding it
            // alone cannot establish the minimum. Rebuild the full forward
            // search, retaining logical cuts/branches and physical exclusions.
            let mut forward = engine.controls.clone();
            forward.bidirectional = false;
            forward.use_symmetric_execution = false;
            match self.new_engine(forward, phase) {
                Ok(mut retry) => {
                    controls.time_limit = call_time_limit.saturating_sub(started.elapsed());
                    answer = retry.solve(&self.graph, &controls);
                    engine = retry;
                }
                Err(error) => answer = Err(error),
            }
        }
        if answer.is_err() {
            engine.half_ready = false;
            engine.full_ready = false;
        }
        if let Ok(result) = &answer {
            if phase.exact() {
                let labels = engine
                    .forward
                    .buckets
                    .iter()
                    .chain(&engine.backward.buckets)
                    .flatten()
                    .map(|b| b.active.len())
                    .sum();
                self.round.collect(
                    started.elapsed().as_secs_f64(),
                    result.statistics.dominance_checks,
                    labels,
                    self.options.get("BALDESdynamicBucketSteps")? != 0.,
                );
            }
        }
        self.engine = Some(engine);
        if !reuse {
            self.prepared = Some(self.topology());
        }
        if let Err(error) = &answer {
            self.requires_rollback =
                self.rollback_point_saved && error.message.contains("time limit");
        }
        answer
    }
}

fn enumerated_cost_order(left: f64, right: f64) -> std::cmp::Ordering {
    // Native cost/index pairs treat the two signed zeros as equal. Stable
    // sorting then retains the canonical exported path index for ties.
    left.partial_cmp(&right)
        .unwrap_or(std::cmp::Ordering::Equal)
}

#[cfg(test)]
mod resource_boundary_tests {
    use super::*;
    fn boundary(alternative: bool) -> Solver {
        let mut s = Solver::new(if alternative { 4 } else { 3 }, 1).unwrap();
        let upper = f64::from_bits(1f64.to_bits() - 1);
        s.graph.initial[0] = Some(0.);
        for v in 0..s.graph.vertices.len() {
            s.graph.set_bounds(v, 0, 0., upper).unwrap();
        }
        let sink = s.graph.sink;
        let mut arcs = vec![(0, 1, 0.7, -2.), (1, sink, 0.3, -2.)];
        if alternative {
            arcs.extend([(0, 2, 0.2, -1.), (2, sink, 0.2, -1.)]);
        }
        for (u, v, consumption, cost) in arcs {
            let id = s.graph.add_arc(u, v, cost).unwrap();
            s.graph.arc_mut(id).unwrap().consumption[0] = consumption;
        }
        s
    }
    #[test]
    fn reverse_only_boundary_path_retries_and_graph_changes_restore_normal_search() {
        let mut s = boundary(false);
        let result = s.price(1, false).unwrap();
        assert!(result.exact && result.columns.is_empty());
        assert!(!s.engine.as_ref().unwrap().controls.bidirectional);
        for v in 0..s.graph.vertices.len() {
            s.graph.set_bounds(v, 0, 0., 1.).unwrap();
        }
        let result = s.price(1, false).unwrap();
        assert!(result.exact);
        assert_eq!(result.columns[0].arcs, [0, 1]);
        assert!(s.engine.as_ref().unwrap().controls.bidirectional);
    }
    #[test]
    fn retry_recovers_displaced_valid_paths_and_preserves_physical_exclusions() {
        let mut s = boundary(true);
        let result = s.price(1, false).unwrap();
        assert!(result.exact);
        assert_eq!(result.columns.len(), 1);
        assert_eq!(result.columns[0].arcs, [2, 3]);
        assert_eq!(result.minimum_reduced_cost, -2.);
        assert!(!s.engine.as_ref().unwrap().controls.bidirectional);
        s.forbidden_arcs = vec![false, false, true, false];
        let result = s.price(1, false).unwrap();
        assert!(result.exact && result.columns.is_empty());
        s.controls.time_limit = std::time::Duration::ZERO;
        let error = s.price(1, false).unwrap_err();
        assert_eq!(error.kind, crate::core::error::ErrorKind::Runtime);
        assert!(error.message.contains("time limit"));
        let engine = s.engine.as_ref().unwrap();
        assert!(!engine.half_ready && !engine.full_ready);
    }
}
