//! Staged pricing and retained-universe repricing.
use super::*;
impl<const D: usize> Solver<D> {
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
    /// Price one phase. `negative_only` filters outputs at the configured zero
    /// threshold; the minimum remains independent of that filter. Retained-path
    /// inspection also keeps its minimum when the output column limit is zero.
    /// Disabled phases over a completed retained domain return a no-op result
    /// before refreshing caches; their empty output does not prove convergence.
    pub fn price_phase(
        &mut self,
        index: usize,
        maximum: usize,
        negative_only: bool,
    ) -> Result<PricingResult> {
        self.price_phase_internal(index, maximum, negative_only, false)
    }

    /// Price with completed columns retained when a label or time limit fires.
    /// `exact == false` forbids interpreting the observed minimum as a bound.
    pub fn price_phase_partial(
        &mut self,
        index: usize,
        maximum: usize,
        negative_only: bool,
    ) -> Result<PricingResult> {
        self.price_phase_internal(index, maximum, negative_only, true)
    }

    fn price_phase_internal(
        &mut self,
        index: usize,
        maximum: usize,
        negative_only: bool,
        allow_partial: bool,
    ) -> Result<PricingResult> {
        if self
            .graph
            .coupled_kernel()
            .is_some_and(|kernel| kernel.has_negative_paired_objectives())
        {
            return Err(incomplete(
                "negative paired objectives are not supported by paired front programs",
            ));
        }
        let phase = *self
            .phases
            .get(index)
            .ok_or_else(|| invalid("pricing phase out of range"))?;
        self.discard_incomplete_enumeration();
        // A disabled retained-path phase is a successful no-op. In particular,
        // it does not recache changed duals or replace the retained working
        // state; the next active phase performs those updates.
        if self.enumeration.is_some() && !phase.exact() {
            return Ok(PricingResult {
                columns: vec![],
                minimum_reduced_cost: 1e12,
                exact: false,
                statistics: Default::default(),
            });
        }
        if self
            .prepared
            .as_ref()
            .is_some_and(|g| g.coupled_kernel() != self.graph.coupled_kernel())
        {
            self.restore_arcs();
        }
        if self.enumeration.is_some() {
            self.check_presence_universe()?;
        }
        self.refresh_cuts()?;
        if self.enumeration.is_some() {
            let enumerated_kpath = self.enumerated_kpath_inputs();
            let universe = self.enumeration.as_mut().unwrap();
            let previous: Vec<f64> = universe
                .paths
                .partitions
                .iter()
                .flatten()
                .map(|path| path.reduced_cost)
                .collect();
            universe.paths.inspect_with_presence(
                &self.graph,
                f64::INFINITY,
                &self.applied_cuts.0,
                &enumerated_kpath,
                &self.knapsack_pricing,
                &self.presence_pricing,
            )?;
            let priced = universe.paths.priced_routes(
                &self.graph,
                maximum.min(self.controls.max_columns),
                negative_only,
                self.controls.zero_threshold,
            );
            let (routes, minimum) = match priced {
                Ok(result) => result,
                Err(error) => {
                    for (path, old) in universe.paths.partitions.iter_mut().flatten().zip(previous)
                    {
                        path.reduced_cost = old;
                    }
                    return Err(error);
                }
            };
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
        let mut answer = if allow_partial {
            engine.solve_partial(&self.graph, &controls)
        } else {
            engine.solve(&self.graph, &controls)
        };
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
                    answer = if allow_partial {
                        retry.solve_partial(&self.graph, &controls)
                    } else {
                        retry.solve(&self.graph, &controls)
                    };
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

#[cfg(test)]
#[path = "../../tests/unit/pricing/solve/resource_boundary_tests.rs"]
mod resource_boundary_tests;
