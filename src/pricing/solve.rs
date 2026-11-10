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

    /// Recovered scalar NG candidate stages. A fork preserves branch and cut
    /// pricing state, while removing the expensive coupled objective. Every
    /// exported route is elementary and replayed with the original resource.
    pub(super) fn scalar_preprice(
        &mut self,
        maximum: usize,
        budget: Duration,
    ) -> Option<PricingResult> {
        if maximum == 0
            || !self.accumulated_branches.is_empty()
            || !self.accumulated_original_bounds.is_empty()
        {
            return None;
        }
        let charges = self.graph.coupled_kernel()?.scalar_pricing_charges()?;
        if charges.len() != self.graph.arcs.len() || charges.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let started = std::time::Instant::now();
        let mut scalar = self.fork_pricing(false);
        scalar.graph.set_coupled_resources(&[]).ok()?;
        scalar.ng_baseline = None;
        scalar.options.set("BALDESdynamicNGmode", 0.).ok()?;
        scalar.graph.edit().elementary = false;
        if scalar.graph.resources.len() > 1
            && scalar.graph.arcs.iter().any(|arc| arc.consumption[1] > 0.)
        {
            scalar.graph.edit().main_resource = Some(1);
        }
        for (arc, charge) in scalar.graph.edit().arcs.iter_mut().zip(charges) {
            arc.cost += charge;
            arc.reduced_cost += charge;
        }
        scalar.controls.max_labels = scalar.controls.max_labels.min(100_000);
        scalar.controls.max_route_arcs = Some(self.graph.vertices.len().saturating_sub(1));
        scalar.controls.max_columns = maximum.min(20);
        let mut columns = Vec::new();
        // The historical workflow starts with a complete relaxation every
        // eighth call. Cyclic minima then refine NG memory before Bellman.
        let complete_first = self.prepricing_calls % 8 == 0;
        let mut stages = if complete_first {
            vec![i64::MAX]
        } else {
            vec![16, 4, 1, i64::MAX]
        }
        .into_iter();
        let mut labels = stages.next()?;
        let mut labels_left = scalar.controls.max_labels;
        loop {
            scalar.controls.time_limit = budget.saturating_sub(started.elapsed());
            scalar.controls.max_labels = labels_left;
            if scalar.controls.time_limit.is_zero() || labels_left == 0 {
                break;
            }
            scalar.phases = vec![phases::Phase {
                check_ng: true,
                jump_arcs: labels == 16 || labels == i64::MAX,
                labels_per_bucket: labels,
            }];
            let priced = scalar.price_phase_partial(0, maximum.min(20), true).ok()?;
            labels_left = labels_left.saturating_sub(priced.statistics.generated.max(1));
            if priced.exact && priced.minimum_reduced_cost.is_finite() {
                let bound = priced.minimum_reduced_cost;
                self.staged_minimum = Some(self.staged_minimum.map_or(bound, |old| old.max(bound)));
            }
            let mut tightened = false;
            for candidate in priced.columns {
                let mut previous = vec![None; scalar.graph.vertices.len()];
                for (position, &node) in candidate.nodes.iter().enumerate() {
                    let node = usize::try_from(node).ok()?;
                    if let Some(before) = previous[node] {
                        let elements = scalar.graph.vertices[node].elements.clone();
                        for &vertex in &candidate.nodes[before..=position] {
                            let vertex = usize::try_from(vertex).ok()?;
                            let memory = &mut scalar.graph.edit().vertices[vertex].memory;
                            for &element in &elements {
                                if !memory.contains(&element)
                                    && memory.len() < crate::graph::ng::NG_MEMORY_CAPACITY
                                {
                                    memory.push(element);
                                    tightened = true;
                                }
                            }
                        }
                    }
                    previous[node] = Some(position);
                }
                let Some(mut route) = self.graph.elementary_route(&candidate.arcs).ok()? else {
                    continue;
                };
                // Preserve the scalar walk's branch and cut prices, replacing
                // only its surrogate original objective after elementary replay.
                route.cost = candidate.cost + route.original_cost - candidate.original_cost;
                if route.cost < self.controls.zero_threshold - crate::labeling::EPSILON
                    && !columns
                        .iter()
                        .any(|old: &crate::labeling::Route| old.arcs == route.arcs)
                {
                    columns.push(route);
                }
            }
            if !columns.is_empty() {
                break;
            }
            if labels == i64::MAX {
                if !priced.exact || !tightened {
                    break;
                }
                // Rebuild preparation after tightening. All elementary routes
                // remain feasible, so completed minima still bound the model.
                scalar.restore_arcs();
            } else if let Some(next) = stages.next() {
                labels = next;
            } else {
                break;
            }
        }
        if columns.is_empty() {
            return None;
        }
        Some(PricingResult {
            columns,
            minimum_reduced_cost: 0.,
            exact: false,
            statistics: Default::default(),
        })
    }

    /// Exact searches below progressively relaxed negative targets recover
    /// useful columns without exhausting the full zero-threshold domain. Only
    /// a finished stage supplies a minimum; it survives later interruptions.
    fn bellman_stages(
        &mut self,
        engine: &mut LabelingEngine<D>,
        controls: &Controls,
        started: std::time::Instant,
        budget: Duration,
    ) -> Result<PricingResult> {
        let mut targets = Vec::new();
        let initial = self.exact_target.or_else(|| {
            let remaining = budget.saturating_sub(started.elapsed());
            let deadline = std::time::Instant::now()
                .checked_add((remaining / 4).min(Duration::from_millis(200)))?;
            let costs: Vec<_> = self.graph.arcs.iter().map(|a| a.reduced_cost).collect();
            self.graph
                .coupled_kernel()?
                .plugin_pricing_minimum(&costs, &mut 1_000_000, deadline)
                .filter(|v| *v < controls.zero_threshold)
                .map(|v| (v - controls.zero_threshold) / 4.)
        });
        if let Some(target) = initial {
            for delta in [target, target / 8.] {
                let threshold = controls.zero_threshold + delta;
                if threshold.is_finite() && threshold < controls.zero_threshold - 1e-6 {
                    targets.push(threshold);
                }
            }
        }
        targets.push(controls.zero_threshold);
        let mut statistics = crate::labeling::Statistics::default();
        let mut outcome = None;
        for threshold in targets {
            let mut stage = controls.clone();
            stage.zero_threshold = threshold;
            stage.time_limit = budget.saturating_sub(started.elapsed());
            if stage.time_limit.is_zero() {
                break;
            }
            let result = engine.solve_partial(&self.graph, &stage);
            match result {
                Ok(mut result) => {
                    statistics.absorb(&result.statistics);
                    if result.exact {
                        let bound = result.minimum_reduced_cost.min(threshold);
                        if bound.is_finite() {
                            self.staged_minimum =
                                Some(self.staged_minimum.map_or(bound, |old| old.max(bound)));
                        }
                    }
                    let best = result
                        .columns
                        .iter()
                        .map(|r| r.cost)
                        .filter(|&v| v < controls.zero_threshold - crate::labeling::EPSILON)
                        .min_by(f64::total_cmp);
                    if let Some(best) = best {
                        self.exact_target = Some((best - controls.zero_threshold) / 4.);
                    }
                    let stop =
                        best.is_some() || !result.exact || threshold == controls.zero_threshold;
                    if threshold != controls.zero_threshold {
                        result.exact = false;
                    }
                    outcome = Some(result);
                    if stop {
                        break;
                    }
                }
                Err(error) => {
                    let exhausted = error.kind == crate::core::error::ErrorKind::Runtime
                        && (error.message.contains("time limit")
                            || error.message.contains("label limit"));
                    if self.staged_minimum.is_none() || !exhausted {
                        return Err(error);
                    }
                    break;
                }
            }
        }
        let mut result = outcome.unwrap_or(PricingResult {
            columns: vec![],
            minimum_reduced_cost: 0.,
            exact: false,
            statistics: Default::default(),
        });
        result.statistics = statistics.clone();
        engine.statistics = statistics;
        Ok(result)
    }

    fn price_phase_internal(
        &mut self,
        index: usize,
        maximum: usize,
        negative_only: bool,
        allow_partial: bool,
    ) -> Result<PricingResult> {
        self.staged_minimum = None;
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
            let mut previous = std::mem::take(&mut self.arc_cost_scratch);
            previous.clear();
            previous.extend(
                universe
                    .paths
                    .partitions
                    .iter()
                    .flatten()
                    .map(|path| path.reduced_cost),
            );
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
                    for (path, old) in universe
                        .paths
                        .partitions
                        .iter_mut()
                        .flatten()
                        .zip(previous.iter().copied())
                    {
                        path.reduced_cost = old;
                    }
                    self.arc_cost_scratch = previous;
                    return Err(error);
                }
            };
            self.arc_cost_scratch = previous;
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
        let bellman = self
            .graph
            .coupled_kernel()
            .is_some_and(|k| k.bellman_pricing());
        let engine_controls = self.engine_controls(phase.exact() || bellman);
        let search_phase = if bellman {
            phases::Phase {
                check_ng: true,
                jump_arcs: true,
                labels_per_bucket: i64::MAX,
            }
        } else {
            phase
        };
        if engine_controls.staged_pricing {
            self.prepricing_calls = self.prepricing_calls.wrapping_add(1);
            let budget = (call_time_limit / 4).min(Duration::from_secs(2));
            if let Some(result) = self.scalar_preprice(controls.max_columns, budget) {
                if let Some(best) = result.columns.iter().map(|r| r.cost).min_by(f64::total_cmp) {
                    self.exact_target = Some((best - controls.zero_threshold) / 4.);
                }
                return Ok(result);
            }
            let remaining = call_time_limit.saturating_sub(started.elapsed());
            if let Some(result) = self.search_route_columns(
                controls.max_columns.min(20),
                (remaining / 4).min(Duration::from_millis(250)),
            ) {
                if let Some(best) = result.columns.iter().map(|r| r.cost).min_by(f64::total_cmp) {
                    self.exact_target = Some((best - controls.zero_threshold) / 4.);
                }
                return Ok(result);
            }
        }
        // Recovered Bellman pricing returns useful improving batches promptly.
        // A batch limit never establishes pricing completion or its minimum.
        let allow_partial = allow_partial || engine_controls.staged_pricing;
        if engine_controls.staged_pricing {
            controls.max_columns = controls.max_columns.min(20);
        }
        // refresh_cuts already validated topology and clears prepared/engine
        // on a mismatch. Nothing above mutates the graph after that check.
        let reuse = self.prepared.is_some()
            && self
                .engine
                .as_ref()
                .is_some_and(|e| e.controls.bidirectional == engine_controls.bidirectional);
        let mut engine = if reuse {
            self.engine.take().unwrap()
        } else {
            self.new_engine(engine_controls.clone(), search_phase)?
        };
        engine.phase = search_phase;
        engine.controls = engine_controls;
        self.requires_rollback = false;
        controls.time_limit = call_time_limit.saturating_sub(started.elapsed());
        let mut answer = if engine.controls.staged_pricing && controls.max_columns > 0 {
            self.bellman_stages(&mut engine, &controls, started, call_time_limit)
        } else if allow_partial {
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
            self.staged_minimum = None;
            let mut forward = engine.controls.clone();
            forward.bidirectional = false;
            forward.backward_only = false;
            forward.use_symmetric_execution = false;
            match self.new_engine(forward, search_phase) {
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
                let labels = result.statistics.forward_labels + result.statistics.backward_labels;
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
