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
        if self
            .enumeration_snapshot
            .as_ref()
            .is_some_and(|snapshot| !snapshot.same_topology(&self.graph))
        {
            self.restore_arcs();
        }
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
            routes.sort_by(|a, b| a.cost.total_cmp(&b.cost));
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
        if self.rollback_point_saved {
            controls.time_limit = Duration::try_from_secs_f64(
                self.options.get("BALDEShardTimeThresholdInPricing")?.max(0.),
            )
            .map_err(|_| invalid("pricing time limit"))?;
        }
        let started = std::time::Instant::now();
        let answer = engine.solve(&self.graph, &controls);
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
