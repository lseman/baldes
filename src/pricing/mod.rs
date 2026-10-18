//! Stateful no-cut pricing and retained elementary route universe.
pub mod controller;
pub mod fixing;
pub mod phases;
use crate::core::options::Options;
use crate::cuts::state::{self, FullInput, FullKPathInput, LimitedInput, LimitedKPathInput};
use crate::cuts::CutRegistry;
use crate::enumeration::{self, Enumeration};
use crate::labeling::engine::{EngineControls, LabelingEngine};
use crate::{incomplete, invalid, Controls, Graph, PricingResult, Result};
use std::time::Duration;
#[derive(Debug)]
pub struct Solver {
    pub graph: Graph,
    pub controls: Controls,
    pub options: Options,
    pub phases: Vec<phases::Phase>,
    pub engine: Option<LabelingEngine>,
    pub enumeration: Option<Enumeration>,
    pub cutoff: f64,
    prepared: Option<Graph>,
    pub bucket_steps: [f64; 2],
    pub rollback_point_saved: bool,
    pub requires_rollback: bool,
    pub round: controller::RoundState,
    pub maximum_variable_id: Option<usize>,
    pub dominance_mode: i32,
    pub accumulated_branches: usize,
    ng_baseline: Option<(Vec<Vec<usize>>, Vec<Vec<usize>>)>,
    pub variable_costs: Vec<f64>,
    enumeration_snapshot: Option<Graph>,
    /// Registered rank-one cuts and their separator.
    pub cuts: CutRegistry,
    /// Cut inputs the engine was last prepared with.
    knapsack_pricing: crate::cuts::knapsack::KnapsackPricing,
    applied_cuts: (
        Vec<LimitedInput>,
        Vec<LimitedKPathInput>,
        Vec<FullInput>,
        Vec<FullKPathInput>,
    ),
}
impl Solver {
    pub fn new(vertices: usize, resources: usize) -> Result<Self> {
        Ok(Self {
            graph: Graph::new(vertices, resources)?,
            controls: Controls::default(),
            options: Options::default(),
            phases: phases::initialize(0, &[])?,
            engine: None,
            enumeration: None,
            cutoff: f64::INFINITY,
            prepared: None,
            bucket_steps: [0.; 2],
            rollback_point_saved: false,
            requires_rollback: false,
            round: controller::RoundState::default(),
            maximum_variable_id: None,
            dominance_mode: 0,
            accumulated_branches: 0,
            ng_baseline: None,
            variable_costs: vec![],
            enumeration_snapshot: None,
            cuts: CutRegistry::default(),
            knapsack_pricing: Default::default(),
            applied_cuts: (vec![], vec![], vec![], vec![]),
        })
    }
    pub fn has_cuts(&self) -> bool {
        self.knapsack_pricing.active || !self.applied_cuts.0.is_empty()
            || !self.applied_cuts.1.is_empty()
            || !self.applied_cuts.2.is_empty()
            || !self.applied_cuts.3.is_empty()
    }
    /// Refreshes the cut inputs. A dual-only change updates the
    /// prepared tables; another change re-prepares them in place. A change
    /// between no active cuts and some builds a new native solver, which in
    /// the C++ engine retains the grid middle, half-way status, dominance
    /// depths, fixed arcs and NG memories, but not the round statistics. A
    /// successful enumeration survives; otherwise it is dropped.
    pub fn refresh_cuts(&mut self) -> Result<()> {
        let mut knapsack = self.knapsack_pricing.clone();
        let resource_indices = (0..self.graph.resources.len()).map(|r| (r, r)).collect();
        knapsack.retrieve(&self.cuts.knapsack_cuts, self.graph.graph_id.max(0), &resource_indices)?;
        if knapsack != self.knapsack_pricing {
            if let Some(engine) = &mut self.engine {
                engine.cuts.knapsack = knapsack.clone();
                engine.half_ready = false;
                engine.full_ready = false;
            }
            self.knapsack_pricing = knapsack;
        }
        let next = state::build_inputs(&self.cuts, &self.graph);
        if next == self.applied_cuts {
            return Ok(());
        }
        let without_duals = |(l, lk, f, fk): &(
            Vec<LimitedInput>,
            Vec<LimitedKPathInput>,
            Vec<FullInput>,
            Vec<FullKPathInput>,
        )| {
            let mut l = l.clone();
            let mut lk = lk.clone();
            let mut f = f.clone();
            let mut fk = fk.clone();
            l.iter_mut().for_each(|c| c.dual = 0.);
            lk.iter_mut().for_each(|c| c.dual = 0.);
            f.iter_mut().for_each(|c| c.dual = 0.);
            fk.iter_mut().for_each(|c| c.dual = 0.);
            (l, lk, f, fk)
        };
        let only_duals = without_duals(&next) == without_duals(&self.applied_cuts);
        let enumerated = self.enumeration.is_some();
        let was_extended = !self.applied_cuts.0.is_empty() || !self.applied_cuts.1.is_empty();
        let needs_extended = !next.0.is_empty() || !next.1.is_empty();
        if let Some(engine) = &mut self.engine {
            if !enumerated && only_duals && was_extended && needs_extended {
                engine.update_cut_duals(next.0.clone(), next.1.clone());
            } else {
                if !engine.load_cuts(&self.graph, next.0.clone(), next.1.clone()) {
                    self.engine = None;
                    self.prepared = None;
                    return Err(invalid("rank-one cut memory exceeds the native label words"));
                }
                if was_extended != needs_extended {
                    self.round = controller::RoundState::default();
                }
            }
        }
        if !enumerated {
            self.enumeration = None;
            self.enumeration_snapshot = None;
        }
        self.applied_cuts = next;
        Ok(())
    }
    fn new_engine(&self, controls: EngineControls, phase: phases::Phase) -> Result<LabelingEngine> {
        let mut engine = LabelingEngine::prepare(&self.graph, controls, phase)?;
        engine.cuts.knapsack = self.knapsack_pricing.clone();
        if !engine.load_cuts(
            &self.graph,
            self.applied_cuts.0.clone(),
            self.applied_cuts.1.clone(),
        ) {
            return Err(invalid("rank-one cut memory exceeds the native label words"));
        }
        Ok(engine)
    }
    pub fn set_option(&mut self, name: &str, value: f64) -> Result<()> {
        self.options.set(name, value)?;
        match name {
            "RCSPhardTimeThresholdInPricing" => {
                self.controls.time_limit = if value <= 0. {
                    Duration::ZERO
                } else {
                    Duration::try_from_secs_f64(value * 50.)
                        .map_err(|_| invalid("pricing time limit"))?
                }
            }
            "RCSPmaxNumOfColsPerExactIteration" => {
                if value < 1. {
                    return Err(invalid("positive exact column limit required"));
                }
                self.controls.max_columns = value as usize;
            }
            "RCSPheurLabelingStrategy" => self.phases = phases::initialize(value as i32, &[])?,
            _ => {}
        }
        self.engine = None;
        Ok(())
    }
    fn engine_controls(&self, exact: bool) -> EngineControls {
        EngineControls {
            bidirectional: self.options.get("RCSPuseBidirectionalSearch").unwrap_or(2.) != 0.,
            bidirectional_mode: self.options.get("RCSPuseBidirectionalSearch").unwrap_or(2.) as i32,
            dominance_mode: self.dominance_mode,
            steps: self.bucket_steps,
            target_buckets: self
                .options
                .get("RCSPnumberOfBucketsPerVertex")
                .unwrap_or(25.) as i32,
            group_limit: if exact {
                0
            } else {
                self.options
                    .get("RCSPmaxNumOfColsPerIteration")
                    .unwrap_or(30.) as i32
            },
            filter_other_buckets: self
                .options
                .get("RCSPcheckDominInOtherBuckets")
                .unwrap_or(1.)
                != 0.,
            ..Default::default()
        }
    }
    pub fn price(&mut self, maximum: usize, negative_only: bool) -> Result<PricingResult> {
        if !negative_only {
            return self.price_phase(0, maximum, false);
        }
        let saved = self.phases.clone();
        let saved_options = self.options.clone();
        let outcome = (|| {
            for (labels, columns) in [(32, 20usize), (128, 50), (i64::MAX, 100)] {
                self.options
                    .set("RCSPmaxNumOfColsPerIteration", columns as f64)?;
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
    fn enumerated_kpath_inputs(&self) -> Vec<LimitedKPathInput> {
        self.applied_cuts.3.iter().map(|cut| LimitedKPathInput {
            dual: cut.dual, arc_memory: false, support: cut.vertex_support.clone(), memory_vertices: vec![],
        }).collect()
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
                .get("RCSPmaxNumOfColsPerIteration")
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
                self.options.get("RCSPhardTimeThresholdInPricing")?.max(0.),
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
                    self.options.get("RCSPdynamicBucketSteps")? != 0.,
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
    pub fn enumerate(&mut self, cutoff: f64) -> Result<Vec<crate::Route>> {
        if self.engine.is_none() {
            self.price_phase(0, 1, false)?;
        }
        self.complete_labels(true)?;
        let controls = enumeration::Controls {
            cutoff,
            label_limit: self.options.get("RCSPmaxNumOfLabelsInEnumeration")?.max(1.) as usize,
            path_limit: self.options.get("RCSPmaxNumOfEnumeratedSolutions")?.max(1.) as usize,
            time_limit: self.controls.time_limit,
            max_route_arcs: self.controls.max_route_arcs,
            // One hash partition,
            // so stored paths keep the native insertion order.
            partitions: 1,
        };
        let has_cuts = self.has_cuts();
        if has_cuts {
            let engine = self.engine.as_mut().unwrap();
            let mut full = state::FullPreparation::default();
            // Completion can materialize backward buckets even when pricing
            // started in forward-only mode. Enumeration uses those buckets.
            let backward = !engine.consumption_joins && !engine.grid.backward.buckets.is_empty();
            if !state::prepare_full_memory(
                &mut engine.cuts.info,
                &mut full,
                &self.graph,
                &self.applied_cuts.2,
                &self.applied_cuts.3,
                backward,
            ) {
                return Err(incomplete("enumeration cut memory exceeds the native label words"));
            }
            engine.cuts.full = Some(full);
        }
        let engine = self.engine.as_ref().unwrap();
        let scan = self.options.get("RCSPuseExactComplBoundsInEnumeration")? != 0.;
        let tables = engine.cuts.full.as_ref().filter(|_| has_cuts).map(|full| enumeration::CutTables {
            info: &engine.cuts.info,
            full,
            knapsack: &engine.cuts.knapsack,
        });
        let universe = Enumeration::run_with_symmetry(
            &self.graph,
            &engine.grid,
            &controls,
            if engine.consumption_joins {
                &engine.reverse
            } else {
                &[]
            },
            tables,
            |forward, label| {
                engine.enumeration_completion(&self.graph, forward, label, cutoff, scan)
            },
        )?;
        let routes = universe.paths.routes(&self.graph)?;
        self.enumeration_snapshot = Some(self.topology());
        self.enumeration = Some(universe);
        Ok(routes)
    }
    pub fn inspect_enumeration(&mut self, cutoff: f64) -> Result<Option<Vec<crate::Route>>> {
        self.refresh_cuts()?;
        let enumerated_kpath = self.enumerated_kpath_inputs();
        if let Some(universe) = &mut self.enumeration {
            universe
                .paths
                .inspect_with_knapsack(&self.graph, cutoff, &self.applied_cuts.0, &enumerated_kpath, &self.knapsack_pricing)?;
            return Ok(Some(universe.paths.routes(&self.graph)?));
        }
        Ok(None)
    }
    pub fn fix(&mut self, threshold: f64) -> Result<fixing::FixingReport> {
        self.complete_labels(true)?;
        self.engine
            .as_mut()
            .ok_or_else(|| invalid("pricing must precede fixing"))?
            .fix_bucket_arcs(&self.graph, threshold)
    }
    /// columnGenerationTerminated  as Alg4EvalByLagrangianDuality
    /// calls it after each converged column generation: the statistics form,
    /// then the end-of-round form that applies a pending bucket-step decrease.
    /// Returns the stop-cut-generation flag of the statistics form.
    pub fn column_generation_terminated(&mut self, flag3: i32, flag5: i32) -> Result<bool> {
        let enumerated = self.enumeration.is_some();
        let grid = self.engine.as_ref().map(|e| &e.grid);
        let (stop, _) = self.round.action(&self.options, grid, enumerated, false, flag3, flag5)?;
        let (_, decrease) = self.round.action(&self.options, grid, enumerated, true, flag3, flag5)?;
        if decrease {
            if let Some(engine) = self.engine.as_mut() {
                // "cannot decrease bucket steps" is reported and ignored.
                let _ = engine.grid.decrease_steps(&self.graph);
                self.bucket_steps = engine.grid.step;
            }
        }
        Ok(stop)
    }
}
impl Solver {
    pub fn complete_labels(&mut self, enforce_time_limit: bool) -> Result<()> {
        if self.accumulated_branches > 0 {
            return Err(invalid(
                "native pricing has no accumulated resource branches",
            ));
        }
        let exact = phases::Phase {
            check_ng: true,
            jump_arcs: true,
            labels_per_bucket: i64::MAX,
        };
        self.refresh_cuts()?;
        if self.engine.is_none() {
            self.engine = Some(self.new_engine(self.engine_controls(true), exact)?);
        }
        let engine = self.engine.as_mut().unwrap();
        let mut controls = self.controls.clone();
        if !enforce_time_limit {
            controls.time_limit = Duration::MAX;
        }
        engine.prepare_full_labels(&self.graph, &controls)
    }
    pub fn restore_arcs(&mut self) {
        self.engine = None;
        self.prepared = None;
        self.enumeration = None;
        self.enumeration_snapshot = None;
    }
}
impl Solver {
    pub fn run_fixing_enumeration(
        &mut self,
        mode: i32,
        zero: f64,
        mut cutoff: f64,
    ) -> Result<fixing::FixingResult> {
        if !(-1..=1).contains(&mode) || !zero.is_finite() || !cutoff.is_finite() {
            return Err(invalid("fixing/enumeration arguments"));
        }
        let mut report = fixing::FixingResult::default();
        if self.options.get("RCSPapplyReducedCostFixing")? == 0. {
            return Ok(report);
        }
        self.refresh_cuts()?;
        if let Some(routes) = self.inspect_enumeration(cutoff)? {
            report.state = 3;
            report.routes = routes.len();
            return Ok(report);
        }
        if mode == 0 {
            let gap = self.options.get("RCSPredCostFixingFalseGap")?;
            if gap > 1. || (gap < -1. && self.has_cuts()) {
                cutoff = (cutoff - zero) / gap.abs() + zero;
            }
        }
        let saved_time = self.controls.time_limit;
        let saved_options = self.options.clone();
        let outcome = (|| {
            if mode == 0 {
                self.controls.time_limit = Duration::try_from_secs_f64(
                    self.options.get("RCSPredCostFixingTimeThreshold")?.max(0.),
                )
                .map_err(|_| invalid("fixing time threshold"))?;
            }
            self.complete_labels(true)?;
            let fixed = self.fix(cutoff)?;
            report.eliminated = fixed.eliminated_forward + fixed.eliminated_backward;
            report.state = 4;
            if mode == -1 {
                return Ok(report);
            }
            let labels = self.options.get(if mode == 0 {
                "RCSPmaxNumOfLabelsInEnumeration"
            } else {
                "RCSPmaxNumOfLabelsInHeurEnumeration"
            })?;
            let paths = self.options.get("RCSPmaxNumOfEnumeratedSolutions")?;
            if labels == 0. || paths == 0. {
                return Ok(report);
            }
            if labels < 0. || paths < 0. {
                return Err(invalid("negative enumeration limit"));
            }
            self.options
                .set("RCSPmaxNumOfLabelsInEnumeration", labels)?;
            match self.enumerate(cutoff) {
                Ok(routes) => {
                    report.state = 2;
                    report.routes = routes.len();
                    Ok(report)
                }
                Err(e) if e.message.contains("budget") || e.message.contains("time limit") => {
                    report.state = 1;
                    Ok(report)
                }
                Err(e) => Err(e),
            }
        })();
        self.controls.time_limit = saved_time;
        self.options = saved_options;
        outcome
    }
}
impl Solver {
    pub fn cache_variable_costs(&mut self, costs: &[f64]) -> Result<()> {
        if costs.iter().any(|v| !v.is_finite())
            || self.maximum_variable_id.is_some_and(|id| id >= costs.len())
        {
            return Err(invalid("variable cost domain"));
        }
        let mut next = vec![];
        for arc in &self.graph.arcs {
            let mut value = arc.cost;
            for &(id, a) in &arc.variables {
                let raw = *costs
                    .get(id)
                    .ok_or_else(|| invalid("variable cost array is too short"))?;
                value += (raw * 1e8 + 0.5).floor() / 1e8 * a;
            }
            if !value.is_finite() {
                return Err(invalid("cached arc cost overflow"));
            }
            next.push(value);
        }
        for (arc, cost) in self.graph.arcs.iter_mut().zip(next) {
            arc.reduced_cost = cost;
        }
        self.variable_costs = costs.to_vec();
        Ok(())
    }
    pub fn corrected_bound(&self, mut bound: f64, route: &crate::Route) -> Result<f64> {
        if self.variable_costs.is_empty() {
            return Ok(bound);
        }
        let mut values = std::collections::BTreeMap::new();
        for &id in &route.arcs {
            let arc = self
                .graph
                .arcs
                .iter()
                .find(|a| a.id == id)
                .ok_or_else(|| invalid("unknown functor route arc"))?;
            for &(id, a) in &arc.variables {
                *values.entry(id).or_insert(0.) += a;
            }
        }
        for (id, value) in values {
            let raw = *self
                .variable_costs
                .get(id)
                .ok_or_else(|| invalid("cached variable cost domain"))?;
            bound += (raw - (raw * 1e8 + 0.5).floor() / 1e8) * value;
        }
        Ok(bound)
    }
}

impl Solver {
    fn topology(&self) -> Graph {
        let mut graph = self.graph.clone();
        for a in &mut graph.arcs {
            a.reduced_cost = 0.;
        }
        graph
    }
}

impl Solver {
    pub fn strengthen_ng(
        &mut self,
        paths: &[crate::graph::relaxation::FractionalPath],
        controls: &crate::graph::relaxation::Controls,
    ) -> Result<crate::graph::relaxation::Counts> {
        let baseline = (
            self.graph
                .vertices
                .iter()
                .map(|v| v.memory.clone())
                .collect(),
            self.graph.arcs.iter().map(|a| a.memory.clone()).collect(),
        );
        let counts = crate::graph::relaxation::strengthen(&mut self.graph, paths, controls)?;
        if counts.columns_forbidden > 0 {
            if self.ng_baseline.is_none() {
                self.ng_baseline = Some(baseline);
            }
            // the solver grows the live elementarity resource: the bucket graph,
            // fixed arcs, half-way point and adaptive depths stay. Only the NG
            // tables, the labels and enumerated paths are invalidated.
            match &mut self.engine {
                Some(engine) => {
                    engine.ng = crate::graph::ng::PreparedNg::new(&self.graph)?;
                    engine.half_ready = false;
                    engine.full_ready = false;
                    self.prepared = Some(self.topology());
                }
                None => self.prepared = None,
            }
            self.enumeration = None;
            self.enumeration_snapshot = None;
        }
        Ok(counts)
    }
    /// Whether a route (vertices and arc ids in path order) is feasible for
    /// the current NG relaxation (the solver satisfiesCurrentRelaxation).
    pub fn route_satisfies_relaxation(&self, vertices: &[i32], arcs: &[i32]) -> Result<bool> {
        if vertices.len() != arcs.len() + 1 {
            return Err(invalid("route node/arc dimensions"));
        }
        let ng = crate::graph::ng::PreparedNg::new(&self.graph)?;
        let Some(&first) = vertices.first() else {
            return Ok(false);
        };
        let mut bits = *usize::try_from(first)
            .ok()
            .and_then(|v| ng.vertex_bits.get(v))
            .ok_or_else(|| invalid("route vertex out of range"))?;
        for (i, &id) in arcs.iter().enumerate() {
            let a = self
                .graph
                .arcs
                .iter()
                .position(|a| a.id == id)
                .ok_or_else(|| invalid("route arc out of range"))?;
            let arc = &self.graph.arcs[a];
            if vertices[i] != arc.tail as i32 || vertices[i + 1] != arc.head as i32 {
                return Ok(false);
            }
            let Some(next) = ng.forward[a].extend(bits, ng.vertex_bits[arc.head]) else {
                return Ok(false);
            };
            bits = next;
        }
        Ok(true)
    }
    pub fn reset_ng(&mut self) -> bool {
        let Some((vertices, arcs)) = self.ng_baseline.take() else {
            return false;
        };
        for (v, memory) in self.graph.vertices.iter_mut().zip(vertices) {
            v.memory = memory;
        }
        for (a, memory) in self.graph.arcs.iter_mut().zip(arcs) {
            a.memory = memory;
        }
        self.restore_arcs();
        true
    }
}
