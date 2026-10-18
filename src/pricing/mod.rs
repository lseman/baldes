//! Stateful pricing, cut preparation, and retained elementary route universe.
pub mod branching;
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
    pub(crate) ryan_foster: Vec<branching::RyanFosterConstraint>,
    pub controls: Controls,
    pub options: Options,
    pub phases: Vec<phases::Phase>,
    pub engine: Option<LabelingEngine>,
    pub enumeration: Option<Enumeration>,
    /// Preserve arc orderings needed by current or future arc-flow branches.
    pub enumeration_preserve_arc_paths: bool,
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
    /// Original knapsack resource ID -> solver coordinate. Empty is identity.
    pub knapsack_resource_indices: std::collections::BTreeMap<usize, usize>,
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
            ryan_foster: vec![],
            controls: Controls::default(),
            options: Options::default(),
            phases: phases::initialize(0, &[])?,
            engine: None,
            enumeration: None,
            enumeration_preserve_arc_paths: false,
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
            knapsack_resource_indices: Default::default(),
            knapsack_pricing: Default::default(),
            applied_cuts: (vec![], vec![], vec![], vec![]),
        })
    }
    /// Independent node state: keep fixing, NG, cuts, options and complete
    /// enumerated paths, while rebuilding labels at the child's duals.
    /// Descendants must retain the incumbent bound used for fixing. Enable
    /// `enumeration_preserve_arc_paths` before enumeration for arc-flow branching.
    #[must_use]
    pub fn fork_for_node(&self) -> Self {
        Self {
            graph: self.graph.clone(),
            ryan_foster: self.ryan_foster.clone(),
            controls: self.controls.clone(),
            options: self.options.clone(),
            phases: self.phases.clone(),
            engine: self.engine.as_ref().map(LabelingEngine::fork_for_node),
            enumeration: self.enumeration.as_ref().map(|e| Enumeration {
                labels: enumeration::Storage::new(e.labels.partitions.len()).unwrap(),
                paths: e.paths.clone(),
                completed: e.completed,
                generated: e.generated,
            }),
            enumeration_preserve_arc_paths: self.enumeration_preserve_arc_paths
                || !self.ryan_foster.is_empty(),
            cutoff: self.cutoff,
            prepared: self.prepared.clone(),
            bucket_steps: self.bucket_steps,
            rollback_point_saved: self.rollback_point_saved,
            requires_rollback: false,
            round: self.round.clone(),
            maximum_variable_id: self.maximum_variable_id,
            dominance_mode: self.dominance_mode,
            accumulated_branches: self.accumulated_branches,
            ng_baseline: self.ng_baseline.clone(),
            variable_costs: self.variable_costs.clone(),
            enumeration_snapshot: self.enumeration_snapshot.clone(),
            cuts: self.cuts.clone(),
            knapsack_resource_indices: self.knapsack_resource_indices.clone(),
            knapsack_pricing: self.knapsack_pricing.clone(),
            applied_cuts: self.applied_cuts.clone(),
        }
    }
    pub fn ryan_foster_constraints(&self) -> &[branching::RyanFosterConstraint] {
        &self.ryan_foster
    }
    /// Install the current node's complete branch list. Fixed bucket arcs are
    /// valid for descendants; dual-dependent labels and enumeration filters refresh.
    pub fn set_ryan_foster(
        &mut self,
        constraints: Vec<branching::RyanFosterConstraint>,
    ) -> Result<()> {
        let prepared = branching::RyanFosterResources::prepare(&self.graph, &constraints)?;
        if constraints != self.ryan_foster {
            if !self.ryan_foster.iter().all(|c| constraints.contains(c)) {
                self.restore_arcs();
            }
            if let Some(engine) = &mut self.engine {
                engine.branching = prepared;
                engine.half_ready = false;
                engine.full_ready = false;
            }
            self.ryan_foster = constraints;
            if let Some(e) = &mut self.enumeration {
                let g = &self.graph;
                let rules = &self.ryan_foster;
                e.paths.retain(|p| {
                    rules.iter().all(|c| {
                        c.satisfied_by(
                            std::iter::once(g.source).chain(p.arcs.iter().map(|&a| g.arcs[a].head)),
                        )
                    })
                });
            }
        }
        Ok(())
    }
    pub fn knapsack_indices(&self) -> std::collections::BTreeMap<usize, usize> {
        if self.knapsack_resource_indices.is_empty() {
            (0..self.graph.resources.len()).map(|r| (r, r)).collect()
        } else {
            self.knapsack_resource_indices.clone()
        }
    }
    pub fn knapsack_coefficient(
        &self,
        cut: &crate::cuts::knapsack::ResourceKnapsackCut,
        resources: &[f64],
    ) -> i32 {
        if self.knapsack_resource_indices.is_empty() {
            cut.coefficient(self.graph.graph_id.max(0), resources)
        } else {
            cut.coefficient_mapped(
                self.graph.graph_id.max(0),
                resources,
                &self.knapsack_resource_indices,
            )
        }
    }
    pub fn has_cuts(&self) -> bool {
        self.knapsack_pricing.active
            || !self.applied_cuts.0.is_empty()
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
        let resource_indices = self.knapsack_indices();
        if resource_indices
            .values()
            .any(|&r| r >= self.graph.resources.len())
        {
            return Err(invalid(
                "knapsack resource mapping exceeds solver coordinates",
            ));
        }
        knapsack.retrieve(
            &self.cuts.knapsack_cuts,
            self.graph.graph_id.max(0),
            &resource_indices,
        )?;
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
                    return Err(invalid(
                        "rank-one cut memory exceeds the native label words",
                    ));
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
        engine.branching = branching::RyanFosterResources::prepare(&self.graph, &self.ryan_foster)?;
        engine.cuts.knapsack = self.knapsack_pricing.clone();
        if !engine.load_cuts(
            &self.graph,
            self.applied_cuts.0.clone(),
            self.applied_cuts.1.clone(),
        ) {
            return Err(invalid(
                "rank-one cut memory exceeds the native label words",
            ));
        }
        Ok(engine)
    }
    pub fn set_option(&mut self, name: &str, value: f64) -> Result<()> {
        let changed = self.options.get(name)? != value;
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
        // Runtime budgets and column limits do not change the bucket graph.
        // Keep inherited fixing when a node installs its own work budget.
        if changed
            && !matches!(
                name,
                "RCSPhardTimeThresholdInPricing"
                    | "RCSPmaxNumOfColsPerExactIteration"
                    | "RCSPmaxNumOfColsPerIteration"
                    | "RCSPheurLabelingStrategy"
            )
        {
            self.engine = None;
        }
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
    /// columnGenerationTerminated  as Alg4EvalByLagrangianDuality
    /// calls it after each converged column generation: the statistics form,
    /// then the end-of-round form that applies a pending bucket-step decrease.
    /// Returns the stop-cut-generation flag of the statistics form.
    pub fn column_generation_terminated(&mut self, flag3: i32, flag5: i32) -> Result<bool> {
        let enumerated = self.enumeration.is_some();
        let grid = self.engine.as_ref().map(|e| &e.grid);
        let (stop, _) = self
            .round
            .action(&self.options, grid, enumerated, false, flag3, flag5)?;
        let (_, decrease) =
            self.round
                .action(&self.options, grid, enumerated, true, flag3, flag5)?;
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
    fn topology(&self) -> Graph {
        let mut graph = self.graph.clone();
        for a in &mut graph.arcs {
            a.reduced_cost = 0.;
        }
        graph
    }
}

mod completion;
mod fixing_workflow;
mod relaxation;
mod solve;
mod universe;
mod variable_costs;
