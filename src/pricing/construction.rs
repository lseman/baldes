//! Solver construction and retained-state forks.
use super::*;

impl<const D: usize> Solver<D> {
    /// Install direct coupled label resources and invalidate objective-dependent
    /// labels, fixing and enumeration. Physical arc and vertex IDs are retained.
    pub fn set_coupled_resources(
        &mut self,
        resources: &[crate::graph::coupled::CoupledResource],
    ) -> Result<()> {
        self.graph.set_coupled_resources(resources)?;
        self.restore_arcs();
        Ok(())
    }
    pub fn new(vertices: usize, resources: usize) -> Result<Self> {
        Ok(Self {
            graph: Graph::new(vertices, resources)?,
            forbidden_arcs: vec![],
            ryan_foster: vec![],
            controls: Controls::default(),
            options: Options::default(),
            phases: phases::initialize(0, &[])?,
            engine: None,
            enumeration: None,
            enumeration_preserve_arc_paths: false,
            cutoff: f64::INFINITY,
            prepared: None,
            bucket_steps: [0.; D].into(),
            rollback_point_saved: false,
            requires_rollback: false,
            round: controller::RoundState::default(),
            maximum_variable_id: None,
            dominance_mode: 0,
            accumulated_branches: vec![],
            accumulated_original_bounds: vec![],
            ng_baseline: None,
            variable_costs: vec![],
            raw_arc_costs: None,
            arc_cost_scratch: vec![],
            raw_pricing: None,
            enumeration_snapshot: None,
            cuts: CutRegistry::default(),
            knapsack_resource_indices: Default::default(),
            knapsack_exact_coefficients: false,
            exact_cut_duals: false,
            knapsack_pricing: Default::default(),
            presence_pricing: Default::default(),
            applied_cuts: (vec![], vec![], vec![], vec![]),
            cut_input_source: None,
        })
    }
    /// Independent node state: keep fixing, NG, cuts, options and complete
    /// enumerated paths, while rebuilding labels at the child's duals.
    /// Descendants must retain the incumbent bound used for fixing. Enable
    /// `enumeration_preserve_arc_paths` before enumeration for arc-flow branching.
    #[must_use]
    pub fn fork_for_node(&self) -> Self {
        self.fork_pricing(true)
    }
    /// Drop cost-dependent working state without copying potentially large
    /// retained path stores. Logical graph and branch restrictions still copy.
    pub(super) fn fork_pricing(&self, keep_cost_state: bool) -> Self {
        Self {
            graph: self.graph.clone(),
            forbidden_arcs: self.forbidden_arcs.clone(),
            ryan_foster: self.ryan_foster.clone(),
            controls: self.controls.clone(),
            options: self.options.clone(),
            phases: self.phases.clone(),
            engine: if keep_cost_state {
                self.engine.as_ref().map(LabelingEngine::fork_for_node)
            } else {
                None
            },
            enumeration: if keep_cost_state {
                self.enumeration.as_ref().map(|e| Enumeration {
                    labels: enumeration::Storage::new(e.labels.partitions.len()).unwrap(),
                    paths: e.paths.clone(),
                    completed: e.completed,
                    generated: e.generated,
                    pruning: e.pruning.clone(),
                })
            } else {
                None
            },
            enumeration_preserve_arc_paths: self.enumeration_preserve_arc_paths
                || !self.ryan_foster.is_empty(),
            cutoff: self.cutoff,
            prepared: if keep_cost_state {
                self.prepared.clone()
            } else {
                None
            },
            bucket_steps: self.bucket_steps,
            rollback_point_saved: self.rollback_point_saved,
            requires_rollback: false,
            round: self.round.clone(),
            maximum_variable_id: self.maximum_variable_id,
            dominance_mode: self.dominance_mode,
            accumulated_branches: self.accumulated_branches.clone(),
            accumulated_original_bounds: self.accumulated_original_bounds.clone(),
            ng_baseline: self.ng_baseline.clone(),
            // Raw validation calls price_phase directly: its arc objectives
            // are built from the parent's costs, so this vector is unused.
            variable_costs: if keep_cost_state {
                self.variable_costs.clone()
            } else {
                vec![]
            },
            raw_arc_costs: if keep_cost_state {
                self.raw_arc_costs.clone()
            } else {
                None
            },
            arc_cost_scratch: vec![],
            raw_pricing: None,
            enumeration_snapshot: if keep_cost_state {
                self.enumeration_snapshot.clone()
            } else {
                None
            },
            cuts: self.cuts.clone(),
            knapsack_resource_indices: self.knapsack_resource_indices.clone(),
            knapsack_exact_coefficients: self.knapsack_exact_coefficients,
            exact_cut_duals: self.exact_cut_duals,
            knapsack_pricing: self.knapsack_pricing.clone(),
            presence_pricing: self.presence_pricing.clone(),
            applied_cuts: self.applied_cuts.clone(),
            cut_input_source: self.cut_input_source.clone(),
        }
    }
}
