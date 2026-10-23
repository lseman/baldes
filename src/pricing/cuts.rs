//! Cut coefficients and labeling-engine preparation.
use super::*;

impl<const D: usize> Solver<D> {
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
        self.presence_pricing.active()
            || self.knapsack_pricing.active
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
        self.invalidate_changed_graph();
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
        if self.knapsack_exact_coefficients
            || self
                .cuts
                .knapsack_cuts
                .iter()
                .any(|cut| cut.cut.kind == crate::cuts::knapsack::KnapsackKind::Fenchel)
        {
            knapsack.retrieve_exact(
                &self.cuts.knapsack_cuts,
                self.graph.graph_id.max(0),
                &resource_indices,
            )?;
        } else {
            knapsack.retrieve(
                &self.cuts.knapsack_cuts,
                self.graph.graph_id.max(0),
                &resource_indices,
            )?;
        }
        if knapsack != self.knapsack_pricing {
            if let Some(engine) = &mut self.engine {
                engine.cuts.knapsack = knapsack.clone();
                engine.half_ready = false;
                engine.full_ready = false;
            }
            self.knapsack_pricing = knapsack;
        }
        let next = if self.exact_cut_duals {
            state::build_inputs_exact(&self.cuts, &self.graph)
        } else {
            state::build_inputs(&self.cuts, &self.graph)
        };
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
    pub(super) fn new_engine(
        &self,
        controls: EngineControls<D>,
        phase: phases::Phase,
    ) -> Result<LabelingEngine<D>> {
        let mut controls = controls;
        // A one-sided arc fixing need not preserve reverse-arc symmetry.
        if self.forbidden_arcs.iter().any(|&b| b) {
            controls.use_symmetric_execution = false;
        }
        let mut engine = LabelingEngine::prepare(&self.graph, controls, phase)?;
        engine.forbidden_arcs = self.forbidden_arcs.clone();
        engine.branching =
            restrictions::RyanFosterResources::prepare(&self.graph, &self.ryan_foster)?;
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
        engine.load_presence(&self.graph, self.presence_pricing.inputs().to_vec())?;
        Ok(engine)
    }
}
