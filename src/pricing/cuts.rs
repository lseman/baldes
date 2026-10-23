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
        let exact_knapsack = self.knapsack_exact_coefficients
            || self
                .cuts
                .knapsack_cuts
                .iter()
                .any(|cut| cut.cut.kind == crate::cuts::knapsack::KnapsackKind::Fenchel);
        if self.knapsack_pricing.refresh_cached(
            &self.cuts.knapsack_cuts,
            self.graph.graph_id.max(0),
            &self.knapsack_resource_indices,
            self.graph.resources.len(),
            exact_knapsack,
        )? {
            if let Some(engine) = &mut self.engine {
                engine.cuts.knapsack = self.knapsack_pricing.clone();
                engine.half_ready = false;
                engine.full_ready = false;
            }
        }
        if self.cuts.cuts.is_empty()
            && self.cuts.pricing_order.is_empty()
            && self.applied_cuts.0.is_empty()
            && self.applied_cuts.1.is_empty()
            && self.applied_cuts.2.is_empty()
            && self.applied_cuts.3.is_empty()
        {
            self.cut_input_source = None;
            return Ok(());
        }
        if let Some(source) = self.cut_input_source.as_ref().filter(|source| {
            source.matches(&self.cuts, &self.graph, self.exact_cut_duals)
                && self.engine.as_ref().map_or(true, |engine| {
                    engine.cut_structure_matches(&self.applied_cuts.0, &self.applied_cuts.1)
                })
        }) {
            if let Some(changed) = source.refresh_duals(&self.cuts, &mut self.applied_cuts) {
                if changed {
                    if let Some(engine) = &mut self.engine {
                        engine
                            .update_cut_duals_borrowed(&self.applied_cuts.0, &self.applied_cuts.1);
                    }
                }
                return Ok(());
            }
        }
        let (next, source) =
            state::build_inputs_cached(&self.cuts, &self.graph, self.exact_cut_duals);
        if next == self.applied_cuts {
            self.cut_input_source = Some(source);
            return Ok(());
        }
        let only_duals = next.0.len() == self.applied_cuts.0.len()
            && next.1.len() == self.applied_cuts.1.len()
            && next.2.len() == self.applied_cuts.2.len()
            && next.3.len() == self.applied_cuts.3.len()
            && next
                .0
                .iter()
                .zip(&self.applied_cuts.0)
                .all(|(a, b)| a.same_structure(b))
            && next
                .1
                .iter()
                .zip(&self.applied_cuts.1)
                .all(|(a, b)| a.same_structure(b))
            && next
                .2
                .iter()
                .zip(&self.applied_cuts.2)
                .all(|(a, b)| a.same_structure(b))
            && next
                .3
                .iter()
                .zip(&self.applied_cuts.3)
                .all(|(a, b)| a.same_structure(b));
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
        self.cut_input_source = Some(source);
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

#[cfg(test)]
#[path = "../../tests/unit/pricing/cut_cache_tests.rs"]
mod cache_tests;
