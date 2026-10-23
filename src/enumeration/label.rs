//! Enumeration labels and full-memory cut transitions.
use super::*;
#[derive(Clone, Debug)]
pub struct Label<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    pub id: usize,
    pub vertex: usize,
    pub resources: crate::graph::ResourceValues<D>,
    pub reduced_cost: f64,
    pub original_cost: f64,
    pub visited: [u64; 16],
    pub element_hash: u32,
    pub parent: Option<usize>,
    pub arc: Option<usize>,
    pub hops: usize,
    /// Full-memory cut state (`State<FullMemSBCutsResources>`).
    pub cut_words: [u32; FULL_WORDS],
    pub cut_cost: f64,
    pub presence_state: crate::cuts::soft_conflicts::LabelState,
    pub coupled_state: Option<Box<crate::graph::coupled::CoupledLabelState>>,
}
/// Full-memory cut tables of an enumeration with cuts.
#[derive(Clone, Copy, Debug)]
pub struct CutTables<'a> {
    pub info: &'a CutResourceInfo,
    pub full: &'a FullPreparation,
    pub knapsack: &'a crate::cuts::knapsack::KnapsackPricing,
    pub presence: Option<&'a crate::cuts::soft_conflicts::PresencePricing>,
}
impl<'a> CutTables<'a> {
    pub(super) fn along_arc<const D: usize>(
        &self,
        label: &mut Label<D>,
        forward: bool,
        arc: usize,
    ) -> Result<()> {
        if let Some(presence) = self.presence.filter(|p| p.active()) {
            let (next, cost) = presence.extend(&label.presence_state, arc)?;
            label.presence_state = next;
            label.cut_cost += cost;
        }
        let table = if forward {
            &self.info.forward
        } else {
            &self.info.backward
        };
        for record in &table[arc].full_memory {
            let mut transition = *record;
            // Preparation retains the observed binary descriptor layout. Its
            // k-path exit descriptor uses the word index as the destination
            // bit shift, which can overwrite a neighboring rank-one field.
            // Enumeration must preserve the visitation bit in its own field.
            if transition.covering
                && transition.denominator == -2
                && transition.increment == 1
                && transition.target_word >= 0
            {
                transition.target_shift = transition.source_shift;
            }
            state::change_full_memory_along_arc(
                &mut label.cut_words,
                &mut label.cut_cost,
                std::slice::from_ref(&transition),
            );
        }
        Ok(())
    }
    pub(super) fn to_vertex<const D: usize>(&self, label: &mut Label<D>) {
        state::apply_vertex_support(
            &mut label.cut_words,
            &mut label.cut_cost,
            &self.info.vertices[label.vertex].full_memory_support,
        );
    }
    /// enumeration_labels_can_be_concatenated cost part, at the forward vertex.
    pub(super) fn join<const D: usize>(
        &self,
        g: &Graph<D>,
        symmetric: bool,
        f: &Label<D>,
        b: &Label<D>,
        cost: f64,
    ) -> Result<f64> {
        let cost = cost
            + self
                .knapsack
                .join_cost(g, &f.resources, &b.resources, symmetric);
        let cost = state::full_memory_concatenation_cost(
            self.info,
            self.full,
            f.vertex,
            (&f.cut_words, f.cut_cost),
            (&b.cut_words, b.cut_cost),
            cost,
        );
        if let Some(presence) = self.presence.filter(|p| p.active()) {
            Ok(cost + presence.join_cost(&f.presence_state, &b.presence_state)?)
        } else {
            Ok(cost)
        }
    }
}
impl<const D: usize> Label<D> {
    /// Copy the non-coupled fields before replacing the resource state with
    /// a physical extension. Avoid cloning a frontier that would be discarded.
    pub(super) fn clone_without_coupled(&self) -> Self {
        // Keep this exhaustive: adding a label field must update this copy.
        let Self {
            id,
            vertex,
            resources,
            reduced_cost,
            original_cost,
            visited,
            element_hash,
            parent,
            arc,
            hops,
            cut_words,
            cut_cost,
            presence_state,
            coupled_state: _,
        } = self;
        Self {
            id: *id,
            vertex: *vertex,
            resources: *resources,
            reduced_cost: *reduced_cost,
            original_cost: *original_cost,
            visited: *visited,
            element_hash: *element_hash,
            parent: *parent,
            arc: *arc,
            hops: *hops,
            cut_words: *cut_words,
            cut_cost: *cut_cost,
            presence_state: presence_state.clone(),
            coupled_state: None,
        }
    }

    /// Enumeration bucket key, ((0.5 potential + additional) + reduced) + 0.
    pub fn key(&self) -> f64 {
        ((0.0 + self.cut_cost) + self.reduced_cost) + 0.0
    }
    pub fn key_with(&self, potential: f64) -> f64 {
        ((0.5 * potential + self.cut_cost) + self.reduced_cost) + 0.0
    }
    /// Test vertex elementarity without adding it to the completion projection.
    pub(crate) fn can_add_elements(&self, elements: &[usize]) -> bool {
        let mut visited = self.visited;
        for &element in elements {
            if element >= 1024 {
                return false;
            }
            let bit = 1u64 << (element % 64);
            let word = &mut visited[element / 64];
            if *word & bit != 0 {
                return false;
            }
            *word |= bit;
        }
        true
    }
    pub fn add_elements(&mut self, elements: &[usize]) -> bool {
        for &e in elements {
            if e >= 1024 {
                return false;
            }
            let word = &mut self.visited[e / 64];
            let bit = 1u64 << (e % 64);
            if *word & bit != 0 {
                return false;
            }
            *word |= bit;
            let x = e as u32;
            self.element_hash = self
                .element_hash
                .wrapping_add(x.wrapping_mul(x).wrapping_mul(x));
        }
        true
    }
}
