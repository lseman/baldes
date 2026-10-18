//! Enumeration labels and full-memory cut transitions.
use super::*;
#[derive(Clone, Debug)]
pub struct Label {
    pub id: usize,
    pub vertex: usize,
    pub resources: [f64; 2],
    pub reduced_cost: f64,
    pub original_cost: f64,
    pub visited: [u64; 16],
    pub element_hash: u32,
    pub parent: Option<usize>,
    pub arc: Option<usize>,
    pub hops: usize,
    /// Full-memory cut state (State<FullMemSBCutsResources>).
    pub cut_words: [u32; FULL_WORDS],
    pub cut_cost: f64,
}
/// Full-memory cut tables of an enumeration with cuts.
#[derive(Clone, Copy, Debug)]
pub struct CutTables<'a> {
    pub info: &'a CutResourceInfo,
    pub full: &'a FullPreparation,
    pub knapsack: &'a crate::cuts::knapsack::KnapsackPricing,
}
impl<'a> CutTables<'a> {
    pub(super) fn along_arc(&self, label: &mut Label, forward: bool, arc: usize) {
        let table = if forward {
            &self.info.forward
        } else {
            &self.info.backward
        };
        state::change_full_memory_along_arc(
            &mut label.cut_words,
            &mut label.cut_cost,
            &table[arc].full_memory,
        );
    }
    pub(super) fn to_vertex(&self, label: &mut Label) {
        state::apply_vertex_support(
            &mut label.cut_words,
            &mut label.cut_cost,
            &self.info.vertices[label.vertex].full_memory_support,
        );
    }
    /// enumeration_labels_can_be_concatenated cost part, at the forward vertex.
    pub(super) fn join(&self, g: &Graph, symmetric: bool, f: &Label, b: &Label, cost: f64) -> f64 {
        let cost = cost
            + self
                .knapsack
                .join_cost(g, &f.resources, &b.resources, symmetric);
        state::full_memory_concatenation_cost(
            self.info,
            self.full,
            f.vertex,
            (&f.cut_words, f.cut_cost),
            (&b.cut_words, b.cut_cost),
            cost,
        )
    }
}
impl Label {
    /// Enumeration bucket key, ((0.5 potential + additional) + reduced) + 0.
    pub fn key(&self) -> f64 {
        ((0.0 + self.cut_cost) + self.reduced_cost) + 0.0
    }
    pub fn key_with(&self, potential: f64) -> f64 {
        ((0.5 * potential + self.cut_cost) + self.reduced_cost) + 0.0
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
