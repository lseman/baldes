//! Immutable source shapes for refreshing prices without rebuilding inputs.
use super::*;
use crate::cuts::RankOneCut;
use std::sync::Arc;

type Inputs = (
    Vec<LimitedInput>,
    Vec<LimitedKPathInput>,
    Vec<FullInput>,
    Vec<FullKPathInput>,
);

#[derive(Debug)]
pub(crate) struct CutInputSource {
    graph: i32,
    vertices: usize,
    exact_duals: bool,
    cuts: Vec<(usize, RankOneCut)>,
    active: Vec<bool>,
    pricing_order: Vec<usize>,
    retired: BTreeSet<usize>,
    rows: [Vec<usize>; 4],
}

fn active(registry: &CutRegistry, index: usize, exact_duals: bool) -> bool {
    !registry.retired.contains(&index)
        && !(registry.cuts[index].dual.abs() <= if exact_duals { 0. } else { 1e-6 })
}

fn price(cut: &RankOneCut, exact_duals: bool) -> f64 {
    if cut.visit_once && !exact_duals {
        (-cut.dual * 1e8 + 0.5).floor() / 1e8
    } else {
        -cut.dual
    }
}

fn same_cut_shape(a: &RankOneCut, b: &RankOneCut) -> bool {
    // Exhaustive destructuring prevents future source fields being silently
    // omitted. Changes to unused fields conservatively rebuild the snapshot.
    let RankOneCut {
        weights,
        graph_support,
        graph_pricing_support,
        native_kpath_support,
        memory,
        graph_dependent_memory,
        graph_memories,
        memory_kind,
        arc_memory,
        graph_arc_memories,
        arc_weights,
        visit_once,
        native_type2,
        denominator,
        dual: _,
        class,
        rhs,
        kind,
        id,
        ordered_support,
    } = a;
    weights == &b.weights
        && graph_support == &b.graph_support
        && graph_pricing_support == &b.graph_pricing_support
        && native_kpath_support == &b.native_kpath_support
        && memory == &b.memory
        && graph_dependent_memory == &b.graph_dependent_memory
        && graph_memories == &b.graph_memories
        && memory_kind == &b.memory_kind
        && arc_memory == &b.arc_memory
        && graph_arc_memories == &b.graph_arc_memories
        && arc_weights == &b.arc_weights
        && visit_once == &b.visit_once
        && native_type2 == &b.native_type2
        && denominator == &b.denominator
        && class == &b.class
        && rhs == &b.rhs
        && kind == &b.kind
        && id == &b.id
        && ordered_support == &b.ordered_support
}

pub(crate) fn build_inputs_cached<const D: usize>(
    registry: &CutRegistry,
    g: &Graph<D>,
    exact_duals: bool,
) -> (Inputs, Arc<CutInputSource>) {
    let inputs = if exact_duals {
        build_inputs_exact(registry, g)
    } else {
        build_inputs(registry, g)
    };
    let graph = g.graph_id.max(0);
    let mut order = registry.pricing_order.clone();
    let mut ordered = vec![false; registry.cuts.len()];
    for &index in &order {
        ordered[index] = true;
    }
    order.extend((0..registry.cuts.len()).filter(|&index| !ordered[index]));
    let mut rows: [Vec<usize>; 4] = std::array::from_fn(|_| Vec::new());
    for index in order {
        if !active(registry, index, exact_duals) {
            continue;
        }
        let cut = &registry.cuts[index];
        if cut.visit_once {
            if cut.native_kpath_support.is_some()
                || !cut.graph_dependent_memory
                || cut.graph_memories.contains_key(&graph)
            {
                rows[1].push(index);
            }
            rows[3].push(index);
        } else {
            rows[0].push(index);
            rows[2].push(index);
        }
    }
    debug_assert_eq!(rows[0].len(), inputs.0.len());
    debug_assert_eq!(rows[1].len(), inputs.1.len());
    debug_assert_eq!(rows[2].len(), inputs.2.len());
    debug_assert_eq!(rows[3].len(), inputs.3.len());
    let source = CutInputSource {
        graph,
        vertices: g.vertices.len(),
        exact_duals,
        cuts: registry
            .cuts
            .iter()
            .enumerate()
            .filter(|&(index, _)| active(registry, index, exact_duals))
            .map(|(index, cut)| (index, cut.clone()))
            .collect(),
        active: (0..registry.cuts.len())
            .map(|index| active(registry, index, exact_duals))
            .collect(),
        pricing_order: registry.pricing_order.clone(),
        retired: registry.retired.clone(),
        rows,
    };
    (inputs, Arc::new(source))
}

impl CutInputSource {
    pub(crate) fn matches<const D: usize>(
        &self,
        registry: &CutRegistry,
        g: &Graph<D>,
        exact_duals: bool,
    ) -> bool {
        self.graph == g.graph_id.max(0)
            && self.vertices == g.vertices.len()
            && self.exact_duals == exact_duals
            && self.pricing_order == registry.pricing_order
            && self.retired == registry.retired
            && self.active.len() == registry.cuts.len()
            && self
                .active
                .iter()
                .enumerate()
                .all(|(index, &before)| before == active(registry, index, exact_duals))
            && self
                .cuts
                .iter()
                .all(|(index, before)| same_cut_shape(before, &registry.cuts[*index]))
    }

    /// Requires a matching source snapshot. Returns None before any mutation
    /// if the owned input layout no longer matches its source-index maps.
    pub(crate) fn refresh_duals(
        &self,
        registry: &CutRegistry,
        inputs: &mut Inputs,
    ) -> Option<bool> {
        if self.rows[0].len() != inputs.0.len()
            || self.rows[1].len() != inputs.1.len()
            || self.rows[2].len() != inputs.2.len()
            || self.rows[3].len() != inputs.3.len()
            || self
                .rows
                .iter()
                .flatten()
                .any(|&index| index >= registry.cuts.len())
        {
            return None;
        }
        let mut changed = false;
        macro_rules! refresh {
            ($lane:tt) => {
                for (row, &index) in inputs.$lane.iter_mut().zip(&self.rows[$lane]) {
                    let dual = price(&registry.cuts[index], self.exact_duals);
                    changed |= row.dual != dual;
                    row.dual = dual;
                }
            };
        }
        refresh!(0);
        refresh!(1);
        refresh!(2);
        refresh!(3);
        Some(changed)
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/cuts/shared/state/source_tests.rs"]
mod tests;
