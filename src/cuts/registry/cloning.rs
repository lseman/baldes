//! Reuse owned registry buffers while preserving independent pricing state.
use super::*;

impl Clone for CutRegistry {
    fn clone(&self) -> Self {
        let mut copy = Self::default();
        copy.clone_from(self);
        copy
    }

    fn clone_from(&mut self, source: &Self) {
        // Raw pricing refreshes this independent registry every logical call.
        // Reuse the cut/order buffers and endpoint hash table across refreshes.
        let Self {
            cuts,
            knapsack_cuts,
            retired,
            pricing_order,
            round_duals,
            parameters,
            separation_phase,
            covering_sets_present,
            covering_search_rng,
            ends,
        } = source;
        self.cuts.truncate(cuts.len());
        for (target, source) in self.cuts.iter_mut().zip(cuts) {
            clone_cut_into(source, target);
        }
        self.cuts.extend_from_slice(&cuts[self.cuts.len()..]);
        self.knapsack_cuts.clone_from(knapsack_cuts);
        self.retired.clone_from(retired);
        self.pricing_order.clone_from(pricing_order);
        self.round_duals = *round_duals;
        self.parameters.clone_from(parameters);
        self.separation_phase = *separation_phase;
        self.covering_sets_present = *covering_sets_present;
        self.covering_search_rng.clone_from(covering_search_rng);
        let source_ends = ends.borrow();
        let target_ends = self.ends.get_mut();
        target_ends.0 = source_ends.0;
        target_ends.1.clone_from(&source_ends.1);
        target_ends.2.clone_from(&source_ends.2);
    }
}

// Refresh an independent raw-pricing copy without allocating new dense
// weight/memory buffers for every unchanged cut. Destructure exhaustively so
// adding a descriptor field requires updating the copy contract.
fn clone_cut_into(source: &RankOneCut, target: &mut RankOneCut) {
    let RankOneCut {
        weights,
        graph_support,
        graph_pricing_support,
        general_kpath_support,
        memory,
        graph_dependent_memory,
        graph_memories,
        memory_kind,
        arc_memory,
        graph_arc_memories,
        arc_weights,
        graph_arc_weights,
        visit_once,
        type2,
        denominator,
        dual,
        class,
        rhs,
        kind,
        id,
        ordered_support,
    } = source;
    target.weights.clone_from(weights);
    target.graph_support.clone_from(graph_support);
    target
        .graph_pricing_support
        .clone_from(graph_pricing_support);
    target
        .general_kpath_support
        .clone_from(general_kpath_support);
    target.memory.clone_from(memory);
    target.graph_memories.clone_from(graph_memories);
    target.arc_memory.clone_from(arc_memory);
    target.graph_arc_memories.clone_from(graph_arc_memories);
    target.arc_weights.clone_from(arc_weights);
    target.graph_arc_weights.clone_from(graph_arc_weights);
    target.ordered_support.clone_from(ordered_support);
    target.graph_dependent_memory = *graph_dependent_memory;
    target.memory_kind = *memory_kind;
    target.visit_once = *visit_once;
    target.type2 = *type2;
    target.denominator = *denominator;
    target.dual = *dual;
    target.class = *class;
    target.rhs = *rhs;
    target.kind = *kind;
    target.id = *id;
}
