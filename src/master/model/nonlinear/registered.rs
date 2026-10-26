//! Normalize existing cut-registry descriptors to one oracle's physical graph.
use super::*;

pub(super) fn normalize<const D: usize>(input: &RankOneCut, g: &Graph<D>) -> Result<RankOneCut> {
    let n = g.vertices.len();
    let graph = g.graph_id;
    // These are the actual packed pricing-kernel limits, not a dynamic scalar
    // resource fallback. Reject before lim_memory_mask can index its table.
    if !(1..8).contains(&input.denominator) {
        return Err(invalid(
            "registered cut denominator must be between one and seven",
        ));
    }
    if input.general_kpath_support.is_some() {
        return Err(invalid(
            "original-set k-path mapping requires an explicit physical descriptor",
        ));
    }
    if input.visit_once && (input.memory_kind == MemoryKind::Arc || !input.arc_weights.is_empty()) {
        return Err(invalid(
            "strong k-path pricing requires vertex memory and support",
        ));
    }
    if input.graph_support.is_empty() && input.weights.len() != n {
        return Err(invalid("registered cut vertex dimensions"));
    }
    for support in [&input.graph_support, &input.graph_pricing_support] {
        if let Some(values) = support.get(&graph) {
            if values.iter().any(|&(v, _)| v >= n) || values.windows(2).any(|w| w[0].0 >= w[1].0) {
                return Err(invalid("sorted unique physical cut support required"));
            }
        }
    }
    let mut cut = input.clone();
    cut.weights = (0..n).map(|v| input.weight_for_graph(graph, v)).collect();
    if (0..n).any(|v| cut.weights[v] != input.pricing_weight_for_graph(graph, v)) {
        return Err(invalid("registered row and pricing supports must agree"));
    }
    if cut.weights.iter().any(|&w| w < 0 || w >= cut.denominator)
        || cut.weights[g.source] != 0
        || cut.weights[g.sink] != 0
    {
        return Err(invalid(
            "registered cut requires nonnegative single-carry weights and zero terminal weights",
        ));
    }
    if cut
        .arc_weights
        .iter()
        .any(|(&id, &w)| w < 0 || w >= cut.denominator || !g.arcs.iter().any(|a| a.id == id))
    {
        return Err(invalid("registered cut physical arc weights"));
    }
    let present = !input.graph_dependent_memory
        || match input.memory_kind {
            MemoryKind::Vertex => input.graph_memories.contains_key(&graph),
            MemoryKind::Arc => input.graph_arc_memories.contains_key(&graph),
        };
    cut.memory = if present {
        input.memory_for_graph(graph)
    } else {
        vec![false; n]
    };
    cut.arc_memory = if present {
        input.arc_memory_for_graph(graph)
    } else {
        Default::default()
    };
    if !present {
        cut.weights.fill(0);
        cut.arc_weights.clear();
    }
    if cut.memory_kind == MemoryKind::Vertex
        && (cut.memory.len() != n
            || cut
                .weights
                .iter()
                .zip(&cut.memory)
                .any(|(&w, &m)| w > 0 && !m))
    {
        return Err(invalid(
            "vertex cut memory must include its weighted support",
        ));
    }
    if cut
        .arc_memory
        .iter()
        .any(|id| !g.arcs.iter().any(|a| a.id == *id))
    {
        return Err(invalid("registered cut memory contains an unknown arc"));
    }
    // Registry coefficient evaluation and label preparation now see exactly
    // the same physical mapping. Other graphs retain independent descriptors.
    cut.graph_support.clear();
    cut.graph_pricing_support.clear();
    cut.graph_dependent_memory = false;
    cut.graph_memories.clear();
    cut.graph_arc_memories.clear();
    if cut.type2 {
        cut.class = CutClass::Packing;
    }
    let mut support: Vec<_> = cut
        .weights
        .iter()
        .enumerate()
        .filter_map(|(v, &w)| (w != 0).then_some((v, w)))
        .collect();
    support.extend(cut.arc_weights.iter().map(|(&id, &w)| (n + id as usize, w)));
    cut.ordered_support = Some(support);
    cut.dual = 0.;
    Ok(cut)
}
