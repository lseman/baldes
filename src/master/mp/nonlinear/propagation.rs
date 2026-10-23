//! Original walk support for exhausted nonnegative registered-cut rows.
use super::*;

struct Work {
    used: usize,
    limit: usize,
    deadline: Instant,
}
impl Work {
    fn take(&mut self, count: usize) -> Option<()> {
        if Instant::now() >= self.deadline || count > self.limit - self.used {
            return None;
        }
        self.used += count;
        Some(())
    }
}

/// A nonnegative sum is zero only when every kernel is zero. Keep the joint
/// packing remainders while allowing all walks, ignoring resources and
/// elementarity. An unsupported arc in this superset cannot occur in a
/// positive-quantity feasible route. No partial traversal publishes exclusions.
pub(in crate::master::mp) fn zero_arcs<const D: usize>(
    graph: &Graph<D>,
    cuts: &[&RankOneCut],
    blocked: impl Fn(usize) -> bool,
    budget: usize,
    deadline: Instant,
) -> (Option<Vec<usize>>, usize) {
    let mut work = Work {
        used: 0,
        limit: budget,
        deadline,
    };
    let result = support(graph, cuts, blocked, &mut work);
    (result, work.used)
}

fn support<const D: usize>(
    graph: &Graph<D>,
    cuts: &[&RankOneCut],
    blocked: impl Fn(usize) -> bool,
    work: &mut Work,
) -> Option<Vec<usize>> {
    // Installed descriptors are normalized to this physical graph. Decline a
    // future unsupported representation rather than interpret it as zero.
    for cut in cuts {
        work.take(1)?;
        if !(1..8).contains(&cut.denominator)
            || cut.weights.len() != graph.vertices.len()
            || cut.graph_dependent_memory
            || !cut.graph_support.is_empty()
            || !cut.graph_pricing_support.is_empty()
            || cut.native_kpath_support.is_some()
            || (cut.visit_once && !cut.arc_weights.is_empty())
        {
            return None;
        }
        for &weight in cut.weights.iter().chain(cut.arc_weights.values()) {
            work.take(1)?;
            if weight < 0 || weight >= cut.denominator {
                return None;
            }
        }
    }
    work.take(graph.vertices.len().saturating_add(graph.arcs.len()))?;
    let mut outgoing = vec![vec![]; graph.vertices.len()];
    // Every remainder transition can be cached per physical arc and kernel.
    // A negative entry means the coefficient becomes positive on that step.
    let mut transitions = vec![];
    for (a, arc) in graph.arcs.iter().enumerate() {
        work.take(cuts.len().saturating_add(1))?;
        let mut kernels = vec![];
        for cut in cuts {
            let vertex = cut.weights[arc.head];
            let edge = cut.arc_weights.get(&arc.id).copied().unwrap_or(0);
            let hit_only = cut.visit_once || (!cut.native_type2 && cut.class == CutClass::Covering);
            let kept = match cut.memory_kind {
                MemoryKind::Vertex => cut.memory.get(arc.tail).copied().unwrap_or(false),
                MemoryKind::Arc => cut.arc_memory.contains(&arc.id),
            };
            let mut table = vec![];
            for remainder in 0..cut.denominator {
                work.take(1)?;
                let next = if hit_only {
                    // Zero covering/entry coefficient means no support was hit
                    // at all. Memory resets cannot erase a previous charge.
                    if vertex != 0 || edge != 0 {
                        -1
                    } else {
                        0
                    }
                } else {
                    let next = if kept { remainder } else { 0 } + vertex + edge;
                    if next >= cut.denominator {
                        -1
                    } else {
                        next
                    }
                };
                table.push(next as i8);
            }
            kernels.push(table);
        }
        transitions.push(kernels);
        if !blocked(a) {
            outgoing[arc.tail].push(a);
        }
    }
    work.take(cuts.len().saturating_add(1))?;
    let root = (graph.source, vec![0u8; cuts.len()]);
    let mut indices = HashMap::from([(root.clone(), 0usize)]);
    let mut states = vec![root];
    let mut incoming: Vec<Vec<(usize, usize)>> = vec![vec![]];
    let mut cursor = 0;
    while cursor < states.len() {
        work.take(cuts.len().saturating_add(1))?;
        let (vertex, remainders) = states[cursor].clone();
        if vertex != graph.sink {
            for &arc in &outgoing[vertex] {
                work.take(cuts.len().saturating_mul(3).saturating_add(1))?;
                let mut next = Vec::with_capacity(cuts.len());
                for (&remainder, table) in remainders.iter().zip(&transitions[arc]) {
                    let value = table[remainder as usize];
                    if value < 0 {
                        break;
                    }
                    next.push(value as u8);
                }
                if next.len() != cuts.len() {
                    continue;
                }
                let key = (graph.arcs[arc].head, next);
                let count = states.len();
                let target = *indices.entry(key.clone()).or_insert_with(|| {
                    states.push(key);
                    incoming.push(vec![]);
                    count
                });
                incoming[target].push((cursor, arc));
            }
        }
        cursor += 1;
    }
    work.take(states.len().saturating_add(graph.arcs.len()))?;
    let mut reaches_sink = vec![false; states.len()];
    let mut queue = vec![];
    let mut viable = vec![false; graph.arcs.len()];
    for (i, (vertex, _)) in states.iter().enumerate() {
        work.take(1)?;
        if *vertex == graph.sink {
            reaches_sink[i] = true;
            queue.push(i);
        }
    }
    while let Some(target) = queue.pop() {
        work.take(1)?;
        for &(parent, arc) in &incoming[target] {
            work.take(1)?;
            viable[arc] = true;
            if !reaches_sink[parent] {
                reaches_sink[parent] = true;
                queue.push(parent);
            }
        }
    }
    work.take(viable.len())?;
    Some(
        viable
            .into_iter()
            .enumerate()
            .filter_map(|(a, used)| (!used).then_some(a))
            .collect(),
    )
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/mp/nonlinear/propagation/tests.rs"]
mod tests;
