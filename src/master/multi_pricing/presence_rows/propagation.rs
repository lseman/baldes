//! Exact zero-conflict walk support, bounded independently of route resources.
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

/// Arcs absent from every zero-conflict source-to-sink walk. Ignoring resources,
/// elementarity and other rows produces a superset of feasible routes. No
/// support conclusion is published unless both traversals finish. A finite
/// state graph merges equal incidence at the same physical vertex, so cycles
/// and repeated event emissions do not require a path-length bound.
pub(in crate::master::multi_pricing) fn zero_arcs<const D: usize>(
    graph: &Graph<D>,
    input: &PricingInput,
    blocked: impl Fn(usize) -> bool,
    budget: usize,
    deadline: Instant,
) -> (Option<Vec<usize>>, usize) {
    let mut work = Work {
        used: 0,
        limit: budget,
        deadline,
    };
    let result = support(graph, input, blocked, &mut work);
    (result, work.used)
}

fn support<const D: usize>(
    graph: &Graph<D>,
    input: &PricingInput,
    blocked: impl Fn(usize) -> bool,
    work: &mut Work,
) -> Option<Vec<usize>> {
    // Compress only events occurring in pairs. Declared event IDs can be large
    // and sparse; neither memory nor work scales with the declared domain.
    let mut positions = HashMap::new();
    let mut pairs = vec![];
    for &(a, b) in input.row.pairs() {
        work.take(3)?;
        let next = positions.len();
        let a = *positions.entry(a).or_insert(next);
        let next = positions.len();
        let b = *positions.entry(b).or_insert(next);
        pairs.push((a, b));
    }
    let words = positions.len() / 64 + usize::from(positions.len() % 64 != 0);
    work.take(graph.vertices.len().saturating_add(graph.arcs.len()))?;
    let mut outgoing = vec![vec![]; graph.vertices.len()];
    let mut emissions = Vec::with_capacity(graph.arcs.len());
    for (a, arc) in graph.arcs.iter().enumerate() {
        work.take(words.saturating_add(1))?;
        let mut bits = vec![0u64; words];
        for event in &input.arc_events[a] {
            work.take(1)?;
            if let Some(&index) = positions.get(event) {
                bits[index / 64] |= 1 << (index % 64);
            }
        }
        emissions.push(bits);
        if !blocked(a) {
            outgoing[arc.tail].push(a);
        }
    }
    work.take(words.saturating_mul(2).saturating_add(1))?;
    let root = (graph.source, vec![0u64; words]);
    let mut indices = HashMap::from([(root.clone(), 0usize)]);
    let mut states = vec![root];
    let mut incoming: Vec<Vec<(usize, usize)>> = vec![vec![]];
    let mut cursor = 0;
    while cursor < states.len() {
        work.take(words.saturating_add(1))?;
        let (vertex, incidence) = states[cursor].clone();
        if vertex != graph.sink {
            for &arc in &outgoing[vertex] {
                work.take(words.saturating_mul(2).saturating_add(1))?;
                let mut next = incidence.clone();
                for (word, emission) in next.iter_mut().zip(&emissions[arc]) {
                    *word |= emission;
                }
                let mut conflict = false;
                for &(left, right) in &pairs {
                    work.take(1)?;
                    if next[left / 64] & (1 << (left % 64)) != 0
                        && next[right / 64] & (1 << (right % 64)) != 0
                    {
                        conflict = true;
                        break;
                    }
                }
                if conflict {
                    continue;
                }
                work.take(words.saturating_mul(3).saturating_add(1))?;
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
#[path = "../../../../tests/unit/master/multi_pricing/presence_rows/propagation/tests.rs"]
mod tests;
