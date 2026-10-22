//! Directed joint-presence walk bounds, with sparse event-state compression.
use super::*;
use crate::cuts::soft_conflicts::{Mode, PricingInput};

struct Work<'a> {
    remaining: &'a mut usize,
    deadline: Instant,
}
impl Work<'_> {
    fn take(&mut self, count: usize) -> Option<()> {
        if Instant::now() >= self.deadline {
            return None;
        }
        *self.remaining = self.remaining.checked_sub(count)?;
        Some(())
    }
}
struct Row {
    pairs: Vec<(usize, usize)>,
    mode: Mode,
    dual: f64,
}
impl Row {
    fn coefficient(&self, bits: &[u64], work: &mut Work<'_>) -> Option<usize> {
        let mut count = 0;
        for &(a, b) in &self.pairs {
            work.take(1)?;
            if bits[a / 64] & (1 << (a % 64)) != 0 && bits[b / 64] & (1 << (b % 64)) != 0 {
                count += 1;
                if self.mode == Mode::AnyPair {
                    break;
                }
            }
        }
        Some(count)
    }
}

/// Every resource-feasible original walk appears in this product graph. Rows
/// share one state but keep separate event namespaces. Directed edge costs
/// charge each row's actual coefficient increment exactly once. Exhaustion or
/// an unbounded adverse cycle returns unknown, never a partial minimum.
pub(super) fn minimum<const D: usize>(
    graph: &Graph<D>,
    weights: &[f64],
    inputs: &[(&PricingInput, f64)],
    blocked: impl Fn(usize) -> bool,
    remaining: &mut usize,
    deadline: Instant,
) -> Option<Option<f64>> {
    let mut work = Work {
        remaining,
        deadline,
    };
    work.take(graph.vertices.len().checked_add(graph.arcs.len())?)?;
    if weights.len() != graph.arcs.len() || weights.iter().any(|w| !w.is_finite()) {
        return None;
    }
    let mut positions = HashMap::new();
    let mut rows = vec![];
    for (i, &(input, dual)) in inputs.iter().enumerate() {
        if !dual.is_finite() || input.arc_events.len() != graph.arcs.len() {
            return None;
        }
        let mut pairs = vec![];
        for &(left, right) in input.row.pairs() {
            work.take(3)?;
            let next = positions.len();
            let left = *positions.entry((i, left)).or_insert(next);
            let next = positions.len();
            let right = *positions.entry((i, right)).or_insert(next);
            pairs.push((left, right));
        }
        rows.push(Row {
            pairs,
            mode: input.row.mode(),
            dual,
        });
    }
    let words = positions.len() / 64 + usize::from(positions.len() % 64 != 0);
    let mut emissions = vec![];
    let mut outgoing = vec![vec![]; graph.vertices.len()];
    for (a, arc) in graph.arcs.iter().enumerate() {
        work.take(words.checked_add(inputs.len())?.checked_add(1)?)?;
        let mut bits = vec![0_u64; words];
        for (i, &(input, _)) in inputs.iter().enumerate() {
            for event in &input.arc_events[a] {
                work.take(1)?;
                if let Some(&b) = positions.get(&(i, *event)) {
                    bits[b / 64] |= 1 << (b % 64);
                }
            }
        }
        emissions.push(bits);
        if !blocked(a) {
            outgoing[arc.tail].push(a);
        }
    }
    work.take(words.checked_mul(3)?.checked_add(1)?)?;
    let root = (graph.source, vec![0_u64; words]);
    let mut indices = HashMap::from([(root.clone(), 0usize)]);
    let mut states = vec![root];
    let mut edges = vec![];
    let mut cursor = 0;
    while cursor < states.len() {
        work.take(words.checked_add(1)?)?;
        let (vertex, bits) = states[cursor].clone();
        if vertex != graph.sink {
            for &arc in &outgoing[vertex] {
                work.take(words.checked_mul(5)?.checked_add(1)?)?;
                let mut next = bits.clone();
                for (value, emission) in next.iter_mut().zip(&emissions[arc]) {
                    *value |= emission;
                }
                let mut weight = weights[arc];
                for row in &rows {
                    let before = row.coefficient(&bits, &mut work)?;
                    let after = row.coefficient(&next, &mut work)?;
                    let delta = after.checked_sub(before)?;
                    if delta as u128 > (1_u128 << 53) {
                        return None;
                    }
                    weight = subtract_product(weight, row.dual, delta as f64)?;
                }
                let key = (graph.arcs[arc].head, next);
                let count = states.len();
                let target = *indices.entry(key.clone()).or_insert_with(|| {
                    states.push(key);
                    count
                });
                edges.push((cursor, target, weight));
            }
        }
        cursor += 1;
    }
    shortest(graph, &states, &edges, &mut work)
}

fn shortest<const D: usize>(
    graph: &Graph<D>,
    states: &[(usize, Vec<u64>)],
    edges: &[(usize, usize, f64)],
    work: &mut Work<'_>,
) -> Option<Option<f64>> {
    work.take(states.len().checked_mul(3)?.checked_add(edges.len())?)?;
    let mut incoming = vec![vec![]; states.len()];
    for &(tail, head, _) in edges {
        incoming[head].push(tail);
    }
    let mut reaches_sink = vec![false; states.len()];
    let mut queue = vec![];
    for (i, (vertex, _)) in states.iter().enumerate() {
        if *vertex == graph.sink {
            reaches_sink[i] = true;
            queue.push(i);
        }
    }
    while let Some(head) = queue.pop() {
        work.take(1)?;
        for &tail in &incoming[head] {
            work.take(1)?;
            if !reaches_sink[tail] {
                reaches_sink[tail] = true;
                queue.push(tail);
            }
        }
    }
    if !reaches_sink[0] {
        return Some(None);
    }
    let hops = super::resource::hop_limit(graph, *work.remaining);
    let mut previous = vec![None; states.len()];
    previous[0] = Some(0.);
    for pass in 0..hops.unwrap_or(states.len()) {
        work.take(states.len().checked_mul(2)?.checked_add(edges.len())?)?;
        let mut next = previous.clone();
        let mut changed = false;
        for &(tail, head, weight) in edges {
            if !reaches_sink[head] {
                continue;
            }
            if let Some(value) = previous[tail] {
                let candidate = add(value, weight)?;
                if next[head].map_or(true, |old| candidate < old) {
                    next[head] = Some(candidate);
                    changed = true;
                }
            }
        }
        previous = next;
        if !changed {
            break;
        }
        if hops.is_none() && pass + 1 == states.len() {
            return None;
        }
    }
    Some(
        states
            .iter()
            .zip(previous)
            .filter_map(|((vertex, _), value)| (*vertex == graph.sink).then_some(value).flatten())
            .min_by(f64::total_cmp),
    )
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/multi_pricing/bounds/presence/tests.rs"]
mod tests;
