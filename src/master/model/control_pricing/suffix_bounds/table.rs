//! Hop-indexed physical suffix recurrence.
use super::*;

pub(in super::super) fn suffix_table<const D: usize>(
    graph: &Graph<D>,
    arc_costs: &[f64],
    hops: usize,
    work: &mut usize,
    deadline: Instant,
) -> Option<SuffixTable> {
    hops.checked_add(1)?;
    // Validate the sink before allocating: an empty vertex set would make
    // the cell-count check zero even for an enormous requested row count.
    if graph.sink >= graph.vertices.len() {
        return None;
    }
    // Bound stored cells by the initial work budget, allocating rows as they
    // are computed. A stable row represents every later hop count, so a
    // large declared cap does not itself require a large allocation.
    let allocation_limit = *work;
    if graph.vertices.len() > allocation_limit {
        return None;
    }
    let mut suffix = vec![vec![None; graph.vertices.len()]];
    *suffix[0].get_mut(graph.sink)? = Some(0.);
    for remaining in 1..=hops {
        if Instant::now() >= deadline {
            return None;
        }
        if (remaining + 1).checked_mul(graph.vertices.len())? > allocation_limit {
            return None;
        }
        suffix.push(suffix.last()?.clone());
        let (before, current) = suffix.split_at_mut(remaining);
        let previous = &before[remaining - 1];
        let current = &mut current[0];
        for (index, arc) in graph.arcs.iter().enumerate() {
            if index % 256 == 0 && Instant::now() >= deadline {
                return None;
            }
            *work = work.checked_sub(1)?;
            let Some(tail) = previous.get(arc.head).copied().flatten() else {
                continue;
            };
            let candidate = crate::core::rounding::add_down(*arc_costs.get(index)?, tail)?;
            let value = current.get_mut(arc.tail)?;
            *value = Some(value.map_or(candidate, |old: f64| old.min(candidate)));
        }
        // The next row is a deterministic function of this row, with the
        // same arc order and directed arithmetic. Once consecutive rows are
        // bit-identical, every later row is identical too. This matters for
        // bounded cyclic walks whose relaxation stabilizes well before the
        // hop cap. Compare bits so signed zeros cannot change unnoticed.
        let fixed = remaining < hops
            && previous
                .iter()
                .zip(current.iter())
                .all(|(a, b)| match (a, b) {
                    (None, None) => true,
                    (Some(a), Some(b)) => a.to_bits() == b.to_bits(),
                    _ => false,
                });
        if fixed {
            return Some(SuffixTable {
                rows: suffix,
                hops,
                stable: true,
            });
        }
    }
    Some(SuffixTable {
        rows: suffix,
        hops,
        stable: false,
    })
}
