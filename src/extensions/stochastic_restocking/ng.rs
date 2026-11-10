//! Adaptive NG walk relaxation recovered from the former scalar prepricing.
//!
//! Only completed sweeps contribute bounds. Unlike the old numerical pricing
//! channel, this implementation rounds every transition downward, so its
//! minimum can also be used by the independent master bound certificate.
use super::RestockingResource;
use crate::core::rounding::{add_down, multiply_down};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

type Key = (usize, u64);
#[derive(Clone)]
struct Walk {
    cost: f64,
    path: Vec<usize>,
}

fn sweep(
    resource: &RestockingResource,
    outgoing: &[Vec<(usize, f64)>],
    memories: &[u64],
    work: &mut usize,
    deadline: Instant,
) -> Option<Walk> {
    let p = &resource.problem;
    // Positive rounded mean loads give an acyclic sweep. Zero-load walks
    // remain covered by the separate q-route fallback, never by a partial NG sweep.
    let capacity = p.capacity().max(1) as f64;
    let scale = (512. / capacity).min(1.);
    let limit = (capacity * scale).ceil() as usize;
    let units: Vec<_> = std::iter::once(0)
        .chain(
            p.demands()
                .iter()
                .map(|d| multiply_down(d.mean(), scale).unwrap_or(0.).max(0.).floor() as usize),
        )
        .collect();
    if units[1..].contains(&0) {
        return None;
    }
    let mut layers: Vec<BTreeMap<Key, Walk>> = vec![BTreeMap::new(); limit + 1];
    layers[0].insert(
        (0, 0),
        Walk {
            cost: 0.,
            path: vec![],
        },
    );
    let mut best: Option<Walk> = None;
    let mut states = 0;
    for load in 0..=limit {
        let layer = std::mem::take(&mut layers[load]);
        for ((last, memory), walk) in layer {
            for &(j, weight) in &outgoing[last] {
                *work = work.checked_sub(1)?;
                if *work % 1024 == 0 && Instant::now() >= deadline {
                    return None;
                }
                let cost = add_down(walk.cost, weight)?;
                if j == 0 {
                    if best.as_ref().is_none_or(|old| cost < old.cost) {
                        best = Some(Walk {
                            cost,
                            path: walk.path.clone(),
                        });
                    }
                    continue;
                }
                if memory & (1 << (j - 1)) != 0 || load + units[j] > limit {
                    continue;
                }
                let key = (j, (memory & memories[j]) | (1 << (j - 1)));
                let next = &mut layers[load + units[j]];
                if next.get(&key).is_none_or(|old| cost < old.cost) {
                    let mut path = walk.path.clone();
                    path.push(j);
                    if next.insert(key, Walk { cost, path }).is_none() {
                        states += 1;
                    }
                    if states > 100_000 {
                        return None;
                    }
                }
            }
        }
    }
    (Instant::now() < deadline).then_some(best).flatten()
}

pub(super) fn minimum(
    resource: &RestockingResource,
    costs: &[f64],
    work: &mut usize,
    deadline: Instant,
) -> Option<f64> {
    let p = &resource.problem;
    let n = p.demands().len();
    if n == 0 || n > 63 || costs.len() != resource.arcs.len() {
        return None;
    }
    let deadline = deadline.min(Instant::now().checked_add(Duration::from_millis(200))?);
    let envelope = super::certificate::objective_bounds(resource, n + 1, work, Some(deadline))?[0];
    let mut outgoing = vec![vec![]; n + 1];
    for (&(i, j), &cost) in resource.arcs.iter().zip(costs) {
        if !cost.is_finite() {
            return None;
        }
        outgoing[i].push((j, cost));
    }
    let mut memories = vec![0; n + 1];
    for (i, memory) in memories.iter_mut().enumerate().skip(1) {
        let mut nearest: Vec<_> = (1..=n).collect();
        nearest.sort_by(|&a, &b| p.costs()[i][a].total_cmp(&p.costs()[i][b]).then(a.cmp(&b)));
        *memory = 1 << (i - 1);
        for j in nearest.into_iter().take(2) {
            *memory |= 1 << (j - 1);
        }
    }
    let mut retained: Option<f64> = None;
    for _ in 0..8 {
        let Some(walk) = sweep(resource, &outgoing, &memories, work, deadline) else {
            break;
        };
        let bound = add_down(walk.cost, envelope)?;
        retained = Some(retained.map_or(bound, |old| old.max(bound)));
        // DSSR: retain the repeated customer throughout the segment that
        // forgot it. Larger memories shrink the relaxation monotonically.
        let mut changed = false;
        for (position, &j) in walk.path.iter().enumerate() {
            if let Some(before) = walk.path[..position].iter().rposition(|&v| v == j) {
                for &v in &walk.path[before..position] {
                    if memories[v].count_ones() < 15 && memories[v] & (1 << (j - 1)) == 0 {
                        memories[v] |= 1 << (j - 1);
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    retained
}

#[cfg(test)]
#[path = "../../../tests/unit/extensions/restocking_ng.rs"]
mod tests;
