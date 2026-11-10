//! Bounded route improvement before expensive coupled exact pricing.
use super::*;
use std::collections::BTreeSet;
use std::time::Instant;

const REPLAYS_PER_STEP: usize = 48;
const MAX_STEPS: usize = 64;

impl<const D: usize> Solver<D> {
    pub(super) fn search_route_columns(
        &self,
        maximum: usize,
        budget: Duration,
    ) -> Option<PricingResult> {
        if maximum == 0
            || self.heuristic_seeds.is_empty()
            || self.graph.vertices.len() > 65
            || !self.accumulated_branches.is_empty()
            || !self.accumulated_original_bounds.is_empty()
            || self.knapsack_pricing.active
            || self.presence_pricing.active()
        {
            return None;
        }
        let deadline = Instant::now().checked_add(budget)?;
        let width = self.graph.vertices.len();
        let mut arcs = vec![None; width.checked_mul(width)?];
        for (index, arc) in self.graph.arcs.iter().enumerate() {
            if self.forbidden_arcs.get(index).copied().unwrap_or(false) {
                continue;
            }
            let entry = &mut arcs[arc.tail * width + arc.head];
            if entry.is_none_or(|old: usize| arc.reduced_cost < self.graph.arcs[old].reduced_cost) {
                *entry = Some(index);
            }
        }
        let path = |customers: &[usize]| -> Option<Vec<usize>> {
            let mut positions = Vec::with_capacity(customers.len() + 1);
            let mut from = self.graph.source;
            for to in customers
                .iter()
                .copied()
                .chain(std::iter::once(self.graph.sink))
            {
                if to >= width {
                    return None;
                }
                positions.push(arcs[from * width + to]?);
                from = to;
            }
            Some(positions)
        };
        let scalar = |customers: &[usize]| -> Option<f64> {
            let mut resources = [0.; D];
            for (r, value) in resources
                .iter_mut()
                .enumerate()
                .take(self.graph.resources.len())
            {
                *value = self.graph.initial[r]
                    .unwrap_or(self.graph.vertices[self.graph.source].bounds[r].lower);
            }
            let mut value = 0.;
            let mut from = self.graph.source;
            for to in customers
                .iter()
                .copied()
                .chain(std::iter::once(self.graph.sink))
            {
                if to >= width {
                    return None;
                }
                let arc = &self.graph.arcs[arcs[from * width + to]?];
                value += arc.reduced_cost;
                for (r, q) in resources
                    .iter_mut()
                    .enumerate()
                    .take(self.graph.resources.len())
                {
                    *q = crate::labeling::extend_resource(&self.graph, arc, r, *q, true)?;
                }
                from = to;
            }
            value.is_finite().then_some(value)
        };
        let replay = |customers: &[usize]| -> Option<crate::Route> {
            if customers.is_empty() {
                return None;
            }
            let positions = path(customers)?;
            let ids: Vec<_> = positions.iter().map(|&p| self.graph.arcs[p].id).collect();
            let mut route = self.graph.elementary_route(&ids).ok()??;
            // Public route replay evaluates the original objective. Add the
            // current arc-price difference explicitly before pricing row cuts.
            route.cost = positions.iter().fold(route.original_cost, |cost, &p| {
                let arc = &self.graph.arcs[p];
                cost + arc.reduced_cost - arc.cost
            });
            if self
                .ryan_foster
                .iter()
                .any(|b| !b.satisfied_by(customers.iter().copied()))
                || self
                    .arc_set_branches
                    .iter()
                    .any(|b| !b.satisfied_by(ids.iter().copied()))
            {
                return None;
            }
            let physical: Vec<_> = positions
                .iter()
                .map(|&p| {
                    let arc = &self.graph.arcs[p];
                    (arc.tail, arc.head, arc.id)
                })
                .collect();
            for (index, cut) in self.cuts.cuts.iter().enumerate() {
                if cut.dual == 0. || self.cuts.retired.contains(&index) {
                    continue;
                }
                let coefficient = self
                    .cuts
                    .route_coefficient(&self.graph, index, &physical, false)
                    .ok()?;
                route.cost -= cut.dual * f64::from(coefficient);
            }
            route.cost.is_finite().then_some(route)
        };
        let mut seeds: Vec<_> = self
            .heuristic_seeds
            .iter()
            .filter_map(|r| Some((scalar(r)?, r.clone())))
            .collect();
        seeds.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut visited = BTreeSet::new();
        let mut found = BTreeSet::new();
        let mut columns = vec![];
        for (_, mut current) in seeds.into_iter().take(32) {
            if Instant::now() >= deadline || columns.len() >= maximum {
                break;
            }
            if !visited.insert(current.clone()) {
                continue;
            }
            let Some(initial) = replay(&current) else {
                continue;
            };
            let mut value = initial.cost;
            for _ in 0..MAX_STEPS {
                if Instant::now() >= deadline || columns.len() >= maximum {
                    break;
                }
                let mut candidates: Vec<_> =
                    neighbors(&current, width, self.graph.source, self.graph.sink)
                        .into_iter()
                        .filter_map(|r| Some((scalar(&r)?, r)))
                        .collect();
                candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
                let mut best: Option<(f64, Vec<usize>)> = None;
                for (_, candidate) in candidates.into_iter().take(REPLAYS_PER_STEP) {
                    if Instant::now() >= deadline || columns.len() >= maximum {
                        break;
                    }
                    if !visited.insert(candidate.clone()) {
                        continue;
                    }
                    let Some(route) = replay(&candidate) else {
                        continue;
                    };
                    let cost = route.cost;
                    if cost < self.controls.zero_threshold - crate::labeling::EPSILON
                        && found.insert(route.arcs.clone())
                    {
                        columns.push(route);
                    }
                    if best.as_ref().is_none_or(|(old, _)| cost < *old) {
                        best = Some((cost, candidate));
                    }
                }
                match best {
                    Some((cost, route)) if cost < value - crate::labeling::EPSILON => {
                        value = cost;
                        current = route;
                    }
                    _ => break,
                }
            }
        }
        if columns.is_empty() {
            return None;
        }
        columns.sort_by(|a, b| a.cost.total_cmp(&b.cost));
        Some(PricingResult {
            columns,
            minimum_reduced_cost: 0.,
            exact: false,
            statistics: Default::default(),
        })
    }
}

fn neighbors(route: &[usize], width: usize, source: usize, sink: usize) -> Vec<Vec<usize>> {
    let outside: Vec<_> = (0..width)
        .filter(|v| *v != source && *v != sink && !route.contains(v))
        .collect();
    let len = route.len();
    let mut out = vec![];
    for &customer in &outside {
        for position in 0..=len {
            let mut next = route.to_vec();
            next.insert(position, customer);
            out.push(next);
        }
    }
    for position in 0..len {
        if len > 1 {
            let mut next = route.to_vec();
            next.remove(position);
            out.push(next);
        }
        for &customer in &outside {
            let mut next = route.to_vec();
            next[position] = customer;
            out.push(next);
        }
        for target in (0..len).filter(|&t| t != position) {
            let mut next = route.to_vec();
            let customer = next.remove(position);
            next.insert(target, customer);
            out.push(next);
        }
        for end in position + 2..=len {
            let mut next = route.to_vec();
            next[position..end].reverse();
            out.push(next);
        }
    }
    out
}

#[cfg(test)]
#[path = "../../tests/unit/pricing/route_search_tests.rs"]
mod tests;
