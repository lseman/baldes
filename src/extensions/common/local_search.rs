//! Feasible local search with route-local replay and versioned pair scans.
use super::validate_plan;
use crate::inputs::Instance;
use std::time::Instant;

const NEIGHBOURS: usize = 50;

/// State after each prefix, including departure time from its last customer.
struct Replay {
    cost: Vec<f64>,
    load: Vec<f64>,
    time: Vec<f64>,
    total: f64,
}
impl Replay {
    fn new(instance: &Instance, route: &[usize]) -> Self {
        let depot = &instance.customers[0];
        let mut result = Self {
            cost: vec![0.],
            load: vec![0.],
            time: vec![depot.ready.max(0.) + depot.service],
            total: 0.,
        };
        let mut previous = 0;
        for (i, &v) in route.iter().enumerate() {
            let c = &instance.customers[v];
            let arc = instance.distance(previous, v, instance.rounding);
            result.cost.push(result.cost[i] + arc);
            result.load.push(result.load[i] + c.demand);
            result
                .time
                .push((result.time[i] + arc).max(c.ready) + c.service);
            previous = v;
        }
        if !route.is_empty() {
            result.total =
                result.cost[route.len()] + instance.distance(previous, 0, instance.rounding);
        }
        result
    }

    /// Reuse the unchanged prefix; replay only the replacement suffix. All
    /// distance lookups remain directed, including reversed route segments.
    fn trial(&self, instance: &Instance, old: &[usize], new: &[usize]) -> Option<f64> {
        if new.is_empty() {
            return Some(0.);
        }
        let prefix = old.iter().zip(new).take_while(|(a, b)| a == b).count();
        let (mut cost, mut load, mut time) =
            (self.cost[prefix], self.load[prefix], self.time[prefix]);
        let mut previous = if prefix == 0 { 0 } else { new[prefix - 1] };
        for &v in &new[prefix..] {
            let c = &instance.customers[v];
            let arc = instance.distance(previous, v, instance.rounding);
            if !arc.is_finite() || arc < 0. {
                return None;
            }
            cost += arc;
            load += c.demand;
            time = (time + arc).max(c.ready);
            if time > c.due + 1e-6 || load > instance.capacity + 1e-6 {
                return None;
            }
            time += c.service;
            previous = v;
        }
        let arc = instance.distance(previous, 0, instance.rounding);
        cost += arc;
        (arc.is_finite()
            && arc >= 0.
            && cost.is_finite()
            && time + arc <= instance.customers[0].due + 1e-6)
            .then_some(cost)
    }
}

/// Symmetric membership built from directed proximity (minimum of both arc
/// costs). Depot boundaries are always eligible, as are small instances.
struct Neighbours(Vec<Vec<usize>>);
impl Neighbours {
    fn new(instance: &Instance, deadline: Instant) -> Option<Self> {
        let n = instance.customers.len();
        let mut lists = Vec::with_capacity(n);
        for u in 0..n {
            if Instant::now() >= deadline {
                return None;
            }
            let customers: Vec<_> = (1..n).filter(|&v| v != u).collect();
            let mut list: Vec<_> = super::nearest_positions(instance, u, &customers, NEIGHBOURS)
                .into_iter()
                .map(|pos| customers[pos])
                .collect();
            list.sort_unstable();
            lists.push(list);
        }
        Some(Self(lists))
    }
    fn contains(&self, u: usize, v: usize) -> bool {
        u == 0
            || v == 0
            || self.0[u].binary_search(&v).is_ok()
            || self.0[v].binary_search(&u).is_ok()
    }
    fn insertion(&self, u: usize, route: &[usize], pos: usize) -> bool {
        self.contains(u, if pos == 0 { 0 } else { route[pos - 1] })
            || self.contains(u, route.get(pos).copied().unwrap_or(0))
    }
}

struct Move {
    a: usize,
    b: usize,
    first: Vec<usize>,
    second: Vec<usize>,
}

fn improving(
    instance: &Instance,
    routes: &[Vec<usize>],
    replay: &[Replay],
    movement: Move,
    active: usize,
) -> Option<Move> {
    let a = movement.a;
    let b = movement.b;
    let first = replay[a].trial(instance, &routes[a], &movement.first)?;
    let (before, after) = if a == b {
        (replay[a].total, first)
    } else {
        let second = replay[b].trial(instance, &routes[b], &movement.second)?;
        // Relocation to an empty slot can increase the fleet when its source
        // remains nonempty. Other operators preserve or reduce the count.
        let next = active - usize::from(!routes[a].is_empty()) - usize::from(!routes[b].is_empty())
            + usize::from(!movement.first.is_empty())
            + usize::from(!movement.second.is_empty());
        if instance.vehicles > 0 && next > instance.vehicles {
            return None;
        }
        (replay[a].total + replay[b].total, first + second)
    };
    (after < before - 1e-8).then_some(movement)
}

/// Instance-bound search data reused across local-search passes. Holding an
/// immutable instance borrow prevents stale neighbours after metric changes.
pub(crate) struct LocalSearch<'a> {
    instance: &'a Instance,
    neighbours: Neighbours,
}

impl<'a> LocalSearch<'a> {
    pub(crate) fn new(instance: &'a Instance, deadline: Instant) -> Option<Self> {
        super::validate_instance(instance).ok()?;
        Some(Self {
            instance,
            neighbours: Neighbours::new(instance, deadline)?,
        })
    }

    pub(crate) fn improve(
        &self,
        routes: &mut [Vec<usize>],
        deadline: Instant,
        initial_changed: &[bool],
    ) {
        improve_with_neighbours(
            self.instance,
            &self.neighbours,
            routes,
            deadline,
            initial_changed,
        );
    }
}

pub(super) fn improve(
    instance: &Instance,
    routes: &mut [Vec<usize>],
    deadline: Instant,
    initial_changed: &[bool],
) {
    if Instant::now() >= deadline {
        return;
    }
    if !initial_changed.iter().any(|&changed| changed) {
        return;
    }
    if let Some(search) = LocalSearch::new(instance, deadline) {
        search.improve(routes, deadline, initial_changed);
    }
}

fn improve_with_neighbours(
    instance: &Instance,
    neighbours: &Neighbours,
    routes: &mut [Vec<usize>],
    deadline: Instant,
    initial_changed: &[bool],
) {
    if Instant::now() >= deadline
        || validate_plan(instance, routes).is_err()
        || !initial_changed.iter().any(|&changed| changed)
    {
        return;
    }
    let mut replay: Vec<_> = routes.iter().map(|r| Replay::new(instance, r)).collect();
    let mut version: Vec<usize> = (0..routes.len())
        .map(|i| usize::from(initial_changed.get(i).copied().unwrap_or(false)))
        .collect();
    // Each entry records the last pair versions for which all moves failed.
    let mut tested = vec![vec![None; routes.len()]; routes.len()];
    loop {
        let active = routes.iter().filter(|r| !r.is_empty()).count();
        let mut chosen = None;
        'pairs: for a in 0..routes.len() {
            for b in a..routes.len() {
                if Instant::now() >= deadline {
                    return;
                }
                if version[a] == 0 && version[b] == 0 {
                    continue;
                }
                let stamp = (version[a], version[b]);
                if tested[a][b] == Some(stamp) {
                    continue;
                }
                // Relocate in both directions, including within a route.
                for (from, to) in [(a, b), (b, a)]
                    .into_iter()
                    .take(if a == b { 1 } else { 2 })
                {
                    for i in 0..routes[from].len() {
                        for pos in 0..=routes[to].len() {
                            if Instant::now() >= deadline {
                                return;
                            }
                            if from == to && (pos == i || pos == i + 1) {
                                continue;
                            }
                            let u = routes[from][i];
                            if !neighbours.insertion(u, &routes[to], pos) {
                                continue;
                            }
                            let mut first = routes[from].clone();
                            first.remove(i);
                            let mut second = if from == to {
                                vec![]
                            } else {
                                routes[to].clone()
                            };
                            if from == to {
                                first.insert(if pos > i { pos - 1 } else { pos }, u);
                            } else {
                                second.insert(pos, u);
                            }
                            if let Some(mv) = improving(
                                instance,
                                routes,
                                &replay,
                                Move {
                                    a: from,
                                    b: to,
                                    first,
                                    second,
                                },
                                active,
                            ) {
                                chosen = Some(mv);
                                break 'pairs;
                            }
                        }
                    }
                }
                for i in 0..routes[a].len() {
                    for j in (if a == b { i + 1 } else { 0 })..routes[b].len() {
                        if Instant::now() >= deadline {
                            return;
                        }
                        let u = routes[a][i];
                        let v = routes[b][j];
                        if !neighbours.contains(u, v) {
                            continue;
                        }
                        for tail in [false, true] {
                            let mut first = routes[a].clone();
                            let mut second = if a == b { vec![] } else { routes[b].clone() };
                            if a == b {
                                if tail {
                                    first[i..=j].reverse();
                                } else {
                                    first.swap(i, j);
                                }
                            } else if tail {
                                let end_a = first.split_off(i + 1);
                                let end_b = second.split_off(j + 1);
                                first.extend(end_b);
                                second.extend(end_a);
                            } else {
                                first[i] = v;
                                second[j] = u;
                            }
                            if let Some(mv) = improving(
                                instance,
                                routes,
                                &replay,
                                Move {
                                    a,
                                    b,
                                    first,
                                    second,
                                },
                                active,
                            ) {
                                chosen = Some(mv);
                                break 'pairs;
                            }
                        }
                        if a == b {
                            continue;
                        }
                        // SWAP*: exchanged customers can use different positions.
                        let mut first = routes[a].clone();
                        first.remove(i);
                        let mut second = routes[b].clone();
                        second.remove(j);
                        for pos_a in 0..=first.len() {
                            if !neighbours.insertion(v, &first, pos_a) {
                                continue;
                            }
                            for pos_b in 0..=second.len() {
                                if Instant::now() >= deadline {
                                    return;
                                }
                                if !neighbours.insertion(u, &second, pos_b) {
                                    continue;
                                }
                                let mut next_a = first.clone();
                                next_a.insert(pos_a, v);
                                let mut next_b = second.clone();
                                next_b.insert(pos_b, u);
                                if let Some(mv) = improving(
                                    instance,
                                    routes,
                                    &replay,
                                    Move {
                                        a,
                                        b,
                                        first: next_a,
                                        second: next_b,
                                    },
                                    active,
                                ) {
                                    chosen = Some(mv);
                                    break 'pairs;
                                }
                            }
                        }
                    }
                }
                tested[a][b] = Some(stamp);
            }
        }
        let Some(mv) = chosen else {
            return;
        };
        // Full replay only on acceptance guards global cost summation and the
        // public feasibility contract; unsuccessful trials never clone a plan.
        let before = super::plan_cost(instance, routes, true).expect("validated search plan");
        let old_a = std::mem::replace(&mut routes[mv.a], mv.first);
        let old_b = if mv.a != mv.b {
            Some(std::mem::replace(&mut routes[mv.b], mv.second))
        } else {
            None
        };
        if !super::plan_cost(instance, routes, true).is_ok_and(|cost| cost < before - 1e-8) {
            routes[mv.a] = old_a;
            if let Some(old) = old_b {
                routes[mv.b] = old;
            }
            return;
        }
        if routes.iter().filter(|r| !r.is_empty()).count() != active {
            // Fleet availability is global: emptying a route can enable a
            // previously rejected split involving an otherwise unchanged pair.
            for row in &mut tested {
                row.fill(None);
            }
        }
        for r in [mv.a, mv.b]
            .into_iter()
            .take(if mv.a == mv.b { 1 } else { 2 })
        {
            replay[r] = Replay::new(instance, &routes[r]);
            version[r] += 1;
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/extensions/common_local_search.rs"]
mod tests;
