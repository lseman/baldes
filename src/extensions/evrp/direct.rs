//! Reference bidirectional, elementary direct pricing for the static EV model.
//! Travel cost is the objective; customer and fleet duals are supplied by a master.
//! Exact energy-domain dominance, joins and reconstruction use the same algebra
//! as fixed-order charging. This is not a reproduction of E1's accelerations.

use super::functions::{Piece, Workspace};
use super::{nonnegative, ChargingSolution, ChargingStatus, LocationKind, Problem, Visit};
use crate::{invalid, Result};

#[derive(Clone, Copy, Debug)]
pub struct DirectOptions {
    /// Maximum number of travel arcs in a complete route. Station repeats count.
    pub max_arcs: usize,
    /// Total labels attempted across both directions, including their roots.
    pub max_labels: usize,
    pub max_work: usize,
    pub dominance: bool,
}

impl Default for DirectOptions {
    fn default() -> Self {
        Self {
            max_arcs: 16,
            max_labels: 20_000,
            max_work: 2_000_000,
            dominance: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvPricingStatus {
    /// All routes under the model were covered, in floating-point arithmetic.
    Complete,
    /// All routes within max_arcs were covered; longer routes remain unproved.
    Bounded,
    WorkLimit,
    NumericalFailure,
}

#[derive(Clone, Debug)]
pub struct PricedRoute {
    pub charging: ChargingSolution,
    pub reduced_cost: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EvPricingStatistics {
    pub labels: usize,
    /// Phase-one subpaths retained by the subpath backend (zero for direct).
    pub subpaths: usize,
    pub dominance_checks: usize,
    pub joins: usize,
    pub work: usize,
}

#[derive(Clone, Debug)]
pub struct EvPricingResult {
    pub status: EvPricingStatus,
    /// Best replay-validated route found, whether or not its reduced cost is negative.
    /// Absence has proof meaning only with Complete (or within the bound with Bounded).
    pub best_route: Option<PricedRoute>,
    pub statistics: EvPricingStatistics,
}

pub(super) struct Label {
    pub(super) parent: Option<usize>,
    pub(super) step: PathStep,
    pub(super) depth: usize,
    pub(super) location: usize,
    pub(super) visited: Vec<bool>,
    pub(super) load: f64,
    pub(super) reduced: f64,
    pub(super) frontier: Vec<usize>,
    pub(super) active: bool,
}

pub(super) enum PathStep {
    Vertex(usize),
    Subpath(std::sync::Arc<[usize]>),
}

/// Label indices remain valid after dominance deactivates an ancestor. Build
/// the stop order only when a join or completed route needs its witness.
pub(super) fn label_path(labels: &[Label], mut id: usize) -> Vec<usize> {
    let mut path = vec![0; labels[id].depth];
    let mut end = path.len();
    loop {
        match &labels[id].step {
            PathStep::Vertex(vertex) => {
                end -= 1;
                path[end] = *vertex;
            }
            PathStep::Subpath(order) => {
                let start = end - (order.len() - 1);
                path[start..end].copy_from_slice(&order[1..]);
                end = start;
            }
        }
        match labels[id].parent {
            Some(parent) => id = parent,
            None => break,
        }
    }
    debug_assert_eq!(end, 0);
    path
}

pub(super) fn failure(status: ChargingStatus) -> EvPricingStatus {
    match status {
        ChargingStatus::WorkLimit => EvPricingStatus::WorkLimit,
        _ => EvPricingStatus::NumericalFailure,
    }
}

fn generate(
    problem: &Problem,
    ws: &mut Workspace,
    duals: &[f64],
    root: Label,
    backward: bool,
    options: DirectOptions,
    stats: &mut EvPricingStatistics,
) -> std::result::Result<Vec<Label>, EvPricingStatus> {
    if stats.labels >= options.max_labels {
        return Err(EvPricingStatus::WorkLimit);
    }
    stats.labels += 1;
    let mut labels = vec![root];
    let mut buckets = vec![Vec::new(); problem.locations.len()];
    buckets[problem.depot].push(0);
    let mut cursor = 0;
    let depth = options.max_arcs / 2 + options.max_arcs % 2;
    while cursor < labels.len() {
        if !labels[cursor].active || labels[cursor].depth > depth {
            cursor += 1;
            continue;
        }
        let at = labels[cursor].location;
        for next in 0..problem.locations.len() {
            ws.tick().map_err(failure)?;
            if next == at
                || matches!(problem.locations[next].kind, LocationKind::Depot)
                || labels[cursor].visited[next]
            {
                continue;
            }
            let (from, to) = if backward { (next, at) } else { (at, next) };
            let Some(arc) = problem.travel[from][to] else {
                continue;
            };
            if stats.labels >= options.max_labels {
                return Err(EvPricingStatus::WorkLimit);
            }
            stats.labels += 1;
            let mut load = labels[cursor].load;
            let mut reduced = labels[cursor].reduced + arc.cost;
            if let LocationKind::Customer { demand } = problem.locations[next].kind {
                load += demand;
                reduced -= duals[next];
            }
            if load > problem.load_capacity {
                continue;
            }
            if !load.is_finite() || !reduced.is_finite() {
                return Err(EvPricingStatus::NumericalFailure);
            }
            let result = if backward {
                ws.extend_backward(problem, &labels[cursor].frontier, from, to)
            } else {
                ws.extend_forward(problem, &labels[cursor].frontier, from, to)
            };
            let frontier = match result {
                Ok(p) => p,
                Err(ChargingStatus::Infeasible) => continue,
                Err(s) => return Err(failure(s)),
            };
            if frontier.is_empty() {
                continue;
            }
            let mut visited = labels[cursor].visited.clone();
            if matches!(problem.locations[next].kind, LocationKind::Customer { .. }) {
                visited[next] = true;
            }
            let mut new = Label {
                parent: Some(cursor),
                step: PathStep::Vertex(next),
                depth: labels[cursor].depth + 1,
                location: next,
                visited,
                load,
                reduced,
                frontier,
                active: true,
            };
            if options.dominance {
                // Equal customer histories preserve even the cheapest positive-cost
                // nonempty route. Pointwise domain coverage, cost and remaining
                // stop budget must all be compatible. Subset-history rules can be
                // added later with a corresponding nonempty-route contract.
                for &old in &buckets[next] {
                    let old: &mut Label = &mut labels[old];
                    if !old.active || old.visited != new.visited {
                        continue;
                    }
                    stats.dominance_checks += 1;
                    if old.depth <= new.depth
                        && old.reduced <= new.reduced
                        && ws
                            .dominates(&old.frontier, &new.frontier)
                            .map_err(failure)?
                    {
                        new.active = false;
                        break;
                    }
                    if new.depth <= old.depth
                        && new.reduced <= old.reduced
                        && ws
                            .dominates(&new.frontier, &old.frontier)
                            .map_err(failure)?
                    {
                        old.active = false;
                    }
                }
            }
            if new.active {
                buckets[next].push(labels.len());
                labels.push(new);
            }
        }
        cursor += 1;
    }
    Ok(labels)
}

impl Problem {
    /// Price elementary customer routes, permitting repeat station visits. The
    /// dual vector has one entry per location, zero at non-customers. Fleet dual
    /// is subtracted once per route. No cuts or branching restrictions are yet
    /// accepted by this oracle. A Complete status is not a rigorous real-number
    /// reduced-cost certificate; the algebra and comparisons use f64.
    pub fn price_direct(
        &self,
        duals: &[f64],
        fleet_dual: f64,
        departure: f64,
        options: DirectOptions,
    ) -> Result<EvPricingResult> {
        self.validate_pricing(duals, fleet_dual, departure, options)?;
        let mut ws = Workspace {
            arena: Vec::new(),
            work: 0,
            limit: options.max_work,
        };
        let mut result = EvPricingResult {
            status: EvPricingStatus::Bounded,
            best_route: None,
            statistics: EvPricingStatistics::default(),
        };
        let run = |ws: &mut Workspace,
                   result: &mut EvPricingResult|
         -> std::result::Result<(), EvPricingStatus> {
            let depot = &self.locations[self.depot];
            let start = departure.max(depot.window.0);
            if start > depot.window.1 {
                result.status = EvPricingStatus::Complete;
                return Ok(());
            }
            let time = start + depot.service;
            let last = depot.window.1 + depot.service;
            if !time.is_finite() || !last.is_finite() {
                return Err(EvPricingStatus::NumericalFailure);
            }
            ws.arena.push(Piece {
                lo: self.battery,
                hi: self.battery,
                a: 0.0,
                b: time,
                p: 1.0,
                q: 0.0,
                incoming: 0.0,
                parent: None,
            });
            ws.arena.push(Piece {
                lo: 0.0,
                hi: self.battery,
                a: 0.0,
                b: -last,
                p: 1.0,
                q: 0.0,
                incoming: 0.0,
                parent: None,
            });
            let root = |id, reduced| Label {
                parent: None,
                step: PathStep::Vertex(self.depot),
                depth: 1,
                location: self.depot,
                visited: vec![false; self.locations.len()],
                load: 0.0,
                reduced,
                frontier: vec![id],
                active: true,
            };
            let forward = generate(
                self,
                ws,
                duals,
                root(0, -fleet_dual),
                false,
                options,
                &mut result.statistics,
            )?;
            let backward = generate(
                self,
                ws,
                duals,
                root(1, 0.0),
                true,
                options,
                &mut result.statistics,
            )?;
            for (fi, f) in forward
                .iter()
                .enumerate()
                .filter(|(_, l)| l.active && l.depth > 1)
            {
                for (bi, b) in backward
                    .iter()
                    .enumerate()
                    .filter(|(_, l)| l.active && l.depth > 1)
                {
                    ws.tick().map_err(failure)?;
                    let at = f.location;
                    if b.location != at || f.depth + b.depth - 2 > options.max_arcs {
                        continue;
                    }
                    if (0..self.locations.len()).any(|i| i != at && f.visited[i] && b.visited[i]) {
                        continue;
                    }
                    let demand = match self.locations[at].kind {
                        LocationKind::Customer { demand } => demand,
                        _ => 0.0,
                    };
                    if f.load + b.load - demand > self.load_capacity {
                        continue;
                    }
                    if !f.visited.iter().zip(&b.visited).any(|(a, b)| *a || *b) {
                        continue;
                    }
                    let rc = f.reduced + b.reduced + duals[at];
                    if !rc.is_finite() {
                        return Err(EvPricingStatus::NumericalFailure);
                    }
                    if result
                        .best_route
                        .as_ref()
                        .map_or(false, |r| rc >= r.reduced_cost)
                    {
                        continue;
                    }
                    result.statistics.joins += 1;
                    let Some((fid, bid, energy)) =
                        ws.join(&f.frontier, &b.frontier).map_err(failure)?
                    else {
                        continue;
                    };
                    let forward_path = label_path(&forward, fi);
                    let backward_path = label_path(&backward, bi);
                    let mut visits = forward_path
                        .iter()
                        .map(|&location| Visit {
                            location,
                            recharge: 0.0,
                        })
                        .collect::<Vec<_>>();
                    let (mut id, mut e) = (fid, energy);
                    for i in (1..forward_path.len()).rev() {
                        let p = ws.arena[id];
                        let arrival = p.p * e + p.q;
                        visits[i].recharge = e - arrival;
                        e = arrival + p.incoming;
                        id = p.parent.ok_or(EvPricingStatus::NumericalFailure)?;
                    }
                    let (mut id, mut e) = (bid, energy);
                    for &location in backward_path[..backward_path.len() - 1].iter().rev() {
                        let p = ws.arena[id];
                        let next_e = p.p * e + p.q;
                        visits.push(Visit {
                            location,
                            recharge: next_e - (e - p.incoming),
                        });
                        e = next_e;
                        id = p.parent.ok_or(EvPricingStatus::NumericalFailure)?;
                    }
                    let schedule = self
                        .replay(&visits, departure)
                        .map_err(|_| EvPricingStatus::NumericalFailure)?;
                    // Recompute reduced cost from original route data, rather than
                    // reporting accumulated partial-path arithmetic.
                    let reduced_cost = visits
                        .iter()
                        .fold(schedule.travel_cost - fleet_dual, |cost, v| {
                            cost - duals[v.location]
                        });
                    if !reduced_cost.is_finite() {
                        return Err(EvPricingStatus::NumericalFailure);
                    }
                    if result
                        .best_route
                        .as_ref()
                        .map_or(true, |r| reduced_cost < r.reduced_cost)
                    {
                        result.best_route = Some(PricedRoute {
                            charging: ChargingSolution { visits, schedule },
                            reduced_cost,
                        });
                    }
                }
            }
            result.status = self.pricing_coverage(time, options.max_arcs);
            Ok(())
        };
        if let Err(status) = run(&mut ws, &mut result) {
            result.status = status;
        }
        result.statistics.work = ws.work;
        Ok(result)
    }
    pub(super) fn validate_pricing(
        &self,
        duals: &[f64],
        fleet_dual: f64,
        departure: f64,
        options: DirectOptions,
    ) -> Result<()> {
        self.validate()?;
        if duals.len() != self.locations.len()
            || !fleet_dual.is_finite()
            || !nonnegative(departure)
            || options.max_arcs < 2
            || options.max_arcs == usize::MAX
            || duals.iter().enumerate().any(|(i, d)| {
                !d.is_finite()
                    || (!matches!(self.locations[i].kind, LocationKind::Customer { .. })
                        && *d != 0.0)
            })
        {
            return Err(invalid("invalid EV pricing duals, departure or arc limit"));
        }
        Ok(())
    }

    pub(super) fn pricing_coverage(&self, time: f64, max_arcs: usize) -> EvPricingStatus {
        let depot = &self.locations[self.depot];
        // No stations gives an elementary finite bound even with zero-time
        // arcs. Otherwise a strictly positive travel-time lower bound limits
        // all feasible station repeats under the depot closing time. Add a
        // conservative extra arc for division rounding at integer boundaries.
        let customers = self
            .locations
            .iter()
            .filter(|l| matches!(l.kind, LocationKind::Customer { .. }))
            .count();
        let no_stations = !self
            .locations
            .iter()
            .any(|l| matches!(l.kind, LocationKind::Station { .. }));
        let min_time = self
            .travel
            .iter()
            .enumerate()
            .flat_map(|(i, r)| {
                r.iter()
                    .enumerate()
                    .filter_map(move |(j, a)| if i == j { None } else { a.map(|a| a.time) })
            })
            .fold(f64::INFINITY, f64::min);
        let time_bound = ((depot.window.1 - time).max(0.0) / min_time).ceil() + 1.0;
        if customers == 0
            || (no_stations && max_arcs >= customers + 1)
            || (min_time > 0.0 && (max_arcs as f64) >= time_bound)
        {
            return EvPricingStatus::Complete;
        }
        EvPricingStatus::Bounded
    }
}

#[cfg(test)]
mod path_tests {
    use super::*;

    #[test]
    fn reconstruction_keeps_inactive_ancestors_and_shared_subpaths() {
        let make = |parent, step, depth, location| Label {
            parent,
            step,
            depth,
            location,
            visited: vec![],
            load: 0.,
            reduced: 0.,
            frontier: vec![],
            active: false,
        };
        let segment: std::sync::Arc<[usize]> = vec![4, 6, 8, 9].into();
        let labels = vec![
            make(None, PathStep::Vertex(9), 1, 9),
            make(Some(0), PathStep::Vertex(4), 2, 4),
            make(Some(1), PathStep::Subpath(segment.clone()), 5, 9),
        ];
        assert_eq!(label_path(&labels, 0), vec![9]);
        assert_eq!(label_path(&labels, 1), vec![9, 4]);
        assert_eq!(label_path(&labels, 2), vec![9, 4, 6, 8, 9]);
        assert_eq!(&*segment, &[4, 6, 8, 9]);
    }
}
