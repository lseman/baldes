//! Two-phase reference pricing: retain customer subpaths between charging
//! locations, then compose full routes with continuous charging functions.
//! Distinct phase-one orders are deliberately retained; no unsafe scalar
//! subpath domination is substituted for the E2 domination framework.

use super::direct::{failure, label_path, Label, PathStep};
use super::functions::{Piece, Workspace};
use super::{
    ChargingSolution, ChargingStatus, DirectOptions, EvPricingResult, EvPricingStatistics,
    EvPricingStatus, LocationKind, PricedRoute, Problem, Visit,
};
use crate::Result;
use std::{borrow::Cow, sync::Arc};

#[derive(Clone, Copy, Debug)]
pub struct SubpathOptions {
    /// Same model, work budget and route-arc bound as direct pricing.
    /// max_labels limits phase-two compositions.
    pub pricing: DirectOptions,
    pub max_subpaths: usize,
}

impl Default for SubpathOptions {
    fn default() -> Self {
        Self {
            pricing: DirectOptions::default(),
            max_subpaths: 20_000,
        }
    }
}

struct Subpath {
    order: Arc<[usize]>,
    visited: Vec<bool>,
    load: f64,
    reduced: f64,
}

struct Generator<'a> {
    problem: &'a Problem,
    duals: &'a [f64],
    options: SubpathOptions,
    count: usize,
}

impl Generator<'_> {
    fn extend(
        &mut self,
        ws: &mut Workspace,
        order: &mut Vec<usize>,
        visited: &mut [bool],
        load: f64,
        energy: f64,
        reduced: f64,
        out: &mut Vec<Subpath>,
    ) -> std::result::Result<(), EvPricingStatus> {
        if order.len() - 1 >= self.options.pricing.max_arcs {
            return Ok(());
        }
        let at = *order.last().unwrap();
        for next in 0..self.problem.locations.len() {
            ws.tick().map_err(failure)?;
            if next == at || visited[next] {
                continue;
            }
            if matches!(self.problem.locations[next].kind, LocationKind::Depot)
                && next != self.problem.depot
            {
                continue;
            }
            let Some(arc) = self.problem.travel[at][next] else {
                continue;
            };
            let total_energy = energy + arc.energy;
            if total_energy > self.problem.battery {
                continue;
            }
            let mut new_load = load;
            let mut new_reduced = reduced + arc.cost;
            let customer = matches!(
                self.problem.locations[next].kind,
                LocationKind::Customer { .. }
            );
            if let LocationKind::Customer { demand } = self.problem.locations[next].kind {
                new_load += demand;
                new_reduced -= self.duals[next];
            }
            if new_load > self.problem.load_capacity {
                continue;
            }
            if !new_reduced.is_finite() {
                return Err(EvPricingStatus::NumericalFailure);
            }
            order.push(next);
            if customer {
                visited[next] = true;
                self.extend(ws, order, visited, new_load, total_energy, new_reduced, out)?;
                visited[next] = false;
            } else {
                if self.count >= self.options.max_subpaths {
                    return Err(EvPricingStatus::WorkLimit);
                }
                self.count += 1;
                out.push(Subpath {
                    order: Arc::from(order.as_slice()),
                    visited: visited.to_vec(),
                    load: new_load,
                    reduced: new_reduced,
                });
            }
            order.pop();
        }
        Ok(())
    }
}

impl Problem {
    /// E2-inspired decomposition prototype. Phase one retains every elementary
    /// customer order between charging/depot endpoints that fits a full battery
    /// and capacity. Phase two composes those orders, checking the full time/
    /// energy functions at every extension. It shares the direct backend's
    /// objective, dual convention, replay and completeness contract. Subpaths
    /// are not master columns; only complete customer-serving routes are returned.
    pub fn price_subpaths(
        &self,
        duals: &[f64],
        fleet_dual: f64,
        departure: f64,
        options: SubpathOptions,
    ) -> Result<EvPricingResult> {
        self.validate_pricing(duals, fleet_dual, departure, options.pricing)?;
        let mut ws = Workspace {
            arena: Vec::new(),
            work: 0,
            limit: options.pricing.max_work,
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
            if !time.is_finite() {
                return Err(EvPricingStatus::NumericalFailure);
            }
            let mut subpaths = (0..self.locations.len())
                .map(|_| Vec::new())
                .collect::<Vec<Vec<Subpath>>>();
            let mut generator = Generator {
                problem: self,
                duals,
                options,
                count: 0,
            };
            for (v, out) in subpaths.iter_mut().enumerate() {
                if v == self.depot || matches!(self.locations[v].kind, LocationKind::Station { .. })
                {
                    let generated = generator.extend(
                        ws,
                        &mut vec![v],
                        &mut vec![false; self.locations.len()],
                        0.0,
                        0.0,
                        0.0,
                        out,
                    );
                    result.statistics.subpaths = generator.count;
                    generated?;
                }
            }
            if options.pricing.max_labels == 0 {
                return Err(EvPricingStatus::WorkLimit);
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
            let mut labels = vec![Label {
                parent: None,
                step: PathStep::Vertex(self.depot),
                depth: 1,
                location: self.depot,
                visited: vec![false; self.locations.len()],
                load: 0.0,
                reduced: -fleet_dual,
                frontier: vec![0],
                active: true,
            }];
            result.statistics.labels = 1;
            let mut buckets = vec![Vec::new(); self.locations.len()];
            buckets[self.depot].push(0);
            let mut cursor = 0;
            while cursor < labels.len() {
                if !labels[cursor].active {
                    cursor += 1;
                    continue;
                }
                let at = labels[cursor].location;
                for subpath in &subpaths[at] {
                    ws.tick().map_err(failure)?;
                    if labels[cursor].depth + subpath.order.len() - 2 > options.pricing.max_arcs {
                        continue;
                    }
                    if labels[cursor]
                        .visited
                        .iter()
                        .zip(&subpath.visited)
                        .any(|(a, b)| *a && *b)
                    {
                        continue;
                    }
                    let load = labels[cursor].load + subpath.load;
                    if load > self.load_capacity {
                        continue;
                    }
                    if result.statistics.labels >= options.pricing.max_labels {
                        return Err(EvPricingStatus::WorkLimit);
                    }
                    result.statistics.labels += 1;
                    let reduced = labels[cursor].reduced + subpath.reduced;
                    if !reduced.is_finite() {
                        return Err(EvPricingStatus::NumericalFailure);
                    }
                    let mut frontier = Cow::Borrowed(labels[cursor].frontier.as_slice());
                    let mut feasible = true;
                    for pair in subpath.order.windows(2) {
                        match ws.extend_forward(self, &frontier, pair[0], pair[1]) {
                            Ok(f) => {
                                frontier = Cow::Owned(f);
                                if frontier.is_empty() {
                                    feasible = false;
                                    break;
                                }
                            }
                            Err(ChargingStatus::Infeasible) => {
                                feasible = false;
                                break;
                            }
                            Err(s) => return Err(failure(s)),
                        }
                    }
                    if !feasible {
                        continue;
                    }
                    let frontier = frontier.into_owned();
                    let visited = labels[cursor]
                        .visited
                        .iter()
                        .zip(&subpath.visited)
                        .map(|(a, b)| *a || *b)
                        .collect::<Vec<_>>();
                    let next = *subpath.order.last().unwrap();
                    if next == self.depot {
                        if !visited.iter().any(|v| *v)
                            || result
                                .best_route
                                .as_ref()
                                .map_or(false, |r| reduced >= r.reduced_cost)
                        {
                            continue;
                        }
                        let mut best: Option<(usize, f64, f64)> = None;
                        for &id in &frontier {
                            let p = ws.arena[id];
                            let e = if p.a < 0.0 { p.hi } else { p.lo };
                            let t = p.value(e);
                            if best.map_or(true, |b| t < b.2) {
                                best = Some((id, e, t));
                            }
                        }
                        let (mut id, mut e, _) = best.ok_or(EvPricingStatus::NumericalFailure)?;
                        let mut path = label_path(&labels, cursor);
                        path.extend_from_slice(&subpath.order[1..]);
                        let mut visits = path
                            .iter()
                            .map(|&location| Visit {
                                location,
                                recharge: 0.0,
                            })
                            .collect::<Vec<_>>();
                        for i in (1..path.len()).rev() {
                            let p = ws.arena[id];
                            let arrival = p.p * e + p.q;
                            visits[i].recharge = e - arrival;
                            e = arrival + p.incoming;
                            id = p.parent.ok_or(EvPricingStatus::NumericalFailure)?;
                        }
                        let schedule = self
                            .replay(&visits, departure)
                            .map_err(|_| EvPricingStatus::NumericalFailure)?;
                        let reduced_cost = visits
                            .iter()
                            .fold(schedule.travel_cost - fleet_dual, |c, v| {
                                c - duals[v.location]
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
                    } else {
                        let mut new = Label {
                            parent: Some(cursor),
                            step: PathStep::Subpath(subpath.order.clone()),
                            depth: labels[cursor].depth + subpath.order.len() - 1,
                            location: next,
                            visited,
                            load,
                            reduced,
                            frontier,
                            active: true,
                        };
                        if options.pricing.dominance {
                            // Compare the complete prefix function. Scalar
                            // subpath summaries cannot justify this deletion.
                            for &old in &buckets[next] {
                                let old: &mut Label = &mut labels[old];
                                if !old.active || old.visited != new.visited {
                                    continue;
                                }
                                result.statistics.dominance_checks += 1;
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
                }
                cursor += 1;
            }
            result.status = self.pricing_coverage(time, options.pricing.max_arcs);
            Ok(())
        };
        if let Err(status) = run(&mut ws, &mut result) {
            result.status = status;
        }
        result.statistics.work = ws.work;
        Ok(result)
    }
}
