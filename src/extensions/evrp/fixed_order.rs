//! Continuous fixed-stop-sequence scheduling by piecewise-affine envelopes.
//!
//! A piece gives earliest departure time at an exact departure energy. At a
//! station with cumulative charging time T, the recurrence is
//! F_out(x) = T(x) + min_{q <= x} (F_in(q) - T(q)).
//! On each joint affine interval the minimum is attained at an endpoint or
//! q=x. This finite construction also works for non-concave charging curves.
//! Earlier time at the same energy dominates because waiting is unrestricted.

use super::functions::{Piece, Workspace};
use super::{nonnegative, LocationKind, Problem, Schedule, Visit};
use crate::{invalid, Result};

#[derive(Clone, Copy, Debug)]
pub struct ChargingOptions {
    /// Shared budget for generated pieces and envelope comparisons.
    pub max_work: usize,
}

impl Default for ChargingOptions {
    fn default() -> Self {
        Self {
            max_work: 1_000_000,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChargingStatus {
    /// Minimum completion time for the supplied sequence, in f64 arithmetic.
    Optimal,
    Infeasible,
    WorkLimit,
    /// Arithmetic overflow or a reconstructed schedule failing strict replay.
    NumericalFailure,
}

#[derive(Clone, Debug)]
pub struct ChargingSolution {
    pub visits: Vec<Visit>,
    pub schedule: Schedule,
}

#[derive(Clone, Debug)]
pub struct FixedOrderResult {
    pub status: ChargingStatus,
    pub solution: Option<ChargingSolution>,
    pub work: usize,
}

impl Problem {
    /// Optimize charging amounts on an explicit depot-to-depot stop sequence.
    /// Station insertion is not performed. The objective is earliest completion;
    /// travel cost is fixed by the sequence. Invalid inputs return Err, whereas
    /// valid but infeasible sequences return Infeasible. A work limit or failed
    /// strict replay never returns an infeasibility certificate.
    pub fn optimize_fixed_order(
        &self,
        order: &[usize],
        departure: f64,
        options: ChargingOptions,
    ) -> Result<FixedOrderResult> {
        self.validate()?;
        if order.len() < 2
            || order[0] != self.depot
            || order[order.len() - 1] != self.depot
            || !nonnegative(departure)
            || order.iter().any(|&v| v >= self.locations.len())
        {
            return Err(invalid("invalid EV fixed stop sequence or departure"));
        }
        let mut seen = vec![false; self.locations.len()];
        let mut load = 0.0;
        for (i, &v) in order.iter().enumerate() {
            match self.locations[v].kind {
                LocationKind::Customer { demand } => {
                    if seen[v] {
                        return Err(invalid("EV fixed sequence repeats a customer"));
                    }
                    seen[v] = true;
                    load += demand;
                }
                LocationKind::Depot if i > 0 && i + 1 < order.len() => {
                    return Err(invalid("intermediate EV depot unsupported"))
                }
                _ => {}
            }
        }
        let mut ws = Workspace {
            arena: Vec::new(),
            work: 0,
            limit: options.max_work,
        };
        let solve = |ws: &mut Workspace| -> std::result::Result<ChargingSolution, ChargingStatus> {
            if load > self.load_capacity {
                return Err(ChargingStatus::Infeasible);
            }
            let depot = &self.locations[self.depot];
            let start = departure.max(depot.window.0);
            if start > depot.window.1 {
                return Err(ChargingStatus::Infeasible);
            }
            let time = start + depot.service;
            if !time.is_finite() {
                return Err(ChargingStatus::NumericalFailure);
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
            let mut frontier = vec![0];
            for pair in order.windows(2) {
                frontier = ws.extend_forward(self, &frontier, pair[0], pair[1])?;
                if frontier.is_empty() {
                    return Err(ChargingStatus::Infeasible);
                }
            }
            let mut best: Option<(usize, f64, f64)> = None;
            for &id in &frontier {
                let p = ws.arena[id];
                let energy = if p.a < 0.0 { p.hi } else { p.lo };
                let time = p.value(energy);
                if best.map_or(true, |b| time < b.2) {
                    best = Some((id, energy, time));
                }
            }
            let (mut id, mut energy, _) = best.ok_or(ChargingStatus::Infeasible)?;
            let mut visits = order
                .iter()
                .map(|&location| Visit {
                    location,
                    recharge: 0.0,
                })
                .collect::<Vec<_>>();
            for i in (1..order.len()).rev() {
                let p = ws.arena[id];
                let arrival = p.p * energy + p.q;
                visits[i].recharge = energy - arrival;
                energy = arrival + p.incoming;
                id = p.parent.ok_or(ChargingStatus::NumericalFailure)?;
            }
            let schedule = self
                .replay(&visits, departure)
                .map_err(|_| ChargingStatus::NumericalFailure)?;
            Ok(ChargingSolution { visits, schedule })
        };
        Ok(match solve(&mut ws) {
            Ok(solution) => FixedOrderResult {
                status: ChargingStatus::Optimal,
                solution: Some(solution),
                work: ws.work,
            },
            Err(status) => FixedOrderResult {
                status,
                solution: None,
                work: ws.work,
            },
        })
    }
}
