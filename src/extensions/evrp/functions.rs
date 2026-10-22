//! Piecewise-affine time/energy algebra and reconstruction witnesses.
use super::{ChargingStatus, LocationKind, Problem};

#[derive(Clone, Copy, Debug)]
pub(super) struct Piece {
    pub(super) lo: f64,
    pub(super) hi: f64,
    pub(super) a: f64,
    pub(super) b: f64,
    // Forward: arrival energy is p*x+q, predecessor departure adds incoming.
    // Backward: successor departure energy is p*x+q; successor arrival is
    // x-incoming. parent points toward the corresponding depot root.
    pub(super) p: f64,
    pub(super) q: f64,
    pub(super) incoming: f64,
    pub(super) parent: Option<usize>,
}

impl Piece {
    pub(super) fn value(self, x: f64) -> f64 {
        self.a * x + self.b
    }

    pub(super) fn intersect(mut self, lo: f64, hi: f64) -> Option<Self> {
        self.lo = self.lo.max(lo);
        self.hi = self.hi.min(hi);
        (self.lo <= self.hi).then_some(self)
    }

    pub(super) fn before(mut self, bound: f64) -> Option<Self> {
        if self.a == 0.0 {
            return (self.b <= bound).then_some(self);
        }
        let x = (bound - self.b) / self.a;
        if self.a > 0.0 {
            self.hi = self.hi.min(x);
        } else {
            self.lo = self.lo.max(x);
        }
        (self.lo <= self.hi).then_some(self)
    }
}

pub(super) struct Workspace {
    pub(super) arena: Vec<Piece>,
    pub(super) work: usize,
    pub(super) limit: usize,
}

impl Workspace {
    /// Backward functions store minus latest departure time. They exclude
    /// processing at `from`, so they join forward functions at the same node
    /// and exact departure energy without processing that node twice.
    pub(super) fn extend_backward(
        &mut self,
        problem: &Problem,
        frontier: &[usize],
        from: usize,
        to: usize,
    ) -> std::result::Result<Vec<usize>, ChargingStatus> {
        let arc = problem.travel[from][to].ok_or(ChargingStatus::Infeasible)?;
        let loc = &problem.locations[to];
        let mut candidates = Vec::new();
        for &parent in frontier {
            self.tick()?;
            let prev = self.arena[parent];
            let station = matches!(loc.kind, LocationKind::Station { .. });
            // At customers latest service start = min(latest departure-s, close).
            // At stations cap departure at closing before undoing charging.
            let base = Piece {
                b: prev.b + if station { 0.0 } else { loc.service },
                parent: Some(parent),
                incoming: arc.energy,
                ..prev
            };
            if !base.b.is_finite() {
                return Err(ChargingStatus::NumericalFailure);
            }
            let early = base.before(-loc.window.1).map(|p| Piece {
                a: 0.0,
                b: -loc.window.1,
                ..p
            });
            let late = Piece {
                a: -base.a,
                b: -base.b,
                ..base
            }
            .before(loc.window.1)
            .map(|p| Piece {
                a: base.a,
                b: base.b,
                ..p
            });
            for p in early.into_iter().chain(late) {
                let mut arrivals = Vec::new();
                if let LocationKind::Station { curve } = &loc.kind {
                    // -A(q) = min_{y>=q}(-B(y)+T(y)) - T(q) + service.
                    for w in curve.knots.windows(2) {
                        let Some(ypiece) = p.intersect(w[0].0, w[1].0) else {
                            continue;
                        };
                        let s = (w[1].1 - w[0].1) / (w[1].0 - w[0].0);
                        let b = w[0].1 - s * w[0].0;
                        let k = ypiece.a + s;
                        let c = ypiece.b + b + loc.service;
                        if ![s, b, k, c].iter().all(|v| v.is_finite()) {
                            return Err(ChargingStatus::NumericalFailure);
                        }
                        let choices = if k > 0.0 {
                            vec![
                                (0.0, ypiece.lo, 0.0, ypiece.lo),
                                (ypiece.lo, ypiece.hi, 1.0, 0.0),
                            ]
                        } else {
                            vec![(0.0, ypiece.hi, 0.0, ypiece.hi)]
                        };
                        for (lo, hi, alpha, beta) in choices {
                            for q in curve.knots.windows(2) {
                                let s = (q[1].1 - q[0].1) / (q[1].0 - q[0].0);
                                let b = q[0].1 - s * q[0].0;
                                let candidate = Piece {
                                    lo,
                                    hi,
                                    a: k * alpha - s,
                                    b: k * beta + c - b,
                                    p: alpha,
                                    q: beta,
                                    ..p
                                };
                                if ![s, b, candidate.a, candidate.b]
                                    .iter()
                                    .all(|v| v.is_finite())
                                {
                                    return Err(ChargingStatus::NumericalFailure);
                                }
                                if let Some(p) = candidate
                                    .intersect(q[0].0, q[1].0)
                                    .and_then(|p| p.before(-loc.window.0))
                                {
                                    self.push(&mut arrivals, p)?;
                                }
                            }
                        }
                    }
                } else if let Some(p) = p.before(-loc.window.0) {
                    self.push(
                        &mut arrivals,
                        Piece {
                            p: 1.0,
                            q: 0.0,
                            ..p
                        },
                    )?;
                }
                for p in arrivals {
                    // Transform successor arrival q into predecessor departure x=q+energy.
                    let p = Piece {
                        lo: p.lo + arc.energy,
                        hi: p.hi + arc.energy,
                        b: p.b - p.a * arc.energy + arc.time,
                        q: p.q - p.p * arc.energy,
                        ..p
                    };
                    if ![p.lo, p.hi, p.b, p.q].iter().all(|v| v.is_finite()) {
                        return Err(ChargingStatus::NumericalFailure);
                    }
                    if let Some(p) = p.intersect(0.0, problem.battery) {
                        self.push(&mut candidates, p)?;
                    }
                }
            }
        }
        self.envelope(candidates)
    }

    /// Pointwise domain-covering comparison. In both directions smaller stored
    /// time is better (backward time is negated). No scalar endpoint shortcut.
    pub(super) fn dominates(
        &mut self,
        better: &[usize],
        worse: &[usize],
    ) -> std::result::Result<bool, ChargingStatus> {
        for &q in worse {
            let q = self.arena[q];
            let mut cover = Vec::new();
            for &p in better {
                self.tick()?;
                let p = self.arena[p];
                let difference = Piece {
                    a: p.a - q.a,
                    b: p.b - q.b,
                    ..p
                };
                if !difference.a.is_finite() || !difference.b.is_finite() {
                    return Err(ChargingStatus::NumericalFailure);
                }
                if let Some(p) = difference.intersect(q.lo, q.hi).and_then(|p| p.before(0.0)) {
                    cover.push((p.lo, p.hi));
                }
            }
            cover.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut end = q.lo;
            let mut covered = false;
            for (lo, hi) in cover {
                if lo > end {
                    break;
                }
                end = end.max(hi);
                if end >= q.hi {
                    covered = true;
                    break;
                }
            }
            if !covered {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Returns a pair of witnesses and a common energy where F(energy)<=B(energy).
    pub(super) fn join(
        &mut self,
        forward: &[usize],
        backward: &[usize],
    ) -> std::result::Result<Option<(usize, usize, f64)>, ChargingStatus> {
        for &f in forward {
            for &b in backward {
                self.tick()?;
                let (fp, bp) = (self.arena[f], self.arena[b]);
                let difference = Piece {
                    a: fp.a + bp.a,
                    b: fp.b + bp.b,
                    ..fp
                };
                if !difference.a.is_finite() || !difference.b.is_finite() {
                    return Err(ChargingStatus::NumericalFailure);
                }
                if let Some(p) = difference
                    .intersect(bp.lo, bp.hi)
                    .and_then(|p| p.before(0.0))
                {
                    return Ok(Some((f, b, if p.a < 0.0 { p.hi } else { p.lo })));
                }
            }
        }
        Ok(None)
    }

    pub(super) fn tick(&mut self) -> std::result::Result<(), ChargingStatus> {
        if self.work >= self.limit {
            return Err(ChargingStatus::WorkLimit);
        }
        self.work += 1;
        Ok(())
    }

    pub(super) fn push(
        &mut self,
        pieces: &mut Vec<Piece>,
        piece: Piece,
    ) -> std::result::Result<(), ChargingStatus> {
        self.tick()?;
        if ![
            piece.lo,
            piece.hi,
            piece.a,
            piece.b,
            piece.p,
            piece.q,
            piece.incoming,
        ]
        .iter()
        .all(|v| v.is_finite())
            || !piece.value(piece.lo).is_finite()
            || !piece.value(piece.hi).is_finite()
        {
            return Err(ChargingStatus::NumericalFailure);
        }
        pieces.push(piece);
        Ok(())
    }

    // Split at every endpoint and crossing; choose the least-time witness on
    // each open interval and at every boundary (including isolated points).
    pub(super) fn envelope(
        &mut self,
        pieces: Vec<Piece>,
    ) -> std::result::Result<Vec<usize>, ChargingStatus> {
        let mut breaks = Vec::new();
        for (i, p) in pieces.iter().enumerate() {
            breaks.extend([p.lo, p.hi]);
            for q in &pieces[..i] {
                self.tick()?;
                if p.a != q.a {
                    let x = (q.b - p.b) / (p.a - q.a);
                    if x.is_finite() && x > p.lo.max(q.lo) && x < p.hi.min(q.hi) {
                        breaks.push(x);
                    }
                }
            }
        }
        breaks.sort_by(f64::total_cmp);
        breaks.dedup();
        let mut result = Vec::new();
        for (i, &lo) in breaks.iter().enumerate() {
            for (hi, x) in std::iter::once((lo, lo))
                .chain(breaks.get(i + 1).map(|&hi| (hi, lo + (hi - lo) * 0.5)))
            {
                let mut best: Option<Piece> = None;
                for &p in &pieces {
                    self.tick()?;
                    // Require full interval coverage, not merely its midpoint.
                    if p.lo <= lo && p.hi >= hi && best.map_or(true, |b| p.value(x) < b.value(x)) {
                        best = Some(p);
                    }
                }
                if let Some(mut p) = best {
                    p.lo = lo;
                    p.hi = hi;
                    // Neighboring fragments with identical witnesses coalesce.
                    if let Some(&last) = result.last() {
                        let prev: &mut Piece = &mut self.arena[last];
                        if prev.hi == lo
                            && prev.a == p.a
                            && prev.b == p.b
                            && prev.p == p.p
                            && prev.q == p.q
                            && prev.parent == p.parent
                            && prev.incoming == p.incoming
                        {
                            prev.hi = hi;
                            continue;
                        }
                    }
                    result.push(self.arena.len());
                    self.arena.push(p);
                }
            }
        }
        Ok(result)
    }
}

impl Workspace {
    pub(super) fn extend_forward(
        &mut self,
        problem: &Problem,
        frontier: &[usize],
        from: usize,
        to: usize,
    ) -> std::result::Result<Vec<usize>, ChargingStatus> {
        let arc = problem.travel[from][to].ok_or(ChargingStatus::Infeasible)?;
        let loc = &problem.locations[to];
        let mut candidates = Vec::new();
        for &parent in frontier {
            self.tick()?;
            let prev = self.arena[parent];
            let arrival = Piece {
                lo: (prev.lo - arc.energy).max(0.0),
                hi: prev.hi - arc.energy,
                a: prev.a,
                b: prev.b + prev.a * arc.energy + arc.time,
                p: 1.0,
                q: 0.0,
                incoming: arc.energy,
                parent: Some(parent),
            };
            if ![arrival.lo, arrival.hi, arrival.a, arrival.b]
                .iter()
                .all(|v| v.is_finite())
            {
                return Err(ChargingStatus::NumericalFailure);
            }
            if arrival.lo > arrival.hi {
                continue;
            }
            // Split max(arrival time, opening time), then enforce latest
            // service start, before adding service and charging.
            let early = arrival.before(loc.window.0).map(|p| Piece {
                a: 0.0,
                b: loc.window.0,
                ..p
            });
            let late = Piece {
                a: -arrival.a,
                b: -arrival.b,
                ..arrival
            }
            .before(-loc.window.0)
            .map(|p| Piece {
                a: arrival.a,
                b: arrival.b,
                ..p
            });
            for p in early.into_iter().chain(late) {
                let Some(mut p) = p.before(loc.window.1) else {
                    continue;
                };
                p.b += loc.service;
                if !p.b.is_finite() {
                    return Err(ChargingStatus::NumericalFailure);
                }
                if let LocationKind::Station { curve } = &loc.kind {
                    for w in curve.knots.windows(2) {
                        let Some(qpiece) = p.intersect(w[0].0, w[1].0) else {
                            continue;
                        };
                        let slope = (w[1].1 - w[0].1) / (w[1].0 - w[0].0);
                        let intercept = w[0].1 - slope * w[0].0;
                        let k = qpiece.a - slope;
                        let c = qpiece.b - intercept;
                        if ![slope, intercept, k, c].iter().all(|v| v.is_finite()) {
                            return Err(ChargingStatus::NumericalFailure);
                        }
                        // Prefix minimum of k*q+c over q in [lo,hi], q<=x.
                        let mut choices = Vec::new();
                        if k < 0.0 {
                            choices.push((qpiece.lo, qpiece.hi, 1.0, 0.0));
                            choices.push((qpiece.hi, problem.battery, 0.0, qpiece.hi));
                        } else {
                            choices.push((qpiece.lo, problem.battery, 0.0, qpiece.lo));
                        }
                        for (lo, hi, alpha, beta) in choices {
                            for out in curve.knots.windows(2) {
                                let s = (out[1].1 - out[0].1) / (out[1].0 - out[0].0);
                                let b = out[0].1 - s * out[0].0;
                                if !s.is_finite() || !b.is_finite() {
                                    return Err(ChargingStatus::NumericalFailure);
                                }
                                let candidate = Piece {
                                    lo,
                                    hi,
                                    a: s + k * alpha,
                                    b: b + c + k * beta,
                                    p: alpha,
                                    q: beta,
                                    ..p
                                };
                                if !candidate.a.is_finite() || !candidate.b.is_finite() {
                                    return Err(ChargingStatus::NumericalFailure);
                                }
                                if let Some(candidate) = candidate
                                    .intersect(out[0].0, out[1].0)
                                    .and_then(|p| p.before(loc.window.1))
                                {
                                    self.push(&mut candidates, candidate)?;
                                }
                            }
                        }
                    }
                } else {
                    self.push(&mut candidates, p)?;
                }
            }
        }
        self.envelope(candidates)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/extensions/evrp/functions/tests.rs"]
mod tests;
