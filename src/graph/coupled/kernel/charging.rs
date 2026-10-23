//! Continuous charging fronts over exact remaining energy. This resource has
//! no cost: earlier departure dominates at the same energy. Backward states
//! defer to physical suffix replay at the actual forward frontier.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Piece {
    lo: f64,
    hi: f64,
    a: f64,
    b: f64,
}
impl Piece {
    fn value(self, x: f64) -> f64 {
        self.a * x + self.b
    }
    fn intersect(mut self, lo: f64, hi: f64) -> Option<Self> {
        self.lo = self.lo.max(lo);
        self.hi = self.hi.min(hi);
        (self.lo <= self.hi).then_some(self)
    }
    fn before(mut self, bound: f64) -> Result<Option<Self>> {
        if self.a == 0. {
            return Ok((self.b <= bound).then_some(self));
        }
        let x = finite(finite(bound - self.b)? / self.a)?;
        if self.a > 0. {
            self.hi = self.hi.min(x);
        } else {
            self.lo = self.lo.max(x);
        }
        Ok((self.lo <= self.hi).then_some(self))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum ChargingLabel {
    Forward(Vec<Piece>),
    Backward,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct PreparedCharging {
    battery: f64,
    initial: Piece,
    arcs: std::sync::Arc<[ChargingResourceArc]>,
    max_work: usize,
}
struct Work(usize);
impl Work {
    fn tick(&mut self) -> Result<()> {
        self.0 = self
            .0
            .checked_sub(1)
            .ok_or_else(|| crate::incomplete("charging frontier work limit"))?;
        Ok(())
    }
    fn push(&mut self, pieces: &mut Vec<Piece>, p: Piece) -> Result<()> {
        self.tick()?;
        for x in [p.lo, p.hi, p.a, p.b, p.value(p.lo), p.value(p.hi)] {
            finite(x)?;
        }
        if p.lo <= p.hi {
            pieces.push(p);
        }
        Ok(())
    }
    fn envelope(&mut self, pieces: Vec<Piece>) -> Result<Vec<Piece>> {
        let mut breaks = Vec::new();
        for (i, p) in pieces.iter().enumerate() {
            self.tick()?;
            breaks.extend([p.lo, p.hi]);
            for q in &pieces[..i] {
                self.tick()?;
                if p.a != q.a {
                    let x = finite(finite(q.b - p.b)? / finite(p.a - q.a)?)?;
                    if x > p.lo.max(q.lo) && x < p.hi.min(q.hi) {
                        breaks.push(x);
                    }
                }
            }
        }
        breaks.sort_by(f64::total_cmp);
        breaks.dedup();
        let mut result: Vec<Piece> = Vec::new();
        for (i, &lo) in breaks.iter().enumerate() {
            for (hi, x) in std::iter::once((lo, lo))
                .chain(breaks.get(i + 1).map(|&hi| (hi, lo + (hi - lo) * 0.5)))
            {
                let mut best: Option<Piece> = None;
                for &p in &pieces {
                    self.tick()?;
                    if p.lo <= lo && p.hi >= hi && best.map_or(true, |b| p.value(x) < b.value(x)) {
                        best = Some(p);
                    }
                }
                if let Some(mut p) = best {
                    p.lo = lo;
                    p.hi = hi;
                    if let Some(last) = result.last_mut() {
                        if last.hi == lo && last.a == p.a && last.b == p.b {
                            last.hi = hi;
                            continue;
                        }
                    }
                    result.push(p);
                }
            }
        }
        Ok(result)
    }
}
impl PreparedCharging {
    pub(super) fn new(
        battery: f64,
        energy: f64,
        time: f64,
        arcs: &[ChargingResourceArc],
        max_work: usize,
    ) -> Self {
        Self {
            battery,
            initial: Piece {
                lo: energy,
                hi: energy,
                a: 0.,
                b: time,
            },
            arcs: arcs.into(),
            max_work,
        }
    }
    pub(super) fn root(&self, forward: bool) -> ChargingLabel {
        if forward {
            ChargingLabel::Forward(vec![self.initial])
        } else {
            ChargingLabel::Backward
        }
    }
    pub(super) fn extend(
        &self,
        state: &ChargingLabel,
        arc: usize,
        forward: bool,
    ) -> Result<Option<ChargingLabel>> {
        match (state, forward) {
            (ChargingLabel::Backward, false) => Ok(Some(ChargingLabel::Backward)),
            (ChargingLabel::Forward(front), true) => {
                let next = self.advance(front, &self.arcs[arc])?;
                Ok((!next.is_empty()).then_some(ChargingLabel::Forward(next)))
            }
            _ => Err(invalid("charging label direction mismatch")),
        }
    }
    pub(super) fn join(
        &self,
        forward: &ChargingLabel,
        suffix: &continuation::Continuation,
    ) -> Result<bool> {
        let ChargingLabel::Forward(front) = forward else {
            return Err(invalid("charging forward join expected"));
        };
        let mut current = front.clone();
        for arc in suffix.iter() {
            current = self.advance(&current, &self.arcs[arc])?;
            if current.is_empty() {
                return Ok(false);
            }
        }
        Ok(true)
    }
    fn advance(&self, front: &[Piece], arc: &ChargingResourceArc) -> Result<Vec<Piece>> {
        let mut work = Work(self.max_work);
        let mut candidates = Vec::new();
        for prev in front {
            work.tick()?;
            let arrival = Piece {
                lo: finite(prev.lo - arc.energy)?.max(0.),
                hi: finite(prev.hi - arc.energy)?,
                a: prev.a,
                b: finite(finite(prev.b + finite(prev.a * arc.energy)?)? + arc.travel_time)?,
            };
            if arrival.lo > arrival.hi {
                continue;
            }
            let early = arrival.before(arc.window.lower)?.map(|p| Piece {
                a: 0.,
                b: arc.window.lower,
                ..p
            });
            let late = Piece {
                a: -arrival.a,
                b: -arrival.b,
                ..arrival
            }
            .before(-arc.window.lower)?
            .map(|p| Piece {
                a: arrival.a,
                b: arrival.b,
                ..p
            });
            for p in early.into_iter().chain(late) {
                let Some(mut p) = p.before(arc.window.upper)? else {
                    continue;
                };
                p.b = finite(p.b + arc.service)?;
                if let Some(curve) = &arc.charging {
                    // T(x) + min_{q<=x}(F(q)-T(q)). On each affine
                    // interval the minimum is at an endpoint or at q=x.
                    for w in curve.points.windows(2) {
                        work.tick()?;
                        let Some(q) = p.intersect(w[0][0], w[1][0]) else {
                            continue;
                        };
                        let slope = finite((w[1][1] - w[0][1]) / (w[1][0] - w[0][0]))?;
                        let intercept = finite(w[0][1] - finite(slope * w[0][0])?)?;
                        let k = finite(q.a - slope)?;
                        let c = finite(q.b - intercept)?;
                        let choices = if k < 0. {
                            vec![(q.lo, q.hi, 1., 0.), (q.hi, self.battery, 0., q.hi)]
                        } else {
                            vec![(q.lo, self.battery, 0., q.lo)]
                        };
                        for (lo, hi, alpha, beta) in choices {
                            for out in curve.points.windows(2) {
                                work.tick()?;
                                let s = finite((out[1][1] - out[0][1]) / (out[1][0] - out[0][0]))?;
                                let b = finite(out[0][1] - finite(s * out[0][0])?)?;
                                let candidate = Piece {
                                    lo,
                                    hi,
                                    a: finite(s + finite(k * alpha)?)?,
                                    b: finite(finite(b + c)? + finite(k * beta)?)?,
                                };
                                if let Some(p) = candidate.intersect(out[0][0], out[1][0]) {
                                    if let Some(p) = p.before(arc.window.upper)? {
                                        work.push(&mut candidates, p)?;
                                    }
                                }
                            }
                        }
                    }
                } else {
                    work.push(&mut candidates, p)?;
                }
            }
        }
        work.envelope(candidates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(close: f64, max_work: usize) -> Graph<2> {
        let mut graph = Graph::<2>::new(4, 0).unwrap();
        for (u, v) in [(0, 1), (1, 2), (2, 3)] {
            graph.add_arc(u, v, -1.).unwrap();
        }
        let curve = PiecewiseLinearFunction::new(vec![[0., 0.], [2., 4.], [4., 5.]]).unwrap();
        let arcs = (0..3)
            .map(|i| ChargingResourceArc {
                travel_time: 1.,
                energy: [3., 2., 1.][i],
                window: Window::new(
                    if i == 1 { 6. } else { 0. },
                    if i == 2 { close } else { 20. },
                )
                .unwrap(),
                service: 0.,
                charging: (i == 0).then(|| curve.clone()),
            })
            .collect();
        graph
            .set_coupled_resources(&[
                CoupledResource::Charging {
                    battery: 4.,
                    initial_energy: 4.,
                    initial_time: 0.,
                    arcs,
                    max_work,
                },
                CoupledResource::Selection {
                    initial: 3,
                    allowed: vec![3, 2, 3],
                },
                CoupledResource::SimultaneousPickupDelivery {
                    capacity: 2.,
                    deliveries: vec![0., 2., 0.],
                    pickups: vec![0.; 3],
                },
            ])
            .unwrap();
        graph
    }

    #[test]
    fn every_native_split_matches_independent_charging_controls() {
        for close in [6., 7., 8., 10.] {
            // Independent quarter-energy controls. At least two units must
            // be charged; the nonconcave profile costs 2+(e-2)/2 above e=2.
            let feasible = (0..=12).any(|i| {
                let recharge = f64::from(i) / 4.;
                let target = 1. + recharge;
                let cumulative = if target <= 2. {
                    2. * target
                } else {
                    4. + (target - 2.) / 2.
                };
                recharge >= 2. && (1. + cumulative - 2. + 1.).max(6.) + 1. <= close
            });
            let graph = fixture(close, 10000);
            let kernel = graph.coupled_kernel().unwrap();
            assert_eq!(kernel.replay(&[0, 1, 2]).unwrap().is_some(), feasible);
            for split in 0..=3 {
                let mut f = kernel.root(true);
                for arc in 0..split {
                    let Some((next, _)) = kernel.extend(&f, arc, true).unwrap() else {
                        break;
                    };
                    f = next;
                }
                // For infeasible complete prefixes, test the forward replay
                // separately above; an absent label has no split join.
                if kernel
                    .replay_state(&(0..split).collect::<Vec<_>>())
                    .unwrap()
                    .is_none()
                {
                    continue;
                }
                let mut b = kernel.root(false);
                for arc in (split..3).rev() {
                    b = kernel.extend(&b, arc, false).unwrap().unwrap().0;
                }
                assert_eq!(
                    kernel.join(&f, &b).unwrap().is_some(),
                    feasible,
                    "close={close}, split={split}"
                );
            }
        }
    }

    #[test]
    fn native_charging_front_exhaustion_is_an_error() {
        let graph = fixture(10., 0);
        let kernel = graph.coupled_kernel().unwrap();
        assert!(kernel
            .replay(&[0, 1, 2])
            .unwrap_err()
            .message
            .contains("work limit"));
        let b = kernel
            .extend(&kernel.root(false), 2, false)
            .unwrap()
            .unwrap()
            .0;
        let b = kernel.extend(&b, 1, false).unwrap().unwrap().0;
        let b = kernel.extend(&b, 0, false).unwrap().unwrap().0;
        assert!(kernel.join(&kernel.root(true), &b).is_err());
    }

    #[cfg(feature = "clp")]
    #[test]
    fn phase_one_keeps_charging_feasibility_and_declines_certification() {
        for close in [6., 7.] {
            let graph = fixture(close, 10000);
            let mut kernel = graph.coupled_kernel().unwrap().clone();
            kernel.suppress_costs();
            assert_eq!(kernel.replay(&[0, 1, 2]).unwrap().is_some(), close >= 7.);
            assert!(kernel.certify(&[0, 1, 2]).is_none());
            let json = serde_json::to_string(kernel.definitions()).unwrap();
            let definitions: Vec<CoupledResource> = serde_json::from_str(&json).unwrap();
            assert_eq!(definitions, kernel.definitions());
        }
    }
}
