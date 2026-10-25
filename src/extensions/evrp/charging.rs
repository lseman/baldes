//! Continuous charging fronts over exact remaining energy, as a coupled
//! labeling plugin. This resource has no cost: earlier departure dominates
//! at the same energy. Backward states defer to physical suffix replay at
//! the actual forward frontier.
use crate::graph::coupled::{CoupledPlugin, PiecewiseLinearFunction, PluginState};
use crate::graph::Window;
use crate::{invalid, Result};

fn finite(x: f64) -> Result<f64> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(invalid("nonfinite coupled resource arithmetic"))
    }
}

/// Travel followed by service and optional continuous charging at the head.
/// Charging maps energy to cumulative time from empty. Customers constrain
/// service start; a charging stop also constrains completion by closing time.
#[derive(Clone, Debug, PartialEq)]
pub struct ChargingResourceArc {
    pub travel_time: f64,
    pub energy: f64,
    pub window: Window,
    pub service: f64,
    pub charging: Option<PiecewiseLinearFunction>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Piece {
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
enum ChargingLabel {
    Forward(Vec<Piece>),
    Backward,
}

/// Earliest departure time as a continuous function of remaining energy.
/// All partial charging choices survive. `max_work` bounds each frontier
/// extension, including envelope work; exhaustion is an error, never a
/// proof of infeasibility. Arcs use the graph's arc vector order.
#[derive(Clone, Debug, PartialEq)]
pub struct ChargingResource {
    /// Battery capacity.
    pub battery: f64,
    /// Energy at the source, at most `battery`.
    pub initial_energy: f64,
    /// Departure time at the source.
    pub initial_time: f64,
    /// One entry per physical arc.
    pub arcs: Vec<ChargingResourceArc>,
    pub max_work: usize,
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
impl ChargingResource {
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

impl CoupledPlugin for ChargingResource {
    fn validate(&self, n: usize) -> Result<()> {
        let battery = &self.battery;
        let initial_energy = &self.initial_energy;
        let initial_time = &self.initial_time;
        let arcs = &self.arcs;
        if !battery.is_finite()
            || *battery <= 0.
            || !initial_energy.is_finite()
            || *initial_energy < 0.
            || initial_energy > battery
            || !initial_time.is_finite()
            || *initial_time < 0.
            || arcs.len() != n
        {
            return Err(invalid(
                "invalid charging resource initial state or arc count",
            ));
        }
        for arc in arcs.iter() {
            Window::new(arc.window.lower, arc.window.upper)?;
            if [
                arc.travel_time,
                arc.energy,
                arc.service,
                arc.window.lower,
                arc.window.upper,
            ]
            .iter()
            .any(|x| !x.is_finite() || *x < 0.)
            {
                return Err(invalid(
                    "invalid charging resource travel, service or window",
                ));
            }
            if let Some(curve) = &arc.charging {
                curve.validate()?;
                if curve.points[0] != [0., 0.]
                    || curve.points.last().unwrap()[0] != *battery
                    || curve.points.windows(2).any(|w| w[0][1] >= w[1][1])
                {
                    return Err(invalid(
                        "charging curve must strictly increase from zero to battery capacity",
                    ));
                }
            }
        }
        Ok(())
    }
    fn root(&self, forward: bool) -> PluginState {
        PluginState::new(if forward {
            ChargingLabel::Forward(vec![Piece {
                lo: self.initial_energy,
                hi: self.initial_energy,
                a: 0.,
                b: self.initial_time,
            }])
        } else {
            ChargingLabel::Backward
        })
    }
    fn extend(
        &self,
        state: &PluginState,
        arc: usize,
        forward: bool,
    ) -> Result<Option<PluginState>> {
        match (state.get::<ChargingLabel>(), forward) {
            (Some(ChargingLabel::Backward), false) => Ok(Some(state.clone())),
            (Some(ChargingLabel::Forward(front)), true) => {
                let next = self.advance(front, &self.arcs[arc])?;
                Ok((!next.is_empty()).then(|| PluginState::new(ChargingLabel::Forward(next))))
            }
            _ => Err(invalid("charging label direction mismatch")),
        }
    }
    fn join(&self, state: &PluginState, suffix: &mut dyn Iterator<Item = usize>) -> Result<bool> {
        let Some(ChargingLabel::Forward(front)) = state.get::<ChargingLabel>() else {
            return Err(invalid("charging forward join expected"));
        };
        let mut current = front.clone();
        for arc in suffix {
            current = self.advance(&current, &self.arcs[arc])?;
            if current.is_empty() {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/extensions/evrp/charging/tests.rs"]
mod tests;
