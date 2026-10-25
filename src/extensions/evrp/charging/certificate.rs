//! Independent charging witnesses and necessary-condition propagation.
//!
//! A witness specifies charging targets; outward arithmetic checks every leg,
//! service start and charging completion. It does not inspect optimized
//! charging fronts. A separate relaxed battery/time image may prove emptiness;
//! exhausting the target search cannot do so.
use super::*;
use crate::core::rounding::{add_down, add_up, multiply_down};
use std::time::Instant;

const MAX_STATES: usize = 512;
const MAX_TARGETS: usize = 256;

#[derive(Clone, Copy)]
struct Range {
    lo: f64,
    hi: f64,
}
impl Range {
    fn point(x: f64) -> Self {
        Self { lo: x, hi: x }
    }
    fn add(self, b: Self) -> Option<Self> {
        Some(Self {
            lo: add_down(self.lo, b.lo)?,
            hi: add_up(self.hi, b.hi)?,
        })
    }
    fn subtract(self, b: Self) -> Option<Self> {
        self.add(Self { lo: -b.hi, hi: -b.lo })
    }
    fn max(self, lower: f64) -> Self {
        Self { lo: self.lo.max(lower), hi: self.hi.max(lower) }
    }
    fn multiply_nonnegative(self, b: Self) -> Option<Self> {
        Some(Self {
            lo: multiply_down(self.lo, b.lo)?,
            hi: -multiply_down(-self.hi, b.hi)?,
        })
    }
}

fn next_down(x: f64) -> f64 {
    if x == 0. {
        -f64::from_bits(1)
    } else {
        f64::from_bits(if x > 0. { x.to_bits() - 1 } else { x.to_bits() + 1 })
    }
}

struct Work {
    remaining: usize,
    deadline: Option<Instant>,
}
impl Work {
    fn take(&mut self, count: usize) -> Option<()> {
        if self.deadline.is_some_and(|limit| Instant::now() >= limit) {
            return None;
        }
        self.remaining = self.remaining.checked_sub(count)?;
        Some(())
    }
}

// All charging curves strictly increase. Endpoint images therefore enclose the
// complete argument interval. Knots return the literal input values; interior
// interpolation includes subtraction, division, multiplication and addition
// rounding independently of the numerical frontier's affine representation.
fn value(curve: &PiecewiseLinearFunction, x: f64, work: &mut Work) -> Option<Range> {
    work.take(curve.points.len())?;
    let upper = curve.points.partition_point(|p| p[0] < x);
    let &[x1, y1] = curve.points.get(upper)?;
    if x1 == x {
        return Some(Range::point(y1));
    }
    let &[x0, y0] = curve.points.get(upper.checked_sub(1)?)?;
    let numerator = Range::point(x).subtract(Range::point(x0))?;
    let denominator = Range::point(x1).subtract(Range::point(x0))?;
    if denominator.lo <= 0. {
        return None;
    }
    let fraction = Range {
        lo: next_down(numerator.lo / denominator.hi).max(0.),
        hi: -next_down(-numerator.hi / denominator.lo),
    };
    let height = Range::point(y1).subtract(Range::point(y0))?;
    let result = Range::point(y0).add(fraction.multiply_nonnegative(height)?)?;
    (result.lo.is_finite() && result.hi.is_finite()).then_some(Range {
        lo: result.lo.max(y0),
        hi: result.hi.min(y1),
    })
}

#[derive(Clone, Copy)]
struct State {
    energy: Range,
    time: Range,
}

pub(super) fn certify(
    resource: &ChargingResource,
    arcs: &[usize],
    shared_work: &mut usize,
    deadline: Option<Instant>,
) -> Option<bool> {
    let allowance = (*shared_work).min(resource.max_work);
    let mut work = Work { remaining: allowance, deadline };
    let result = run(resource, arcs, &mut work);
    *shared_work = shared_work.checked_sub(allowance - work.remaining)?;
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        None
    } else {
        result
    }
}

fn run(resource: &ChargingResource, arcs: &[usize], work: &mut Work) -> Option<bool> {
    work.take(1)?;
    let mut outer_energy = Range::point(resource.initial_energy);
    let mut outer_time = Range::point(resource.initial_time);
    let mut states = vec![State { energy: outer_energy, time: outer_time }];
    for (position, &index) in arcs.iter().enumerate() {
        work.take(1)?;
        let arc = resource.arcs.get(index)?;
        outer_energy = outer_energy.subtract(Range::point(arc.energy))?;
        if outer_energy.hi < 0. {
            return Some(false);
        }
        outer_energy.lo = outer_energy.lo.max(0.);
        outer_time = outer_time.add(Range::point(arc.travel_time))?.max(arc.window.lower);
        if outer_time.lo > arc.window.upper {
            return Some(false);
        }
        outer_time.hi = outer_time.hi.min(arc.window.upper);
        outer_time = outer_time.add(Range::point(arc.service))?;
        if arc.charging.is_some() {
            if outer_time.lo > arc.window.upper {
                return Some(false);
            }
            // Zero charging duration is a relaxation, never an actual witness.
            outer_energy.hi = resource.battery;
            outer_time.hi = arc.window.upper;
        }

        let mut targets = vec![resource.battery];
        if let Some(curve) = &arc.charging {
            work.take(curve.points.len())?;
            targets.extend(curve.points.iter().map(|p| p[0]));
            let mut required = 0.;
            for &future in &arcs[position + 1..] {
                work.take(1)?;
                let leg = resource.arcs.get(future)?;
                required = add_up(required, leg.energy)?;
                targets.push(required);
                if leg.charging.is_some() {
                    break;
                }
            }
        }
        work.take(targets.len().checked_mul(usize::BITS as usize)?)?;
        targets.retain(|target| *target >= 0. && *target <= resource.battery);
        targets.sort_by(f64::total_cmp);
        targets.dedup();
        if targets.len() > MAX_TARGETS {
            return None;
        }
        let mut next: Vec<State> = vec![];
        for state in states {
            work.take(1)?;
            let energy = state.energy.subtract(Range::point(arc.energy))?;
            let time = state.time.add(Range::point(arc.travel_time))?.max(arc.window.lower);
            if energy.lo < 0. || time.hi > arc.window.upper {
                continue;
            }
            let time = time.add(Range::point(arc.service))?;
            if let Some(curve) = &arc.charging {
                if time.hi > arc.window.upper {
                    continue;
                }
                // Retain the independent no-charge control as well as targets.
                insert(&mut next, State { energy, time }, work)?;
                let initial = Range {
                    lo: value(curve, energy.lo, work)?.lo,
                    hi: value(curve, energy.hi, work)?.hi,
                };
                for &target in &targets {
                    work.take(1)?;
                    if target < energy.hi {
                        continue;
                    }
                    let duration = value(curve, target, work)?.subtract(initial)?.max(0.);
                    let completion = time.add(duration)?;
                    if completion.hi <= arc.window.upper {
                        insert(&mut next, State { energy: Range::point(target), time: completion }, work)?;
                    }
                }
            } else {
                insert(&mut next, State { energy, time }, work)?;
            }
        }
        states = next;
    }
    (!states.is_empty()).then_some(true)
}

fn insert(states: &mut Vec<State>, candidate: State, work: &mut Work) -> Option<()> {
    work.take(states.len().checked_add(1)?)?;
    if let Some(old) = states.iter_mut().find(|old| {
        old.energy.lo == candidate.energy.lo && old.energy.hi == candidate.energy.hi
    }) {
        if candidate.time.hi < old.time.hi {
            *old = candidate;
        }
    } else {
        if states.len() == MAX_STATES {
            return None;
        }
        states.push(candidate);
    }
    Some(())
}
