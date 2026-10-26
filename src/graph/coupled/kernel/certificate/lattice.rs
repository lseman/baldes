//! Independent exact certificates for unit-slope native paired programs.
//!
//! Integer breakpoints and values, slopes in {-1,0,1}, translated arrivals,
//! constant increments/bounds and unit signed waiting preserve this lattice.
//! On a contiguous input domain, the waiting envelope selects the latest
//! departure (waiting=1), the earliest departure (waiting=-1), or a running
//! minimum (waiting=0). Its crossings and constant-bound crossings are integer.
//! Thus an integer control attains every continuous minimum, and an empty
//! integer frontier proves emptiness. We decline disconnected output domains
//! before another transition, rather than assuming this invariant across gaps.
//!
//! This proof never reads a numerical optimized front. Integer spacing also
//! keeps unequal coordinates outside native alignment tolerances. The bounds
//! below keep endpoint and expanded-intercept arithmetic exact in binary64.
use super::*;
use crate::graph::coupled::descriptor::{PairedCostNormalization, PreparedSlopeFunction};

const LIMIT: f64 = 1_000_000.;
const MAX_POINTS: usize = 100_000;

fn integer(x: f64) -> bool {
    x.is_finite() && x.fract() == 0. && x.abs() <= LIMIT
}

fn domain(f: &PreparedSlopeFunction, work: &mut usize, deadline: Option<Instant>) -> Option<()> {
    debit(work, f.pairs().len(), deadline)?;
    f.pairs().iter().all(|p| integer(p.1)).then_some(())
}

fn constant(f: &PreparedSlopeFunction, work: &mut usize, deadline: Option<Instant>) -> Option<f64> {
    domain(f, work, deadline)?;
    (integer(f.pairs()[0].0) && f.pairs()[1..].iter().all(|p| p.0 == 0.)).then_some(f.pairs()[0].0)
}

fn count(lo: f64, hi: f64, work: &mut usize, deadline: Option<Instant>) -> Option<usize> {
    if !integer(lo) || !integer(hi) || lo > hi {
        return None;
    }
    let n = (hi - lo) as usize + 1;
    if n > MAX_POINTS {
        return None;
    }
    // Account for allocation and every grid position, including empty ones.
    debit(work, n, deadline)?;
    Some(n)
}

pub(super) fn front_algebra_pair(
    definition: &CoupledResource,
    path: &[usize],
    scale: f64,
    work: &mut usize,
    deadline: Option<Instant>,
) -> Option<(bool, [f64; 2])> {
    let CoupledResource::SlopePaired {
        initial,
        initial_base,
        arcs,
        mode,
        normalization,
        ..
    } = definition
    else {
        return None;
    };
    if !scale.is_finite() || scale < 0. {
        return None;
    }
    domain(initial, work, deadline)?;
    if !integer(initial.pairs()[0].0)
        || !initial.pairs()[1..]
            .iter()
            .all(|p| matches!(p.0, -1. | 0. | 1.))
        || !integer(initial_base.lower)
        || !integer(initial_base.upper)
    {
        return None;
    }
    let (start, end) = initial.domain();
    let mut lo = start.max(initial_base.lower);
    let hi = end.min(initial_base.upper);
    if lo > hi {
        return Some((false, [0.; 2]));
    }
    let n = count(lo, hi, work, deadline)?;
    let mut values = Vec::with_capacity(n);
    for offset in 0..n {
        debit(work, 1, deadline)?;
        let value = paired::slope_range(
            initial,
            bounds::Range::point(lo + offset as f64),
            work,
            deadline,
        )?;
        if value.lo != value.hi || !integer(value.lo) {
            return None;
        }
        values.push(value.lo);
    }
    let initial_minimum = values.iter().copied().reduce(f64::min)?;
    for &index in path {
        debit(work, 1, deadline)?;
        let arc = arcs.get(index)?;
        if !matches!(arc.waiting, -1..=1) {
            return None;
        }
        domain(&arc.arrival, work, deadline)?;
        if arc.arrival.pairs().len() > 2
            || arc.arrival.pairs()[0].1 != 0.
            || !arc.arrival.pairs()[1..].iter().all(|p| p.0 == 1.)
        {
            return None;
        }
        let delta = arc.arrival.pairs()[0].0;
        if !integer(delta) || delta < 0. {
            return None;
        }
        let increment = constant(&arc.increment, work, deadline)?;
        let lower = constant(&arc.lower, work, deadline)?;
        let upper = constant(&arc.upper, work, deadline)?;
        let (al, ah) = arc.arrival.domain();
        let (il, ih) = arc.increment.domain();
        let tail_lo = lo.max(al).max(il);
        let tail_hi = (lo + (values.len() - 1) as f64).min(ah).min(ih);
        if tail_lo > tail_hi || lower > upper {
            return Some((false, [0.; 2]));
        }
        let (ll, lh) = arc.lower.domain();
        let (ul, uh) = arc.upper.domain();
        let head_lo = ll.max(ul).max(tail_lo + delta);
        let head_hi = lh.min(uh);
        if head_lo > head_hi {
            return Some((false, [0.; 2]));
        }
        let n = count(head_lo, head_hi, work, deadline)?;
        let waiting = f64::from(arc.waiting);
        let mut prefix = f64::INFINITY;
        let mut tail = tail_lo;
        let mut next = Vec::with_capacity(n);
        let mut next_lo = None;
        let mut gap = false;
        for offset in 0..n {
            debit(work, 1, deadline)?;
            let head = head_lo + offset as f64;
            while tail <= tail_hi && tail + delta <= head {
                debit(work, 1, deadline)?;
                let value = values[(tail - lo) as usize];
                prefix = prefix.min(value - waiting * tail);
                tail += 1.;
            }
            let dependent = (prefix + increment + waiting * (head - delta)).max(lower);
            if dependent <= upper {
                if gap
                    || !integer(dependent)
                    || next
                        .last()
                        .is_some_and(|old: &f64| (dependent - old).abs() > 1.)
                {
                    // Disconnected domains need a separate waiting proof.
                    return None;
                }
                next_lo.get_or_insert(head);
                next.push(dependent);
            } else if next_lo.is_some() {
                gap = true;
            }
        }
        let Some(first) = next_lo else {
            return Some((false, [0.; 2]));
        };
        lo = first;
        values = next;
    }
    if path.is_empty() || *mode == PairedResourceMode::Feasibility || scale == 0. {
        return Some((true, [0.; 2]));
    }
    let baseline = match normalization {
        PairedCostNormalization::AbsoluteFinalValue => 0.,
        PairedCostNormalization::FinalMinusInitialMinimum => initial_minimum,
    };
    let minimum = values.into_iter().reduce(f64::min)?;
    let cost = bounds::Range::point(minimum)
        .subtract(bounds::Range::point(baseline))
        .multiply(bounds::Range::point(scale));
    cost.finite().then_some((true, [cost.lo, cost.hi]))
}
