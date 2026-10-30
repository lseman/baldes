//! Independent proofs of the declared raw MACHINE front program.
//!
//! Source descriptors and physical arcs are the only proof inputs. Arithmetic
//! is implemented on decoded integer significands, with nearest/even packing;
//! no production optimized frontier or minimum supplies evidence. This scope
//! does not assert equivalence to sequential source-control replay or to an
//! exact-real control problem. See `CoupledKernel::has_machine_front_contracts`.
use super::*;
use crate::graph::coupled::descriptor::{FrontOperationPolicy, PreparedSlopeFunction};

pub(super) type Function = Vec<(f64, f64)>;
pub(super) type Front = Vec<Function>;

const FRACTION: u64 = (1u64 << 52) - 1;
const SIGN: u64 = 1u64 << 63;
const MAX_PAIRS: usize = 4096;

#[derive(Clone, Copy)]
struct Binary {
    negative: bool,
    significand: u64,
    exponent: i32,
}

fn decode(x: f64) -> Option<Binary> {
    let bits = x.to_bits();
    let exponent = ((bits >> 52) & 2047) as i32;
    if exponent == 2047 {
        return None;
    }
    Some(Binary {
        negative: bits & SIGN != 0,
        significand: (bits & FRACTION) | if exponent == 0 { 0 } else { 1u64 << 52 },
        exponent: if exponent == 0 {
            -1074
        } else {
            exponent - 1075
        },
    })
}

/// Round a positive integer and a possible positive fractional remainder.
/// The remainder is smaller than one input integer unit. It can turn an exact
/// halfway case into round-up, but cannot cross any other rounding boundary.
fn rounded_shift(value: u128, shift: u32, sticky: bool) -> Option<u128> {
    if shift == 0 {
        return (!sticky).then_some(value);
    }
    if shift > 128 {
        return Some(0);
    }
    if shift == 128 {
        let halfway = 1u128 << 127;
        return Some(u128::from(value > halfway || value == halfway && sticky));
    }
    let quotient = value >> shift;
    let remainder = value & ((1u128 << shift) - 1);
    let halfway = 1u128 << (shift - 1);
    Some(
        quotient
            + u128::from(
                remainder > halfway || remainder == halfway && (sticky || quotient & 1 != 0),
            ),
    )
}

fn pack(negative: bool, value: u128, exponent: i32, sticky: bool) -> Option<f64> {
    let sign = if negative { SIGN } else { 0 };
    if value == 0 {
        return (!sticky).then_some(f64::from_bits(sign));
    }
    let length = 128 - value.leading_zeros();
    let mut top = exponent + length as i32 - 1;
    if top > 1023 {
        return None;
    }
    if top < -1022 {
        let shift = -1074 - exponent;
        let significand = if shift >= 0 {
            rounded_shift(value, shift as u32, sticky)?
        } else {
            if sticky {
                return None;
            }
            value.checked_shl((-shift) as u32)?
        };
        // A rounded subnormal may carry into the smallest normal value.
        return (significand <= 1u128 << 52).then_some(f64::from_bits(sign | significand as u64));
    }
    let shift = length as i32 - 53;
    let mut significand = if shift >= 0 {
        rounded_shift(value, shift as u32, sticky)?
    } else {
        if sticky {
            return None;
        }
        value.checked_shl((-shift) as u32)?
    };
    if significand == 1u128 << 53 {
        significand >>= 1;
        top += 1;
    }
    if top > 1023 || !(1u128 << 52..1u128 << 53).contains(&significand) {
        return None;
    }
    Some(f64::from_bits(
        sign | ((top + 1023) as u64) << 52 | significand as u64 & FRACTION,
    ))
}

fn ieee_add(a: f64, b: f64) -> Option<f64> {
    let mut left = decode(a)?;
    let mut right = decode(b)?;
    if left.significand == 0 && right.significand == 0 {
        return Some(f64::from_bits(if left.negative && right.negative {
            SIGN
        } else {
            0
        }));
    }
    if left.significand == 0 {
        return Some(b);
    }
    if right.significand == 0 {
        return Some(a);
    }
    let mut large = a;
    if left.exponent < right.exponent {
        std::mem::swap(&mut left, &mut right);
        large = b;
    }
    let distance = (left.exponent - right.exponent) as u32;
    if distance > 74 {
        // The larger operand is normal and has a full53-bit significand.
        // The other is <2^-22 of its least input significand unit, also below
        // half the smaller neighboring ulp at a power-of-two boundary.
        return Some(large);
    }
    let l = u128::from(left.significand) << distance;
    let r = u128::from(right.significand);
    let (negative, magnitude) = if left.negative == right.negative {
        (left.negative, l.checked_add(r)?)
    } else if l >= r {
        (left.negative && l != r, l - r)
    } else {
        (right.negative, r - l)
    };
    pack(negative, magnitude, right.exponent, false)
}

fn ieee_multiply(a: f64, b: f64) -> Option<f64> {
    let a = decode(a)?;
    let b = decode(b)?;
    pack(
        a.negative ^ b.negative,
        u128::from(a.significand) * u128::from(b.significand),
        a.exponent + b.exponent,
        false,
    )
}

pub(super) struct Machine<'a> {
    work: &'a mut usize,
    deadline: Option<Instant>,
}

impl<'a> Machine<'a> {
    pub(super) fn new(work: &'a mut usize, deadline: Option<Instant>) -> Self {
        Self { work, deadline }
    }
    pub(super) fn take(&mut self, count: usize) -> Option<()> {
        debit(self.work, count, self.deadline)
    }
    pub(super) fn add(&mut self, a: f64, b: f64) -> Option<f64> {
        self.take(1)?;
        ieee_add(a, b)
    }
    pub(super) fn sub(&mut self, a: f64, b: f64) -> Option<f64> {
        self.add(a, f64::from_bits(b.to_bits() ^ SIGN))
    }
    pub(super) fn mul(&mut self, a: f64, b: f64) -> Option<f64> {
        self.take(1)?;
        ieee_multiply(a, b)
    }
    pub(super) fn near(&mut self, a: f64, b: f64, policy: FrontOperationPolicy) -> Option<bool> {
        self.take(1)?;
        if a == b {
            return Some(true);
        }
        if policy == FrontOperationPolicy::StrictDomain {
            return Some(false);
        }
        let difference = self.sub(a, b)?.abs();
        if difference < 0.0001 {
            return Some(true);
        }
        let sum = self.add(a, b)?.abs();
        Some(difference < self.mul(2.220446049250313e-10, sum)?)
    }
}

pub(super) fn prepare(pairs: &[(f64, f64)], m: &mut Machine<'_>) -> Option<Function> {
    m.take(pairs.len())?;
    if pairs.is_empty()
        || pairs.len() > MAX_PAIRS
        || pairs.iter().any(|p| !p.0.is_finite() || !p.1.is_finite())
    {
        return None;
    }
    let (mut value, mut x) = pairs[0];
    for &(slope, end) in &pairs[1..] {
        if end <= x {
            return None;
        }
        let span = m.sub(end, x)?;
        let product = m.mul(slope, span)?;
        value = m.add(value, product)?;
        x = end;
    }
    Some(pairs.to_vec())
}

pub(super) fn validate(front: &Front, m: &mut Machine<'_>) -> Option<()> {
    m.take(front.len())?;
    if front.len() > MAX_PAIRS {
        return None;
    }
    let mut count = 0usize;
    let mut previous = None;
    for f in front {
        count = count.checked_add(f.len())?;
        if count > MAX_PAIRS || f.is_empty() || previous.is_some_and(|x| f[0].1 <= x) {
            return None;
        }
        prepare(f, m)?;
        previous = Some(f.last()?.1);
    }
    Some(())
}

pub(super) fn evaluate(
    f: &Function,
    x: f64,
    policy: FrontOperationPolicy,
    m: &mut Machine<'_>,
) -> Option<f64> {
    let (mut value, mut previous) = *f.first()?;
    if m.near(x, previous, policy)? {
        return Some(value);
    }
    if x < previous || policy == FrontOperationPolicy::StrictDomain && x > f.last()?.1 {
        return None;
    }
    for &(slope, end) in &f[1..] {
        m.take(1)?;
        let span = m.sub(x.min(end), previous)?;
        let product = m.mul(slope, span)?;
        value = m.add(value, product)?;
        if x <= end {
            return Some(value);
        }
        previous = end;
    }
    m.near(x, previous, policy)?.then_some(value)
}

pub(super) fn restrict(
    front: &Front,
    lower: f64,
    upper: f64,
    policy: FrontOperationPolicy,
    m: &mut Machine<'_>,
) -> Option<Front> {
    if !lower.is_finite() || !upper.is_finite() || lower > upper {
        return None;
    }
    let mut out = vec![];
    for f in front {
        m.take(f.len())?;
        let lo = f.first()?.1.max(lower);
        let hi = f.last()?.1.min(upper);
        if lo > hi {
            continue;
        }
        if policy == FrontOperationPolicy::FrontAlgebra {
            for &(_, boundary) in f {
                for point in [lo, hi] {
                    if point != boundary && m.near(point, boundary, policy)? {
                        return None;
                    }
                }
            }
        }
        let mut pairs = vec![(evaluate(f, lo, policy, m)?, lo)];
        if lo != hi {
            for &(slope, end) in &f[1..] {
                m.take(1)?;
                if end <= lo {
                    continue;
                }
                pairs.push((slope, end.min(hi)));
                if end >= hi {
                    break;
                }
            }
        }
        out.push(prepare(&pairs, m)?);
    }
    validate(&out, m)?;
    Some(out)
}

fn sum_component(
    a: &Function,
    b: &Function,
    policy: FrontOperationPolicy,
    m: &mut Machine<'_>,
) -> Option<Option<Function>> {
    let (alo, ahi, blo, bhi) = (a.first()?.1, a.last()?.1, b.first()?.1, b.last()?.1);
    if alo > bhi || blo > ahi {
        if policy == FrontOperationPolicy::FrontAlgebra
            && (m.near(alo, bhi, policy)? || m.near(blo, ahi, policy)?)
        {
            return None;
        }
        return Some(None);
    }
    let same = m.near(alo, blo, policy)?;
    let start = if same { alo } else { alo.max(blo) };
    let av = if same {
        a[0].0
    } else {
        evaluate(a, start, policy, m)?
    };
    let bv = if same {
        b[0].0
    } else {
        evaluate(b, start, policy, m)?
    };
    let mut out = vec![(m.add(av, bv)?, start)];
    if a.len() == 1 || b.len() == 1 || start == ahi.min(bhi) {
        return (start == ahi.min(bhi)).then(|| Some(out));
    }
    let (mut i, mut j) = (1, 1);
    while i < a.len() && a[i].1 <= start {
        m.take(1)?;
        i += 1;
    }
    while j < b.len() && b[j].1 <= start {
        m.take(1)?;
        j += 1;
    }
    let mut previous = start;
    while i < a.len() && j < b.len() {
        m.take(1)?;
        let (sa, xa) = a[i];
        let (sb, xb) = b[j];
        let same = m.near(xa, xb, policy)?;
        let end = if same { xa } else { xa.min(xb) };
        if end <= previous {
            return None;
        }
        let slope = m.add(sa, sb)?;
        if policy == FrontOperationPolicy::FrontAlgebra
            && out.len() > 1
            && m.near(slope, out.last()?.0, policy)?
        {
            out.last_mut()?.1 = end;
        } else {
            out.push((slope, end));
        }
        if out.len() > MAX_PAIRS {
            return None;
        }
        previous = end;
        if same {
            i += 1;
            j += 1;
        } else if xa < xb {
            i += 1;
        } else {
            j += 1;
        }
    }
    Some(Some(prepare(&out, m)?))
}

pub(super) fn add(
    a: &Front,
    b: &Front,
    policy: FrontOperationPolicy,
    m: &mut Machine<'_>,
) -> Option<Front> {
    let mut out = vec![];
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        m.take(1)?;
        if let Some(f) = sum_component(&a[i], &b[j], policy, m)? {
            out.push(f);
        }
        let (ae, be) = (a[i].last()?.1, b[j].last()?.1);
        if ae <= be {
            i += 1;
        }
        if be <= ae {
            j += 1;
        }
    }
    validate(&out, m)?;
    Some(out)
}

pub(super) fn minimum(front: &Front, m: &mut Machine<'_>) -> Option<Option<f64>> {
    let mut best = None;
    for f in front {
        m.take(f.len())?;
        let (mut value, mut x) = *f.first()?;
        if best.is_none_or(|old| value < old) {
            best = Some(value);
        }
        for &(slope, end) in &f[1..] {
            let span = m.sub(end, x)?;
            let product = m.mul(slope, span)?;
            value = m.add(value, product)?;
            if best.is_none_or(|old| value < old) {
                best = Some(value);
            }
            x = end;
        }
    }
    Some(best)
}

/// Independent source-domain controls for DEFAULT raw feasibility.
///
/// The machine cost contract does not supply this witness. Outward replay of
/// one shared initial control encloses both exact-real and sequential binary64
/// increments. Staying in nominal source domains is a sufficient subset of
/// native leaf acceptance. Failure to find a witness is unknown. Only an empty
/// hard-window OUTER image (dropping source restrictions) proves infeasibility.
pub(super) fn source_feasibility(
    initial: Window,
    bounds: Window,
    increments: &[f64],
    functions: &[PreparedSlopeFunction],
    path: &[usize],
    work: &mut usize,
    deadline: Option<Instant>,
) -> Option<bool> {
    let mut outer = bounds::Range {
        lo: initial.lower,
        hi: initial.upper,
    };
    let mut shift = bounds::Range::point(0.);
    let (mut inner_lo, mut inner_hi) = (initial.lower, initial.upper);
    let mut candidates = vec![
        initial.lower,
        initial.upper,
        initial.lower / 2. + initial.upper / 2.,
    ];
    for &index in path {
        debit(work, 1, deadline)?;
        let delta = bounds::Range::point(*increments.get(index)?);
        outer = outer.add(delta);
        shift = shift.add(delta);
        if !outer.finite() || !shift.finite() {
            return None;
        }
        outer.lo = outer.lo.max(bounds.lower);
        outer.hi = outer.hi.min(bounds.upper);
        if outer.lo > outer.hi {
            return Some(false);
        }
        let source = functions.get(index)?;
        let (lo, hi) = source.domain();
        let lower = bounds::Range::point(bounds.lower.max(lo)).subtract(shift);
        let upper = bounds::Range::point(bounds.upper.min(hi)).subtract(shift);
        if lower.finite() && upper.finite() {
            inner_lo = inner_lo.max(lower.hi);
            inner_hi = inner_hi.min(upper.lo);
        }
        if candidates.len() < 256 {
            debit(work, source.pairs().len(), deadline)?;
            for &(_, x) in source.pairs() {
                if candidates.len() >= 256 {
                    break;
                }
                let suggested = bounds::Range::point(x).subtract(shift);
                if suggested.finite() {
                    candidates.extend([suggested.lo, suggested.hi]);
                }
            }
        }
    }
    candidates.extend([inner_lo, inner_hi, inner_lo / 2. + inner_hi / 2.]);
    for initial_control in candidates {
        debit(work, 1, deadline)?;
        if !initial_control.is_finite()
            || initial_control < initial.lower
            || initial_control > initial.upper
        {
            continue;
        }
        let mut state = bounds::Range::point(initial_control);
        let mut valid = true;
        for &index in path {
            debit(work, 1, deadline)?;
            state = state.add(bounds::Range::point(*increments.get(index)?));
            let source = functions.get(index)?;
            let (lo, hi) = source.domain();
            if !state.finite()
                || state.lo < bounds.lower
                || state.hi > bounds.upper
                || state.lo < lo
                || state.hi > hi
            {
                valid = false;
                break;
            }
            // Native first-coordinate testing must not overflow before a
            // source value is returned; retaining a nominal-domain control
            // alone would not justify that arithmetic branch.
            if !state.add(bounds::Range::point(lo)).finite()
                || !state.subtract(bounds::Range::point(lo)).finite()
                || super::paired::slope_range(source, state, work, deadline).is_none()
            {
                valid = false;
                break;
            }
        }
        if valid {
            return Some(true);
        }
    }
    None
}

pub(super) fn raw(
    initial: Window,
    bounds: Window,
    increments: &[f64],
    functions: &[PreparedSlopeFunction],
    path: &[usize],
    scale: f64,
    work: &mut usize,
    deadline: Option<Instant>,
) -> Option<(bool, [f64; 2])> {
    if !(scale == 0. || scale == 1.) {
        return None;
    }
    let policy = FrontOperationPolicy::FrontAlgebra;
    let mut m = Machine::new(work, deadline);
    let root = if initial.lower == initial.upper {
        vec![(0., initial.lower)]
    } else {
        vec![(0., initial.lower), (0., initial.upper)]
    };
    let mut front = vec![prepare(&root, &mut m)?];
    for &index in path {
        m.take(1)?;
        let delta = *increments.get(index)?;
        let mut translated = vec![];
        for f in &front {
            let mut pairs = vec![];
            for &(s, x) in f {
                pairs.push((s, m.add(x, delta)?));
            }
            translated.push(prepare(&pairs, &mut m)?);
        }
        front = restrict(&translated, bounds.lower, bounds.upper, policy, &mut m)?;
        if front.is_empty() {
            return Some((false, [0.; 2]));
        }
        let source = functions.get(index)?;
        let pairs = if scale == 0. {
            m.take(source.pairs().len())?;
            source.pairs().iter().map(|&(_, x)| (0., x)).collect()
        } else {
            prepare(source.pairs(), &mut m)?
        };
        front = add(&front, &vec![pairs], policy, &mut m)?;
        if front.is_empty() {
            return Some((false, [0.; 2]));
        }
        minimum(&front, &mut m)??;
    }
    let value = if path.is_empty() {
        0.
    } else {
        minimum(&front, &mut m)??
    };
    Some((true, [value, value]))
}

#[cfg(test)]
#[path = "../../../../../tests/unit/graph/coupled/kernel/machine_certificate_tests.rs"]
mod tests;
