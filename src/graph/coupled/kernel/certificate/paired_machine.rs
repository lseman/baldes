//! Independent interpreter for the declared IEEE native paired-front program.
//!
//! Software binary64 operations reconstruct components from raw descriptors.
//! This certifies the MACHINE front contract, never the source-control domain
//! or an exact-real envelope. Production optimized fronts/minima are not read.
use super::machine::{self, Front, Function, Machine};
use super::*;
use crate::graph::coupled::descriptor::{
    FrontOperationPolicy, PairedCostNormalization, SlopeRelationArc,
};

macro_rules! add {
    ($m:expr,$a:expr,$b:expr) => {{
        let a = $a;
        let b = $b;
        $m.add(a, b)?
    }};
}
macro_rules! sub {
    ($m:expr,$a:expr,$b:expr) => {{
        let a = $a;
        let b = $b;
        $m.sub(a, b)?
    }};
}
macro_rules! mul {
    ($m:expr,$a:expr,$b:expr) => {{
        let a = $a;
        let b = $b;
        $m.mul(a, b)?
    }};
}
macro_rules! div {
    ($m:expr,$a:expr,$b:expr) => {{
        let a = $a;
        let b = $b;
        $m.div(a, b)?
    }};
}
const NATIVE: FrontOperationPolicy = FrontOperationPolicy::FrontAlgebra;
const STRICT: FrontOperationPolicy = FrontOperationPolicy::StrictDomain;
const MAX_PAIRS: usize = 4096;

fn domain(f: &Function) -> (f64, f64) {
    (f[0].1, f.last().unwrap().1)
}
fn at(m: &mut Machine<'_>, x: f64, y: f64, slope: f64, argument: f64) -> Option<f64> {
    if argument == x {
        Some(y)
    } else {
        Some(add!(m, y, mul!(m, slope, sub!(m, argument, x))))
    }
}
fn crossing(m: &mut Machine<'_>, x: f64, y: f64, slope: f64, target: f64) -> Option<f64> {
    Some(add!(m, x, div!(m, sub!(m, target, y), slope)))
}
fn inverse(f: &Function, m: &mut Machine<'_>) -> Option<Function> {
    m.take(f.len())?;
    let (mut y, mut x) = f[0];
    let mut out = vec![(x, y)];
    for &(slope, end) in &f[1..] {
        m.take(1)?;
        if slope <= 0. {
            return None;
        }
        let mapped = add!(m, y, mul!(m, sub!(m, end, x), slope));
        if mapped <= y {
            return None;
        }
        out.push((div!(m, 1., slope), mapped));
        x = end;
        y = mapped;
    }
    machine::prepare(&out, m)
}

struct Cached<'a> {
    function: &'a Function,
    knots: Vec<(f64, f64)>,
}
impl<'a> Cached<'a> {
    fn new(function: &'a Function, increasing: bool, m: &mut Machine<'_>) -> Option<Self> {
        m.take(function.len())?;
        let (mut y, mut x) = function[0];
        let mut knots = vec![(x, y)];
        for &(slope, end) in &function[1..] {
            if increasing && slope <= 0. {
                return None;
            }
            let next = add!(m, y, mul!(m, slope, sub!(m, end, x)));
            if increasing && next <= y {
                return None;
            }
            knots.push((end, next));
            x = end;
            y = next;
        }
        Some(Self { function, knots })
    }
    fn value(&self, argument: f64, m: &mut Machine<'_>) -> Option<f64> {
        m.take(self.knots.len())?;
        let (lo, hi) = domain(self.function);
        if argument < lo || argument > hi {
            return None;
        }
        if argument == lo || m.near(argument, lo, NATIVE)? {
            return Some(self.knots[0].1);
        }
        let i = self.knots.partition_point(|p| p.0 < argument);
        if self.knots.get(i)?.0 == argument {
            return Some(self.knots[i].1);
        }
        let (x, y) = self.knots[i - 1];
        at(m, x, y, self.function[i].0, argument)
    }
}

fn emit(
    out: &mut Function,
    last: &mut Option<(f64, f64)>,
    x: f64,
    value: f64,
    inner_endpoint: bool,
    m: &mut Machine<'_>,
) -> Option<()> {
    m.take(1)?;
    let (px, py) = (*last)?;
    if x == px && value == py {
        return Some(());
    }
    if inner_endpoint && m.near(x, px, NATIVE)? {
        return Some(());
    }
    if x <= px {
        return None;
    }
    let slope = div!(m, sub!(m, value, py), sub!(m, x, px));
    let merge = out.len() > 1 && m.near(out.last()?.0, slope, NATIVE)?;
    if merge {
        let pair = out.last_mut()?;
        if inner_endpoint {
            pair.0 = slope;
        }
        pair.1 = x;
    } else {
        if out.len() >= MAX_PAIRS {
            return None;
        }
        out.push((slope, x));
    }
    *last = Some((x, value));
    Some(())
}

/// Paired callers intentionally clip the feasible preimage. This is not the
/// standalone native helper's exhausted-outer-domain return contract.
fn compose(
    outer: &Function,
    inner: &Function,
    lower: f64,
    upper: f64,
    m: &mut Machine<'_>,
) -> Option<Option<Function>> {
    let inner = Cached::new(inner, true, m)?;
    let outer = Cached::new(outer, false, m)?;
    if !lower.is_finite() || !upper.is_finite() || lower > upper {
        return None;
    }
    if lower < domain(inner.function).0 && m.near(lower, domain(inner.function).0, NATIVE)? {
        return None;
    }
    let lo = lower.max(domain(inner.function).0);
    let hi = upper.min(domain(inner.function).1);
    if lo > hi {
        return if m.near(lo, hi, NATIVE)? {
            None
        } else {
            Some(None)
        };
    }
    if m.near(lo, hi, NATIVE)? {
        let y = machine::evaluate(inner.function, lo, NATIVE, m)?;
        let value = machine::evaluate(outer.function, y, NATIVE, m)?;
        let out = if lo == hi {
            vec![(value, lo)]
        } else {
            vec![(value, lo), (0., hi)]
        };
        return Some(Some(machine::prepare(&out, m)?));
    }
    let (outer_lo, outer_hi) = domain(outer.function);
    if inner.knots.len() == 1 {
        let (x, y) = inner.knots[0];
        if y < outer_lo || y > outer_hi {
            return if m.near(y, outer_lo, NATIVE)? || m.near(y, outer_hi, NATIVE)? {
                None
            } else {
                Some(None)
            };
        }
        return Some(Some(machine::prepare(&[(outer.value(y, m)?, x)], m)?));
    }
    let mut out = vec![];
    let mut last = None;
    let mut inner_end = None;
    let mut outer_anchor: Option<(f64, f64)> = None;
    let mut outer_next = None;
    for i in 1..inner.knots.len() {
        m.take(1)?;
        let (gx, gy) = inner.knots[i - 1];
        let (ge, _) = inner.knots[i];
        let slope = inner.function[i].0;
        let mut x0 = gx.max(lo);
        let mut x1 = ge.min(hi);
        if x0 > x1 {
            continue;
        }
        let anchor_y = match inner_end {
            Some(y) => y,
            None if m.near(lo, domain(inner.function).0, NATIVE)? => inner.knots[0].1,
            None => at(m, gx, gy, slope, x0)?,
        };
        let anchor_x = x0;
        let mut y0 = at(m, anchor_x, anchor_y, slope, x0)?;
        let mut y1 = at(m, anchor_x, anchor_y, slope, x1)?;
        inner_end = Some(y1);
        if y0 > outer_hi || y1 < outer_lo {
            if m.near(y0, outer_hi, NATIVE)? || m.near(y1, outer_lo, NATIVE)? {
                return None;
            }
            continue;
        }
        if y0 < outer_lo {
            x0 = crossing(m, anchor_x, anchor_y, slope, outer_lo)?;
            y0 = outer_lo;
        }
        if y1 > outer_hi {
            x1 = crossing(m, anchor_x, anchor_y, slope, outer_hi)?;
            y1 = outer_hi;
        }
        if x0 < lo || x1 > hi || x0 > x1 || (x0 == x1 && y0 != y1) {
            return None;
        }
        let f0 = match last {
            Some((x, y)) if x == x0 => y,
            _ => outer.value(y0, m)?,
        };
        if outer_anchor.is_none() {
            outer_anchor = Some((y0, f0));
        }
        match last {
            None => {
                out.push((f0, x0));
                last = Some((x0, f0));
            }
            Some((x, y)) if x == x0 && y == f0 => (),
            _ => return None,
        }
        let mut next = match outer_next {
            Some(n) => n,
            None => {
                let n = outer.knots.partition_point(|p| p.0 < y0);
                if m.near(y0, outer_lo, NATIVE)? {
                    1
                } else if n < outer.knots.len() && m.near(outer.knots[n].0, y0, NATIVE)? {
                    n + 1
                } else {
                    n
                }
            }
        };
        while next < outer.knots.len()
            && (outer.knots[next].0 < y1 || m.near(outer.knots[next].0, y1, NATIVE)?)
        {
            m.take(1)?;
            let (target, cached) = outer.knots[next];
            let value = if let Some((y, v)) = outer_anchor {
                let v = at(m, y, v, outer.function[next].0, target)?;
                outer_anchor = Some((target, v));
                v
            } else {
                cached
            };
            let x = if m.near(target, y1, NATIVE)? {
                x1
            } else {
                crossing(m, anchor_x, anchor_y, slope, target)?
            };
            if x < x0 || x > x1 {
                return None;
            }
            emit(&mut out, &mut last, x, value, false, m)?;
            next += 1;
        }
        let end = if last.is_some_and(|(x, _)| x == x1) {
            last?.1
        } else if let Some((y, v)) = outer_anchor {
            if y1 == y {
                v
            } else {
                at(m, y, v, outer.function.get(next)?.0, y1)?
            }
        } else {
            outer.value(y1, m)?
        };
        if outer.knots.get(next).is_some_and(|p| p.0 == y1) {
            outer_anchor = Some((y1, end));
        }
        outer_next = Some(next);
        let inner_endpoint = !outer.knots.get(next).is_some_and(|p| p.0 == y1);
        emit(&mut out, &mut last, x1, end, inner_endpoint, m)?;
    }
    if out.is_empty() {
        Some(None)
    } else {
        Some(Some(machine::prepare(&out, m)?))
    }
}

#[derive(Clone, Copy)]
struct Segment {
    lo: f64,
    hi: f64,
    y: f64,
    slope: f64,
}
impl Segment {
    fn at(self, x: f64, m: &mut Machine<'_>) -> Option<f64> {
        Some(add!(m, self.y, mul!(m, self.slope, sub!(m, x, self.lo))))
    }
}
#[derive(Clone, Copy)]
struct Translation {
    delta: f64,
    phi: f64,
    tail_lo: f64,
    tail_hi: f64,
    head_lo: f64,
    head_hi: f64,
}
fn constant(f: &[(f64, f64)]) -> Option<f64> {
    f[1..].iter().all(|p| p.0 == 0.).then_some(f[0].0)
}
fn translation(a: &SlopeRelationArc) -> Option<Translation> {
    if a.arrival.pairs().len() > 2
        || a.arrival.pairs()[0].1 != 0.
        || !a.arrival.pairs()[1..].iter().all(|p| p.0 == 1.)
        || a.arrival.pairs()[0].0 < 0.
    {
        return None;
    }
    let phi = constant(a.increment.pairs())?;
    let (al, ah) = a.arrival.domain();
    let (pl, ph) = a.increment.domain();
    let (ll, lh) = a.lower.domain();
    let (ul, uh) = a.upper.domain();
    Some(Translation {
        delta: a.arrival.pairs()[0].0,
        phi,
        tail_lo: al.max(pl),
        tail_hi: ah.min(ph),
        head_lo: ll.max(ul),
        head_hi: lh.min(uh),
    })
}
fn exact_transport(front: &Front, t: Translation, m: &mut Machine<'_>) -> Option<bool> {
    let integer = |x: f64| x.is_finite() && x.abs() <= 1_099_511_627_776. && x.fract() == 0.;
    if ![t.delta, t.phi, t.tail_lo, t.tail_hi]
        .into_iter()
        .all(integer)
    {
        return Some(false);
    }
    for f in front {
        m.take(f.len())?;
        let (mut y, mut x) = f[0];
        y = add!(m, y, t.phi);
        if !integer(x) || !integer(add!(m, x, t.delta)) || !integer(f[0].0) || !integer(y) {
            return Some(false);
        }
        for &(s, e) in &f[1..] {
            let p = mul!(m, s, sub!(m, e, x));
            y = add!(m, y, p);
            if ![s, e, add!(m, e, t.delta), p, y].into_iter().all(integer) {
                return Some(false);
            }
            x = e;
        }
    }
    Some(true)
}
fn push(out: &mut Vec<Segment>, s: Segment, m: &mut Machine<'_>) -> Option<()> {
    m.take(1)?;
    if out.len() >= MAX_PAIRS {
        return None;
    }
    out.push(s);
    Some(())
}
fn assemble_waiting(
    parts: Vec<Segment>,
    head_lo: f64,
    head_hi: f64,
    m: &mut Machine<'_>,
) -> Option<Front> {
    let mut components = vec![];
    let mut current: Function = vec![];
    let mut current_y = 0.;
    for s in parts {
        m.take(1)?;
        let lo = s.lo.max(head_lo);
        let hi = s.hi.min(head_hi);
        if lo > hi {
            continue;
        }
        if (lo != s.lo && m.near(lo, s.lo, NATIVE)?) || (hi != s.hi && m.near(hi, s.hi, NATIVE)?) {
            return None;
        }
        let y = s.at(lo, m)?;
        let f = if lo == hi {
            vec![(y, lo)]
        } else {
            vec![(y, lo), (s.slope, hi)]
        };
        if let Some(last) = current.last() {
            if domain(&f).0 < last.1 {
                if f.len() == 1 {
                    continue;
                }
                return None;
            }
            if domain(&f).0 == last.1 {
                if current_y != f[0].0 {
                    return None;
                }
                current_y = machine::evaluate(&f, domain(&f).1, STRICT, m)?;
                current.extend_from_slice(&f[1..]);
                continue;
            }
            components.push(machine::prepare(&std::mem::take(&mut current), m)?);
        }
        current_y = machine::evaluate(&f, domain(&f).1, STRICT, m)?;
        current.extend_from_slice(&f);
    }
    if !current.is_empty() {
        components.push(machine::prepare(&current, m)?);
    }
    machine::validate(&components, m)?;
    Some(components)
}
fn waiting(front: &Front, t: Translation, k: f64, m: &mut Machine<'_>) -> Option<Front> {
    if t.tail_lo > t.tail_hi || t.head_lo > t.head_hi {
        return Some(vec![]);
    }
    let mut input = vec![];
    for f in front {
        m.take(f.len())?;
        let (mut y, mut x) = f[0];
        y = add!(m, y, t.phi);
        if f.len() == 1 && x >= t.tail_lo && x <= t.tail_hi {
            let lo = add!(m, x, t.delta);
            push(
                &mut input,
                Segment {
                    lo,
                    hi: lo,
                    y,
                    slope: 0.,
                },
                m,
            )?;
        }
        for &(s, e) in &f[1..] {
            let lo = x.max(t.tail_lo);
            let hi = e.min(t.tail_hi);
            if lo <= hi {
                if (lo != x && m.near(lo, x, NATIVE)?) || (hi != e && m.near(hi, e, NATIVE)?) {
                    return None;
                }
                let left = add!(m, lo, t.delta);
                let right = add!(m, hi, t.delta);
                if lo < hi && left >= right {
                    return None;
                }
                push(
                    &mut input,
                    Segment {
                        lo: left,
                        hi: right,
                        y: add!(m, y, mul!(m, s, sub!(m, lo, x))),
                        slope: s,
                    },
                    m,
                )?;
            }
            y = add!(m, y, mul!(m, s, sub!(m, e, x)));
            x = e;
        }
    }
    if input.is_empty() {
        return Some(vec![]);
    }
    let mut out = vec![];
    let mut end = input[0].lo;
    let mut record = input[0].y;
    for (i, s) in input.into_iter().enumerate() {
        m.take(1)?;
        if s.lo > end {
            push(
                &mut out,
                Segment {
                    lo: end,
                    hi: s.lo,
                    y: record,
                    slope: k,
                },
                m,
            )?;
            record = add!(m, record, mul!(m, k, sub!(m, s.lo, end)));
        }
        if s.y < record {
            if i > 0 {
                return None;
            }
            record = s.y;
        }
        let difference = sub!(m, s.y, record);
        if s.slope < k && difference > 0. {
            let cross = add!(m, s.lo, div!(m, difference, sub!(m, k, s.slope)));
            if cross < s.hi {
                let dx = sub!(m, cross, s.lo);
                let cy = add!(m, s.y, mul!(m, s.slope, dx));
                if cy != add!(m, record, mul!(m, k, dx)) {
                    return None;
                }
                push(
                    &mut out,
                    Segment {
                        lo: s.lo,
                        hi: cross,
                        y: record,
                        slope: k,
                    },
                    m,
                )?;
                push(
                    &mut out,
                    Segment {
                        lo: cross,
                        hi: s.hi,
                        y: cy,
                        slope: s.slope,
                    },
                    m,
                )?;
                record = add!(m, cy, mul!(m, s.slope, sub!(m, s.hi, cross)));
            } else {
                push(
                    &mut out,
                    Segment {
                        lo: s.lo,
                        hi: s.hi,
                        y: record,
                        slope: k,
                    },
                    m,
                )?;
                record = add!(m, record, mul!(m, k, sub!(m, s.hi, s.lo)));
            }
        } else {
            let slope = if difference == 0. { s.slope.min(k) } else { k };
            push(
                &mut out,
                Segment {
                    lo: s.lo,
                    hi: s.hi,
                    y: record,
                    slope,
                },
                m,
            )?;
            record = add!(m, record, mul!(m, slope, sub!(m, s.hi, s.lo)));
        }
        end = s.hi;
    }
    if end < t.head_hi {
        push(
            &mut out,
            Segment {
                lo: end,
                hi: t.head_hi,
                y: record,
                slope: k,
            },
            m,
        )?;
    }
    assemble_waiting(out, t.head_lo, t.head_hi, m)
}

fn segments(f: &Function, m: &mut Machine<'_>) -> Option<Vec<Segment>> {
    m.take(f.len())?;
    let (mut y, mut x) = f[0];
    let mut out = vec![];
    if f.len() == 1 {
        out.push(Segment {
            lo: x,
            hi: x,
            y,
            slope: 0.,
        });
    }
    for &(s, e) in &f[1..] {
        out.push(Segment {
            lo: x,
            hi: e,
            y,
            slope: s,
        });
        y = add!(m, y, mul!(m, s, sub!(m, e, x)));
        x = e;
    }
    Some(out)
}
fn segment_index(s: &[Segment], x: f64) -> usize {
    s.partition_point(|s| s.hi < x).min(s.len() - 1)
}
fn near_bounds(a: f64, b: f64, m: &mut Machine<'_>) -> Option<bool> {
    let distance = sub!(m, a, b).abs();
    let sum = add!(m, a, b).abs();
    Some(distance < 0.0001 || distance < mul!(m, sum, 2.220446049250313e-10))
}
fn bound_cross(
    a: Segment,
    b: Segment,
    lo: f64,
    hi: f64,
    m: &mut Machine<'_>,
) -> Option<Option<f64>> {
    m.take(1)?;
    let al = a.at(lo, m)?;
    let bl = b.at(lo, m)?;
    let ah = a.at(hi, m)?;
    let bh = b.at(hi, m)?;
    if al == bl && ah == bh && a.slope != b.slope {
        return None;
    }
    if (al <= bl && ah <= bh) || (al >= bl && ah >= bh) {
        return Some(None);
    }
    let denominator = sub!(m, a.slope, b.slope);
    if denominator == 0. {
        return None;
    }
    let x = div!(
        m,
        sub!(
            m,
            sub!(m, bh, mul!(m, hi, b.slope)),
            sub!(m, ah, mul!(m, hi, a.slope))
        ),
        denominator
    );
    if x <= lo || x >= hi || a.at(x, m)? != b.at(x, m)? {
        return None;
    }
    Some(Some(x))
}
fn bound_point(
    out: &mut Vec<Segment>,
    x: f64,
    f: Segment,
    l: Segment,
    u: Segment,
    m: &mut Machine<'_>,
) -> Option<()> {
    let y = f.at(x, m)?.max(l.at(x, m)?);
    if y <= u.at(x, m)? {
        push(
            out,
            Segment {
                lo: x,
                hi: x,
                y,
                slope: 0.,
            },
            m,
        )?;
    }
    Some(())
}
fn bound_cell(
    out: &mut Vec<Segment>,
    lo: f64,
    hi: f64,
    f: Segment,
    l: Segment,
    u: Segment,
    m: &mut Machine<'_>,
) -> Option<()> {
    m.take(1)?;
    let mid = add!(m, mul!(m, lo, 0.5), mul!(m, hi, 0.5));
    if mid <= lo || mid >= hi {
        return None;
    }
    let (selected, other) = if f.at(mid, m)? >= l.at(mid, m)? {
        (f, l)
    } else {
        (l, f)
    };
    if selected.at(lo, m)? < other.at(lo, m)? || selected.at(hi, m)? < other.at(hi, m)? {
        return None;
    }
    let left = selected.at(lo, m)?;
    let right = selected.at(hi, m)?;
    let ul = u.at(lo, m)?;
    let ur = u.at(hi, m)?;
    if left <= ul && right <= ur {
        let s = Segment {
            lo,
            hi,
            y: left,
            slope: selected.slope,
        };
        if s.at(hi, m)? != right {
            return None;
        }
        push(out, s, m)?;
    } else if left >= ul && right >= ur {
        if left == ul && right == ur {
            return None;
        }
    } else {
        return None;
    }
    Some(())
}
fn assemble_bounds(mut pieces: Vec<Segment>, m: &mut Machine<'_>) -> Option<Front> {
    m.take(pieces.len().checked_mul(64)?)?;
    pieces.sort_by(|a, b| a.lo.total_cmp(&b.lo).then(a.hi.total_cmp(&b.hi)));
    let mut front = vec![];
    let mut f: Function = vec![];
    let (mut end, mut value) = (0., 0.);
    for s in pieces {
        m.take(1)?;
        if !f.is_empty() && s.lo < end {
            return None;
        }
        if !f.is_empty() && s.lo == end {
            if value != s.y {
                return None;
            }
            if s.hi == end {
                continue;
            }
            f.push((s.slope, s.hi));
            end = s.hi;
            value = s.at(end, m)?;
            continue;
        }
        if !f.is_empty() {
            front.push(machine::prepare(&std::mem::take(&mut f), m)?);
        }
        f.push((s.y, s.lo));
        end = s.lo;
        value = s.y;
        if s.hi > s.lo {
            f.push((s.slope, s.hi));
            end = s.hi;
            value = s.at(end, m)?;
        }
    }
    if !f.is_empty() {
        front.push(machine::prepare(&f, m)?);
    }
    machine::validate(&front, m)?;
    Some(front)
}
fn clamp_filter(
    front: &Front,
    lower: &Function,
    upper: &Function,
    m: &mut Machine<'_>,
) -> Option<Front> {
    let ls = segments(lower, m)?;
    let us = segments(upper, m)?;
    let constant_bounds = constant(lower).is_some() && constant(upper).is_some();
    let (ll, lh) = domain(lower);
    let (ul, uh) = domain(upper);
    let mut accepted = vec![];
    for f in front {
        let fs = segments(f, m)?;
        let (fl, fh) = domain(f);
        let lo = fl.max(ll).max(ul);
        let hi = fh.min(lh).min(uh);
        if lo > hi {
            if near_bounds(lo, hi, m)? {
                return None;
            }
            continue;
        }
        let point_unchanged =
            f.len() == 1 && constant_bounds && f[0].0 >= ls[0].y && f[0].0 <= us[0].y;
        if !point_unchanged {
            for endpoint in [lo, hi] {
                for source in [fl, fh, ll, lh, ul, uh] {
                    if endpoint != source && near_bounds(endpoint, source, m)? {
                        return None;
                    }
                }
            }
        }
        let (mut fi, mut li, mut ui) = (
            segment_index(&fs, lo),
            segment_index(&ls, lo),
            segment_index(&us, lo),
        );
        if lo == hi {
            bound_point(&mut accepted, lo, fs[fi], ls[li], us[ui], m)?;
            continue;
        }
        let mut x = lo;
        while x < hi {
            m.take(1)?;
            while fs[fi].hi <= x && fi + 1 < fs.len() {
                fi += 1;
            }
            while ls[li].hi <= x && li + 1 < ls.len() {
                li += 1;
            }
            while us[ui].hi <= x && ui + 1 < us.len() {
                ui += 1;
            }
            let (f, l, u) = (fs[fi], ls[li], us[ui]);
            let end = hi.min(f.hi).min(l.hi).min(u.hi);
            if end <= x {
                return None;
            }
            let mut cuts = vec![x, end];
            for (a, b) in [(f, l), (f, u), (l, u)] {
                if let Some(c) = bound_cross(a, b, x, end, m)? {
                    cuts.push(c);
                }
            }
            cuts.sort_by(f64::total_cmp);
            cuts.dedup();
            for p in cuts.windows(2) {
                if near_bounds(p[0], p[1], m)? {
                    return None;
                }
            }
            for &cut in &cuts {
                bound_point(&mut accepted, cut, f, l, u, m)?;
            }
            for p in cuts.windows(2) {
                bound_cell(&mut accepted, p[0], p[1], f, l, u, m)?;
            }
            x = end;
        }
    }
    assemble_bounds(accepted, m)
}

/// A point objective interval for the declared computed-front program. This
/// entry deliberately accepts only FrontAlgebra raw paired descriptors;
/// source-control paired resources keep their independent outward verifier.
pub(super) fn certify(
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
        policy: NATIVE,
    } = definition
    else {
        return None;
    };
    if !scale.is_finite() || scale < 0. {
        return None;
    }
    let mut m = Machine::new(work, deadline);
    let initial = machine::prepare(initial.pairs(), &mut m)?;
    let mut front = machine::restrict(
        &vec![initial],
        initial_base.lower,
        initial_base.upper,
        NATIVE,
        &mut m,
    )?;
    let Some(initial_minimum) = machine::minimum(&front, &mut m)? else {
        return Some((false, [0.; 2]));
    };
    for &index in path {
        m.take(1)?;
        let source = arcs.get(index)?;
        let arrival = machine::prepare(source.arrival.pairs(), &mut m)?;
        let increment = machine::prepare(source.increment.pairs(), &mut m)?;
        let lower = machine::prepare(source.lower.pairs(), &mut m)?;
        let upper = machine::prepare(source.upper.pairs(), &mut m)?;
        let k = f64::from(source.waiting);
        let candidate = translation(source);
        let t = if let Some(t) = candidate {
            if exact_transport(&front, t, &mut m)? {
                Some(t)
            } else {
                None
            }
        } else {
            None
        };
        let waited = if let Some(t) = t {
            waiting(&front, t, k, &mut m)?
        } else {
            let inverse = inverse(&arrival, &mut m)?;
            let (lo, hi) = domain(&inverse);
            // Preparation composes Phi independently of F, preserving the
            // native separate rounding before their ordered slope addition.
            let Some(phi) = compose(&increment, &inverse, lo, hi, &mut m)? else {
                return Some((false, [0.; 2]));
            };
            let mut transformed = vec![];
            for component in &front {
                if let Some(f) = compose(component, &inverse, lo, hi, &mut m)? {
                    transformed.push(f);
                }
            }
            machine::validate(&transformed, &mut m)?;
            if transformed.is_empty() {
                return Some((false, [0.; 2]));
            }
            let charged = machine::add(&transformed, &vec![phi], NATIVE, &mut m)?;
            if charged.is_empty() {
                return Some((false, [0.; 2]));
            }
            let (ll, lh) = domain(&lower);
            let (ul, uh) = domain(&upper);
            waiting(
                &charged,
                Translation {
                    delta: 0.,
                    phi: 0.,
                    tail_lo: lo,
                    tail_hi: hi,
                    head_lo: ll.max(ul),
                    head_hi: lh.min(uh),
                },
                k,
                &mut m,
            )?
        };
        front = clamp_filter(&waited, &lower, &upper, &mut m)?;
        if front.is_empty() {
            return Some((false, [0.; 2]));
        }
    }
    if path.is_empty() || *mode == PairedResourceMode::Feasibility || scale == 0. {
        m.take(1)?;
        return Some((true, [0.; 2]));
    }
    let minimum = machine::minimum(&front, &mut m)??;
    let amount = match normalization {
        PairedCostNormalization::AbsoluteFinalValue => minimum,
        PairedCostNormalization::FinalMinusInitialMinimum => sub!(&mut m, minimum, initial_minimum),
    };
    let cost = mul!(&mut m, amount, scale);
    m.take(1)?;
    Some((true, [cost, cost]))
}

#[cfg(test)]
#[path = "../../../../../tests/unit/graph/coupled/kernel/paired_machine_tests.rs"]
mod tests;
