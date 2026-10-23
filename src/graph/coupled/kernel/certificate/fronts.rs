//! Conservative interval-front witnesses. Exact integral state translations
//! establish a complete feasible domain; objective ranges remain outward.
use super::*;

fn exact(x: f64) -> bool {
    x.is_finite() && x.fract() == 0. && x.abs() <= 1_125_899_906_842_624.
}
impl CoupledKernel {
    pub(super) fn certify_fronts(&self, arcs: &[usize]) -> Option<(bool, [f64; 2])> {
        let mut total = Interval::point(0.);
        for &index in &self.front_indices {
            let CoupledResource::IntervalCost {
                initial,
                bounds,
                increments,
                functions,
            } = &self.definitions[index]
            else {
                return None;
            };
            // Do not infer real-domain feasibility from a rounded inverse map.
            // Nonintegral/large translations remain explicitly unproved.
            if ![initial.lower, initial.upper, bounds.lower, bounds.upper]
                .into_iter()
                .all(exact)
            {
                return None;
            }
            let (mut lo, mut hi, mut shift) = (initial.lower, initial.upper, 0.);
            let mut shifts = Vec::with_capacity(arcs.len());
            let mut knots = vec![lo, hi];
            for &arc in arcs {
                let delta = *increments.get(arc)?;
                if !exact(delta) {
                    return None;
                }
                shift += delta;
                if !exact(shift) {
                    return None;
                }
                let function = functions.get(arc)?;
                if !function.points.iter().all(|p| exact(p[0])) {
                    return None;
                }
                let domain = function.prepared_domain();
                lo = lo.max(bounds.lower - shift).max(domain.lower - shift);
                hi = hi.min(bounds.upper - shift).min(domain.upper - shift);
                shifts.push(shift);
                knots.extend(function.points.iter().map(|p| p[0] - shift));
            }
            if lo > hi {
                return Some((false, [0.; 2]));
            }
            knots.extend([lo, hi]);
            knots.retain(|x| *x >= lo && *x <= hi);
            knots.sort_by(f64::total_cmp);
            knots.dedup();
            let evaluate = |lo: f64, hi: f64| -> Option<Interval> {
                let mut cost = Interval::point(0.);
                for (&arc, &shift) in arcs.iter().zip(&shifts) {
                    // All arguments are exact integers at endpoints. The range
                    // includes every interpolated/rounded interior value too.
                    let range = functions[arc].range(bounds::Range {
                        lo: lo + shift,
                        hi: hi + shift,
                    })?;
                    cost = cost.add(
                        Interval {
                            lo: range.lo,
                            hi: range.hi,
                        }
                        .multiply(self.cost_scale)?,
                    )?;
                }
                Some(cost)
            };
            let mut upper = f64::INFINITY;
            for &x in &knots {
                upper = upper.min(evaluate(x, x)?.hi);
            }
            let lower = if knots.len() == 1 {
                evaluate(knots[0], knots[0])?.lo
            } else {
                let mut lower = f64::INFINITY;
                for pair in knots.windows(2) {
                    lower = lower.min(evaluate(pair[0], pair[1])?.lo);
                }
                lower
            };
            if !lower.is_finite() || !upper.is_finite() {
                return None;
            }
            total = total.add(Interval {
                lo: lower,
                hi: upper,
            })?;
        }
        Some((true, [total.lo, total.hi]))
    }
}
