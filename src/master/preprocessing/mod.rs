//! Node-local linear bound propagation using constraint activities.
//! Callers must supply a complete formulation (never
//! just the currently generated columns of an unfinished pricing problem).
use super::RowSense;
use crate::{invalid, Result};
use std::collections::{BTreeMap, VecDeque};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds {
    pub lower: f64,
    pub upper: f64,
}

#[derive(Clone, Debug)]
pub struct Constraint {
    pub entries: Vec<(usize, f64)>,
    pub sense: RowSense,
    pub rhs: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Report {
    pub infeasible: bool,
    pub bound_updates: usize,
    pub rows_processed: usize,
    /// Logical budget spent: row visits for `propagate`, arithmetic work units
    /// for conservative propagation and its network policy coordinator.
    pub work_used: usize,
    /// False means the work budget stopped propagation; all returned bounds
    /// remain valid, but more deductions may be possible.
    pub complete: bool,
}

/// Propagate continuous bounds with signed coefficients and infinite domains.
/// No integer rounding is applied: the resulting bounds preserve the LP.
pub fn propagate(
    bounds: &mut [Bounds],
    constraints: &[Constraint],
    tolerance: f64,
    work_limit: usize,
) -> Result<Report> {
    if !tolerance.is_finite() || tolerance <= 0. {
        return Err(invalid("positive finite preprocessing tolerance required"));
    }
    for b in bounds.iter() {
        if b.lower.is_nan()
            || b.upper.is_nan()
            || b.lower == f64::INFINITY
            || b.upper == f64::NEG_INFINITY
        {
            return Err(invalid("invalid preprocessing bounds"));
        }
    }
    let mut rows = Vec::with_capacity(constraints.len());
    let mut incidence = vec![vec![]; bounds.len()];
    for (r, constraint) in constraints.iter().enumerate() {
        if !constraint.rhs.is_finite() {
            return Err(invalid("finite preprocessing RHS required"));
        }
        let mut entries = BTreeMap::new();
        for &(v, a) in &constraint.entries {
            if v >= bounds.len() || !a.is_finite() {
                return Err(invalid("invalid preprocessing coefficient"));
            }
            *entries.entry(v).or_insert(0.) += a;
        }
        let mut row = vec![];
        for (v, a) in entries {
            if !a.is_finite() {
                return Err(invalid("preprocessing coefficient overflow"));
            }
            if a != 0. {
                incidence[v].push(r);
                row.push((v, a));
            }
        }
        rows.push(row);
    }
    let mut report = Report::default();
    if bounds.iter().any(|b| b.lower > b.upper + tolerance) {
        report.infeasible = true;
        report.complete = true;
        return Ok(report);
    }
    let mut queue: VecDeque<_> = (0..rows.len()).collect();
    let mut queued = vec![true; rows.len()];
    while let Some(r) = queue.pop_front() {
        if report.rows_processed == work_limit {
            return Ok(report);
        }
        queued[r] = false;
        report.rows_processed += 1;
        report.work_used += 1;
        let row = &rows[r];
        let c = &constraints[r];
        // Count infinite contributions separately to avoid infinity-infinity
        // when computing the residual activity excluding one variable.
        let mut min = 0.;
        let mut max = 0.;
        let mut min_inf = 0;
        let mut max_inf = 0;
        for &(v, a) in row {
            let b = bounds[v];
            let (lo, hi) = if a > 0. {
                (a * b.lower, a * b.upper)
            } else {
                (a * b.upper, a * b.lower)
            };
            if lo.is_finite() {
                min += lo;
            } else {
                min_inf += 1;
            }
            if hi.is_finite() {
                max += hi;
            } else {
                max_inf += 1;
            }
        }
        // Overflow cannot be used as evidence of infeasibility.
        if !min.is_finite() || !max.is_finite() {
            continue;
        }
        let le = c.sense != RowSense::GreaterEqual;
        let ge = c.sense != RowSense::LessEqual;
        if (le && min_inf == 0 && min > c.rhs + tolerance)
            || (ge && max_inf == 0 && max < c.rhs - tolerance)
        {
            report.infeasible = true;
            report.complete = true;
            return Ok(report);
        }
        // Use the same activity snapshot for every deduction from this row.
        let mut updates = vec![];
        for &(v, a) in row {
            let b = bounds[v];
            let (lo, hi) = if a > 0. {
                (a * b.lower, a * b.upper)
            } else {
                (a * b.upper, a * b.lower)
            };
            let mut next = b;
            if le && min_inf == usize::from(!lo.is_finite()) {
                let residual = min - if lo.is_finite() { lo } else { 0. };
                let limit = (c.rhs - residual) / a;
                if limit.is_finite() {
                    if a > 0. {
                        next.upper = next.upper.min(limit);
                    } else {
                        next.lower = next.lower.max(limit);
                    }
                }
            }
            if ge && max_inf == usize::from(!hi.is_finite()) {
                let residual = max - if hi.is_finite() { hi } else { 0. };
                let limit = (c.rhs - residual) / a;
                if limit.is_finite() {
                    if a > 0. {
                        next.lower = next.lower.max(limit);
                    } else {
                        next.upper = next.upper.min(limit);
                    }
                }
            }
            if next.lower > next.upper + tolerance {
                report.infeasible = true;
                report.complete = true;
                return Ok(report);
            }
            // Ignore changes below tolerance; never widen the input domain.
            if next.lower > b.lower + tolerance || next.upper < b.upper - tolerance {
                if next.lower <= next.upper {
                    updates.push((v, next));
                }
            }
        }
        for (v, b) in updates {
            bounds[v] = b;
            report.bound_updates += 1;
            for &affected in &incidence[v] {
                if !queued[affected] {
                    queued[affected] = true;
                    queue.push_back(affected);
                }
            }
        }
    }
    report.complete = true;
    Ok(report)
}

mod conservative;
pub use conservative::{propagate_conservative, propagate_mixed_conservative};

#[cfg(feature = "clp")]
pub(crate) mod flow;
