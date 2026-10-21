//! Outward-rounded activity propagation for complete original formulations.
use super::*;
use crate::master::rounding::{add_down, multiply_down};

fn sum_down(a: f64, b: f64) -> f64 {
    add_down(a, b).unwrap_or(f64::NEG_INFINITY)
}
fn quotient_up(a: f64, b: f64) -> Option<f64> {
    // Division by one is exact, including subnormals. Preserve exact unit
    // coefficients so flow conservation can deduce an actual zero endpoint.
    if b == 1. {
        return a.is_finite().then_some(a);
    }
    let value = a / b;
    if !value.is_finite() {
        return None;
    }
    if a == 0. {
        return Some(0.);
    }
    Some(if value == 0. {
        f64::from_bits(1)
    } else {
        f64::from_bits(if value > 0. {
            value.to_bits() + 1
        } else {
            value.to_bits() - 1
        })
    })
}

/// Propagate a complete formulation with directed arithmetic. Coefficients
/// within each row must already be coalesced (each variable appears once).
/// A budget limit returns valid partial deductions, never an infeasibility claim
/// based on missing work. No integer rounding or tolerance-based fixing occurs.
pub fn propagate_conservative(
    bounds: &mut [Bounds],
    constraints: &[Constraint],
    work_limit: usize,
) -> Result<Report> {
    propagate_mixed_conservative(bounds, constraints, &vec![false; bounds.len()], work_limit)
}

/// Directed propagation with optional integral domains. Integer rounding is
/// applied only to variables explicitly marked integral by the original model.
pub fn propagate_mixed_conservative(
    bounds: &mut [Bounds],
    constraints: &[Constraint],
    integers: &[bool],
    work_limit: usize,
) -> Result<Report> {
    if integers.len() != bounds.len() {
        return Err(invalid("preprocessing integrality dimensions"));
    }
    for b in bounds.iter() {
        if b.lower.is_nan()
            || b.upper.is_nan()
            || b.lower == f64::INFINITY
            || b.upper == f64::NEG_INFINITY
        {
            return Err(invalid("invalid conservative preprocessing bounds"));
        }
    }
    let mut rows = vec![];
    let mut incidence = vec![vec![]; bounds.len()];
    for c in constraints {
        if !c.rhs.is_finite() {
            return Err(invalid("finite preprocessing RHS required"));
        }
        let mut seen = std::collections::HashSet::new();
        for &(v, a) in &c.entries {
            if v >= bounds.len() || !a.is_finite() || !seen.insert(v) {
                return Err(invalid(
                    "coalesced finite preprocessing coefficients required",
                ));
            }
        }
        for sign in [1., -1.] {
            if sign > 0. && c.sense == RowSense::GreaterEqual
                || sign < 0. && c.sense == RowSense::LessEqual
            {
                continue;
            }
            let entries: Vec<_> = c
                .entries
                .iter()
                .filter(|&&(_, a)| a != 0.)
                .map(|&(v, a)| (v, sign * a))
                .collect();
            let index = rows.len();
            for &(v, _) in &entries {
                incidence[v].push(index);
            }
            rows.push((entries, sign * c.rhs));
        }
    }
    let mut report = Report::default();
    for (b, &integer) in bounds.iter_mut().zip(integers) {
        if integer {
            b.lower = b.lower.ceil();
            b.upper = b.upper.floor();
        }
    }
    if bounds.iter().any(|b| b.lower > b.upper) {
        report.infeasible = true;
        report.complete = true;
        return Ok(report);
    }
    let mut queue: VecDeque<_> = (0..rows.len()).collect();
    let mut queued = vec![true; rows.len()];
    while let Some(r) = queue.pop_front() {
        let (entries, rhs) = &rows[r];
        let cost = entries.len().saturating_mul(4).saturating_add(1);
        if cost > work_limit.saturating_sub(report.work_used) {
            return Ok(report);
        }
        report.work_used += cost;
        report.rows_processed += 1;
        queued[r] = false;
        let products: Vec<_> = entries
            .iter()
            .map(|&(v, a)| {
                multiply_down(
                    a,
                    if a > 0. {
                        bounds[v].lower
                    } else {
                        bounds[v].upper
                    },
                )
                .unwrap_or(f64::NEG_INFINITY)
            })
            .collect();
        let mut prefix = vec![0.; entries.len() + 1];
        let mut suffix = vec![0.; entries.len() + 1];
        for i in 0..entries.len() {
            prefix[i + 1] = sum_down(prefix[i], products[i]);
        }
        for i in (0..entries.len()).rev() {
            suffix[i] = sum_down(products[i], suffix[i + 1]);
        }
        if prefix[entries.len()] > *rhs {
            report.infeasible = true;
            report.complete = true;
            return Ok(report);
        }
        for (i, &(v, a)) in entries.iter().enumerate() {
            let residual = sum_down(prefix[i], suffix[i + 1]);
            let Some(numerator) = add_down(residual, -rhs).map(|v| -v) else {
                continue;
            };
            let Some(limit) = quotient_up(numerator, a.abs()) else {
                continue;
            };
            let old = bounds[v];
            if a > 0. {
                bounds[v].upper = old.upper.min(limit);
            } else {
                bounds[v].lower = old.lower.max(-limit);
            }
            if integers[v] {
                bounds[v].lower = bounds[v].lower.ceil();
                bounds[v].upper = bounds[v].upper.floor();
            }
            if bounds[v].lower > bounds[v].upper {
                report.infeasible = true;
                report.complete = true;
                return Ok(report);
            }
            if bounds[v] != old {
                report.bound_updates += 1;
                for &affected in &incidence[v] {
                    if !queued[affected] {
                        queued[affected] = true;
                        queue.push_back(affected);
                    }
                }
            }
        }
    }
    report.complete = true;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn signed_propagation_preserves_all_exact_grid_solutions() {
        let constraints = [
            Constraint {
                entries: vec![(0, 2.), (1, -1.)],
                sense: RowSense::Equal,
                rhs: 1.,
            },
            Constraint {
                entries: vec![(1, 1.), (2, 1.)],
                sense: RowSense::LessEqual,
                rhs: 2.,
            },
        ];
        let mut bounds = [Bounds {
            lower: 0.,
            upper: 4.,
        }; 3];
        let report = propagate_conservative(&mut bounds, &constraints, 10000).unwrap();
        assert!(report.complete && !report.infeasible && report.bound_updates > 0);
        for x in 0..=16 {
            for y in 0..=16 {
                for z in 0..=16 {
                    let q = [x as f64 / 4., y as f64 / 4., z as f64 / 4.];
                    if 2. * q[0] - q[1] == 1. && q[1] + q[2] <= 2. {
                        assert!(q
                            .iter()
                            .zip(bounds)
                            .all(|(&v, b)| b.lower <= v && v <= b.upper));
                    }
                }
            }
        }
        let report = propagate_conservative(&mut bounds, &constraints, 0).unwrap();
        assert!(!report.complete && !report.infeasible);
    }
    #[test]
    fn cancellation_does_not_invent_a_zero_bound() {
        let mut bounds = [
            Bounds {
                lower: 1e16,
                upper: 1e16,
            },
            Bounds {
                lower: 1.,
                upper: 1.,
            },
            Bounds {
                lower: 0.,
                upper: 2.,
            },
        ];
        // x-y+z <= 1e16 permits z=1. Subtracting a rounded full activity
        // instead of outward prefix/suffix sums could incorrectly infer zero.
        let constraints = [Constraint {
            entries: vec![(0, 1.), (1, -1.), (2, 1.)],
            sense: RowSense::LessEqual,
            rhs: 1e16,
        }];
        let report = propagate_conservative(&mut bounds, &constraints, 1000).unwrap();
        assert!(!report.infeasible && bounds[2].upper >= 1.);
    }
}
