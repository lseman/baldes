// Terminal-box LP proposals shared by path pricing and witness recovery.
use super::*;
// Both route-local proposals and cross-path witness recovery use the same
// shifted terminal-box LP. Its solution is a proposal, never a proof bound.
pub(super) type TerminalRow = (RowSense, f64, Vec<(usize, f64)>);

// The cost optimum is useful for witness recovery, but it can land on a
// source-unattainable point of a coupled face. Also propose a feasible point
// selected by a zero objective and both extrema of every terminal
// coordinate. These targets are candidates only; source replay remains the
// acceptance test.
pub(super) fn terminal_objective(costs: &[f64], proposal: usize) -> Option<Vec<f64>> {
    if proposal == 0 {
        return Some(costs.to_vec());
    }
    if proposal == 1 {
        return Some(vec![0.; costs.len()]);
    }
    let coordinate = (proposal - 2) / 2;
    let sign = if proposal % 2 == 0 { -1. } else { 1. };
    let mut objective = vec![0.; costs.len()];
    *objective.get_mut(coordinate)? = sign;
    Some(objective)
}

pub(super) fn solve_terminal_targets(
    domains: &[Interval],
    rows: &[TerminalRow],
    costs: &[f64],
    iterations: &mut usize,
    deadline: Instant,
) -> Option<Vec<f64>> {
    use crate::master::clp::{ClpMaster, ClpSolveOutcome};
    if *iterations == 0 || Instant::now() >= deadline {
        return None;
    }
    let mut master = ClpMaster::new().ok()?;
    for (domain, &cost) in domains.iter().zip(costs) {
        let column = master.add_column(cost, &[]).ok()?;
        master.set_upper(column, domain.hi - domain.lo).ok()?;
    }
    for (sense, rhs, entries) in rows {
        master.add_row(*sense, *rhs, entries).ok()?;
    }
    let before = master.simplex_iterations;
    let solved = master
        .solve_limited(
            *iterations,
            deadline.saturating_duration_since(Instant::now()),
        )
        .ok()?;
    *iterations =
        iterations.checked_sub(master.simplex_iterations.saturating_sub(before).max(1))?;
    if solved != ClpSolveOutcome::Optimal {
        return None;
    }
    Some(
        domains
            .iter()
            .zip(master.values())
            .map(|(domain, v)| domain.lo + v)
            .collect(),
    )
}

/// Enumerate vertices of a bounded shifted terminal LP, up to shared work.
/// The returned points are numerical proposals only; callers must realize
/// every coordinate in the source model and validate the complete row set.
pub(super) fn terminal_vertices(
    domains: &[Interval],
    rows: &[TerminalRow],
    work: &mut usize,
    deadline: Instant,
) -> Vec<Vec<f64>> {
    #[derive(Clone)]
    struct Constraint {
        coefficients: Vec<f64>,
        rhs: f64,
    }

    let n = domains.len();
    if n == 0 {
        return vec![];
    }
    let Some(bound_constraints) = n.checked_mul(2) else {
        return vec![];
    };
    let Some(setup_work) = rows
        .len()
        .checked_add(bound_constraints)
        .and_then(|constraints| constraints.checked_mul(n))
    else {
        return vec![];
    };
    let Some(remaining) = work.checked_sub(setup_work) else {
        return vec![];
    };
    *work = remaining;
    let mut constraints = Vec::with_capacity(rows.len() + n * 2);
    for &(_sense, rhs, ref entries) in rows {
        let mut coefficients = vec![0.; n];
        for &(index, coefficient) in entries {
            let Some(value) = coefficients.get_mut(index) else {
                return vec![];
            };
            *value += coefficient;
        }
        if coefficients.iter().any(|value| !value.is_finite()) || !rhs.is_finite() {
            return vec![];
        }
        constraints.push(Constraint { coefficients, rhs });
    }
    for (index, domain) in domains.iter().enumerate() {
        let width = domain.hi - domain.lo;
        if !width.is_finite() || width < 0. {
            return vec![];
        }
        let mut upper = vec![0.; n];
        upper[index] = 1.;
        constraints.push(Constraint {
            coefficients: upper,
            rhs: width,
        });
        let mut lower = vec![0.; n];
        lower[index] = -1.;
        constraints.push(Constraint {
            coefficients: lower,
            rhs: 0.,
        });
    }

    pub(super) fn solve_active(active: &[&Constraint], n: usize) -> Option<Vec<f64>> {
        let mut matrix = vec![vec![0.; n + 1]; n];
        for (row, constraint) in active.iter().enumerate() {
            matrix[row][..n].copy_from_slice(&constraint.coefficients);
            matrix[row][n] = constraint.rhs;
        }
        for column in 0..n {
            let pivot = (column..n)
                .max_by(|&a, &b| matrix[a][column].abs().total_cmp(&matrix[b][column].abs()))?;
            let scale = matrix[pivot][..n]
                .iter()
                .fold(0_f64, |largest, value| largest.max(value.abs()));
            if matrix[pivot][column].abs() <= 64. * f64::EPSILON * scale.max(1.) {
                return None;
            }
            matrix.swap(column, pivot);
            let divisor = matrix[column][column];
            for value in &mut matrix[column][column..=n] {
                *value /= divisor;
            }
            for row in 0..n {
                if row == column {
                    continue;
                }
                let factor = matrix[row][column];
                for index in column..=n {
                    matrix[row][index] -= factor * matrix[column][index];
                }
            }
        }
        let point: Vec<_> = matrix.into_iter().map(|row| row[n]).collect();
        point.iter().all(|value| value.is_finite()).then_some(point)
    }

    pub(super) fn feasible(point: &[f64], domains: &[Interval], rows: &[TerminalRow]) -> bool {
        for (value, domain) in point.iter().zip(domains) {
            let width = domain.hi - domain.lo;
            let tolerance = 1e-9 * (1. + width.abs() + value.abs());
            if *value < -tolerance || *value > width + tolerance {
                return false;
            }
        }
        for &(sense, rhs, ref entries) in rows {
            let lhs = entries
                .iter()
                .map(|&(index, coefficient)| point[index] * coefficient)
                .sum::<f64>();
            if !lhs.is_finite() {
                return false;
            }
            let tolerance =
                1e-9 * (1. + rhs.abs() + entries.iter().map(|&(_, a)| a.abs()).sum::<f64>());
            match sense {
                RowSense::LessEqual if lhs > rhs + tolerance => return false,
                RowSense::GreaterEqual if lhs < rhs - tolerance => return false,
                RowSense::Equal if (lhs - rhs).abs() > tolerance => return false,
                _ => {}
            }
        }
        true
    }

    pub(super) fn visit(
        start: usize,
        active: &mut Vec<usize>,
        constraints: &[Constraint],
        domains: &[Interval],
        rows: &[TerminalRow],
        work: &mut usize,
        deadline: Instant,
        vertices: &mut Vec<Vec<f64>>,
        seen: &mut std::collections::HashSet<Vec<u64>>,
    ) {
        if Instant::now() >= deadline || *work == 0 {
            return;
        }
        let n = domains.len();
        if active.len() == n {
            let Some(cost) = n.checked_mul(n).and_then(|value| value.checked_add(1)) else {
                return;
            };
            let Some(remaining) = work.checked_sub(cost) else {
                *work = 0;
                return;
            };
            *work = remaining;
            let selected: Vec<_> = active.iter().map(|&index| &constraints[index]).collect();
            let Some(point) = solve_active(&selected, n) else {
                return;
            };
            if !feasible(&point, domains, rows) {
                return;
            }
            let key: Vec<_> = point
                .iter()
                .map(|value| if *value == 0. { 0 } else { value.to_bits() })
                .collect();
            if seen.insert(key) {
                vertices.push(point);
            }
            return;
        }
        let needed = n - active.len();
        if constraints.len().saturating_sub(start) < needed {
            return;
        }
        for index in start..constraints.len() {
            active.push(index);
            visit(
                index + 1,
                active,
                constraints,
                domains,
                rows,
                work,
                deadline,
                vertices,
                seen,
            );
            active.pop();
            if Instant::now() >= deadline || *work == 0 {
                break;
            }
        }
    }

    let mut vertices = Vec::new();
    let mut seen = std::collections::HashSet::new();
    visit(
        0,
        &mut Vec::with_capacity(n),
        &constraints,
        domains,
        rows,
        work,
        deadline,
        &mut vertices,
        &mut seen,
    );
    vertices
}
