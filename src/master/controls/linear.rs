//! Limited LP proposals and independently verified finite-matrix bounds.
use super::program::{row, shifted, Cell};
use super::{exact_mul, exact_sub, Budget};
use crate::master::{
    certificates::{Column, FiniteMaster},
    clp::{ClpMaster, ClpSolveOutcome},
    preprocessing::Constraint,
    RowSense,
};
struct Matrix {
    rows: Vec<Constraint>,
    costs: Vec<f64>,
    columns: Vec<Vec<(usize, f64)>>,
    multiplicity: usize,
    offset: f64,
}
impl Matrix {
    fn new(
        mut rows: Vec<Constraint>,
        widths: &[f64],
        target: usize,
        sign: f64,
        offset: f64,
        phase_one: bool,
        budget: &mut Budget,
    ) -> Option<Self> {
        budget.take(rows.len().checked_mul(widths.len().checked_add(1)?)?)?;
        // Knot equations may carry a common segment-width factor (for
        // example7*y-7*x=0). Remove it only when every quotient is exact.
        // Equivalent source rows then admit representable dual certificates
        // instead of requiring a rounded1/7 multiplier.
        for row in &mut rows {
            budget.take(row.entries.len().checked_add(1)?)?;
            let scale = row.entries.iter().map(|&(_, a)| a.abs()).fold(0., f64::max);
            if scale == 0. || scale == 1. {
                continue;
            }
            let divide = |a: f64| {
                let q = a / scale;
                (q.is_finite() && exact_mul(q, scale) == Some(a)).then_some(q)
            };
            let Some(rhs) = divide(row.rhs) else {
                continue;
            };
            let Some(entries) = row
                .entries
                .iter()
                .map(|&(v, a)| divide(a).map(|q| (v, q)))
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            row.rhs = rhs;
            row.entries = entries;
        }
        let mut costs = vec![0.; widths.len()];
        if !phase_one {
            costs[target] = sign;
        }
        let mut caps = widths.to_vec();
        if phase_one {
            let original = rows.len();
            for current in rows.iter_mut().take(original) {
                let signs = match current.sense {
                    RowSense::Equal => vec![1., -1.],
                    RowSense::LessEqual => vec![-1.],
                    RowSense::GreaterEqual => vec![1.],
                };
                for a in signs {
                    current.entries.push((costs.len(), a));
                    caps.push(current.rhs.abs());
                    costs.push(1.);
                }
            }
        }
        let mut bound = 0.;
        for &cap in &caps {
            bound = crate::core::rounding::add_up(bound, cap)?;
        }
        let multiplicity = rows.len();
        rows.push(row(
            (0..costs.len()).map(|v| (v, 1.)).collect(),
            RowSense::LessEqual,
            bound,
        ));
        let mut columns = vec![vec![]; costs.len()];
        for (r, row) in rows.iter().enumerate() {
            for &(v, a) in &row.entries {
                columns[v].push((r, a));
            }
        }
        Some(Self {
            rows,
            costs,
            columns,
            multiplicity,
            offset: if phase_one { 0. } else { offset },
        })
    }
    fn master(&self, budget: &mut Budget) -> Option<ClpMaster> {
        budget.take(
            self.rows
                .len()
                .checked_add(self.columns.iter().map(Vec::len).sum())?,
        )?;
        let mut master = ClpMaster::new().ok()?;
        master.zero_tolerance = 0.;
        // Bounds live in the matrix. This backend default must not add an
        // unproved 1e12 cap to source coordinates.
        master.set_column_default_upper(f64::MAX).ok()?;
        master
            .add_rows(
                &self
                    .rows
                    .iter()
                    .map(|r| (r.sense, r.rhs, &[][..]))
                    .collect::<Vec<_>>(),
            )
            .ok()?;
        master
            .add_columns(
                &self
                    .costs
                    .iter()
                    .zip(&self.columns)
                    .map(|(&c, a)| (c, a.as_slice()))
                    .collect::<Vec<_>>(),
            )
            .ok()?;
        Some(master)
    }
    fn bound(&self, master: &ClpMaster, budget: &mut Budget) -> Option<f64> {
        let rhs: Vec<_> = self.rows.iter().map(|r| r.rhs).collect();
        let senses: Vec<_> = self.rows.iter().map(|r| r.sense).collect();
        let columns: Vec<_> = self
            .costs
            .iter()
            .zip(&self.columns)
            .map(|(c, a)| Column {
                cost_terms: std::slice::from_ref(c),
                entries: a,
            })
            .collect();
        FiniteMaster {
            rhs: &rhs,
            senses: &senses,
            columns: &columns,
            offset_terms: &[(self.offset, 1.)],
            multiplicity_row: self.multiplicity,
        }
        .lower_bound_checked(&master.duals, || budget.take(1))
    }
}

// Reconstruct a representable vertex proposal from equalities and nearby
// active rows/bounds. Exact guarded elimination is only a proposal generator;
// independent source replay remains authoritative, including skipped rows.
fn vertex_proposal(matrix: &Matrix, values: &[f64], budget: &mut Budget) -> Option<Vec<f64>> {
    let n = matrix.costs.len();
    budget.take(
        matrix
            .rows
            .len()
            .checked_add(n)?
            .checked_mul(n.checked_add(1)?)?,
    )?;
    let mut candidates = vec![];
    for row in &matrix.rows {
        let mut coefficients = vec![0.; n];
        let mut activity = 0.;
        let mut scale = 1.;
        for &(v, a) in &row.entries {
            coefficients[v] = a;
            activity += a * values[v];
            scale += a.abs();
        }
        let score = if row.sense == RowSense::Equal {
            -1.
        } else {
            ((activity - row.rhs) / scale).abs()
        };
        candidates.push((score, coefficients, row.rhs));
    }
    for v in 0..n {
        let mut coefficients = vec![0.; n];
        coefficients[v] = 1.;
        candidates.push((values[v].abs(), coefficients, 0.));
    }
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut basis: Vec<(usize, Vec<f64>, f64)> = vec![];
    for (_, mut coefficients, mut rhs) in candidates {
        budget.take(n.checked_mul(n.checked_add(1)?)?.checked_mul(8)?)?;
        let reduced = (|| {
            for (pivot, row, value) in &basis {
                let factor = coefficients[*pivot];
                if factor == 0. {
                    continue;
                }
                for (a, &b) in coefficients.iter_mut().zip(row) {
                    *a = exact_sub(*a, exact_mul(factor, b)?)?;
                }
                rhs = exact_sub(rhs, exact_mul(factor, *value)?)?;
            }
            let pivot = coefficients.iter().position(|&value| value != 0.)?;
            let divisor = coefficients[pivot];
            let divide = |value: f64| -> Option<f64> {
                let q = value / divisor;
                (q.is_finite() && exact_mul(q, divisor) == Some(value)).then_some(q)
            };
            for value in &mut coefficients {
                *value = divide(*value)?;
            }
            rhs = divide(rhs)?;
            Some(pivot)
        })();
        let Some(pivot) = reduced else {
            continue;
        };
        // Maintain a reduced basis without committing a partially rounded edit.
        let mut next = basis.clone();
        let valid = (|| {
            for (_, row, value) in &mut next {
                let factor = row[pivot];
                if factor == 0. {
                    continue;
                }
                for (a, &b) in row.iter_mut().zip(&coefficients) {
                    *a = exact_sub(*a, exact_mul(factor, b)?)?;
                }
                *value = exact_sub(*value, exact_mul(factor, rhs)?)?;
            }
            Some(())
        })()
        .is_some();
        if !valid {
            continue;
        }
        next.push((pivot, coefficients, rhs));
        next.sort_by_key(|row| row.0);
        basis = next;
        if basis.len() == n {
            let mut values = vec![0.; n];
            for (pivot, _, rhs) in basis {
                values[pivot] = rhs;
            }
            return Some(values);
        }
    }
    None
}

pub(super) enum CellResult {
    Infeasible,
    Bounded { bound: f64, bases: Vec<Vec<f64>> },
    Unknown,
}
pub(super) fn solve_cell(cell: &Cell, maximize: bool, budget: &mut Budget) -> CellResult {
    let Some((rows, widths)) = shifted(cell, budget) else {
        return CellResult::Unknown;
    };
    let sign = if maximize { -1. } else { 1. };
    let target = cell.bounds.len() - 1;
    let Some(matrix) = Matrix::new(
        rows.clone(),
        &widths,
        target,
        sign,
        sign * cell.bounds[target].lower,
        false,
        budget,
    ) else {
        return CellResult::Unknown;
    };
    let Some(mut master) = matrix.master(budget) else {
        return CellResult::Unknown;
    };
    match budget.solve(&mut master) {
        Some(ClpSolveOutcome::Optimal) => {
            let Some(bound) = matrix.bound(&master, budget) else {
                return CellResult::Unknown;
            };
            let base = (0..cell.base_count)
                .map(|v| {
                    (master.values[v] + cell.bounds[v].lower)
                        .max(cell.bounds[v].lower)
                        .min(cell.bounds[v].upper)
                })
                .collect();
            let mut bases = vec![base];
            if let Some(vertex) = vertex_proposal(&matrix, &master.values, budget) {
                bases.push(
                    (0..cell.base_count)
                        .map(|v| {
                            (vertex[v] + cell.bounds[v].lower)
                                .max(cell.bounds[v].lower)
                                .min(cell.bounds[v].upper)
                        })
                        .collect(),
                );
            }
            CellResult::Bounded { bound, bases }
        }
        Some(ClpSolveOutcome::PrimalInfeasible) => {
            let Some(matrix) = Matrix::new(rows, &widths, target, sign, 0., true, budget) else {
                return CellResult::Unknown;
            };
            let Some(mut master) = matrix.master(budget) else {
                return CellResult::Unknown;
            };
            if budget.solve(&mut master) == Some(ClpSolveOutcome::Optimal)
                && matrix
                    .bound(&master, budget)
                    .is_some_and(|bound| bound > 0.)
            {
                CellResult::Infeasible
            } else {
                CellResult::Unknown
            }
        }
        _ => CellResult::Unknown,
    }
}
