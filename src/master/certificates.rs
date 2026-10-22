//! Conservative lower bounds for explicitly supplied finite master matrices.
//!
//! A bound for this matrix is not automatically a bound for an original routing
//! problem: callers must prove domain completeness or separately cover omitted
//! columns, and validate any generated rows for that original domain.
use super::RowSense;
use crate::core::rounding::{add_down, add_up, multiply_down};

/// A nonnegative variable with an exact sum of supplied floating cost terms.
/// Sparse entries may repeat; their coefficients are interpreted as exact sums.
#[derive(Clone, Copy, Debug)]
pub struct Column<'a> {
    pub cost_terms: &'a [f64],
    pub entries: &'a [(usize, f64)],
}

/// A finite minimization model with nonnegative variables. Additional variable
/// upper bounds can be omitted: relaxing them preserves the lower bound.
#[derive(Clone, Copy, Debug)]
pub struct FiniteMaster<'a> {
    pub rhs: &'a [f64],
    pub senses: &'a [RowSense],
    pub columns: &'a [Column<'a>],
    /// Constant objective terms (value, multiplier), e.g. original fixed arcs.
    pub offset_terms: &'a [(f64, f64)],
    /// A <= row whose exact coefficient in every column must equal one.
    /// This proves `sum(x) <= rhs[row]` without assuming an oracle multiplicity.
    pub multiplicity_row: usize,
}

impl FiniteMaster<'_> {
    /// Returns a downward enclosure of D + M min(0, min_j(c_j - y A_j)).
    /// Duals are projected to their required row signs. The exact unit
    /// multiplicity coefficients are checked using enclosing arithmetic.
    /// Invalid shapes, nonfinite data, unsupported arithmetic or unproved
    /// multiplicity coefficients return None, never a guessed finite bound.
    /// Assumes ordinary IEEE binary64 operations without fast-math reassociation.
    pub fn lower_bound(&self, duals: &[f64]) -> Option<f64> {
        self.lower_bound_checked(duals, || Some(()))
    }

    /// Internal cooperative budget checkpoints; interruption discards the proof.
    pub(crate) fn lower_bound_checked(
        &self,
        duals: &[f64],
        mut checkpoint: impl FnMut() -> Option<()>,
    ) -> Option<f64> {
        checkpoint()?;
        let rows = self.rhs.len();
        if self.senses.len() != rows
            || duals.len() != rows
            || self.multiplicity_row >= rows
            || self.senses[self.multiplicity_row] != RowSense::LessEqual
            || self.rhs[self.multiplicity_row] < 0.
            || self.rhs.iter().chain(duals).any(|v| !v.is_finite())
        {
            return None;
        }
        let projected = duals
            .iter()
            .zip(self.senses)
            .map(|(&dual, sense)| {
                checkpoint()?;
                Some(match sense {
                    RowSense::LessEqual => dual.min(0.),
                    RowSense::GreaterEqual => dual.max(0.),
                    RowSense::Equal => dual,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        let mut value = 0.;
        for &(term, multiplier) in self.offset_terms {
            checkpoint()?;
            if !term.is_finite() || !multiplier.is_finite() {
                return None;
            }
            value = add_down(value, multiply_down(term, multiplier)?)?;
        }
        for (&dual, &rhs) in projected.iter().zip(self.rhs) {
            checkpoint()?;
            value = add_down(value, multiply_down(dual, rhs)?)?;
        }
        let mut minimum = 0f64;
        for column in self.columns {
            checkpoint()?;
            let mut reduced = 0.;
            for &term in column.cost_terms {
                checkpoint()?;
                if !term.is_finite() {
                    return None;
                }
                reduced = add_down(reduced, term)?;
            }
            let mut unit_lower = 0.;
            let mut unit_upper = 0.;
            for &(row, coefficient) in column.entries {
                checkpoint()?;
                if row >= rows || !coefficient.is_finite() {
                    return None;
                }
                if row == self.multiplicity_row {
                    unit_lower = add_down(unit_lower, coefficient)?;
                    unit_upper = add_up(unit_upper, coefficient)?;
                }
                reduced = add_down(reduced, multiply_down(-projected[row], coefficient)?)?;
            }
            if unit_lower != 1. || unit_upper != 1. {
                return None;
            }
            minimum = minimum.min(reduced);
        }
        let bound = add_down(
            value,
            multiply_down(self.rhs[self.multiplicity_row], minimum)?,
        )?;
        checkpoint()?;
        bound.is_finite().then_some(bound)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/master/certificates/tests.rs"]
mod tests;
