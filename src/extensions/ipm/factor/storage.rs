//! Numeric lower-factor storage; compressed patterns are immutable across refactors.
use std::sync::Arc;

#[derive(Debug, Clone)]
pub(super) struct Pattern {
    pub ptr: Vec<usize>,
    pub rows: Vec<usize>,
}

#[derive(Debug, Clone)]
pub(super) enum LowerFactor {
    Columns(Vec<Vec<(usize, f64)>>),
    Compressed {
        pattern: Arc<Pattern>,
        values: Vec<f64>,
    },
}

impl LowerFactor {
    pub fn forward(&self, x: &mut [f64]) {
        match self {
            Self::Columns(columns) => {
                for (j, column) in columns.iter().enumerate() {
                    let xj = x[j];
                    for &(i, value) in column {
                        x[i] -= value * xj;
                    }
                }
            }
            Self::Compressed { pattern, values } => {
                for j in 0..pattern.ptr.len() - 1 {
                    let xj = x[j];
                    for p in pattern.ptr[j]..pattern.ptr[j + 1] {
                        x[pattern.rows[p]] -= values[p] * xj;
                    }
                }
            }
        }
    }

    pub fn backward(&self, x: &mut [f64]) {
        match self {
            Self::Columns(columns) => {
                for (j, column) in columns.iter().enumerate().rev() {
                    let mut xj = x[j];
                    for &(i, value) in column {
                        xj -= value * x[i];
                    }
                    x[j] = xj;
                }
            }
            Self::Compressed { pattern, values } => {
                for j in (0..pattern.ptr.len() - 1).rev() {
                    let mut xj = x[j];
                    for p in pattern.ptr[j]..pattern.ptr[j + 1] {
                        xj -= values[p] * x[pattern.rows[p]];
                    }
                    x[j] = xj;
                }
            }
        }
    }
}
