//! Sparse up-looking row updates for diagonal-primal, assembly-dominated KKT systems.
//! This path does not pivot. Unsafe pivots return an error so HiPO can retry
//! with the existing pivoted multifrontal factorization.
use super::storage::{LowerFactor, Pattern};
use super::{Block, Error, Factor, FactorStats, Result, SparseMatrix, Symbolic};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub(crate) struct Uplooking {
    pattern: Arc<Pattern>,
    links: Vec<Vec<(usize, usize)>>,
    primal: usize,
}

impl Uplooking {
    pub fn new(symbolic: &Symbolic, primal: usize) -> Option<Self> {
        let n = symbolic.order.len();
        let dual = n - primal;
        let nnz: usize = symbolic
            .fronts
            .iter()
            .map(|front| {
                let width = front.end - front.lo;
                width * (width - 1) / 2 + width * front.updates.len()
            })
            .sum();
        // Use a conservative sparse regime. Wider/dense factors retain the
        // pivoted multifrontal path; no global dense matrix is introduced.
        if n == 0 || primal < 2 * dual || nnz > 32 * n {
            return None;
        }
        let mut ptr = Vec::with_capacity(n + 1);
        let mut rows = Vec::with_capacity(nnz);
        ptr.push(0);
        for front in &symbolic.fronts {
            for j in front.lo..front.end {
                rows.extend(j + 1..front.end);
                rows.extend_from_slice(&front.updates);
                ptr.push(rows.len());
            }
        }
        let mut links = vec![vec![]; n];
        for j in 0..n {
            for position in ptr[j]..ptr[j + 1] {
                links[rows[position]].push((j, position));
            }
        }
        Some(Self {
            pattern: Arc::new(Pattern { ptr, rows }),
            links,
            primal,
        })
    }

    pub fn factorize(&self, symbolic: &Symbolic, a: &SparseMatrix) -> Result<Factor> {
        let n = symbolic.order.len();
        let mut values = vec![0.; self.pattern.rows.len()];
        let mut diagonal = vec![0.; n];
        let mut work = vec![0.; n];
        let mut stats = FactorStats {
            min_abs_pivot: f64::INFINITY,
            ..Default::default()
        };
        for k in 0..n {
            // Read the permuted upper triangle: this is row k of the
            // symmetric system. Symbolic row links give its triangular reach.
            for (i, value) in a.column(symbolic.order[k]) {
                let row = symbolic.inverse[i];
                if row <= k {
                    work[row] = value;
                }
            }
            let mut d = work[k];
            work[k] = 0.;
            for &(j, position) in &self.links[k] {
                let value = work[j];
                work[j] = 0.;
                // Only the prefix of column j has been computed. Its rows
                // precede k and belong to the same symbolic reach.
                for p in self.pattern.ptr[j]..position {
                    work[self.pattern.rows[p]] -= values[p] * value;
                }
                let entry = value / diagonal[j];
                if !entry.is_finite() {
                    return Err(Error::NumericalFailure("sparse LDL overflow".into()));
                }
                values[position] = entry;
                d -= value * entry;
            }
            if !d.is_finite() || d.abs() <= 1e-11 || (d < 0.) != (k < self.primal) {
                return Err(Error::NumericalFailure("unsafe sparse LDL pivot".into()));
            }
            diagonal[k] = d;
            stats.min_abs_pivot = stats.min_abs_pivot.min(d.abs());
        }
        Ok(Factor {
            l: LowerFactor::Compressed {
                pattern: self.pattern.clone(),
                values,
            },
            blocks: diagonal.into_iter().map(Block::One).collect(),
            order: symbolic.order.clone(),
            stats,
        })
    }
}
