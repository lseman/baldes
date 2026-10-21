use super::{finite, Error, Result, SparseMatrix};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sense {
    LessEqual,
    Equal,
    GreaterEqual,
}

/// A minimization LP. Infinite bounds are allowed; all coefficients must be finite.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelData {
    pub a: SparseMatrix,
    pub b: Vec<f64>,
    pub c: Vec<f64>,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub sense: Vec<Sense>,
}

impl ModelData {
    pub fn validate(&self) -> Result<()> {
        let n = self.a.cols();
        let m = self.a.rows();
        if self.c.len() != n
            || self.lower.len() != n
            || self.upper.len() != n
            || self.b.len() != m
            || self.sense.len() != m
        {
            return Err(Error::InvalidInput("model dimensions do not match".into()));
        }
        if !finite(&self.b) || !finite(&self.c) {
            return Err(Error::InvalidInput("non-finite objective or RHS".into()));
        }
        for (&l, &u) in self.lower.iter().zip(&self.upper) {
            if l.is_nan() || u.is_nan() || l > u || l == f64::INFINITY || u == f64::NEG_INFINITY {
                return Err(Error::InvalidInput("invalid variable bounds".into()));
            }
        }
        Ok(())
    }
    pub(crate) fn standard_form(&self) -> Result<StandardForm> {
        self.validate()?;
        let mut b = self.b.clone();
        let mut c = vec![];
        let mut ptr = vec![0];
        let mut idx = vec![];
        let mut val = vec![];
        let mut recovery = vec![];
        let mut upper = vec![];
        let row_sign: Vec<_> = self
            .sense
            .iter()
            .map(|s| if *s == Sense::GreaterEqual { -1.0 } else { 1.0 })
            .collect();
        for (bi, &s) in b.iter_mut().zip(&row_sign) {
            *bi *= s;
        }
        for j in 0..self.a.cols() {
            let l = self.lower[j];
            let u = self.upper[j];
            let offset = if l.is_finite() {
                l
            } else if u.is_finite() {
                u
            } else {
                0.0
            };
            let mut terms = vec![];
            if l != u {
                let sign = if l.is_finite() || !u.is_finite() {
                    1.0
                } else {
                    -1.0
                };
                let k = c.len();
                c.push(sign * self.c[j]);
                terms.push((k, sign));
                if l.is_finite() && u.is_finite() {
                    upper.push((k, u - l));
                }
                if !l.is_finite() && !u.is_finite() {
                    terms.push((c.len(), -1.0));
                    c.push(-self.c[j]);
                }
            }
            for (i, v) in self.a.column(j) {
                b[i] -= row_sign[i] * v * offset;
            }
            // Each transformed column inherits the canonical row order of A.
            // Building CSC directly avoids sorting the entire master again.
            for &(_, sign) in &terms {
                for (i, v) in self.a.column(j) {
                    idx.push(i);
                    val.push(row_sign[i] * v * sign);
                }
                ptr.push(idx.len());
            }
            recovery.push((offset, terms));
        }
        for (i, s) in self.sense.iter().enumerate() {
            if *s != Sense::Equal {
                idx.push(i);
                val.push(1.0);
                ptr.push(idx.len());
                c.push(0.0);
            }
        }
        if !finite(&b) || upper.iter().any(|&(_, u)| !u.is_finite()) {
            return Err(Error::InvalidInput("bound transformation overflow".into()));
        }
        Ok(StandardForm {
            a: SparseMatrix {
                rows: b.len(),
                cols: c.len(),
                ptr,
                idx,
                val,
            },
            b,
            c,
            upper,
            recovery,
            row_sign,
        })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct StandardForm {
    pub a: SparseMatrix,
    pub b: Vec<f64>,
    pub c: Vec<f64>,
    pub upper: Vec<(usize, f64)>,
    pub recovery: Vec<(f64, Vec<(usize, f64)>)>,
    pub row_sign: Vec<f64>,
}
