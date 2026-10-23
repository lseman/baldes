use super::{Error, Result};
use std::collections::BTreeMap;

/// Canonical compressed sparse column matrix. Constructors sort and sum duplicates.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseMatrix {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) ptr: Vec<usize>,
    pub(crate) idx: Vec<usize>,
    pub(crate) val: Vec<f64>,
}

impl SparseMatrix {
    pub fn from_triplets(
        rows: usize,
        cols: usize,
        entries: &[(usize, usize, f64)],
    ) -> Result<Self> {
        let mut columns = vec![BTreeMap::new(); cols];
        for &(i, j, v) in entries {
            if i >= rows || j >= cols || !v.is_finite() {
                return Err(Error::InvalidInput(
                    "matrix entry out of bounds or non-finite".into(),
                ));
            }
            *columns[j].entry(i).or_insert(0.0) += v;
        }
        let mut a = Self {
            rows,
            cols,
            ptr: vec![0],
            idx: vec![],
            val: vec![],
        };
        for col in columns {
            for (i, v) in col {
                if !v.is_finite() {
                    return Err(Error::InvalidInput("duplicate entry sum overflow".into()));
                }
                if v != 0.0 {
                    a.idx.push(i);
                    a.val.push(v);
                }
            }
            a.ptr.push(a.idx.len());
        }
        Ok(a)
    }

    pub fn from_csr(
        rows: usize,
        cols: usize,
        row_ptr: &[usize],
        col_idx: &[usize],
        values: &[f64],
    ) -> Result<Self> {
        if row_ptr.len() != rows + 1
            || row_ptr.first() != Some(&0)
            || row_ptr.last() != Some(&values.len())
            || col_idx.len() != values.len()
            || row_ptr.windows(2).any(|w| w[0] > w[1])
        {
            return Err(Error::InvalidInput("malformed CSR matrix".into()));
        }
        let mut entries = Vec::with_capacity(values.len());
        for i in 0..rows {
            for k in row_ptr[i]..row_ptr[i + 1] {
                entries.push((i, col_idx[k], values[k]));
            }
        }
        Self::from_triplets(rows, cols, &entries)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn nnz(&self) -> usize {
        self.val.len()
    }
    pub fn column(&self, j: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        (self.ptr[j]..self.ptr[j + 1]).map(|p| (self.idx[p], self.val[p]))
    }
    pub fn multiply(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.cols {
            return Err(Error::InvalidInput(
                "matrix-vector dimension mismatch".into(),
            ));
        }
        Ok(self.mul(x))
    }
    pub fn transpose_multiply(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.rows {
            return Err(Error::InvalidInput(
                "transpose-vector dimension mismatch".into(),
            ));
        }
        Ok(self.tmul(x))
    }
    pub(crate) fn mul(&self, x: &[f64]) -> Vec<f64> {
        let mut y = Vec::new();
        self.mul_into(x, &mut y);
        y
    }
    pub(crate) fn mul_into(&self, x: &[f64], y: &mut Vec<f64>) {
        y.resize(self.rows, 0.);
        y.fill(0.);
        for (j, &xj) in x.iter().enumerate() {
            for (i, v) in self.column(j) {
                y[i] += v * xj;
            }
        }
    }
    pub(crate) fn tmul(&self, x: &[f64]) -> Vec<f64> {
        let mut y = Vec::new();
        self.tmul_into(x, &mut y);
        y
    }
    pub(crate) fn tmul_into(&self, x: &[f64], y: &mut Vec<f64>) {
        y.clear();
        y.extend((0..self.cols).map(|j| self.column(j).map(|(i, v)| v * x[i]).sum::<f64>()));
    }
}
