//! Square matrix stored as a flat vector (row-major).
//!
//! Mirrors the C++ `Matrix` class used for distance/time data in HGS.

#[derive(Clone, Debug)]
pub struct Matrix {
    cols: usize,
    data: Vec<i32>,
}

impl Matrix {
    /// Create a square matrix of size `dimension × dimension`, initialised to zero.
    pub fn new(dimension: usize) -> Self {
        Self {
            cols: dimension,
            data: vec![0; dimension * dimension],
        }
    }

    /// Number of rows (equals columns for square matrices).
    #[inline]
    pub fn size(&self) -> usize {
        self.cols
    }

    /// Set value at `(row, col)`.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, val: i32) {
        self.data[self.cols * row + col] = val;
    }

    /// Get value at `(row, col)`.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> i32 {
        self.data[self.cols * row + col]
    }

    /// Row slice (read-only).
    #[inline]
    pub fn row(&self, row: usize) -> &[i32] {
        &self.data[self.cols * row..self.cols * row + self.cols]
    }

    /// Mutable row slice.
    #[inline]
    pub fn row_mut(&mut self, row: usize) -> &mut [i32] {
        &mut self.data[self.cols * row..self.cols * row + self.cols]
    }
}

impl std::ops::Index<(usize, usize)> for Matrix {
    type Output = i32;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.data[self.cols * row + col]
    }
}

impl std::ops::IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        &mut self.data[self.cols * row + col]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_access() {
        let mut m = Matrix::new(3);
        assert_eq!(m.get(0, 0), 0);
        m.set(1, 2, 42);
        assert_eq!(m.get(1, 2), 42);
        assert_eq!(m[(1, 2)], 42);
        m[(1, 2)] = 99;
        assert_eq!(m.get(1, 2), 99);
    }

    #[test]
    fn row_slice() {
        let mut m = Matrix::new(3);
        m.set(0, 0, 1);
        m.set(0, 1, 2);
        m.set(0, 2, 3);
        assert_eq!(m.row(0), &[1, 2, 3]);
    }
}
