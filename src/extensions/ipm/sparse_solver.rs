use super::factor::{block_order, minimum_degree, Factor, Symbolic, Uplooking};
use super::{finite, Error, Result, SparseMatrix};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SolverType {
    #[default]
    HiPO,
    NormalEquations,
    Auto,
}

#[derive(Debug, Clone, Copy)]
pub struct FillEstimate {
    pub normal_eq_nnz: usize,
    pub augmented_nnz: usize,
}

/// Upper-triangle structural estimate, including all diagonal entries.
/// Contraction is over *columns* of A, because the normal matrix is A D⁻¹ Aᵀ.
pub fn estimate_fill(a: &SparseMatrix) -> FillEstimate {
    let pairs = (0..a.cols).fold(a.rows, |s, j| {
        let d = a.ptr[j + 1] - a.ptr[j];
        s.saturating_add(d.saturating_mul(d.saturating_sub(1)) / 2)
    });
    FillEstimate {
        normal_eq_nnz: pairs.min(a.rows.saturating_mul(a.rows.saturating_add(1)) / 2),
        augmented_nnz: a.cols.saturating_add(a.rows).saturating_add(a.nnz()),
    }
}

fn validate(a: &SparseMatrix, theta: &[f64], rp: &[f64], rd: &[f64]) -> Result<()> {
    if theta.len() != a.cols
        || rp.len() != a.cols
        || rd.len() != a.rows
        || !finite(theta)
        || !finite(rp)
        || !finite(rd)
        || theta.iter().any(|&v| v < 0.)
        || rp.iter().any(|&v| v <= 0.)
        || rd.iter().any(|&v| v <= 0.)
        || theta.iter().zip(rp).any(|(t, r)| !(t + r).is_finite())
    {
        return Err(Error::InvalidInput(
            "invalid KKT diagonal dimensions or regularization".into(),
        ));
    }
    Ok(())
}

fn augmented(a: &SparseMatrix, theta: &[f64], rp: &[f64], rd: &[f64]) -> Result<SparseMatrix> {
    let n = a.cols;
    let mut counts = vec![1; n + a.rows];
    for j in 0..n {
        counts[j] += a.ptr[j + 1] - a.ptr[j];
        for (i, _) in a.column(j) {
            counts[n + i] += 1;
        }
    }
    let mut ptr = Vec::with_capacity(counts.len() + 1);
    ptr.push(0);
    for count in counts {
        ptr.push(ptr.last().unwrap() + count);
    }
    let nnz = *ptr.last().unwrap();
    let mut idx = vec![0; nnz];
    let mut val = vec![0.; nnz];
    let mut next = ptr[n..n + a.rows].to_vec();
    for j in 0..n {
        idx[ptr[j]] = j;
        val[ptr[j]] = -theta[j] - rp[j];
        for (k, (i, v)) in a.column(j).enumerate() {
            idx[ptr[j] + 1 + k] = n + i;
            val[ptr[j] + 1 + k] = v;
            idx[next[i]] = j;
            val[next[i]] = v;
            next[i] += 1;
        }
    }
    for (i, &r) in rd.iter().enumerate() {
        idx[next[i]] = n + i;
        val[next[i]] = r;
    }
    if !finite(&val) {
        return Err(Error::InvalidInput("non-finite KKT matrix".into()));
    }
    Ok(SparseMatrix {
        rows: n + a.rows,
        cols: n + a.rows,
        ptr,
        idx,
        val,
    })
}

/// Sparse up-looking or block-ordered multifrontal augmented-system LDLᵀ.
/// Unsafe sparse pivots retry the multifrontal path with restricted BK pivots.
#[derive(Debug, Clone)]
pub struct HiPOLDLT {
    a: SparseMatrix,
    symbolic: Symbolic,
    factor: Option<Factor>,
    matrix: SparseMatrix,
    diagonal: Vec<usize>,
    uplooking: Option<Uplooking>,
}

impl HiPOLDLT {
    pub fn new(a: SparseMatrix) -> Result<Self> {
        let pattern = augmented(&a, &vec![1.; a.cols], &vec![1.; a.cols], &vec![1.; a.rows])?;
        let symbolic = Symbolic::analyze(&pattern, block_order(&a), a.cols);
        Self::from_symbolic(a, pattern, symbolic)
    }
    fn from_symbolic(a: SparseMatrix, pattern: SparseMatrix, symbolic: Symbolic) -> Result<Self> {
        let uplooking = Uplooking::new(&symbolic, a.cols);
        let diagonal = (0..pattern.cols)
            .map(|j| {
                let start = pattern.ptr[j];
                start
                    + pattern.idx[start..pattern.ptr[j + 1]]
                        .binary_search(&j)
                        .unwrap()
            })
            .collect();
        Ok(Self {
            a,
            symbolic,
            factor: None,
            matrix: pattern,
            diagonal,
            uplooking,
        })
    }
    fn with_appended_columns(&self, a: SparseMatrix) -> Result<Option<Self>> {
        let symbolic = match self.symbolic.extend_primal(&self.a, &a) {
            Some(symbolic) => symbolic,
            None => return Ok(None),
        };
        let pattern = augmented(&a, &vec![1.; a.cols], &vec![1.; a.cols], &vec![1.; a.rows])?;
        Self::from_symbolic(a, pattern, symbolic).map(Some)
    }
    /// Numeric refactorization reuses symbolic analysis. Failure invalidates the old factor.
    pub fn factorize(&mut self, theta: &[f64], rp: &[f64], rd: &[f64]) -> Result<()> {
        self.factor = None;
        validate(&self.a, theta, rp, rd)?;
        // A and the KKT sparsity pattern are fixed for this solver's lifetime.
        // Only diagonal values change between Newton iterations; retain the
        // canonical CSC matrix used by factorization and refinement.
        for j in 0..self.a.cols {
            self.matrix.val[self.diagonal[j]] = -theta[j] - rp[j];
        }
        for (i, &v) in rd.iter().enumerate() {
            self.matrix.val[self.diagonal[self.a.cols + i]] = v;
        }
        let factor = match self
            .uplooking
            .as_ref()
            .and_then(|plan| plan.factorize(&self.symbolic, &self.matrix).ok())
        {
            Some(factor) => factor,
            None => self.symbolic.factorize(&self.matrix, true)?,
        };
        self.factor = Some(factor);
        Ok(())
    }
    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>> {
        self.factor.as_ref().ok_or(Error::NotFactorized)?.solve(rhs)
    }
    pub fn solve_with_refinement(&self, rhs: &[f64]) -> Result<(Vec<f64>, usize)> {
        let mut workspace = SolveWorkspace::default();
        let iterations = self.solve_refined_into(rhs, &mut workspace)?;
        Ok((workspace.solution, iterations))
    }

    fn solve_refined_into(&self, rhs: &[f64], workspace: &mut SolveWorkspace) -> Result<usize> {
        let factor = self.factor.as_ref().ok_or(Error::NotFactorized)?;
        let SolveWorkspace {
            solution: x,
            permuted,
            residual,
            denominator,
            correction,
            ..
        } = workspace;
        factor.solve_into(rhs, x, permuted)?;
        residual.resize(rhs.len(), 0.);
        denominator.resize(rhs.len(), 0.);
        for attempt in 0..=2 {
            let error = refinement_error(&self.matrix, rhs, x, residual, denominator)?;
            if error <= 1e-10 {
                return Ok(attempt);
            }
            if !error.is_finite() || attempt == 2 {
                return Err(Error::NumericalFailure("inaccurate augmented solve".into()));
            }
            factor.solve_into(residual, correction, permuted)?;
            for (xi, di) in x.iter_mut().zip(correction.iter()) {
                *xi += di;
            }
        }
        unreachable!("refinement returns after checking the final residual")
    }
    pub fn perturbed_pivots(&self) -> usize {
        self.factor.as_ref().map_or(0, |f| f.stats.perturbed_pivots)
    }
    pub fn num_2x2_pivots(&self) -> usize {
        self.factor.as_ref().map_or(0, |f| f.stats.num_2x2)
    }
    pub fn min_abs_pivot(&self) -> Option<f64> {
        self.factor.as_ref().map(|f| f.stats.min_abs_pivot)
    }
}

/// SPD normal equations, with recovery of the primal part of the KKT solution.
#[derive(Debug, Clone)]
pub struct NormalEqLDLT {
    a: SparseMatrix,
    symbolic: Symbolic,
    factor: Option<Factor>,
    dinv: Vec<f64>,
    matrix: SparseMatrix,
    row_entries: Vec<Vec<(usize, usize)>>,
    work: Vec<f64>,
}

impl NormalEqLDLT {
    pub fn new(a: SparseMatrix) -> Result<Self> {
        let mut graph = vec![BTreeSet::new(); a.rows];
        let mut row_entries = vec![Vec::new(); a.rows];
        for j in 0..a.cols {
            for p in a.ptr[j]..a.ptr[j + 1] {
                row_entries[a.idx[p]].push((j, p));
            }
            let rows: Vec<_> = a.column(j).map(|(i, _)| i).collect();
            for (p, &i) in rows.iter().enumerate() {
                for &k in &rows[p + 1..] {
                    graph[i].insert(k);
                    graph[k].insert(i);
                }
            }
        }
        let mut entries = Vec::new();
        for (j, neighbors) in graph.iter().enumerate() {
            entries.push((j, j, 1.));
            entries.extend(neighbors.iter().map(|&i| (i, j, 1.)));
        }
        let pattern = SparseMatrix::from_triplets(a.rows, a.rows, &entries)?;
        let symbolic = Symbolic::analyze(&pattern, minimum_degree(graph), a.rows);
        let work = vec![0.; a.rows];
        Ok(Self {
            a,
            symbolic,
            factor: None,
            dinv: vec![],
            matrix: pattern,
            row_entries,
            work,
        })
    }
    pub fn factorize(&mut self, theta: &[f64], rp: &[f64], rd: &[f64]) -> Result<()> {
        self.factor = None;
        validate(&self.a, theta, rp, rd)?;
        self.dinv.clear();
        self.dinv
            .extend(theta.iter().zip(rp).map(|(t, r)| 1. / (t + r)));
        // The row-to-column correspondence and CSC structure are fixed.
        // Accumulate values directly, avoiding duplicate triplets and sorting
        // on every Newton step. Clear scratch after failed refactorizations too.
        self.work.fill(0.);
        for (j, &regularization) in rd.iter().enumerate() {
            for &(col, position) in &self.row_entries[j] {
                let value = self.a.val[position];
                for p in self.a.ptr[col]..self.a.ptr[col + 1] {
                    self.work[self.a.idx[p]] += value * self.a.val[p] * self.dinv[col];
                }
            }
            for p in self.matrix.ptr[j]..self.matrix.ptr[j + 1] {
                let i = self.matrix.idx[p];
                self.matrix.val[p] = self.work[i] + if i == j { regularization } else { 0. };
                self.work[i] = 0.;
            }
        }
        self.factor = Some(self.symbolic.factorize(&self.matrix, false)?);
        Ok(())
    }
    pub fn solve(&self, xi_d: &[f64], xi_p: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
        let factor = self.factor.as_ref().ok_or(Error::NotFactorized)?;
        if xi_d.len() != self.a.cols || xi_p.len() != self.a.rows || !finite(xi_d) || !finite(xi_p)
        {
            return Err(Error::InvalidInput("invalid KKT RHS".into()));
        }
        let scaled: Vec<_> = xi_d.iter().zip(&self.dinv).map(|(d, s)| d * s).collect();
        let rhs: Vec<_> = self
            .a
            .mul(&scaled)
            .iter()
            .zip(xi_p)
            .map(|(a, p)| a + p)
            .collect();
        let dy = factor.solve(&rhs)?;
        let dx: Vec<_> = self
            .a
            .tmul(&dy)
            .iter()
            .zip(xi_d)
            .zip(&self.dinv)
            .map(|((a, d), s)| (a - d) * s)
            .collect();
        if !finite(&dx) {
            return Err(Error::NumericalFailure("primal recovery overflow".into()));
        }
        Ok((dx, dy))
    }
}

// Slice arguments give this hot kernel disjoint output buffers. Keep it out
// of the caller that owns Vec headers, so those alias guarantees remain visible.
#[inline(never)]
fn refinement_error(
    matrix: &SparseMatrix,
    rhs: &[f64],
    x: &[f64],
    residual: &mut [f64],
    denominator: &mut [f64],
) -> Result<f64> {
    residual.fill(0.);
    for (d, b) in denominator.iter_mut().zip(rhs) {
        *d = b.abs();
    }
    // Compute A*x and |A|*|x| in one CSC traversal. Each row keeps
    // precisely the same accumulation order as the separate products.
    for (j, &xj) in x.iter().enumerate() {
        for (i, v) in matrix.column(j) {
            residual[i] += v * xj;
            denominator[i] += v.abs() * xj.abs();
        }
    }
    for (r, b) in residual.iter_mut().zip(rhs) {
        *r = b - *r;
    }
    if !finite(x) || !finite(residual) || !finite(denominator) {
        return Err(Error::NumericalFailure("refinement overflow".into()));
    }
    let error = residual
        .iter()
        .zip(denominator.iter())
        .map(|(r, d)| r.abs() / d.max(1e-300))
        .fold(0.0_f64, f64::max);
    Ok(error)
}

/// Per-IPM scratch; no mutable workspace is shared between solver clones.
#[derive(Debug, Clone, Default)]
pub(crate) struct SolveWorkspace {
    pub(crate) rhs: Vec<f64>,
    pub(crate) solution: Vec<f64>,
    permuted: Vec<f64>,
    residual: Vec<f64>,
    denominator: Vec<f64>,
    correction: Vec<f64>,
}

#[derive(Debug, Clone)]
enum Backend {
    HiPO(HiPOLDLT),
    Normal(NormalEqLDLT),
}

/// Owns matrix values and cached symbolic analysis. Recreate when A changes.
#[derive(Debug, Clone)]
pub struct SparseSolver {
    backend: Backend,
    n: usize,
    m: usize,
}
impl SparseSolver {
    pub fn new(a: SparseMatrix, kind: SolverType) -> Result<Self> {
        let n = a.cols;
        let m = a.rows;
        let fill = estimate_fill(&a);
        let normal = kind == SolverType::NormalEquations
            || (kind == SolverType::Auto && fill.normal_eq_nnz < fill.augmented_nnz);
        let backend = if normal {
            Backend::Normal(NormalEqLDLT::new(a)?)
        } else {
            Backend::HiPO(HiPOLDLT::new(a)?)
        };
        Ok(Self { backend, n, m })
    }
    pub(crate) fn with_appended_columns(
        &self,
        a: SparseMatrix,
        kind: SolverType,
    ) -> Result<Option<Self>> {
        // Auto can select a different backend after growth; re-run selection.
        if kind != SolverType::HiPO {
            return Ok(None);
        }
        match &self.backend {
            Backend::HiPO(s) => {
                let (n, m) = (a.cols, a.rows);
                Ok(s.with_appended_columns(a)?.map(|s| Self {
                    backend: Backend::HiPO(s),
                    n,
                    m,
                }))
            }
            Backend::Normal(_) => Ok(None),
        }
    }
    pub fn active_solver_type(&self) -> SolverType {
        match self.backend {
            Backend::HiPO(_) => SolverType::HiPO,
            Backend::Normal(_) => SolverType::NormalEquations,
        }
    }
    pub fn factorize(&mut self, theta: &[f64], rp: &[f64], rd: &[f64]) -> Result<()> {
        match &mut self.backend {
            Backend::HiPO(s) => s.factorize(theta, rp, rd),
            Backend::Normal(s) => s.factorize(theta, rp, rd),
        }
    }
    pub(crate) fn solve_into(&self, rhs: &[f64], workspace: &mut SolveWorkspace) -> Result<()> {
        if rhs.len() != self.n + self.m {
            return Err(Error::InvalidInput("KKT RHS dimension mismatch".into()));
        }
        match &self.backend {
            Backend::HiPO(s) => {
                s.solve_refined_into(rhs, workspace)?;
            }
            Backend::Normal(s) => {
                let (mut x, y) = s.solve(&rhs[..self.n], &rhs[self.n..])?;
                x.extend(y);
                workspace.solution = x;
            }
        }
        Ok(())
    }
    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>> {
        if rhs.len() != self.n + self.m {
            return Err(Error::InvalidInput("KKT RHS dimension mismatch".into()));
        }
        match &self.backend {
            Backend::HiPO(s) => Ok(s.solve_with_refinement(rhs)?.0),
            Backend::Normal(s) => {
                let (mut x, y) = s.solve(&rhs[..self.n], &rhs[self.n..])?;
                x.extend(y);
                Ok(x)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reused_solve_workspace_matches_full_kkt_residuals_after_errors_and_resizing() {
        let mut workspace = SolveWorkspace::default();
        for (m, n) in [(5, 20), (1, 3), (0, 0), (7, 24)] {
            let mut entries = Vec::new();
            for j in 0..n {
                for i in 0..m {
                    if (i + 3 * j) % 4 != 0 {
                        entries.push((i, j, ((i + j) % 7) as f64 / 9. - 0.25));
                    }
                }
            }
            let a = SparseMatrix::from_triplets(m, n, &entries).unwrap();
            let mut solver = SparseSolver::new(a.clone(), SolverType::HiPO).unwrap();
            for scale in [0.1, 10.] {
                let theta = vec![scale; n];
                let rp = vec![0.1; n];
                let rd = vec![0.2; m];
                solver.factorize(&theta, &rp, &rd).unwrap();
                let matrix = augmented(&a, &theta, &rp, &rd).unwrap();
                let expected: Vec<_> = (0..n + m).map(|i| i as f64 / 13. - 0.5).collect();
                let rhs = matrix.mul(&expected);
                assert!(solver
                    .solve_into(&vec![0.; n + m + 1], &mut workspace)
                    .is_err());
                if n + m > 0 {
                    assert!(solver
                        .solve_into(&vec![f64::NAN; n + m], &mut workspace)
                        .is_err());
                }
                workspace.permuted.fill(f64::NAN);
                workspace.solution.fill(f64::NAN);
                workspace.residual.fill(f64::NAN);
                workspace.denominator.fill(f64::NAN);
                workspace.correction.fill(f64::NAN);
                solver.solve_into(&rhs, &mut workspace).unwrap();
                for (&actual, &want) in workspace.solution.iter().zip(&expected) {
                    assert!((actual - want).abs() < 1e-9);
                }
                let full_residual: Vec<_> = rhs
                    .iter()
                    .zip(matrix.mul(&workspace.solution))
                    .map(|(b, ax)| b - ax)
                    .collect();
                let mut full_denominator: Vec<_> = rhs.iter().map(|b| b.abs()).collect();
                for (j, x) in workspace.solution.iter().enumerate() {
                    for (i, v) in matrix.column(j) {
                        full_denominator[i] += v.abs() * x.abs();
                    }
                }
                assert_eq!(workspace.residual, full_residual);
                assert_eq!(workspace.denominator, full_denominator);
            }
        }
    }

    #[test]
    fn appended_primal_symbolics_preserve_kkt_solutions_and_reject_new_fill() {
        let mut entries = Vec::new();
        for j in 0..8 {
            entries.push((j % 3, j, 1.));
            entries.push((j % 3 + 1, j, 2.));
        }
        let a = SparseMatrix::from_triplets(4, 8, &entries).unwrap();
        let original = HiPOLDLT::new(a).unwrap();
        entries.extend([(0, 8, 0.5), (1, 8, 3.)]);
        // New numerical values can include changed equilibration on old entries.
        for entry in &mut entries {
            entry.2 *= 0.25;
        }
        let a = SparseMatrix::from_triplets(4, 9, &entries).unwrap();
        let extended = original.with_appended_columns(a.clone()).unwrap().unwrap();
        for uplooking in [true, false] {
            let mut solver = extended.clone();
            if !uplooking {
                solver.uplooking = None;
            }
            let expected: Vec<_> = (0..13).map(|i| i as f64 / 7. - 0.5).collect();
            for diagonal in [0.1, 3.] {
                let theta = vec![diagonal; 9];
                let rp = vec![0.1; 9];
                let rd = vec![0.2; 4];
                let matrix = augmented(&a, &theta, &rp, &rd).unwrap();
                let rhs = matrix.mul(&expected);
                solver.factorize(&theta, &rp, &rd).unwrap();
                let solution = solver.solve_with_refinement(&rhs).unwrap().0;
                for (&x, &want) in solution.iter().zip(&expected) {
                    assert!((x - want).abs() < 1e-9, "{x} != {want}");
                }
            }
        }
        entries.push((2, 9, 1.));
        let a = SparseMatrix::from_triplets(4, 10, &entries).unwrap();
        let extended = extended.with_appended_columns(a).unwrap().unwrap();
        entries.extend([(0, 10, 1.), (3, 10, 1.)]);
        let new_fill = SparseMatrix::from_triplets(4, 11, &entries).unwrap();
        assert!(extended.with_appended_columns(new_fill).unwrap().is_none());
        // A structural edit to an existing column also invalidates extension.
        entries.push((3, 0, 1.));
        let edited = SparseMatrix::from_triplets(4, 11, &entries).unwrap();
        assert!(extended.with_appended_columns(edited).unwrap().is_none());
    }

    #[test]
    fn cached_normal_equations_refactor_and_recover_after_failure() {
        let a = SparseMatrix::from_triplets(
            3,
            4,
            &[
                (0, 0, 1.),
                (2, 0, -2.),
                (1, 1, 3.),
                (0, 2, 0.5),
                (1, 2, -1.),
                (2, 3, 2.),
            ],
        )
        .unwrap();
        let mut solver = NormalEqLDLT::new(a.clone()).unwrap();
        let expected = [0.5, -1., 2., 0.25, 1., -0.5, 0.75];
        for theta in [[1., 2., 3., 4.], [4., 0.5, 2., 1.]] {
            let rp = [0.1; 4];
            let rd = [0.2; 3];
            solver.factorize(&theta, &rp, &rd).unwrap();
            let rhs = augmented(&a, &theta, &rp, &rd).unwrap().mul(&expected);
            let (x, y) = solver.solve(&rhs[..4], &rhs[4..]).unwrap();
            for (actual, expected) in x.iter().chain(&y).zip(expected) {
                assert!((actual - expected).abs() < 1e-10);
            }
            assert!(solver.factorize(&theta, &[0.; 4], &rd).is_err());
            assert_eq!(
                solver.solve(&rhs[..4], &rhs[4..]).unwrap_err(),
                Error::NotFactorized
            );
        }
    }

    #[test]
    fn sparse_ldl_clones_share_only_immutable_structure() {
        let n = 64;
        let entries: Vec<_> = (0..n).map(|j| (0, j, 1.)).collect();
        let mut original =
            HiPOLDLT::new(SparseMatrix::from_triplets(1, n, &entries).unwrap()).unwrap();
        assert!(original.uplooking.is_some());
        original
            .factorize(&vec![1.; n], &vec![0.1; n], &[0.2])
            .unwrap();
        let mut sibling = original.clone();
        sibling
            .factorize(&vec![2.; n], &vec![0.1; n], &[0.2])
            .unwrap();
        let mut rhs = vec![-0.1; n];
        rhs.push(n as f64 + 0.2);
        for x in original.solve(&rhs).unwrap() {
            assert!((x - 1.).abs() < 1e-10);
        }
        let mut rhs = vec![-0.8; n];
        rhs.push(n as f64 * 0.5 + 0.05);
        let x = sibling.solve(&rhs).unwrap();
        for &x in &x[..n] {
            assert!((x - 0.5).abs() < 1e-10);
        }
        assert!((x[n] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn sparse_ldl_unsafe_pivots_retry_pivoted_factorization_and_invalidate_on_error() {
        let n = 64;
        let a = SparseMatrix::from_triplets(1, n, &[]).unwrap();
        let mut solver = HiPOLDLT::new(a).unwrap();
        assert!(solver.uplooking.is_some());
        solver
            .factorize(&vec![1.; n], &vec![0.1; n], &[0.2])
            .unwrap();
        assert_eq!(solver.perturbed_pivots(), 0);
        // The fast path rejects these tiny pivots. The existing multifrontal
        // path must regularize them instead of publishing an invalid factor.
        solver
            .factorize(&vec![1e-20; n], &vec![1e-20; n], &[1e-20])
            .unwrap();
        assert_eq!(solver.perturbed_pivots(), n + 1);
        assert_eq!(solver.solve(&vec![0.; n + 1]).unwrap(), vec![0.; n + 1]);
        // The perturbed factor is finite but does not accurately solve the
        // requested system. Refinement must report failure to its caller.
        assert!(matches!(
            solver.solve_with_refinement(&vec![1.; n + 1]),
            Err(Error::NumericalFailure(_))
        ));
        assert!(solver.factorize(&vec![1.; n], &vec![0.; n], &[1.]).is_err());
        assert_eq!(
            solver.solve(&vec![0.; n + 1]).unwrap_err(),
            Error::NotFactorized
        );
    }
}
