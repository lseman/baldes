use super::*;

impl HighsMaster {
    pub fn new() -> Result<Self> {
        // SAFETY: Highs_create returns an exclusively owned handle.
        let highs = NonNull::new(unsafe { Highs_create() })
            .ok_or_else(|| incomplete("HiGHS model allocation failed"))?;
        let master = Self {
            highs,
            matrix_cache: OnceCell::new(),
            rhs: vec![],
            senses: vec![],
            costs: vec![],
            upper: vec![],
            duals: vec![],
            values: vec![],
            reduced_costs: vec![],
            row_status: vec![],
            row_values: vec![],
            col_status: vec![],
            primal_tolerance: 1e-7,
            iteration_limit: None,
            objective: 0.,
            objective_offset: 0.,
            status: -1,
            solved: false,
            dual_tolerance: 1e-7,
            zero_tolerance: 0.,
        };
        master.set_bool("output_flag", false)?;
        // Adding columns preserves primal feasibility but can invalidate dual
        // feasibility; adding cuts or changing bounds can do the reverse.
        // Let simplex choose the direction from the retained basis instead of
        // HiGHS's default of always using dual simplex.
        // SAFETY: live owned handle and a static NUL-terminated option name.
        unsafe {
            Self::check(
                Highs_setIntOptionValue(highs.as_ptr(), b"simplex_strategy\0".as_ptr().cast(), 0),
                "set adaptive simplex strategy",
            )?;
        }
        Ok(master)
    }

    /// Independently copy the model and install its current native basis.
    /// Native factorization storage is private to each instance; the child
    /// refactorizes this basis once, then reuses its own solver state.
    pub fn try_clone(&self) -> Result<Self> {
        let mut copy = Self::new()?;
        copy.set_tolerances(self.primal_tolerance, self.dual_tolerance)?;
        let rows = self
            .senses
            .iter()
            .zip(&self.rhs)
            .map(|(&s, &b)| (s, b, &[][..]))
            .collect::<Vec<_>>();
        copy.add_rows(&rows)?;
        let count = self.costs.len();
        if count > 0 {
            let matrix = self.sparse_columns()?;
            let lower = vec![0.; count];
            // SAFETY: the immutable CSC snapshot has one start per column and
            // the declared nonzeros. HiGHS copies it into the child's own model.
            unsafe {
                Self::check(
                    Highs_addCols(
                        copy.highs.as_ptr(),
                        count as HighsInt,
                        self.costs.as_ptr(),
                        lower.as_ptr(),
                        self.upper.as_ptr(),
                        matrix.indices.len() as HighsInt,
                        matrix.starts.as_ptr(),
                        matrix.indices.as_ptr(),
                        matrix.values.as_ptr(),
                    ),
                    "add clone columns",
                )?;
            }
            copy.costs.clone_from(&self.costs);
            copy.upper.clone_from(&self.upper);
        }
        // Query the current basis directly: edits may invalidate solution info
        // while retaining a valid (possibly infeasible) simplex warm basis.
        let mut columns = vec![0; self.costs.len()];
        let mut rows = vec![0; self.rhs.len()];
        // SAFETY: source is borrowed, destination independently owns its basis.
        unsafe {
            if Highs_getBasis(self.highs.as_ptr(), columns.as_mut_ptr(), rows.as_mut_ptr()) == 0 {
                Self::check(
                    Highs_setBasis(copy.highs.as_ptr(), columns.as_ptr(), rows.as_ptr()),
                    "setBasis for clone",
                )?;
            }
        }
        copy.values.clone_from(&self.values);
        copy.duals.clone_from(&self.duals);
        copy.reduced_costs.clone_from(&self.reduced_costs);
        copy.row_status.clone_from(&self.row_status);
        copy.col_status.clone_from(&self.col_status);
        copy.objective = self.objective;
        copy.objective_offset = self.objective_offset;
        copy.zero_tolerance = self.zero_tolerance;
        copy.status = self.status;
        copy.solved = self.solved;
        Ok(copy)
    }

    /// HiGHS range queries have substantial per-call overhead on large LPs.
    /// Snapshot the CSC once per matrix revision instead of querying each column
    /// separately during cut replay, cleanup, certification, or node restoration.
    pub(super) fn sparse_columns(&self) -> Result<&SparseColumns> {
        if self.matrix_cache.get().is_none() {
            let count = self.costs.len();
            let mut matrix = SparseColumns {
                starts: vec![0; count + 1],
                indices: vec![],
                values: vec![],
            };
            if count > 0 {
                let (mut num_col, mut nnz) = (0, 0);
                let mut costs = vec![0.; count];
                let mut lower = vec![0.; count];
                let mut upper = vec![0.; count];
                // SAFETY: the source is immutably borrowed. The first query
                // sizes the CSC; the second fills exactly those owned buffers.
                unsafe {
                    Self::check(
                        Highs_getColsByRange(
                            self.highs.as_ptr(),
                            0,
                            count as HighsInt - 1,
                            &mut num_col,
                            costs.as_mut_ptr(),
                            lower.as_mut_ptr(),
                            upper.as_mut_ptr(),
                            &mut nnz,
                            std::ptr::null_mut(),
                            std::ptr::null_mut(),
                            std::ptr::null_mut(),
                        ),
                        "size matrix snapshot",
                    )?;
                    if nnz < 0 || num_col != count as HighsInt {
                        return Err(incomplete("HiGHS matrix snapshot dimensions"));
                    }
                    matrix.indices.resize(nnz as usize, 0);
                    matrix.values.resize(nnz as usize, 0.);
                    Self::check(
                        Highs_getColsByRange(
                            self.highs.as_ptr(),
                            0,
                            count as HighsInt - 1,
                            &mut num_col,
                            costs.as_mut_ptr(),
                            lower.as_mut_ptr(),
                            upper.as_mut_ptr(),
                            &mut nnz,
                            matrix.starts.as_mut_ptr(),
                            matrix.indices.as_mut_ptr(),
                            matrix.values.as_mut_ptr(),
                        ),
                        "read matrix snapshot",
                    )?;
                }
                matrix.starts[count] = nnz;
            }
            // No reentrant callbacks or concurrent mutation can populate it
            // between these calls; native edits require an exclusive borrow.
            let _ = self.matrix_cache.set(matrix);
        }
        Ok(self.matrix_cache.get().expect("matrix snapshot populated"))
    }

    /// Replace RHS values without rebuilding the model or discarding the basis.
    pub fn set_rhs(&mut self, rhs: &[f64]) -> Result<()> {
        if rhs.len() != self.rhs.len() || rhs.iter().any(|v| !v.is_finite()) {
            return Err(invalid("HiGHS RHS dimensions or values"));
        }
        if rhs == self.rhs {
            return Ok(());
        }
        let infinity = self.infinity();
        let lower: Vec<_> = rhs
            .iter()
            .zip(&self.senses)
            .map(|(&b, &s)| {
                if s == RowSense::LessEqual {
                    -infinity
                } else {
                    b
                }
            })
            .collect();
        let upper: Vec<_> = rhs
            .iter()
            .zip(&self.senses)
            .map(|(&b, &s)| {
                if s == RowSense::GreaterEqual {
                    infinity
                } else {
                    b
                }
            })
            .collect();
        if !rhs.is_empty() {
            // SAFETY: one lower/upper bound for every native row; HiGHS copies.
            unsafe {
                Self::check(
                    Highs_changeRowsBoundsByRange(
                        self.highs.as_ptr(),
                        0,
                        (rhs.len() - 1) as HighsInt,
                        lower.as_ptr(),
                        upper.as_ptr(),
                    ),
                    "changeRowsBoundsByRange",
                )?;
            }
        }
        self.rhs.copy_from_slice(rhs);
        self.solved = false;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn native_identity(&self) -> usize {
        self.highs.as_ptr() as usize
    }

    /// Simplex pivots from the most recent native solve.
    pub fn simplex_iterations(&self) -> Result<usize> {
        let mut count = 0;
        // SAFETY: live instance and one integer output.
        unsafe {
            Self::check(
                Highs_getIntInfoValue(
                    self.highs.as_ptr(),
                    b"simplex_iteration_count\0".as_ptr().cast(),
                    &mut count,
                ),
                "simplex_iteration_count",
            )?;
        }
        usize::try_from(count).map_err(|_| incomplete("negative HiGHS iteration count"))
    }

    pub(crate) fn set_lp_iteration_limit(&mut self, iterations: usize) -> Result<()> {
        if self.iteration_limit == Some(iterations) {
            return Ok(());
        }
        let limit = i32::try_from(iterations).map_err(|_| invalid("HiGHS iteration limit"))?;
        // SAFETY: live owned handle and static NUL-terminated option names.
        unsafe {
            Self::check(
                Highs_setIntOptionValue(
                    self.highs.as_ptr(),
                    b"simplex_iteration_limit\0".as_ptr().cast(),
                    limit,
                ),
                "simplex_iteration_limit",
            )?;
            Self::check(
                Highs_setIntOptionValue(
                    self.highs.as_ptr(),
                    b"ipm_iteration_limit\0".as_ptr().cast(),
                    limit,
                ),
                "ipm_iteration_limit",
            )?;
        }
        self.iteration_limit = Some(iterations);
        Ok(())
    }

    pub(super) fn set_bool(&self, name: &str, value: bool) -> Result<()> {
        let name_c = CString::new(name).map_err(|_| invalid("HiGHS option name"))?;
        // SAFETY: live handle, NUL-terminated option name.
        let rc = unsafe {
            Highs_setBoolOptionValue(self.highs.as_ptr(), name_c.as_ptr(), value as HighsInt)
        };
        if rc == STATUS_ERROR {
            return Err(invalid(format!("HiGHS option {name}")));
        }
        Ok(())
    }

    pub(super) fn set_double(&self, name: &str, value: f64) -> Result<()> {
        let name_c = CString::new(name).map_err(|_| invalid("HiGHS option name"))?;
        // SAFETY: live handle, NUL-terminated option name.
        let rc = unsafe { Highs_setDoubleOptionValue(self.highs.as_ptr(), name_c.as_ptr(), value) };
        if rc == STATUS_ERROR {
            return Err(invalid(format!("HiGHS option {name}")));
        }
        Ok(())
    }

    pub(super) fn infinity(&self) -> f64 {
        // SAFETY: live handle.
        unsafe { Highs_getInfinity(self.highs.as_ptr()) }
    }

    pub(super) fn check(rc: HighsInt, what: &str) -> Result<()> {
        if rc == STATUS_ERROR {
            Err(invalid(format!("HiGHS {what} failed")))
        } else {
            Ok(())
        }
    }
}
