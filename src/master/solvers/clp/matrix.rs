use super::*;

impl ClpMaster {
    /// Replace all current row bounds, preserving row identities and the basis.
    pub fn set_rhs(&mut self, rhs: &[f64]) -> Result<()> {
        if rhs.len() != self.rhs.len() || rhs.iter().any(|x| !x.is_finite()) {
            return Err(invalid("Clp RHS dimensions or values"));
        }
        let lower: Vec<_> = rhs
            .iter()
            .zip(&self.senses)
            .map(|(&b, &s)| {
                if s == RowSense::LessEqual {
                    -f64::MAX
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
                    f64::MAX
                } else {
                    b
                }
            })
            .collect();
        // SAFETY: both buffers contain one bound for every live model row;
        // Clp copies them during these calls and the model is exclusively owned.
        unsafe {
            Clp_chgRowLower(self.model.as_ptr(), lower.as_ptr());
            Clp_chgRowUpper(self.model.as_ptr(), upper.as_ptr());
        }
        self.rhs.copy_from_slice(rhs);
        #[cfg(feature = "highs")]
        self.edit_highs(|backend| backend.set_rhs(rhs))?;
        self.solved = false;
        Ok(())
    }

    pub fn add_row(
        &mut self,
        sense: RowSense,
        rhs: f64,
        entries: &[(usize, f64)],
    ) -> Result<usize> {
        Ok(self.add_rows(&[(sense, rhs, entries)])?.start)
    }

    /// Adds the rows in order with one Clp call.
    pub fn add_rows(
        &mut self,
        rows: &[(RowSense, f64, &[(usize, f64)])],
    ) -> Result<std::ops::Range<usize>> {
        let first = self.rhs.len();
        if rows.is_empty() {
            return Ok(first..first);
        }
        let (mut lower, mut upper, mut starts) = (vec![], vec![], vec![0i32]);
        let (mut indices, mut values) = (vec![], vec![]);
        for &(sense, rhs, entries) in rows {
            if !rhs.is_finite() {
                return Err(invalid("finite row RHS required"));
            }
            let (i, v) = entries_of(entries, self.costs.len())?;
            lower.push(if sense == RowSense::LessEqual {
                -f64::MAX
            } else {
                rhs
            });
            upper.push(if sense == RowSense::GreaterEqual {
                f64::MAX
            } else {
                rhs
            });
            indices.extend(i);
            values.extend(v);
            starts.push(indices.len() as i32);
        }
        // SAFETY: All arrays contain the declared row and nonzero counts; Clp copies them.
        unsafe {
            Clp_addRows(
                self.model.as_ptr(),
                rows.len() as i32,
                lower.as_ptr(),
                upper.as_ptr(),
                starts.as_ptr(),
                indices.as_ptr(),
                values.as_ptr(),
            );
        }
        for &(sense, rhs, _) in rows {
            self.rhs.push(rhs);
            self.senses.push(sense);
        }
        self.solved = false;
        #[cfg(feature = "highs")]
        self.edit_highs(|backend| backend.add_rows(rows).map(|_| ()))?;
        Ok(first..self.rhs.len())
    }

    pub fn add_column(&mut self, cost: f64, entries: &[(usize, f64)]) -> Result<usize> {
        Ok(self.add_columns(&[(cost, entries)])?.start)
    }

    /// Adds the columns in order with one Clp call: Clp copies the whole
    /// matrix on every append, so adding them one by one was quadratic.
    /// The model and bookkeeping match adding each with `add_column`.
    pub fn add_columns(
        &mut self,
        columns: &[(f64, &[(usize, f64)])],
    ) -> Result<std::ops::Range<usize>> {
        let first = self.costs.len();
        if columns.is_empty() {
            return Ok(first..first);
        }
        let mut starts = vec![0i32];
        let mut indices = vec![];
        let mut values = vec![];
        let mut costs = Vec::with_capacity(columns.len());
        // `add_column` kept a zero value per column while `solved` held.
        let mut preserve = self.solved;
        let mut preserved = 0;
        for &(cost, entries) in columns {
            if !cost.is_finite() {
                return Err(invalid("finite column cost required"));
            }
            let (ids, vals) = entries_of(entries, self.rhs.len())?;
            let mut reduced = cost;
            for (&r, &a) in ids.iter().zip(&vals) {
                reduced -= self.duals.get(r as usize).copied().unwrap_or(0.) * a;
            }
            preserve = preserve && reduced >= -self.dual_tolerance;
            preserved += preserve as usize;
            indices.extend(ids);
            values.extend(vals);
            starts.push(indices.len() as i32);
            costs.push(cost);
        }
        let lower = vec![0.; columns.len()];
        let upper = vec![1e12; columns.len()];
        // SAFETY: Model is exclusively borrowed, Clp copies the supplied arrays.
        unsafe {
            Clp_addColumns(
                self.model.as_ptr(),
                columns.len() as i32,
                lower.as_ptr(),
                upper.as_ptr(),
                costs.as_ptr(),
                starts.as_ptr(),
                indices.as_ptr(),
                values.as_ptr(),
            );
        }
        self.costs.extend(&costs);
        self.upper.extend(&upper);
        self.solved = preserve;
        self.values.resize(self.values.len() + preserved, 0.);
        #[cfg(feature = "highs")]
        self.edit_highs(|backend| backend.add_columns(columns).map(|_| ()))?;
        Ok(first..self.costs.len())
    }

    pub fn set_coefficient(&mut self, row: usize, column: usize, value: f64) -> Result<()> {
        if row >= self.rhs.len() || column >= self.costs.len() || !value.is_finite() {
            return Err(invalid("master coefficient index/value"));
        }
        unsafe {
            Clp_modifyCoefficient(self.model.as_ptr(), row as i32, column as i32, value, true);
        }
        #[cfg(feature = "highs")]
        self.edit_highs(|backend| backend.set_coefficient(row, column, value))?;
        self.solved = false;
        Ok(())
    }

    pub fn set_cost(&mut self, column: usize, cost: f64) -> Result<()> {
        if column >= self.costs.len() || !cost.is_finite() {
            return Err(invalid("master cost index/value"));
        }
        self.costs[column] = cost;
        unsafe {
            Clp_chgObjCoefficients(self.model.as_ptr(), self.costs.as_ptr());
        }
        #[cfg(feature = "highs")]
        self.edit_highs(|backend| backend.set_cost(column, cost))?;
        self.solved = false;
        Ok(())
    }

    pub fn set_upper(&mut self, column: usize, upper: f64) -> Result<()> {
        if column >= self.upper.len() || !upper.is_finite() || upper < 0. {
            return Err(invalid("master upper bound"));
        }
        self.upper[column] = upper;
        unsafe {
            Clp_chgColumnUpper(self.model.as_ptr(), self.upper.as_ptr());
        }
        #[cfg(feature = "highs")]
        self.edit_highs(|backend| backend.set_upper(column, upper))?;
        self.solved = false;
        Ok(())
    }

    pub fn row_is_basic(&self, row: usize) -> Result<bool> {
        if row >= self.rhs.len() {
            return Err(invalid("master row index"));
        }
        if self.solver == MasterSolver::Ipm {
            return Ok(false);
        }
        if self.solver == MasterSolver::Highs {
            return self
                .highs_row_basic
                .get(row)
                .copied()
                .ok_or_else(|| invalid("HiGHS basis requires a solved master"));
        }
        Ok(unsafe { Clp_getRowStatus(self.model.as_ptr(), row as i32) } == 1)
    }

    /// Cut cleanup with default threshold 1
    /// and retention ratio 0.66. The normalization divides by the SIGNED RHS
    /// when |rhs| > 1: a slack packing cut with rhs < -1 is retained.
    pub fn cleanup_cut_rows(&self, initial_rows: usize) -> Result<Vec<usize>> {
        crate::master::lp_master::cleanup_cut_rows(self, initial_rows)
    }

    /// Sparse column of the current matrix, sorted by row.
    pub fn column_entries(&self, column: usize) -> Result<Vec<(usize, f64)>> {
        if column >= self.costs.len() {
            return Err(invalid("master column index"));
        }
        let m = self.model.as_ptr();
        let mut entries = unsafe {
            let start = *Clp_getVectorStarts(m).add(column) as usize;
            let length = *Clp_getVectorLengths(m).add(column) as usize;
            let indices = std::slice::from_raw_parts(Clp_getIndices(m).add(start), length);
            let values = std::slice::from_raw_parts(Clp_getElements(m).add(start), length);
            indices
                .iter()
                .zip(values)
                .map(|(&r, &v)| (r as usize, v))
                .collect::<Vec<_>>()
        };
        entries.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok(entries)
    }

    /// ClpMaster::remove_cuts: basic cut rows and their nonbasic artificials.
    /// The current solution stays available, compacted to the kept indices.
    pub fn remove_cuts(
        &mut self,
        rows: &[usize],
        columns: &[usize],
    ) -> Result<(Vec<Option<usize>>, Vec<Option<usize>>)> {
        let solved = self.solved;
        let duals = self.duals.clone();
        let values = self.values.clone();
        let reduced_costs = self.reduced_costs.clone();
        let (rm, cm) = self.remove(rows, columns)?;
        if solved {
            self.duals = duals
                .into_iter()
                .zip(&rm)
                .filter_map(|(x, m)| m.map(|_| x))
                .collect();
            self.values = values
                .into_iter()
                .zip(&cm)
                .filter_map(|(x, m)| m.map(|_| x))
                .collect();
            self.reduced_costs = reduced_costs
                .into_iter()
                .zip(&cm)
                .filter_map(|(x, m)| m.map(|_| x))
                .collect();
            self.solved = true;
        }
        Ok((rm, cm))
    }

    pub fn remove(
        &mut self,
        rows: &[usize],
        columns: &[usize],
    ) -> Result<(Vec<Option<usize>>, Vec<Option<usize>>)> {
        fn mapping(size: usize, removed: &[usize]) -> Result<(Vec<Option<usize>>, Vec<i32>)> {
            let mut ids = removed.to_vec();
            ids.sort_unstable();
            ids.dedup();
            if ids.iter().any(|&i| i >= size) {
                return Err(invalid("master deletion index"));
            }
            let mut next = 0;
            let map = (0..size)
                .map(|i| {
                    if ids.binary_search(&i).is_ok() {
                        None
                    } else {
                        let id = next;
                        next += 1;
                        Some(id)
                    }
                })
                .collect();
            Ok((map, ids.into_iter().map(|i| i as i32).collect()))
        }
        let (rm, ri) = mapping(self.rhs.len(), rows)?;
        let (cm, ci) = mapping(self.costs.len(), columns)?;
        // Skip empty deletions: a Clp call
        // with no index still resets internal state.
        unsafe {
            if !ri.is_empty() {
                Clp_deleteRows(self.model.as_ptr(), ri.len() as i32, ri.as_ptr());
            }
            if !ci.is_empty() {
                Clp_deleteColumns(self.model.as_ptr(), ci.len() as i32, ci.as_ptr());
            }
        }
        #[cfg(feature = "highs")]
        self.edit_highs(|backend| backend.remove(rows, columns).map(|_| ()))?;
        self.highs_row_basic = self
            .highs_row_basic
            .iter()
            .zip(&rm)
            .filter_map(|(&value, mapped)| mapped.map(|_| value))
            .collect();
        self.rhs = self
            .rhs
            .iter()
            .zip(&rm)
            .filter_map(|(&x, m)| m.map(|_| x))
            .collect();
        self.senses = self
            .senses
            .iter()
            .zip(&rm)
            .filter_map(|(&x, m)| m.map(|_| x))
            .collect();
        self.costs = self
            .costs
            .iter()
            .zip(&cm)
            .filter_map(|(&x, m)| m.map(|_| x))
            .collect();
        self.upper = self
            .upper
            .iter()
            .zip(&cm)
            .filter_map(|(&x, m)| m.map(|_| x))
            .collect();
        self.solved = false;
        Ok((rm, cm))
    }
}
