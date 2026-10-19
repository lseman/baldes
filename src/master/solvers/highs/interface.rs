use super::*;

impl LpMaster for HighsMaster {
    fn add_row(&mut self, sense: RowSense, rhs: f64, entries: &[(usize, f64)]) -> Result<usize> {
        Ok(self.add_rows(&[(sense, rhs, entries)])?.start)
    }

    fn add_rows(
        &mut self,
        rows: &[(RowSense, f64, &[(usize, f64)])],
    ) -> Result<std::ops::Range<usize>> {
        let first = self.rhs.len();
        if rows.is_empty() {
            return Ok(first..first);
        }
        let infinity = self.infinity();
        let mut lower = Vec::with_capacity(rows.len());
        let mut upper = Vec::with_capacity(rows.len());
        let mut starts = vec![0];
        let mut indices = vec![];
        let mut values = vec![];
        for &(sense, rhs, entries) in rows {
            if !rhs.is_finite() {
                return Err(invalid("finite row RHS required"));
            }
            let (i, v) = entries_of(entries, self.costs.len())?;
            lower.push(if sense == RowSense::LessEqual {
                -infinity
            } else {
                rhs
            });
            upper.push(if sense == RowSense::GreaterEqual {
                infinity
            } else {
                rhs
            });
            indices.extend(i);
            values.extend(v);
            starts.push(indices.len() as HighsInt);
        }
        // SAFETY: buffers contain all row bounds and declared sparse entries;
        // HiGHS copies them without invalidating the surviving basis columns.
        unsafe {
            Self::check(
                Highs_addRows(
                    self.highs.as_ptr(),
                    rows.len() as HighsInt,
                    lower.as_ptr(),
                    upper.as_ptr(),
                    indices.len() as HighsInt,
                    starts.as_ptr(),
                    indices.as_ptr(),
                    values.as_ptr(),
                ),
                "addRows",
            )?;
        }
        for &(sense, rhs, _) in rows {
            self.senses.push(sense);
            self.rhs.push(rhs);
        }
        self.solved = false;
        Ok(first..self.rhs.len())
    }

    fn add_column(&mut self, cost: f64, entries: &[(usize, f64)]) -> Result<usize> {
        Ok(self.add_columns(&[(cost, entries)])?.start)
    }

    /// As `ClpMaster::add_columns`: one call; a solution is kept, with zero
    /// values, for the leading columns that are dual feasible.
    fn add_columns(
        &mut self,
        columns: &[(f64, &[(usize, f64)])],
    ) -> Result<std::ops::Range<usize>> {
        let first = self.costs.len();
        if columns.is_empty() {
            return Ok(first..first);
        }
        let mut starts = vec![];
        let mut indices = vec![];
        let mut values = vec![];
        let mut costs = Vec::with_capacity(columns.len());
        let mut reduced = Vec::with_capacity(columns.len());
        let mut preserve = self.solved;
        let mut preserved = 0;
        for &(cost, entries) in columns {
            if !cost.is_finite() {
                return Err(invalid("finite column cost required"));
            }
            let (ids, vals) = entries_of(entries, self.rhs.len())?;
            let mut rc = cost;
            for (&r, &a) in ids.iter().zip(&vals) {
                rc -= self.duals.get(r as usize).copied().unwrap_or(0.) * a;
            }
            preserve = preserve && rc >= -self.dual_tolerance;
            preserved += preserve as usize;
            starts.push(indices.len() as HighsInt);
            indices.extend(ids);
            values.extend(vals);
            costs.push(cost);
            reduced.push(rc);
        }
        let lower = vec![0.; columns.len()];
        let upper = vec![COLUMN_UPPER; columns.len()];
        // SAFETY: `starts` has one entry per column, `indices`/`values` the
        // declared nonzeros; HiGHS copies them.
        let rc = unsafe {
            Highs_addCols(
                self.highs.as_ptr(),
                columns.len() as HighsInt,
                costs.as_ptr(),
                lower.as_ptr(),
                upper.as_ptr(),
                indices.len() as HighsInt,
                starts.as_ptr(),
                indices.as_ptr(),
                values.as_ptr(),
            )
        };
        Self::check(rc, "addCols")?;
        self.costs.extend(&costs);
        self.upper.extend(&upper);
        self.solved = preserve;
        self.values.resize(self.values.len() + preserved, 0.);
        self.reduced_costs.extend(&reduced[..preserved]);
        Ok(first..self.costs.len())
    }

    fn set_coefficient(&mut self, row: usize, column: usize, value: f64) -> Result<()> {
        if row >= self.rhs.len() || column >= self.costs.len() || !value.is_finite() {
            return Err(invalid("master coefficient index/value"));
        }
        // SAFETY: indices checked above.
        let rc = unsafe {
            Highs_changeCoeff(
                self.highs.as_ptr(),
                row as HighsInt,
                column as HighsInt,
                value,
            )
        };
        Self::check(rc, "changeCoeff")?;
        self.solved = false;
        Ok(())
    }

    fn set_cost(&mut self, column: usize, cost: f64) -> Result<()> {
        if column >= self.costs.len() || !cost.is_finite() {
            return Err(invalid("master cost index/value"));
        }
        // SAFETY: index checked above.
        let rc = unsafe { Highs_changeColCost(self.highs.as_ptr(), column as HighsInt, cost) };
        Self::check(rc, "changeColCost")?;
        self.costs[column] = cost;
        self.solved = false;
        Ok(())
    }

    fn set_upper(&mut self, column: usize, upper: f64) -> Result<()> {
        if column >= self.upper.len() || !upper.is_finite() || upper < 0. {
            return Err(invalid("master upper bound"));
        }
        // SAFETY: index checked above.
        let rc =
            unsafe { Highs_changeColBounds(self.highs.as_ptr(), column as HighsInt, 0., upper) };
        Self::check(rc, "changeColBounds")?;
        self.upper[column] = upper;
        self.solved = false;
        Ok(())
    }

    fn set_tolerances(&mut self, primal: f64, dual: f64) -> Result<()> {
        if !primal.is_finite() || !dual.is_finite() || primal <= 0. || dual <= 0. {
            return Err(invalid("positive finite HiGHS tolerances required"));
        }
        self.set_double("primal_feasibility_tolerance", primal)?;
        self.set_double("dual_feasibility_tolerance", dual)?;
        if self.primal_tolerance != primal || self.dual_tolerance != dual {
            self.solved = false;
        }
        self.primal_tolerance = primal;
        self.dual_tolerance = dual;
        Ok(())
    }

    fn solve(&mut self) -> bool {
        if self.costs.is_empty() {
            // No variable can satisfy a nonzero required activity. Do not
            // interpret a backend "empty model" status as feasibility.
            let feasible = self
                .senses
                .iter()
                .zip(&self.rhs)
                .all(|(sense, &rhs)| match sense {
                    RowSense::Equal => rhs == 0.,
                    RowSense::LessEqual => rhs >= 0.,
                    RowSense::GreaterEqual => rhs <= 0.,
                });
            self.status = if feasible { 0 } else { 1 };
            self.solved = feasible;
            if feasible {
                self.objective = self.objective_offset;
                self.values.clear();
                self.reduced_costs.clear();
                self.duals = vec![0.; self.rhs.len()];
                self.row_status = vec![BASIS_BASIC; self.rhs.len()];
            }
            return feasible;
        }
        let h = self.highs.as_ptr();
        // SAFETY: live handle; the buffers below have the model's dimensions.
        unsafe {
            let mut result = Highs_run(h);
            if matches!(Highs_getModelStatus(h), MODEL_SOLVE_ERROR | MODEL_UNKNOWN) {
                // A retained simplex basis can fail numerically after column
                // edits even though a cold solve of the same LP succeeds.
                // Retry once, discarding only solver state. Options (including
                // the cumulative time limit) and the model stay unchanged.
                if Highs_clearSolver(h) != STATUS_ERROR {
                    result = Highs_run(h);
                }
            }
            if result == STATUS_ERROR {
                self.status = 4;
                self.solved = false;
                return false;
            }
            self.status = match Highs_getModelStatus(h) {
                MODEL_OPTIMAL => 0,
                MODEL_INFEASIBLE => 1,
                MODEL_UNBOUNDED => 2,
                MODEL_TIME_LIMIT | MODEL_ITERATION_LIMIT => 3,
                MODEL_UNBOUNDED_OR_INFEASIBLE => -1,
                _ => -1,
            };
            self.solved = self.status == 0;
            if !self.solved {
                return false;
            }
            self.objective = Highs_getObjectiveValue(h) + self.objective_offset;
            let (rows, cols) = (self.rhs.len(), self.costs.len());
            self.values.resize(cols, 0.);
            self.reduced_costs.resize(cols, 0.);
            self.duals.resize(rows, 0.);
            self.row_values.resize(rows, 0.);
            Highs_getSolution(
                h,
                self.values.as_mut_ptr(),
                self.reduced_costs.as_mut_ptr(),
                self.row_values.as_mut_ptr(),
                self.duals.as_mut_ptr(),
            );
            self.col_status.resize(cols, 0);
            self.row_status.resize(rows, 0);
            Highs_getBasis(
                h,
                self.col_status.as_mut_ptr(),
                self.row_status.as_mut_ptr(),
            );
        }
        let tolerance = self.zero_tolerance;
        if tolerance > 0. {
            for x in self
                .duals
                .iter_mut()
                .chain(self.values.iter_mut())
                .chain(self.reduced_costs.iter_mut())
            {
                if !(*x > tolerance || *x < -tolerance) {
                    *x = 0.;
                }
            }
        }
        true
    }

    fn solve_with_time_limit(&mut self, remaining: Duration) -> Result<LpSolveOutcome> {
        let started = Instant::now();
        if remaining.is_zero() {
            self.status = 3;
            self.solved = false;
            return Ok(LpSolveOutcome::TimeLimit);
        }
        let mut previous = 0.;
        // SAFETY: live handle; option getter writes one double. HiGHS run time
        // is cumulative across Highs_run calls, as is its time_limit option.
        let runtime = unsafe {
            Self::check(
                Highs_getDoubleOptionValue(
                    self.highs.as_ptr(),
                    b"time_limit\0".as_ptr().cast(),
                    &mut previous,
                ),
                "get time_limit",
            )?;
            Highs_getRunTime(self.highs.as_ptr())
        };
        if !runtime.is_finite() || runtime < 0. || previous.is_nan() || previous < 0. {
            return Err(invalid("HiGHS time budget"));
        }
        if previous <= runtime {
            self.status = 3;
            self.solved = false;
            return Ok(LpSolveOutcome::TimeLimit);
        }
        self.set_double(
            "time_limit",
            previous.min(runtime + remaining.as_secs_f64()),
        )?;
        let mut guard = ScopedTimeLimit {
            highs: self.highs,
            previous,
            active: true,
        };
        let solved = self.solve();
        if let Err(error) = guard.restore() {
            self.status = 4;
            self.solved = false;
            return Err(error);
        }
        if started.elapsed() >= remaining {
            self.status = 3;
            self.solved = false;
            return Ok(LpSolveOutcome::TimeLimit);
        }
        Ok(if solved {
            LpSolveOutcome::Optimal
        } else {
            match self.status {
                1 => LpSolveOutcome::PrimalInfeasible,
                2 => LpSolveOutcome::DualInfeasible,
                3 => {
                    // SAFETY: live model; only inspect a successfully returned stop.
                    match unsafe { Highs_getModelStatus(self.highs.as_ptr()) } {
                        MODEL_TIME_LIMIT => LpSolveOutcome::TimeLimit,
                        MODEL_ITERATION_LIMIT => LpSolveOutcome::IterationLimit,
                        _ => LpSolveOutcome::Stopped,
                    }
                }
                4 => LpSolveOutcome::Failed,
                _ => LpSolveOutcome::Unknown,
            }
        })
    }

    fn duals(&self) -> &[f64] {
        &self.duals
    }
    fn values(&self) -> &[f64] {
        &self.values
    }
    fn reduced_costs(&self) -> &[f64] {
        &self.reduced_costs
    }
    fn objective(&self) -> f64 {
        self.objective
    }
    fn objective_offset(&self) -> f64 {
        self.objective_offset
    }
    fn status(&self) -> i32 {
        self.status
    }
    fn solved(&self) -> bool {
        self.solved
    }
    fn rhs(&self) -> &[f64] {
        &self.rhs
    }
    fn senses(&self) -> &[RowSense] {
        &self.senses
    }
    fn costs(&self) -> &[f64] {
        &self.costs
    }
    fn upper(&self) -> &[f64] {
        &self.upper
    }

    fn row_is_basic(&self, row: usize) -> Result<bool> {
        if row >= self.rhs.len() {
            return Err(invalid("master row index"));
        }
        if !self.solved {
            return Err(invalid("HiGHS basis requested before an optimal solve"));
        }
        Ok(self.row_status[row] == BASIS_BASIC)
    }

    fn cleanup_cut_rows(&self, initial_rows: usize) -> Result<Vec<usize>> {
        crate::master::lp_master::cleanup_cut_rows(self, initial_rows)
    }

    fn column_entries(&self, column: usize) -> Result<Vec<(usize, f64)>> {
        if column >= self.costs.len() {
            return Err(invalid("master column index"));
        }
        let h = self.highs.as_ptr();
        let col = column as HighsInt;
        let (mut num_col, mut num_nz) = (0, 0);
        let (mut cost, mut lower, mut upper) = (0., 0., 0.);
        let mut start = 0;
        // SAFETY: first call sizes the column; the second fills buffers of that size.
        unsafe {
            Self::check(
                Highs_getColsByRange(
                    h,
                    col,
                    col,
                    &mut num_col,
                    &mut cost,
                    &mut lower,
                    &mut upper,
                    &mut num_nz,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                ),
                "getColsByRange",
            )?;
            let mut index = vec![0; num_nz as usize];
            let mut value = vec![0.; num_nz as usize];
            Self::check(
                Highs_getColsByRange(
                    h,
                    col,
                    col,
                    &mut num_col,
                    &mut cost,
                    &mut lower,
                    &mut upper,
                    &mut num_nz,
                    &mut start,
                    index.as_mut_ptr(),
                    value.as_mut_ptr(),
                ),
                "getColsByRange",
            )?;
            let mut entries: Vec<(usize, f64)> = index
                .into_iter()
                .zip(value)
                .filter(|&(_, v)| v != 0.)
                .map(|(r, v)| (r as usize, v))
                .collect();
            entries.sort_by_key(|&(r, _)| r);
            Ok(entries)
        }
    }

    fn remove(
        &mut self,
        rows: &[usize],
        columns: &[usize],
    ) -> Result<(Vec<Option<usize>>, Vec<Option<usize>>)> {
        let (rm, ri) = mapping(self.rhs.len(), rows)?;
        let (cm, ci) = mapping(self.costs.len(), columns)?;
        // SAFETY: sets hold distinct, in-range, ascending indices.
        unsafe {
            if !ri.is_empty() {
                Self::check(
                    Highs_deleteRowsBySet(self.highs.as_ptr(), ri.len() as HighsInt, ri.as_ptr()),
                    "deleteRowsBySet",
                )?;
            }
            if !ci.is_empty() {
                Self::check(
                    Highs_deleteColsBySet(self.highs.as_ptr(), ci.len() as HighsInt, ci.as_ptr()),
                    "deleteColsBySet",
                )?;
            }
        }
        let keep = |values: &mut Vec<f64>, map: &[Option<usize>]| {
            *values = values
                .iter()
                .zip(map)
                .filter_map(|(&x, m)| m.map(|_| x))
                .collect();
        };
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
        keep(&mut self.costs, &cm);
        keep(&mut self.upper, &cm);
        self.solved = false;
        Ok((rm, cm))
    }

    fn remove_cuts(
        &mut self,
        rows: &[usize],
        columns: &[usize],
    ) -> Result<(Vec<Option<usize>>, Vec<Option<usize>>)> {
        let solved = self.solved;
        let (duals, values, reduced_costs, row_status) = (
            self.duals.clone(),
            self.values.clone(),
            self.reduced_costs.clone(),
            self.row_status.clone(),
        );
        let (rm, cm) = self.remove(rows, columns)?;
        if solved {
            let pick = |x: Vec<f64>, map: &[Option<usize>]| -> Vec<f64> {
                x.into_iter()
                    .zip(map)
                    .filter_map(|(x, m)| m.map(|_| x))
                    .collect()
            };
            self.duals = pick(duals, &rm);
            self.values = pick(values, &cm);
            self.reduced_costs = pick(reduced_costs, &cm);
            self.row_status = row_status
                .into_iter()
                .zip(&rm)
                .filter_map(|(x, m)| m.map(|_| x))
                .collect();
            self.solved = true;
        }
        Ok((rm, cm))
    }
}
