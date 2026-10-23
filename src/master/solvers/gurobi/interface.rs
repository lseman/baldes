use super::*;

impl LpMaster for GurobiMaster {
    fn try_clone_for_transaction(&self) -> Result<Option<Self>> {
        self.try_clone().map(Some)
    }
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
        let mut starts = vec![];
        let mut indices = vec![];
        let mut values = vec![];
        let mut senses = vec![];
        let mut rhs_values = vec![];
        for &(sense, rhs, entries) in rows {
            if !rhs.is_finite() {
                return Err(invalid("finite row RHS required"));
            }
            let (i, v) = entries_of(entries, self.costs.len())?;
            starts.push(count(indices.len())?);
            indices.extend(i);
            values.extend(v);
            senses.push(sense_code(sense));
            rhs_values.push(rhs);
        }
        let n = count(rows.len())?;
        let nz = count(indices.len())?;
        self.update()?;
        // SAFETY: arrays contain the declared row and nonzero counts; Gurobi copies them.
        unsafe {
            self.check(
                GRBaddconstrs(
                    self.model.as_ptr(),
                    n,
                    nz,
                    starts.as_ptr(),
                    indices.as_ptr(),
                    values.as_ptr(),
                    senses.as_ptr(),
                    rhs_values.as_ptr(),
                    std::ptr::null(),
                ),
                "add rows",
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
            starts.push(count(indices.len())?);
            indices.extend(ids);
            values.extend(vals);
            costs.push(cost);
            reduced.push(rc);
        }
        let lower = vec![0.; columns.len()];
        let upper = vec![COLUMN_UPPER; columns.len()];
        let n = count(columns.len())?;
        let nz = count(indices.len())?;
        self.update()?;
        // SAFETY: arrays contain the declared columns and nonzero counts.
        unsafe {
            self.check(
                GRBaddvars(
                    self.model.as_ptr(),
                    n,
                    nz,
                    starts.as_ptr(),
                    indices.as_ptr(),
                    values.as_ptr(),
                    costs.as_ptr(),
                    lower.as_ptr(),
                    upper.as_ptr(),
                    std::ptr::null(),
                    std::ptr::null(),
                ),
                "add columns",
            )?;
        }
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
        self.update()?;
        let row = count(row)?;
        let column = count(column)?;
        // SAFETY: indices were checked and each pointer addresses one value.
        unsafe {
            self.check(
                GRBchgcoeffs(self.model.as_ptr(), 1, &row, &column, &value),
                "change coefficient",
            )?;
        }
        self.solved = false;
        Ok(())
    }

    fn set_cost(&mut self, column: usize, cost: f64) -> Result<()> {
        if column >= self.costs.len() || !cost.is_finite() {
            return Err(invalid("master cost index/value"));
        }
        self.set_element(b"Obj\0", column, cost)?;
        self.costs[column] = cost;
        self.solved = false;
        Ok(())
    }

    fn set_upper(&mut self, column: usize, upper: f64) -> Result<()> {
        if column >= self.upper.len() || !upper.is_finite() || upper < 0. {
            return Err(invalid("master upper bound"));
        }
        self.set_element(b"UB\0", column, upper)?;
        self.upper[column] = upper;
        self.solved = false;
        Ok(())
    }

    fn set_tolerances(&mut self, primal: f64, dual: f64) -> Result<()> {
        if !primal.is_finite() || !dual.is_finite() || primal <= 0. || dual <= 0. {
            return Err(invalid("positive finite Gurobi tolerances required"));
        }
        self.set_double(b"FeasibilityTol\0", primal)?;
        self.set_double(b"OptimalityTol\0", dual)?;
        if self.primal_tolerance != primal || self.dual_tolerance != dual {
            self.solved = false;
        }
        self.primal_tolerance = primal;
        self.dual_tolerance = dual;
        Ok(())
    }

    fn solve(&mut self) -> bool {
        matches!(
            self.solve_with_time_limit(Duration::MAX),
            Ok(LpSolveOutcome::Optimal)
        )
    }

    fn solve_with_time_limit(&mut self, remaining: Duration) -> Result<LpSolveOutcome> {
        self.solved = false;
        self.status = 4;
        if remaining.is_zero() {
            self.status = 3;
            return Ok(LpSolveOutcome::TimeLimit);
        }
        let previous = self.get_parameter(b"TimeLimit\0")?;
        self.set_double(b"TimeLimit\0", previous.min(remaining.as_secs_f64()))?;
        // SAFETY: exclusively owned live model; restore the per-model parameter even on error.
        let result = unsafe { GRBoptimize(self.model.as_ptr()) };
        let restored = self.set_double(b"TimeLimit\0", previous);
        self.check(result, "optimize")?;
        restored?;
        let native_status = self.int_attr(b"Status\0")?;
        let outcome = match native_status {
            2 => LpSolveOutcome::Optimal,
            3 => LpSolveOutcome::PrimalInfeasible,
            5 => LpSolveOutcome::DualInfeasible,
            7 => LpSolveOutcome::IterationLimit,
            9 => LpSolveOutcome::TimeLimit,
            4 => LpSolveOutcome::Unknown,
            12 => LpSolveOutcome::Failed,
            _ => LpSolveOutcome::Stopped,
        };
        self.status = match outcome {
            LpSolveOutcome::Optimal => 0,
            LpSolveOutcome::PrimalInfeasible => 1,
            LpSolveOutcome::DualInfeasible => 2,
            LpSolveOutcome::TimeLimit
            | LpSolveOutcome::IterationLimit
            | LpSolveOutcome::Stopped => 3,
            _ => 4,
        };
        if outcome == LpSolveOutcome::Optimal {
            let values = self.double_array(b"X\0", self.costs.len())?;
            let reduced = self.double_array(b"RC\0", self.costs.len())?;
            let duals = self.double_array(b"Pi\0", self.rhs.len())?;
            let mut basis = vec![0; self.rhs.len()];
            if !basis.is_empty() {
                // SAFETY: one output slot per row.
                unsafe {
                    self.check(
                        GRBgetintattrarray(
                            self.model.as_ptr(),
                            b"CBasis\0".as_ptr().cast(),
                            0,
                            count(basis.len())?,
                            basis.as_mut_ptr(),
                        ),
                        "row basis",
                    )?;
                }
            }
            self.objective = self.double_attr(b"ObjVal\0")? + self.objective_offset;
            self.values = values;
            self.reduced_costs = reduced;
            self.duals = duals;
            self.row_status = basis;
            for x in self
                .values
                .iter_mut()
                .chain(&mut self.reduced_costs)
                .chain(&mut self.duals)
            {
                if x.abs() <= self.zero_tolerance {
                    *x = 0.;
                }
            }
            self.solved = true;
        }
        Ok(outcome)
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
            return Err(invalid("Gurobi basis requested before an optimal solve"));
        }
        Ok(self.row_status[row] == 0)
    }

    fn cleanup_cut_rows(&self, initial_rows: usize) -> Result<Vec<usize>> {
        crate::master::lp_master::cleanup_cut_rows(self, initial_rows)
    }

    fn column_entries(&self, column: usize) -> Result<Vec<(usize, f64)>> {
        if column >= self.costs.len() {
            return Err(invalid("master column index"));
        }
        self.update()?;
        let mut nz = 0;
        let mut start = 0;
        // SAFETY: query the required allocation first, then provide buffers of that size.
        unsafe {
            self.check(
                GRBgetvars(
                    self.model.as_ptr(),
                    &mut nz,
                    &mut start,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    count(column)?,
                    1,
                ),
                "column size",
            )?;
            let size = usize::try_from(nz).map_err(|_| invalid("Gurobi nonzero count"))?;
            let mut ids = vec![0; size];
            let mut vals = vec![0.; size];
            self.check(
                GRBgetvars(
                    self.model.as_ptr(),
                    &mut nz,
                    &mut start,
                    ids.as_mut_ptr(),
                    vals.as_mut_ptr(),
                    count(column)?,
                    1,
                ),
                "column entries",
            )?;
            let mut entries: Vec<_> = ids
                .into_iter()
                .zip(vals)
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
        self.update()?;
        // SAFETY: distinct checked indices. Flush deletions before indices can be reused.
        unsafe {
            if !ri.is_empty() {
                self.check(
                    GRBdelconstrs(self.model.as_ptr(), count(ri.len())?, ri.as_ptr()),
                    "delete rows",
                )?;
            }
            if !ci.is_empty() {
                self.check(
                    GRBdelvars(self.model.as_ptr(), count(ci.len())?, ci.as_ptr()),
                    "delete columns",
                )?;
            }
        }
        self.update()?;
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
