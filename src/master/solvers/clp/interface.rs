use super::*;

impl LpMaster for ClpMaster {
    fn try_clone_for_transaction(&self) -> Result<Option<Self>> {
        let mut copy = self.try_clone()?;
        // This continues one solve rather than starting a descendant's counters.
        copy.simplex_iterations = self.simplex_iterations;
        copy.ipm_iterations = self.ipm_iterations;
        Ok(Some(copy))
    }
    fn add_row(&mut self, sense: RowSense, rhs: f64, entries: &[(usize, f64)]) -> Result<usize> {
        Self::add_row(self, sense, rhs, entries)
    }

    fn add_rows(
        &mut self,
        rows: &[(RowSense, f64, &[(usize, f64)])],
    ) -> Result<std::ops::Range<usize>> {
        Self::add_rows(self, rows)
    }

    fn add_column(&mut self, cost: f64, entries: &[(usize, f64)]) -> Result<usize> {
        Self::add_column(self, cost, entries)
    }

    fn add_columns(
        &mut self,
        columns: &[(f64, &[(usize, f64)])],
    ) -> Result<std::ops::Range<usize>> {
        Self::add_columns(self, columns)
    }

    fn set_coefficient(&mut self, row: usize, column: usize, value: f64) -> Result<()> {
        Self::set_coefficient(self, row, column, value)
    }

    fn set_cost(&mut self, column: usize, cost: f64) -> Result<()> {
        Self::set_cost(self, column, cost)
    }

    fn set_upper(&mut self, column: usize, upper: f64) -> Result<()> {
        Self::set_upper(self, column, upper)
    }

    fn set_tolerances(&mut self, primal: f64, dual: f64) -> Result<()> {
        Self::set_tolerances(self, primal, dual)
    }

    fn solve(&mut self) -> bool {
        Self::solve(self)
    }

    fn solve_for_column_generation(
        &mut self,
        remaining: Duration,
        gap_tolerance: Option<f64>,
    ) -> Result<LpSolveOutcome> {
        if self.solver != MasterSolver::Ipm || gap_tolerance.is_none() {
            return self.solve_with_time_limit(remaining);
        }
        let started = Instant::now();
        Ok(match self.solve_ipm(500, remaining, gap_tolerance)? {
            ClpSolveOutcome::Optimal => LpSolveOutcome::Suboptimal,
            ClpSolveOutcome::PrimalInfeasible => LpSolveOutcome::PrimalInfeasible,
            ClpSolveOutcome::DualInfeasible => LpSolveOutcome::DualInfeasible,
            ClpSolveOutcome::Failed => LpSolveOutcome::Failed,
            ClpSolveOutcome::Stopped => {
                if remaining.is_zero() || started.elapsed() >= remaining {
                    LpSolveOutcome::TimeLimit
                } else {
                    LpSolveOutcome::IterationLimit
                }
            }
        })
    }

    fn solve_with_time_limit(&mut self, remaining: Duration) -> Result<LpSolveOutcome> {
        if self.solver == MasterSolver::Highs {
            return self.solve_highs(i32::MAX as usize, remaining);
        }
        let started = Instant::now();
        Ok(match self.solve_limited(i32::MAX as usize, remaining)? {
            ClpSolveOutcome::Optimal => LpSolveOutcome::Optimal,
            ClpSolveOutcome::PrimalInfeasible => LpSolveOutcome::PrimalInfeasible,
            ClpSolveOutcome::DualInfeasible => LpSolveOutcome::DualInfeasible,
            ClpSolveOutcome::Failed => LpSolveOutcome::Failed,
            ClpSolveOutcome::Stopped => {
                // SAFETY: live, exclusively borrowed model. Clp secondary
                // status 9 identifies a time stop rather than an iteration stop.
                if remaining.is_zero()
                    || started.elapsed() >= remaining
                    || (self.solver == MasterSolver::Clp
                        && unsafe {
                            Clp_secondaryStatus(self.model.expect("Clp backend is active").as_ptr())
                        } == 9)
                {
                    LpSolveOutcome::TimeLimit
                } else {
                    LpSolveOutcome::IterationLimit
                }
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
        Self::row_is_basic(self, row)
    }
    fn cleanup_cut_rows(&self, initial_rows: usize) -> Result<Vec<usize>> {
        Self::cleanup_cut_rows(self, initial_rows)
    }
    fn column_entries(&self, column: usize) -> Result<Vec<(usize, f64)>> {
        Self::column_entries(self, column)
    }
    fn remove(
        &mut self,
        rows: &[usize],
        columns: &[usize],
    ) -> Result<(Vec<Option<usize>>, Vec<Option<usize>>)> {
        Self::remove(self, rows, columns)
    }
    fn remove_cuts(
        &mut self,
        rows: &[usize],
        columns: &[usize],
    ) -> Result<(Vec<Option<usize>>, Vec<Option<usize>>)> {
        Self::remove_cuts(self, rows, columns)
    }
}
