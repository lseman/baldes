use super::*;

impl ClpMaster {
    pub fn set_tolerances(&mut self, primal: f64, dual: f64) -> Result<()> {
        if !primal.is_finite() || !dual.is_finite() || primal <= 0. || dual <= 0. {
            return Err(invalid("positive finite Clp tolerances required"));
        }
        unsafe {
            Clp_setPrimalTolerance(self.model.as_ptr(), primal);
            Clp_setDualTolerance(self.model.as_ptr(), dual);
        }
        self.dual_tolerance = dual;
        self.primal_tolerance = primal;
        #[cfg(feature = "highs")]
        self.edit_highs(|backend| backend.set_tolerances(primal, dual))?;
        Ok(())
    }

    pub fn solve(&mut self) -> bool {
        if self.solver == MasterSolver::Highs {
            return matches!(
                self.solve_highs(i32::MAX as usize, Duration::MAX),
                Ok(LpSolveOutcome::Optimal)
            );
        }
        if self.solver == MasterSolver::Ipm {
            return matches!(
                self.solve_ipm(500, Duration::MAX, None),
                Ok(ClpSolveOutcome::Optimal)
            );
        }
        // SAFETY: The model is exclusively borrowed and owned by this handle.
        unsafe {
            Clp_dual(self.model.as_ptr(), 0);
        }
        self.read_solution()
    }

    /// Cooperatively limit one LP solve by elapsed wall time and simplex pivots.
    /// Existing stricter backend limits remain effective and are restored
    /// exactly. Only `Optimal` marks the cached solution as solved; other
    /// outcomes do not validate the cached primal/dual arrays for this call.
    pub fn solve_limited(
        &mut self,
        iterations: usize,
        time_limit: Duration,
    ) -> Result<ClpSolveOutcome> {
        if self.solver == MasterSolver::Highs {
            return Ok(match self.solve_highs(iterations, time_limit)? {
                LpSolveOutcome::Optimal => ClpSolveOutcome::Optimal,
                LpSolveOutcome::PrimalInfeasible => ClpSolveOutcome::PrimalInfeasible,
                LpSolveOutcome::DualInfeasible => ClpSolveOutcome::DualInfeasible,
                LpSolveOutcome::TimeLimit
                | LpSolveOutcome::IterationLimit
                | LpSolveOutcome::Stopped => ClpSolveOutcome::Stopped,
                _ => ClpSolveOutcome::Failed,
            });
        }
        if self.solver == MasterSolver::Ipm {
            return self.solve_ipm(iterations.min(500), time_limit, None);
        }
        let iterations = i32::try_from(iterations).map_err(|_| invalid("Clp iteration limit"))?;
        let started = Instant::now();
        if iterations == 0 || time_limit.is_zero() {
            self.status = 3;
            self.solved = false;
            return Ok(ClpSolveOutcome::Stopped);
        }
        // SAFETY: The native scope guard restores this owned model's limits
        // before returning, including on caught C++ exceptions.
        let failed = unsafe {
            baldes_clp_dual_limited(self.model.as_ptr(), iterations, time_limit.as_secs_f64())
        } != 0;
        if failed {
            self.status = 4;
            self.solved = false;
            return Ok(ClpSolveOutcome::Failed);
        }
        self.read_solution();
        if started.elapsed() >= time_limit {
            self.status = 3;
            self.solved = false;
        }
        Ok(match self.status {
            0 => ClpSolveOutcome::Optimal,
            1 => ClpSolveOutcome::PrimalInfeasible,
            2 => ClpSolveOutcome::DualInfeasible,
            3 => ClpSolveOutcome::Stopped,
            _ => ClpSolveOutcome::Failed,
        })
    }

    pub(super) fn solve_highs(
        &mut self,
        iterations: usize,
        remaining: Duration,
    ) -> Result<LpSolveOutcome> {
        self.solved = false;
        self.status = 4;
        self.highs_row_basic.clear();
        self.solver.ensure_available()?;
        if remaining.is_zero() || iterations == 0 {
            self.status = 3;
            return Ok(if remaining.is_zero() {
                LpSolveOutcome::TimeLimit
            } else {
                LpSolveOutcome::IterationLimit
            });
        }
        #[cfg(feature = "highs")]
        {
            let started = Instant::now();
            if self.highs_backend.is_none() {
                let mut backend = crate::master::solvers::highs::HighsMaster::new()?;
                backend.set_tolerances(self.primal_tolerance, self.dual_tolerance)?;
                let rows = self
                    .senses
                    .iter()
                    .zip(&self.rhs)
                    .map(|(&sense, &rhs)| (sense, rhs, &[][..]))
                    .collect::<Vec<_>>();
                backend.add_rows(&rows)?;
                let columns = (0..self.costs.len())
                    .map(|j| self.column_entries(j))
                    .collect::<Result<Vec<_>>>()?;
                let batch = self
                    .costs
                    .iter()
                    .zip(&columns)
                    .map(|(&cost, entries)| (cost, entries.as_slice()))
                    .collect::<Vec<_>>();
                backend.add_columns(&batch)?;
                for (column, &upper) in self.upper.iter().enumerate() {
                    if backend.upper[column] != upper {
                        backend.set_upper(column, upper)?;
                    }
                }
                self.highs_backend = Some(backend);
            }
            let backend = self.highs_backend.as_mut().unwrap();
            backend.set_lp_iteration_limit(iterations)?;
            let reuse_solution = backend.solved
                && backend.reduced_costs.len() == self.costs.len()
                && backend.values.len() == self.costs.len()
                && backend.duals.len() == self.rhs.len()
                && backend.zero_tolerance == self.zero_tolerance
                && backend.objective_offset == self.objective_offset;
            backend.zero_tolerance = self.zero_tolerance;
            backend.objective_offset = self.objective_offset;
            let budget = remaining.saturating_sub(started.elapsed());
            let outcome = if budget.is_zero() {
                LpSolveOutcome::TimeLimit
            } else if reuse_solution {
                LpSolveOutcome::Optimal
            } else {
                backend.solve_with_time_limit(budget)?
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
                if !reuse_solution {
                    self.simplex_iterations = self
                        .simplex_iterations
                        .saturating_add(backend.simplex_iterations()?);
                }
                self.highs_row_basic = (0..self.rhs.len())
                    .map(|r| backend.row_is_basic(r))
                    .collect::<Result<Vec<_>>>()?;
                self.objective = backend.objective;
                self.values.clone_from(&backend.values);
                self.duals.clone_from(&backend.duals);
                self.reduced_costs.clone_from(&backend.reduced_costs);
                self.solved = true;
            }
            Ok(outcome)
        }
        #[cfg(not(feature = "highs"))]
        unreachable!("availability checked above")
    }

    pub(super) fn solve_ipm(
        &mut self,
        iterations: usize,
        time_limit: Duration,
        gap_tolerance: Option<f64>,
    ) -> Result<ClpSolveOutcome> {
        self.solved = false;
        self.status = 4;
        if iterations == 0 || time_limit.is_zero() {
            self.status = 3;
            return Ok(ClpSolveOutcome::Stopped);
        }
        let started = Instant::now();
        let mut entries = Vec::new();
        for column in 0..self.costs.len() {
            for (row, value) in self.column_entries(column)? {
                entries.push((row, column, value));
            }
        }
        let model = crate::ipm::ModelData {
            a: crate::ipm::SparseMatrix::from_triplets(self.rhs.len(), self.costs.len(), &entries)
                .map_err(|e| incomplete(e.to_string()))?,
            b: self.rhs.clone(),
            c: self.costs.clone(),
            lower: vec![0.; self.costs.len()],
            upper: self.upper.clone(),
            sense: self
                .senses
                .iter()
                .map(|s| match s {
                    RowSense::Equal => crate::ipm::Sense::Equal,
                    RowSense::LessEqual => crate::ipm::Sense::LessEqual,
                    RowSense::GreaterEqual => crate::ipm::Sense::GreaterEqual,
                })
                .collect(),
        };
        let solution = crate::ipm::solve(
            &model,
            &crate::ipm::Options {
                tolerance: 1e-9,
                gap_tolerance,
                max_iterations: iterations,
                time_limit: Some(time_limit.saturating_sub(started.elapsed())),
                ..Default::default()
            },
        )
        .map_err(|e| incomplete(format!("IPM master: {e}")))?;
        self.status = match solution.status {
            crate::ipm::Status::Optimal => 0,
            crate::ipm::Status::Infeasible => 1,
            crate::ipm::Status::Unbounded => 2,
            _ => 3,
        };
        if started.elapsed() >= time_limit {
            self.status = 3;
        }
        if self.status == 0 {
            self.objective = solution.objective + self.objective_offset;
            self.values = solution.primal;
            self.duals = solution.dual;
            self.reduced_costs = (0..self.costs.len())
                .map(|j| {
                    self.column_entries(j).map(|entries| {
                        self.costs[j] - entries.iter().map(|&(r, a)| a * self.duals[r]).sum::<f64>()
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            self.solved = true;
        }
        Ok(match self.status {
            0 => ClpSolveOutcome::Optimal,
            1 => ClpSolveOutcome::PrimalInfeasible,
            2 => ClpSolveOutcome::DualInfeasible,
            _ => ClpSolveOutcome::Stopped,
        })
    }

    pub(super) fn read_solution(&mut self) -> bool {
        // SAFETY: All returned arrays are owned by this model. Copy before further mutation.
        unsafe {
            self.simplex_iterations += Clp_numberIterations(self.model.as_ptr()).max(0) as usize;
            self.status = Clp_status(self.model.as_ptr());
            self.solved = self.status == 0;
            if self.solved {
                self.objective = Clp_objectiveValue(self.model.as_ptr()) + self.objective_offset;
                self.duals = if self.rhs.is_empty() {
                    vec![]
                } else {
                    std::slice::from_raw_parts(
                        Clp_dualRowSolution(self.model.as_ptr()),
                        self.rhs.len(),
                    )
                    .to_vec()
                };
                self.values = if self.costs.is_empty() {
                    vec![]
                } else {
                    std::slice::from_raw_parts(
                        Clp_primalColumnSolution(self.model.as_ptr()),
                        self.costs.len(),
                    )
                    .to_vec()
                };
                self.reduced_costs = if self.costs.is_empty() {
                    vec![]
                } else {
                    std::slice::from_raw_parts(
                        Clp_getReducedCost(self.model.as_ptr()),
                        self.costs.len(),
                    )
                    .to_vec()
                };
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
            }
        }
        self.solved
    }
}
