use super::*;

impl ClpMaster {
    pub fn solver(&self) -> MasterSolver {
        self.solver
    }

    /// Set the HiGHS/Gurobi worker budget before materializing either backend.
    /// Zero uses all CPUs available to this process. Forks retain this setting.
    /// A live backend's budget cannot change, including HiGHS's shared scheduler.
    pub fn set_backend_threads(&mut self, threads: usize) -> Result<()> {
        if threads > i32::MAX as usize {
            return Err(invalid("master thread count exceeds native integer range"));
        }
        if threads == self.backend_threads {
            return Ok(());
        }
        #[cfg(feature = "highs")]
        if self.highs_backend.is_some() {
            return Err(invalid(
                "set master threads before creating the HiGHS backend",
            ));
        }
        #[cfg(feature = "gurobi")]
        if self.gurobi_backend.is_some() {
            return Err(invalid(
                "set master threads before creating the Gurobi backend",
            ));
        }
        self.backend_threads = threads;
        Ok(())
    }

    pub fn backend_threads(&self) -> usize {
        self.backend_threads
    }

    /// Changing algorithms invalidates the cached solution. Switching between
    /// Clp/IPM, HiGHS, and Gurobi converts storage on the next edit or solve and starts
    /// a fresh native basis; edits within one backend retain its warm basis.
    pub fn set_solver(&mut self, solver: MasterSolver) {
        if self.solver != solver {
            self.solved = false;
            self.native_row_basic.clear();
        }
        self.solver = solver;
    }

    /// Materialize the requested backend before a mutation or solve. Conversion
    /// copies the matrix once, then releases the previous native model. Merely
    /// selecting a solver remains infallible; allocation errors surface here.
    pub(super) fn ensure_backend(&mut self) -> Result<()> {
        self.solver.ensure_available()?;
        #[cfg(feature = "highs")]
        if self.solver == MasterSolver::Highs {
            if self.highs_backend.is_none() {
                let mut backend =
                    super::super::highs::HighsMaster::with_worker_budget(self.backend_threads)?;
                backend.set_tolerances(self.primal_tolerance, self.dual_tolerance)?;
                let rows = self
                    .senses
                    .iter()
                    .zip(&self.rhs)
                    .map(|(&s, &b)| (s, b, &[][..]))
                    .collect::<Vec<_>>();
                backend.add_rows(&rows)?;
                let columns = (0..self.costs.len())
                    .map(|j| self.column_entries(j))
                    .collect::<Result<Vec<_>>>()?;
                let batch = self
                    .costs
                    .iter()
                    .zip(&columns)
                    .map(|(&c, e)| (c, e.as_slice()))
                    .collect::<Vec<_>>();
                backend.add_columns(&batch)?;
                for (j, &upper) in self.upper.iter().enumerate() {
                    if backend.upper[j] != upper {
                        backend.set_upper(j, upper)?;
                    }
                }
                self.highs_backend = Some(backend);
                #[cfg(feature = "gurobi")]
                {
                    self.gurobi_backend = None;
                }
                if let Some(model) = self.model.take() {
                    // SAFETY: ownership has moved to the new backend; no Clp
                    // pointers escape the wrapper.
                    unsafe { Clp_deleteModel(model.as_ptr()) };
                }
                self.ipm = Default::default();
                self.solved = false;
            }
            return Ok(());
        }
        #[cfg(feature = "gurobi")]
        if self.solver == MasterSolver::Gurobi {
            if self.gurobi_backend.is_none() {
                let mut backend =
                    super::super::gurobi::GurobiMaster::with_threads(self.backend_threads)?;
                backend.set_tolerances(self.primal_tolerance, self.dual_tolerance)?;
                let rows = self
                    .senses
                    .iter()
                    .zip(&self.rhs)
                    .map(|(&s, &b)| (s, b, &[][..]))
                    .collect::<Vec<_>>();
                backend.add_rows(&rows)?;
                let columns = (0..self.costs.len())
                    .map(|j| self.column_entries(j))
                    .collect::<Result<Vec<_>>>()?;
                let batch = self
                    .costs
                    .iter()
                    .zip(&columns)
                    .map(|(&c, e)| (c, e.as_slice()))
                    .collect::<Vec<_>>();
                backend.add_columns(&batch)?;
                for (j, &upper) in self.upper.iter().enumerate() {
                    if backend.upper[j] != upper {
                        backend.set_upper(j, upper)?;
                    }
                }
                self.gurobi_backend = Some(backend);
                #[cfg(feature = "highs")]
                {
                    self.highs_backend = None;
                }
                if let Some(model) = self.model.take() {
                    // SAFETY: ownership has moved to the new backend; no Clp
                    // pointers escape the wrapper.
                    unsafe { Clp_deleteModel(model.as_ptr()) };
                }
                self.ipm = Default::default();
                self.solved = false;
            }
            return Ok(());
        }
        if self.model.is_none() {
            let mut copy = Self::new()?;
            copy.set_clp_options(self.clp_options);
            copy.set_tolerances(self.primal_tolerance, self.dual_tolerance)?;
            let rows = self
                .senses
                .iter()
                .zip(&self.rhs)
                .map(|(&s, &b)| (s, b, &[][..]))
                .collect::<Vec<_>>();
            copy.add_rows(&rows)?;
            let columns = (0..self.costs.len())
                .map(|j| self.column_entries(j))
                .collect::<Result<Vec<_>>>()?;
            let batch = self
                .costs
                .iter()
                .zip(&columns)
                .map(|(&c, e)| (c, e.as_slice()))
                .collect::<Vec<_>>();
            copy.add_columns(&batch)?;
            for (j, &upper) in self.upper.iter().enumerate() {
                if copy.upper[j] != upper {
                    copy.set_upper(j, upper)?;
                }
            }
            self.model = copy.model.take();
            #[cfg(feature = "highs")]
            {
                self.highs_backend = None;
            }
            #[cfg(feature = "gurobi")]
            {
                self.gurobi_backend = None;
            }
            self.ipm = Default::default();
            self.solved = false;
        }
        Ok(())
    }

    /// Apply an edit to the sole HiGHS matrix. Keep it on failure: there is
    /// no shadow matrix from which to reconstruct this backend.
    #[cfg(feature = "highs")]
    pub(super) fn edit_highs(
        &mut self,
        edit: impl FnOnce(&mut super::super::highs::HighsMaster) -> Result<()>,
    ) -> Result<()> {
        if let Some(backend) = self.highs_backend.as_mut() {
            if let Err(error) = edit(backend) {
                self.solved = false;
                return Err(error);
            }
        }
        Ok(())
    }

    #[cfg(feature = "gurobi")]
    pub(super) fn edit_gurobi(
        &mut self,
        edit: impl FnOnce(&mut super::super::gurobi::GurobiMaster) -> Result<()>,
    ) -> Result<()> {
        if let Some(backend) = self.gurobi_backend.as_mut() {
            if let Err(error) = edit(backend) {
                self.solved = false;
                return Err(error);
            }
        }
        Ok(())
    }

    /// Reuse a preceding solve when deleting its trailing rows/columns gives
    /// this exact ordered matrix and costs. Bounds remain those of this model.
    /// Returns false without changing either model when they do not match.
    pub(crate) fn continue_from(&mut self, previous: &Self) -> Result<bool> {
        let rows = self.rhs.len();
        let columns = self.costs.len();
        if previous.rhs.len() < rows
            || previous.costs.len() < columns
            || previous.costs[..columns] != self.costs
            || previous.senses[..rows] != self.senses
            || previous.objective_offset != self.objective_offset
        {
            return Ok(false);
        }
        for column in 0..columns {
            let retained: Vec<_> = previous
                .column_entries(column)?
                .into_iter()
                .filter(|&(row, _)| row < rows)
                .collect();
            if retained != self.column_entries(column)? {
                return Ok(false);
            }
        }
        let mut continued = previous.try_clone()?;
        // Native node restoration deletes branch artificials before rows.
        let removed_columns: Vec<_> = (columns..previous.costs.len()).collect();
        let removed_rows: Vec<_> = (rows..previous.rhs.len()).collect();
        continued.remove(&[], &removed_columns)?;
        continued.remove(&removed_rows, &[])?;
        continued.duals.truncate(rows);
        continued.values.truncate(columns);
        continued.reduced_costs.truncate(columns);
        continued.set_rhs(&self.rhs)?;
        for column in 0..columns {
            if continued.upper[column] != self.upper[column] {
                continued.set_upper(column, self.upper[column])?;
            }
        }
        continued.set_clp_options(self.clp_options);
        continued.set_solver(self.solver);
        continued.zero_tolerance = self.zero_tolerance;
        continued.set_tolerances(1e-7, self.dual_tolerance)?;
        *self = continued;
        Ok(true)
    }

    /// Fork only the active native backend, retaining its basis and solution.
    /// The fork owns all data independently.
    pub fn try_clone(&self) -> Result<Self> {
        #[cfg(feature = "highs")]
        let highs_backend = self
            .highs_backend
            .as_ref()
            .map(|backend| backend.try_clone())
            .transpose()?;
        #[cfg(feature = "gurobi")]
        let gurobi_backend = self
            .gurobi_backend
            .as_ref()
            .map(|backend| backend.try_clone())
            .transpose()?;
        // SAFETY: source stays alive and is not mutated; the C++ bridge returns
        // an independently owned Clp wrapper, released by this type's Drop.
        let model = self
            .model
            .map(|model| {
                let copy = NonNull::new(unsafe { baldes_clp_clone(model.as_ptr()) })
                    .ok_or_else(|| incomplete("Clp model copy failed"))?;
                unsafe { Clp_setLogLevel(copy.as_ptr(), 0) };
                Ok(copy)
            })
            .transpose()?;
        Ok(Self {
            model,
            solver: self.solver,
            backend_threads: self.backend_threads,
            clp_options: self.clp_options,
            ipm: self.ipm.clone(),
            #[cfg(feature = "highs")]
            highs_backend,
            #[cfg(feature = "gurobi")]
            gurobi_backend,
            primal_tolerance: self.primal_tolerance,
            native_row_basic: self.native_row_basic.clone(),
            rhs: self.rhs.clone(),
            senses: self.senses.clone(),
            costs: self.costs.clone(),
            upper: self.upper.clone(),
            column_default_upper: self.column_default_upper,
            duals: self.duals.clone(),
            values: self.values.clone(),
            reduced_costs: self.reduced_costs.clone(),
            objective: self.objective,
            objective_offset: self.objective_offset,
            status: self.status,
            solved: self.solved,
            dual_tolerance: self.dual_tolerance,
            zero_tolerance: self.zero_tolerance,
            simplex_iterations: 0,
            ipm_iterations: 0,
        })
    }

    /// Returns a copy of the dual (shadow) prices for each row.
    pub fn duals(&self) -> Vec<f64> {
        self.duals.clone()
    }

    /// Returns a copy of the primal column values.
    pub fn values(&self) -> Vec<f64> {
        self.values.clone()
    }

    /// Set the backend bound for newly created columns only. Original-domain
    /// proof state remains the owning controller's responsibility.
    pub(crate) fn set_column_default_upper(&mut self, upper: f64) -> Result<()> {
        if !upper.is_finite() || upper < 0. {
            return Err(invalid("default column upper bound"));
        }
        self.column_default_upper = upper;
        Ok(())
    }

    pub fn new() -> Result<Self> {
        // SAFETY: Clp returns an exclusively owned model, freed by Drop below.
        let model = NonNull::new(unsafe { Clp_newModel() })
            .ok_or_else(|| incomplete("Clp model allocation failed"))?;
        unsafe {
            Clp_setLogLevel(model.as_ptr(), 0);
            Clp_setOptimizationDirection(model.as_ptr(), 1.);
            Clp_scaling(model.as_ptr(), ClpOptions::default().scaling.code());
        }
        Ok(Self {
            model: Some(model),
            solver: MasterSolver::Clp,
            backend_threads: 0,
            clp_options: ClpOptions::default(),
            ipm: Default::default(),
            native_row_basic: vec![],
            #[cfg(feature = "highs")]
            highs_backend: None,
            #[cfg(feature = "gurobi")]
            gurobi_backend: None,
            primal_tolerance: 1e-7,
            rhs: vec![],
            senses: vec![],
            costs: vec![],
            upper: vec![],
            column_default_upper: 1e12,
            duals: vec![],
            values: vec![],
            reduced_costs: vec![],
            objective: 0.,
            objective_offset: 0.,
            status: -1,
            solved: false,
            dual_tolerance: 1e-7,
            simplex_iterations: 0,
            ipm_iterations: 0,
            zero_tolerance: 0.,
        })
    }
}
