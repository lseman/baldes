use super::*;

impl ClpMaster {
    pub fn solver(&self) -> MasterSolver {
        self.solver
    }

    /// Changing algorithms invalidates the cached solution.
    pub fn set_solver(&mut self, solver: MasterSolver) {
        if self.solver != solver {
            self.solved = false;
            self.highs_row_basic.clear();
            #[cfg(feature = "highs")]
            {
                self.highs_backend = None;
            }
        }
        self.solver = solver;
    }

    /// Mutations mirror into an existing HiGHS model, preserving its basis.
    /// On a mirror error discard that backend: the canonical model is retained
    /// and a subsequent solve can reconstruct it without using stale state.
    #[cfg(feature = "highs")]
    pub(super) fn edit_highs(
        &mut self,
        edit: impl FnOnce(&mut crate::master::solvers::highs::HighsMaster) -> Result<()>,
    ) -> Result<()> {
        if let Some(mut backend) = self.highs_backend.take() {
            if let Err(error) = edit(&mut backend) {
                self.solved = false;
                return Err(error);
            }
            self.highs_backend = Some(backend);
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
        continued.solver = self.solver;
        continued.zero_tolerance = self.zero_tolerance;
        continued.set_tolerances(1e-7, self.dual_tolerance)?;
        *self = continued;
        Ok(true)
    }

    /// Fork the model through Clp's deep-copy operation, retaining its basis,
    /// primal/dual solution, scaling and simplex state. The fork owns all data.
    pub fn try_clone(&self) -> Result<Self> {
        #[cfg(feature = "highs")]
        let highs_backend = self
            .highs_backend
            .as_ref()
            .map(|backend| backend.try_clone())
            .transpose()?;
        // SAFETY: source stays alive and is not mutated; the C++ bridge returns
        // an independently owned Clp wrapper, released by this type's Drop.
        let model = NonNull::new(unsafe { baldes_clp_clone(self.model.as_ptr()) })
            .ok_or_else(|| incomplete("Clp model copy failed"))?;
        unsafe { Clp_setLogLevel(model.as_ptr(), 0) };
        Ok(Self {
            model,
            solver: self.solver,
            #[cfg(feature = "highs")]
            highs_backend,
            primal_tolerance: self.primal_tolerance,
            highs_row_basic: self.highs_row_basic.clone(),
            rhs: self.rhs.clone(),
            senses: self.senses.clone(),
            costs: self.costs.clone(),
            upper: self.upper.clone(),
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

    pub fn new() -> Result<Self> {
        // SAFETY: Clp returns an exclusively owned model, freed by Drop below.
        let model = NonNull::new(unsafe { Clp_newModel() })
            .ok_or_else(|| incomplete("Clp model allocation failed"))?;
        unsafe {
            Clp_setLogLevel(model.as_ptr(), 0);
            Clp_setOptimizationDirection(model.as_ptr(), 1.);
        }
        Ok(Self {
            model,
            solver: MasterSolver::Clp,
            highs_row_basic: vec![],
            #[cfg(feature = "highs")]
            highs_backend: None,
            primal_tolerance: 1e-7,
            rhs: vec![],
            senses: vec![],
            costs: vec![],
            upper: vec![],
            duals: vec![],
            values: vec![],
            reduced_costs: vec![],
            objective: 0.,
            objective_offset: 0.,
            status: -1,
            solved: false,
            dual_tolerance: 1e-7,
            simplex_iterations: 0,
            zero_tolerance: 0.,
        })
    }
}
