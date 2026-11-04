//! Pointer-free model transport. Native models are created on the receiving thread.
use super::*;

pub(crate) struct MasterTransfer {
    solver: MasterSolver,
    threads: usize,
    options: ClpOptions,
    primal_tolerance: f64,
    dual_tolerance: f64,
    zero_tolerance: f64,
    default_upper: f64,
    rhs: Vec<f64>,
    senses: Vec<RowSense>,
    columns: Vec<Vec<(usize, f64)>>,
    costs: Vec<f64>,
    upper: Vec<f64>,
    objective_offset: f64,
    basis: Option<Vec<u8>>,
    solved: bool,
    duals: Vec<f64>,
    values: Vec<f64>,
    reduced_costs: Vec<f64>,
    objective: f64,
    status: i32,
    native_row_basic: Vec<bool>,
    simplex_iterations: usize,
    ipm_iterations: usize,
}

impl ClpMaster {
    pub(crate) fn export_for_worker(&self) -> Result<MasterTransfer> {
        let basis = self.model.and_then(|model| {
            // SAFETY: the exclusively owned source model exposes exactly one
            // status byte per row/column. Copy bytes; no native pointer escapes.
            unsafe {
                let count = self.rhs.len() + self.costs.len();
                if count == 0 || Clp_statusExists(model.as_ptr()) == 0 {
                    None
                } else {
                    Some(
                        std::slice::from_raw_parts(Clp_statusArray(model.as_ptr()), count).to_vec(),
                    )
                }
            }
        });
        Ok(MasterTransfer {
            solver: self.solver,
            threads: self.backend_threads,
            options: self.clp_options,
            primal_tolerance: self.primal_tolerance,
            dual_tolerance: self.dual_tolerance,
            zero_tolerance: self.zero_tolerance,
            default_upper: self.column_default_upper,
            rhs: self.rhs.clone(),
            senses: self.senses.clone(),
            columns: (0..self.costs.len())
                .map(|i| self.column_entries(i))
                .collect::<Result<_>>()?,
            costs: self.costs.clone(),
            upper: self.upper.clone(),
            objective_offset: self.objective_offset,
            basis,
            solved: self.solved,
            duals: self.duals.clone(),
            values: self.values.clone(),
            reduced_costs: self.reduced_costs.clone(),
            objective: self.objective,
            status: self.status,
            native_row_basic: self.native_row_basic.clone(),
            simplex_iterations: self.simplex_iterations,
            ipm_iterations: self.ipm_iterations,
        })
    }
}

impl MasterTransfer {
    pub(crate) fn reset_iterations(&mut self) {
        self.simplex_iterations = 0;
        self.ipm_iterations = 0;
    }

    pub(crate) fn restore(self) -> Result<ClpMaster> {
        let mut master = ClpMaster::new()?;
        master.set_clp_options(self.options);
        master.set_tolerances(self.primal_tolerance, self.dual_tolerance)?;
        master.zero_tolerance = self.zero_tolerance;
        master.set_column_default_upper(self.default_upper)?;
        let rows: Vec<_> = self
            .senses
            .iter()
            .zip(&self.rhs)
            .map(|(&s, &b)| (s, b, &[][..]))
            .collect();
        master.add_rows(&rows)?;
        let columns: Vec<_> = self
            .costs
            .iter()
            .zip(&self.columns)
            .map(|(&c, a)| (c, a.as_slice()))
            .collect();
        master.add_columns(&columns)?;
        for (column, &upper) in self.upper.iter().enumerate() {
            master.set_upper(column, upper)?;
        }
        if let Some(basis) = self.basis {
            // SAFETY: copyinStatus allocates the destination status array;
            // the owned bytes match this newly reconstructed matrix exactly.
            unsafe {
                Clp_copyinStatus(
                    master.model.expect("new Clp model").as_ptr(),
                    basis.as_ptr(),
                );
            }
        }
        master.objective_offset = self.objective_offset;
        master.set_solver(self.solver);
        master.set_backend_threads(self.threads)?;
        // Cached values belong to this exact matrix and objective. Transport
        // them unchanged: restoring a result must neither solve outside the
        // tree deadline nor replace an IPM/degenerate LP point with another one.
        // The receiver's next solve rebuilds backend workspaces as needed.
        master.duals = self.duals;
        master.values = self.values;
        master.reduced_costs = self.reduced_costs;
        master.objective = self.objective;
        master.status = self.status;
        master.native_row_basic = self.native_row_basic;
        master.solved = self.solved;
        master.simplex_iterations = self.simplex_iterations;
        master.ipm_iterations = self.ipm_iterations;
        Ok(master)
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/solvers/clp/transfer_tests.rs"]
mod tests;
