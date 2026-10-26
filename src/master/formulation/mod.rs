//! Construction and row insertion for restricted masters of any model.
use super::{
    clp::{ClpMaster, ClpOptions, MasterSolver},
    LpMaster, RowSense,
};
use crate::{graph::Window, invalid, Result};

/// A constraint in the original master formulation.
#[derive(Clone, Debug)]
pub struct Row {
    pub sense: RowSense,
    pub rhs: f64,
}

/// Backend settings apply equally to new and inherited restricted masters.
pub(crate) struct Backend {
    pub solver: MasterSolver,
    pub threads: usize,
    pub clp: ClpOptions,
    pub zero_tolerance: f64,
    pub column_upper: f64,
}
impl Default for Backend {
    fn default() -> Self {
        Self {
            solver: MasterSolver::Clp,
            threads: 0,
            clp: ClpOptions::default(),
            zero_tolerance: 0.,
            column_upper: f64::MAX,
        }
    }
}
impl Backend {
    pub fn configure(&self, master: &mut ClpMaster) -> Result<()> {
        self.solver.ensure_available()?;
        master.set_solver(self.solver);
        master.set_backend_threads(self.threads)?;
        master.set_clp_options(self.clp);
        master.set_tolerances(1e-7, 1e-7)?;
        master.zero_tolerance = self.zero_tolerance;
        master.set_column_default_upper(self.column_upper)
    }
}

pub(crate) struct BuiltMaster {
    pub master: ClpMaster,
    pub quantity_rows: Vec<Vec<(usize, f64)>>,
    pub shared_offset: usize,
}

/// Build the same canonical model for every input adapter.
pub(crate) fn build_model<const D: usize>(
    problem: &crate::master::model::Problem<D>,
    backend: &Backend,
) -> Result<BuiltMaster> {
    problem.validate()?;
    let quantities: Vec<_> = problem
        .networks
        .iter()
        .map(|network| network.multiplicity)
        .collect();
    build(&problem.rows, &quantities, backend)
}

/// Quantity constraints precede shared rows. All route coefficients refer to
/// this returned layout; adapters never reconstruct physical row offsets.
pub(crate) fn build(rows: &[Row], quantities: &[Window], backend: &Backend) -> Result<BuiltMaster> {
    let mut master = ClpMaster::new()?;
    backend.configure(&mut master)?;
    let mut pending = PendingRows::new(0);
    let mut quantity_rows = Vec::with_capacity(quantities.len());
    for quantity in quantities {
        if !quantity.lower.is_finite()
            || quantity.lower < 0.
            || quantity.upper.is_nan()
            || quantity.upper < quantity.lower
        {
            return Err(invalid("invalid network multiplicity"));
        }
        let mut entries = vec![];
        if quantity.upper.is_finite() {
            entries.push((
                pending.push(RowSense::LessEqual, quantity.upper, vec![]),
                1.,
            ));
        }
        if quantity.lower > 0. {
            entries.push((
                pending.push(RowSense::GreaterEqual, quantity.lower, vec![]),
                1.,
            ));
        }
        quantity_rows.push(entries);
    }
    let shared_offset = pending.len();
    for row in rows {
        pending.push(row.sense, row.rhs, vec![]);
    }
    pending.install(&mut master)?;
    Ok(BuiltMaster {
        master,
        quantity_rows,
        shared_offset,
    })
}

/// A batch of dynamic rows. Assigned row identities remain valid only while
/// the master has the same prefix; check that before touching its matrix.
pub(crate) struct PendingRows {
    base: usize,
    rows: Vec<(RowSense, f64, Vec<(usize, f64)>)>,
}
impl PendingRows {
    pub fn new(base: usize) -> Self {
        Self { base, rows: vec![] }
    }
    pub fn len(&self) -> usize {
        self.rows.len()
    }
    pub fn push(&mut self, sense: RowSense, rhs: f64, entries: Vec<(usize, f64)>) -> usize {
        self.rows.push((sense, rhs, entries));
        self.base + self.rows.len() - 1
    }
    pub fn install(&self, master: &mut ClpMaster) -> Result<()> {
        if master.rhs().len() != self.base {
            return Err(invalid("pending rows have a stale master prefix"));
        }
        let batch: Vec<_> = self
            .rows
            .iter()
            .map(|(sense, rhs, entries)| (*sense, *rhs, entries.as_slice()))
            .collect();
        master.add_rows(&batch).map(|_| ())
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/master/formulation/tests.rs"]
mod tests;
