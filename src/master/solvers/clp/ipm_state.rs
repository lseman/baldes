use super::*;

#[derive(Debug, Clone, Default)]
pub(super) struct IpmState {
    pub model: Option<crate::extensions::ipm::ModelData>,
    pub solver: crate::extensions::ipm::Solver,
    pub matrix_dirty: bool,
}

impl ClpMaster {
    fn refresh_ipm_model(&mut self) -> Result<()> {
        use crate::extensions::ipm::{ModelData, Sense, SparseMatrix};
        let mut model = self.ipm.model.take().unwrap_or_else(|| ModelData {
            a: SparseMatrix {
                rows: 0,
                cols: 0,
                ptr: vec![0],
                idx: vec![],
                val: vec![],
            },
            b: vec![],
            c: vec![],
            lower: vec![],
            upper: vec![],
            sense: vec![],
        });
        if self.ipm.matrix_dirty
            || model.a.rows != self.rhs.len()
            || model.a.cols > self.costs.len()
        {
            model.a.ptr.clear();
            model.a.ptr.push(0);
            model.a.idx.clear();
            model.a.val.clear();
            model.a.cols = 0;
        }
        // The matrix methods mark edits/removals dirty. Otherwise only newly
        // appended columns need copying from Clp; existing CSC buffers survive.
        self.ipm.matrix_dirty = true;
        for j in model.a.cols..self.costs.len() {
            for (i, value) in self.column_entries(j)? {
                if value != 0. {
                    model.a.idx.push(i);
                    model.a.val.push(value);
                }
            }
            model.a.ptr.push(model.a.idx.len());
        }
        model.a.rows = self.rhs.len();
        model.a.cols = self.costs.len();
        model.b.clone_from(&self.rhs);
        model.c.clone_from(&self.costs);
        model.lower.resize(self.costs.len(), 0.);
        model.upper.clear();
        model.upper.extend(self.upper.iter().map(
            |&u| {
                if u == f64::MAX {
                    f64::INFINITY
                } else {
                    u
                }
            },
        ));
        model.sense.clear();
        model.sense.extend(self.senses.iter().map(|s| match s {
            RowSense::Equal => Sense::Equal,
            RowSense::LessEqual => Sense::LessEqual,
            RowSense::GreaterEqual => Sense::GreaterEqual,
        }));
        self.ipm.model = Some(model);
        self.ipm.matrix_dirty = false;
        Ok(())
    }

    pub(super) fn solve_persistent_ipm(
        &mut self,
        options: &crate::extensions::ipm::Options,
    ) -> Result<crate::extensions::ipm::Solution> {
        let started = Instant::now();
        self.ensure_backend()?;
        self.refresh_ipm_model()?;
        let mut options = options.clone();
        options.time_limit = options
            .time_limit
            .map(|t| t.saturating_sub(started.elapsed()));
        self.ipm
            .solver
            .solve(self.ipm.model.as_ref().unwrap(), &options)
            .map_err(|e| incomplete(format!("IPM master: {e}")))
    }
}
