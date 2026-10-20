//! Column-generation callback adapter; implementations are grouped by responsibility.
use super::*;
mod capacity;
mod checkpoint;
mod enumeration;
mod knapsack;
mod kpath;
mod pool;
mod rank_one;
mod relaxation;
mod separation;

pub(super) use checkpoint::PricingCheckpoint;
#[cfg(test)]
use pool::synchronize_pool;
use separation::separation_gap_closed;
#[cfg(test)]
#[path = "tests.rs"]
mod synchronization_tests;

pub(super) struct RootController<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    pub(super) problem: Rc<RootProblem<D>>,
    pub(super) state: Rc<RefCell<State<D>>>,
    pub(super) checkpoint: Option<PricingCheckpoint<D>>,
    pub(super) certificate_search: super::certificate::SearchCache,
}

impl<const D: usize> Controller<ClpMaster> for RootController<D> {
    fn restore_feasible_columns(&mut self, duals: &[f64]) -> Result<Vec<Column>> {
        super::restoration::columns(&self.problem, &mut self.state.borrow_mut(), duals)
    }

    fn certified_pricing_bound(&self, master: &ClpMaster, _duals: &[f64]) -> Result<Option<f64>> {
        let bound = super::certificate::lower_bound(
            &self.problem,
            &self.state.borrow(),
            master,
            &self.certificate_search,
        );
        Ok(Some(bound.unwrap_or(f64::NEG_INFINITY)))
    }

    fn save_pricing_checkpoint(&mut self, master: &ClpMaster, output: &Output) -> Result<()> {
        self.store_pricing_checkpoint(master, output.clone())
    }

    fn store_pricing_checkpoint(&mut self, master: &ClpMaster, output: Output) -> Result<()> {
        self.store_checkpoint(master, output)
    }

    fn unstabilized_rows(&self) -> Vec<usize> {
        let map = self.problem.row_map.borrow();
        std::iter::once(map[0])
            .chain(
                self.problem
                    .branch_constraints
                    .iter()
                    .filter_map(|(row, constraint)| {
                        matches!(constraint, BranchConstraint::NumberOfVehicles { .. })
                            .then_some(map[*row])
                    }),
            )
            .collect()
    }

    fn protected_rows(&self) -> Vec<usize> {
        self.problem.row_map.borrow().clone()
    }

    fn master_solved(
        &mut self,
        master: &ClpMaster,
        tags: &[i32],
        artificials: &[usize],
    ) -> Result<()> {
        if artificials.iter().any(|&i| master.values()[i] > 1e-6) {
            return Ok(());
        }
        let mut state = self.state.borrow_mut();
        if let Some(upper) =
            super::incumbent::validated_upper(&self.problem, &state.routes, tags, &master.values())
        {
            state.incumbent = state.incumbent.min(upper);
        }
        Ok(())
    }

    fn incumbent(&self) -> f64 {
        self.state.borrow().incumbent
    }

    fn refresh_generated_columns(&mut self, pool: &mut [Column]) -> Result<()> {
        self.refresh_pool(pool)
    }

    fn synchronize_generated_columns(
        &mut self,
        pool: &mut [Column],
        loaded: &[Option<Column>],
    ) -> Result<()> {
        self.synchronize_pool_entries(pool, loaded)
    }

    fn enumerated_universe(&mut self, duals: &[f64], bound: f64) -> Result<Option<Vec<Column>>> {
        self.enumerate_columns(duals, bound)
    }

    fn regenerate_enumerated_columns(
        &mut self,
        pool: &[Column],
        duals: &[f64],
        bound: f64,
        universe: &[Column],
    ) -> Result<Vec<Column>> {
        self.regenerate_pool(pool, duals, bound, universe)
    }

    fn remap_rows(&mut self, rows: &[Option<usize>]) -> Result<()> {
        self.remap_pool_rows(rows)
    }

    fn separates(&self) -> bool {
        !self.state.borrow().cut.separators().is_empty()
    }

    fn separate(&mut self, master: &mut ClpMaster, tags: &[i32]) -> Result<usize> {
        self.separate_round(master, tags)
    }

    fn take_removed_columns(&mut self) -> Vec<usize> {
        std::mem::take(&mut self.state.borrow_mut().cut.removed_columns)
    }
}
