//! Complete an output with resumable restricted-master state.
use super::*;

pub(super) struct Checkpoint {
    retain_smoothing: bool,
    initial_rows: usize,
}

impl Checkpoint {
    pub fn new(retain_smoothing: bool, initial_rows: usize) -> Self {
        Self {
            retain_smoothing,
            initial_rows,
        }
    }

    pub fn finish(
        &self,
        mut out: Output,
        loaded: Vec<Option<Column>>,
        cut_artificials: Vec<(usize, usize)>,
        pool: Vec<Column>,
        smoothing: DualPriceSmoothing,
    ) -> Result<Output> {
        out.warm_start = WarmStart {
            bound: out.bound,
            bound_mode: out.bound_mode,
            numerical_bound: out.numerical_bound,
            certified_bound: out.certified_bound,
            stabilization: self.retain_smoothing.then_some(smoothing),
            proximal: out.warm_start.proximal.take(),
            generated_pool: pool,
            columns: loaded,
            artificials: out.artificials.clone(),
            cut_artificials,
            initial_rows: self.initial_rows,
        };
        Ok(out)
    }
}
