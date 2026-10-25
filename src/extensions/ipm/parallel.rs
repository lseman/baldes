//! Per-solver CPU pool; nested tree/node work shares the same thread budget.
use super::{Error, Result};
use std::sync::Arc;

/// Whether to use parallel IPM preparation and factorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Parallel {
    Off,
    On,
    #[default]
    Choose,
}

/// Factorization parallelism when `Parallel::On` is selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ParallelType {
    Tree,
    Node,
    #[default]
    Both,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct Execution {
    pub pool: Option<Arc<rayon::ThreadPool>>,
    pub mode: Parallel,
    pub kind: ParallelType,
}
impl Execution {
    pub fn new(mode: Parallel, kind: ParallelType, threads: usize) -> Result<Self> {
        let count = if threads == 0 {
            std::thread::available_parallelism().map_or(1, usize::from)
        } else {
            threads
        };
        let pool = if mode != Parallel::Off && count > 1 {
            Some(Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(count)
                    .build()
                    .map_err(|e| {
                        Error::InvalidInput(format!("cannot create IPM worker pool: {e}"))
                    })?,
            ))
        } else {
            None
        };
        Ok(Self { pool, mode, kind })
    }
    pub fn levels(&self, wide_tree: bool, large_front: bool) -> (bool, bool) {
        if self.pool.is_none() {
            return (false, false);
        }
        match self.mode {
            Parallel::Off => (false, false),
            Parallel::Choose => (wide_tree, large_front),
            Parallel::On => (
                self.kind != ParallelType::Node,
                self.kind != ParallelType::Tree,
            ),
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/extensions/ipm/parallel/tests.rs"]
mod tests;
