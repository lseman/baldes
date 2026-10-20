//! Full-memory elementary enumeration and label concatenation.
use super::*;
#[derive(Debug)]
pub struct Enumeration<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    pub labels: Storage<D>,
    pub paths: PathStorage,
    pub completed: bool,
    pub generated: usize,
    pub pruning: PruningFrontier,
}

mod directional;
mod forward;
