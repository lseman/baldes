//! Full-memory elementary enumeration and label concatenation.
use super::*;
#[derive(Debug)]
pub struct Enumeration {
    pub labels: Storage,
    pub paths: PathStorage,
    pub completed: bool,
    pub generated: usize,
    pub pruning: PruningFrontier,
}

mod directional;
mod forward;
