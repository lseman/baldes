//! Termination outcomes for all branch-and-price input adapters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TreeStatus {
    Optimal,
    Infeasible,
    TimeLimit,
    NodeLimit,
    Incomplete,
    BatchLimit,
}
