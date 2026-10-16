//! Restricted-master and dual-price smoothing ports. Cuts are outside this scope.
#[cfg(feature = "clp")]
pub mod clp;
#[cfg(feature = "clp")]
pub mod column_generation;
pub mod stabilization;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowSense {
    Equal,
    LessEqual,
    GreaterEqual,
}

#[cfg(feature = "clp")]
pub mod root;
