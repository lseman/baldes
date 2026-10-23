//! Algorithms, native container ordering and pricing state shared by cut families.
pub mod bits;
pub mod mincut;
pub mod pulse;
pub mod rbtree;
pub mod state;
pub mod stdhash;
pub mod stdsort;

/// Conservative capacity for public cut-separator RHS rounding. Even when every
/// elementary route fits the supplied capacity exactly, summing the demands of
/// their union can round upward and make a raw `ceil(sum / capacity)` too large.
/// The nonnegative-sum error bound, with two extra operations for division and
/// the bound itself, only relaxes machine-near integer boundaries. Original
/// model capacities remain unchanged in the public preparation state.
pub(crate) fn rhs_rounding_capacity(capacity: f64, customers: usize) -> f64 {
    let error = (customers as f64 + 2.) * f64::EPSILON;
    let adjusted = capacity * (1. + error / (1. - error));
    if !adjusted.is_finite() || adjusted == f64::MAX {
        f64::MAX
    } else {
        // `next_up` is newer than the crate's MSRV. Positive finite inputs
        // have monotonically ordered bit representations.
        f64::from_bits(adjusted.to_bits() + 1)
    }
}
