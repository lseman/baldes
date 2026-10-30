//! Paired front bucket configuration.
use std::num::NonZeroUsize;

/// Disabled by default: subdivision can increase indexing work and allocations
/// in its consumer. Enable only after measuring the actual bucket workload.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum FrontBucketPolicy {
    #[default]
    Disabled,
    /// Group adjacent stored pieces without changing any endpoint or waiting
    /// provenance. If the requested view count exceeds the budget, use one view.
    PieceGroups {
        pieces_per_view: NonZeroUsize,
        max_views: NonZeroUsize,
    },
}
