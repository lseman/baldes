//! Optional, allocation-free views of a paired frontier for bucket indexing.
//!
//! A view contains whole original pieces. We deliberately do not manufacture
//! bucket-boundary knots: interpolating a cut and then interpolating the two
//! shortened segments need not preserve literal binary64 evaluation. Nor do we
//! rebase a split label's cost to its local minimum. Both operations would need
//! a stronger arithmetic contract than the numerical frontier currently has.
//!
//! These are indexing views, not independently extendable labels. They retain
//! the complete source frontier, path and original cost normalization. A bucket
//! implementation may index a label through several views, but must deduplicate
//! by the physical label identity before extension and charge its objective
//! exactly once. This avoids both losing a cheaper piece and counting an
//! overlapping endpoint twice.

use super::relation::{RelationFront, RelationPiece};
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

#[derive(Clone, Copy, Debug)]
pub(super) struct FrontBucketView<'a> {
    /// Keep this identity with each index entry; splitting never changes it.
    pub(super) source: &'a RelationFront,
    pub(super) pieces: &'a [RelationPiece],
    /// All views retain this same minimum. Do not replace it by a view minimum.
    pub(super) normalization_minimum: f64,
}

impl FrontBucketView<'_> {
    /// Conservative base-coordinate hull for selecting intersecting buckets.
    /// Gaps stay in `pieces`; membership in the hull is not feasibility.
    pub(super) fn base_hull(&self) -> Option<[f64; 2]> {
        let first = self.pieces.first()?;
        Some(
            self.pieces
                .iter()
                .fold([first.left[0], first.right[0]], |[lo, hi], piece| {
                    [lo.min(piece.left[0]), hi.max(piece.right[0])]
                }),
        )
    }
}

/// The iterator borrows original storage and does no floating arithmetic.
/// Its nonempty views partition the stored pieces exactly once, including
/// singleton pieces, overlapping endpoints, gaps and waiting markers.
pub(super) struct FrontBucketViews<'a> {
    source: &'a RelationFront,
    remaining: &'a [RelationPiece],
    group_size: usize,
    normalization_minimum: f64,
}

impl<'a> FrontBucketViews<'a> {
    pub(super) fn new(source: &'a RelationFront, policy: FrontBucketPolicy) -> Self {
        let pieces = source.pieces();
        let group_size = match policy {
            FrontBucketPolicy::Disabled => pieces.len().max(1),
            FrontBucketPolicy::PieceGroups {
                pieces_per_view,
                max_views,
            } => {
                let width = pieces_per_view.get();
                // Overflow-free ceil division, including empty fronts.
                let count = pieces.len() / width + usize::from(pieces.len() % width != 0);
                if count <= max_views.get() {
                    width
                } else {
                    pieces.len().max(1)
                }
            }
        };
        Self {
            source,
            remaining: pieces,
            group_size,
            normalization_minimum: source.minimum(),
        }
    }
}

impl<'a> Iterator for FrontBucketViews<'a> {
    type Item = FrontBucketView<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining.is_empty() {
            return None;
        }
        let (pieces, remaining) = self
            .remaining
            .split_at(self.group_size.min(self.remaining.len()));
        self.remaining = remaining;
        Some(FrontBucketView {
            source: self.source,
            pieces,
            normalization_minimum: self.normalization_minimum,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let count = self.remaining.len() / self.group_size
            + usize::from(self.remaining.len() % self.group_size != 0);
        (count, Some(count))
    }
}

impl ExactSizeIterator for FrontBucketViews<'_> {}
impl std::iter::FusedIterator for FrontBucketViews<'_> {}

#[cfg(test)]
#[path = "../../../../tests/unit/graph/coupled/kernel/bucketing/tests.rs"]
mod tests;
