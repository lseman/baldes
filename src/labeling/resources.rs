//! Fixed-array scalar kernels specialized for each compiled dimension.
use super::{equality_resource_dominates, extend_resource, EPSILON};
use crate::graph::{Arc, Graph, ResourceKind};

/// A statically sized scalar kernel; two dimensions are the default.
/// LLVM sees `D` in each instantiation and can unroll its loops and eliminate
/// resource-coordinate bounds checks. Graph count mismatches are rejected.
pub struct ScalarResourceKernel<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }>;
impl<const D: usize> ScalarResourceKernel<D> {
    pub const DIMENSIONS: usize = D;

    #[inline(always)]
    pub fn extend(g: &Graph<D>, arc: &Arc<D>, mut q: [f64; D], forward: bool) -> Option<[f64; D]> {
        if g.resources.len() != D {
            return None;
        }
        for r in 0..D {
            q[r] = extend_resource(g, arc, r, q[r], forward)?;
        }
        Some(q)
    }

    #[inline(always)]
    pub(crate) fn extend_active(
        g: &Graph<D>,
        arc: &Arc<D>,
        mut q: [f64; D],
        forward: bool,
    ) -> Option<[f64; D]> {
        for r in 0..D {
            if r < g.resources.len() {
                q[r] = extend_resource(g, arc, r, q[r], forward)?;
            }
        }
        Some(q)
    }

    #[inline(always)]
    pub fn dominates(kinds: &[ResourceKind], a: &[f64; D], b: &[f64; D], forward: bool) -> bool {
        Self::dominates_from::<0>(kinds, a, b, forward)
    }

    #[inline(always)]
    pub(crate) fn dominates_from<const START: usize>(
        kinds: &[ResourceKind],
        a: &[f64],
        b: &[f64],
        forward: bool,
    ) -> bool {
        if kinds.len() > D || a.len() < D || b.len() < D {
            return false;
        }
        for r in START..D {
            if r >= kinds.len() {
                break;
            }
            let pass = match kinds[r] {
                ResourceKind::Disposable => {
                    if forward {
                        a[r] <= b[r] + EPSILON
                    } else {
                        b[r] - EPSILON <= a[r]
                    }
                }
                ResourceKind::Equality => equality_resource_dominates(a[r], b[r]),
            };
            if !pass {
                return false;
            }
        }
        true
    }
}
