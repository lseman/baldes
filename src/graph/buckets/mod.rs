//! Native disposable-resource bucket grids. References use stable (vertex,
//! bucket) identities rather than C++ pointers into vectors that may move.
use super::Graph;
use crate::{invalid, Result};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BucketId {
    pub vertex: usize,
    pub bucket: usize,
}
#[derive(Clone, Copy, Debug)]
pub struct BucketArc {
    pub id: usize,
    pub from: BucketId,
    pub physical: usize,
    pub jump: Option<usize>,
    pub flags: u8,
}
#[derive(Clone, Copy, Debug)]
pub struct Predecessor {
    pub bucket: BucketId,
    pub minimum: usize,
}
#[derive(Clone, Debug)]
pub struct Bucket {
    pub id: BucketId,
    pub component: Option<usize>,
    pub reachable: bool,
    pub predecessors: Vec<Predecessor>,
    pub arcs: Vec<BucketArc>,
}
#[derive(Clone, Debug, Default)]
pub struct Direction {
    pub buckets: Vec<Vec<Bucket>>,
    pub corners: Vec<Vec<[f64; 2]>>,
    pub components: Vec<Vec<BucketId>>,
    pub depth: Vec<i32>,
    pub arc_counter: usize,
    pub maximum_arcs: usize,
    pub current_arcs: usize,
    pub jump_attempts: usize,
}
#[derive(Clone, Debug)]
pub struct BucketGraph {
    pub forward: Direction,
    pub backward: Direction,
    pub bidirectional: bool,
    pub dimensions: usize,
    pub lower: Vec<[f64; 2]>,
    pub upper: Vec<[f64; 2]>,
    pub stride: Vec<[usize; 2]>,
    pub consumption: Vec<[f64; 2]>,
    pub arc_lower: Vec<[f64; 2]>,
    pub arc_upper: Vec<[f64; 2]>,
    pub step: [f64; 2],
    pub middle: f64,
    pub jump_arcs: bool,
    pub version: i32,
    pub backward_version: i32,
    /// Shared by every clone of this grid, like the single native manager.
    pub(crate) versions: Arc<StepVersions>,
    pub target_buckets: i32,
    pub resource_slots: Vec<usize>,
    /// Original vertex IDs in native internal order and its inverse.
    pub order: Vec<usize>,
    pub internal: Vec<usize>,
    pub arc_order: Vec<usize>,
}
/// Step versions of a grid lineage: `next` issues a fresh version to each step
/// decrease, and `current` is the version last decreased or restored.
#[derive(Default)]
pub(crate) struct StepVersions {
    next: AtomicI32,
    current: AtomicI32,
}
/// Lineage state is shared bookkeeping, not grid content.
impl std::fmt::Debug for StepVersions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("StepVersions")
    }
}
impl BucketGraph {
    /// Native `restoreBucketGraphs` when entering a stored node. If the steps
    /// changed elsewhere after the store, reachability, components and
    /// predecessors are recomputed on this grid's own (post-fixing) arcs.
    /// The stored depths are kept.
    pub(crate) fn restore_step_version<const D: usize>(&mut self, g: &Graph<D>) -> Result<()> {
        if self.versions.current.swap(self.version, Ordering::Relaxed) == self.version {
            return Ok(());
        }
        for forward in [true, false] {
            if !forward && (!self.bidirectional || self.backward.buckets.is_empty()) {
                continue;
            }
            self.order_components_preserving_depth(g, forward)?;
            self.refresh_counts(g, forward);
            self.regenerate_jumps(g, forward);
        }
        Ok(())
    }

    /// Give this grid its own lineage, so tree workers never observe each
    /// other's step decreases.
    pub(crate) fn detach_step_versions(&mut self) {
        self.versions = Arc::new(StepVersions {
            next: AtomicI32::new(self.versions.next.load(Ordering::Relaxed)),
            current: AtomicI32::new(self.version),
        });
    }

    fn issue_step_version(&mut self) {
        self.version = self.versions.next.fetch_add(1, Ordering::Relaxed) + 1;
        self.versions.current.store(self.version, Ordering::Relaxed);
    }
}
fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}
fn round8(v: f64) -> f64 {
    (v * 1e8 + 0.5).floor() / 1e8
}

/// `x.floor() as isize` without the libm call that baseline x86-64 makes for
/// `floor` (no SSE4.1 round): the same value for every input, ±inf and NaN too.
#[inline(always)]
fn floor_isize(x: f64) -> isize {
    // Bucket coordinates are normally non-negative: truncation is already
    // floor in this case, including the saturating positive overflow result.
    if x >= 0. {
        return x as isize;
    }
    let t = x as isize;
    if t != isize::MIN && t as f64 > x {
        t - 1
    } else {
        t
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/graph/buckets/tests.rs"]
mod tests;

mod grid;

impl BucketGraph {
    pub fn direction(&self, forward: bool) -> &Direction {
        if forward {
            &self.forward
        } else {
            &self.backward
        }
    }

    pub(super) fn direction_mut(&mut self, forward: bool) -> &mut Direction {
        if forward {
            &mut self.forward
        } else {
            &mut self.backward
        }
    }
}

mod adaptive;
mod arcs;
mod components;
mod jumps;
