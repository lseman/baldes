//! Native disposable-resource bucket grids. References use stable (vertex,
//! bucket) identities rather than C++ pointers into vectors that may move.
use super::Graph;
use crate::{invalid, Result};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BucketId {
    pub vertex: usize,
    pub bucket: usize,
}
#[derive(Clone, Debug)]
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
    pub target_buckets: i32,
    pub resource_slots: Vec<usize>,
    /// Original vertex IDs in native internal order and its inverse.
    pub order: Vec<usize>,
    pub internal: Vec<usize>,
    pub arc_order: Vec<usize>,
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
