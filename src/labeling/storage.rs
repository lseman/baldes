//! Per-bucket native insertion with immutable ancestry in an arena. Removing a
//! dominated active label never destroys a predecessor needed for reconstruction.
use super::{Statistics, EPSILON};
use crate::cuts::state::{packed_dominate, DominanceDescriptor, LIMITED_WORDS};
use crate::graph::buckets::BucketId;
use crate::graph::{Graph, ResourceKind};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InsertionTest {
    Full,
    ResourcesAndCost,
    ResourcesAndNg,
    Resources,
}
#[derive(Clone, Debug)]
pub struct StoredLabel {
    pub id: usize,
    pub local_id: u64,
    pub bucket: BucketId,
    pub vertex: usize,
    pub cost: f64,
    pub original_cost: f64,
    pub resources: [f64; 2],
    pub ng: u16,
    pub parent: Option<usize>,
    pub arc: Option<usize>,
    pub hops: usize,
    pub expanded: bool,
    /// Packed limited-memory cut state and its accumulated cost.
    pub cut_words: [u32; LIMITED_WORDS],
    pub additional_cost: f64,
}
impl StoredLabel {
    /// Bucket ordering key, (0.5 potential + additional) + (0 + reduced);
    /// the rank-one potential is zero. Equals the standard key without cuts.
    pub fn key(&self) -> f64 {
        (0.0 + self.additional_cost) + (0.0 + self.cost)
    }
}
#[derive(Clone, Debug, Default)]
pub struct LabelBucket {
    pub active: Vec<usize>,
    pub archived: Vec<usize>,
    pub next_id: u64,
    pub lower_cost: f64,
}
#[derive(Clone, Debug)]
pub struct LabelStorage {
    pub labels: Vec<StoredLabel>,
    pub buckets: Vec<Vec<LabelBucket>>,
    pub crossing: Vec<Vec<usize>>,
    pub root: Option<usize>,
}
impl LabelStorage {
    pub fn new(counts: impl IntoIterator<Item = usize>) -> Self {
        let buckets: Vec<Vec<_>> = counts
            .into_iter()
            .map(|n| {
                (0..n)
                    .map(|_| LabelBucket {
                        lower_cost: 1e12,
                        ..Default::default()
                    })
                    .collect()
            })
            .collect();
        let crossing = vec![vec![]; buckets.len()];
        Self {
            labels: vec![],
            buckets,
            crossing,
            root: None,
        }
    }
    pub fn retain(&mut self, mut label: StoredLabel) -> usize {
        let id = self.labels.len();
        label.id = id;
        self.labels.push(label);
        id
    }
    pub fn dominates(
        g: &Graph,
        a: &StoredLabel,
        b: &StoredLabel,
        forward: bool,
        test: InsertionTest,
        hop_limit: bool,
        cuts: &[DominanceDescriptor],
    ) -> bool {
        if hop_limit && a.hops > b.hops {
            return false;
        }
        if !g.resources.iter().enumerate().all(|(r, kind)| match kind {
            ResourceKind::Disposable => {
                if forward {
                    a.resources[r] <= b.resources[r] + EPSILON
                } else {
                    b.resources[r] <= a.resources[r] + EPSILON
                }
            }
            ResourceKind::Equality => (a.resources[r] - b.resources[r]).abs() <= EPSILON,
        }) {
            return false;
        }
        if matches!(test, InsertionTest::Full | InsertionTest::ResourcesAndNg) && a.ng & !b.ng != 0
        {
            return false;
        }
        match test {
            // Full dominance: the additional cost joins the gap, then
            // the vertex's packed cut fields.
            InsertionTest::Full => {
                let mut gap = (a.cost - b.cost) + (a.additional_cost - b.additional_cost);
                if gap >= EPSILON {
                    return false;
                }
                packed_dominate(cuts, &a.cut_words, &b.cut_words, &mut gap)
            }
            InsertionTest::ResourcesAndCost => a.cost - b.cost < EPSILON,
            _ => true,
        }
    }
    pub fn insert(
        &mut self,
        g: &Graph,
        mut label: StoredLabel,
        forward: bool,
        test: InsertionTest,
        capacity: usize,
        hop_limit: bool,
        cuts: &[DominanceDescriptor],
        stats: &mut Statistics,
    ) -> bool {
        let b = label.bucket;
        let old = self.buckets[b.vertex][b.bucket].active.clone();
        // Native single-label phases retain the cheapest prefix, regardless of
        // resource/NG dominance, and notify the SCC scheduler on replacement.
        if capacity == 1 {
            if let Some(&previous) = old.first() {
                if !(self.labels[previous].key() - EPSILON > label.key()) {
                    return false;
                }
                if self.labels[previous].expanded {
                    self.buckets[b.vertex][b.bucket].archived.push(previous);
                }
            }
            label.local_id = self.buckets[b.vertex][b.bucket].next_id;
            self.buckets[b.vertex][b.bucket].next_id += 1;
            let id = self.retain(label);
            self.buckets[b.vertex][b.bucket].active = vec![id];
            stats.generated += 1;
            return true;
        }
        let threshold = EPSILON + label.key();
        let mut position = 0;
        while position < old.len() {
            let previous = &self.labels[old[position]];
            if previous.key() > threshold {
                break;
            }
            stats.dominance_checks += 1;
            if Self::dominates(g, previous, &label, forward, test, hop_limit, cuts) {
                stats.dominated += 1;
                return false;
            }
            position += 1;
        }
        if position == old.len() && capacity <= old.len() {
            stats.capped += 1;
            return false;
        }
        let local = self.buckets[b.vertex][b.bucket].next_id;
        self.buckets[b.vertex][b.bucket].next_id += 1;
        label.local_id = local;
        let inserted = self.retain(label);
        let mut active = old[..position].to_vec();
        active.push(inserted);
        for (offset, &id) in old[position..].iter().enumerate() {
            if offset > 0 {
                stats.later_label_checks += 1;
            }
            stats.dominance_checks += 1;
            if Self::dominates(
                g,
                &self.labels[inserted],
                &self.labels[id],
                forward,
                test,
                hop_limit,
                cuts,
            ) {
                stats.dominated += 1;
                if self.labels[id].expanded
                    && matches!(test, InsertionTest::Full | InsertionTest::ResourcesAndCost)
                {
                    self.buckets[b.vertex][b.bucket].archived.push(id);
                }
            } else {
                active.push(id);
            }
        }
        // Native insertion keeps the newly inserted candidate even if the last
        // carried label cannot fit, but reports no new expansion in that case.
        let accepted = active.len() <= capacity;
        if !accepted {
            stats.capped += 1;
            active.truncate(capacity);
        }
        self.buckets[b.vertex][b.bucket].active = active;
        stats.generated += 1;
        accepted
    }
}
