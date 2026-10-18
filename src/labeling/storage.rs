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
    pub branch_state: crate::pricing::branching::BranchState,
    pub parent: Option<usize>,
    pub arc: Option<usize>,
    pub hops: usize,
    pub expanded: bool,
    /// Packed limited-memory cut state and its accumulated cost.
    pub cut_words: [u32; LIMITED_WORDS],
    pub additional_cost: f64,
    pub half_potential: f64,
}
impl StoredLabel {
    /// Bucket ordering key, (0.5 potential + additional) + (0 + reduced);
    /// the rank-one potential is zero. Equals the standard key without cuts.
    pub fn key(&self) -> f64 {
        (self.half_potential + self.additional_cost) + (0.0 + self.cost)
    }
}
/// The fields dominance reads before the cut words, packed so bucket scans
/// touch 48 bytes per label instead of the whole `StoredLabel`.
#[derive(Clone, Copy, Debug)]
pub struct HotLabel {
    pub key: f64,
    pub cost: f64,
    pub additional_cost: f64,
    pub resources: [f64; 2],
    pub hops: usize,
    pub ng: u16,
}
impl HotLabel {
    #[inline]
    pub fn of(l: &StoredLabel) -> Self {
        Self {
            key: l.key(),
            cost: l.cost,
            additional_cost: l.additional_cost,
            resources: l.resources,
            hops: l.hops,
            ng: l.ng,
        }
    }
}
/// `HotLabel`s as columns, so a bucket's prefix scan compares eight labels
/// at a time.
#[derive(Clone, Debug, Default)]
pub struct HotColumns {
    pub key: Vec<f64>,
    pub cost: Vec<f64>,
    pub additional_cost: Vec<f64>,
    pub resources: [Vec<f64>; 2],
    pub hops: Vec<usize>,
    pub ng: Vec<u16>,
}
impl HotColumns {
    #[inline(always)]
    pub fn get(&self, i: usize) -> HotLabel {
        HotLabel {
            key: self.key[i],
            cost: self.cost[i],
            additional_cost: self.additional_cost[i],
            resources: [self.resources[0][i], self.resources[1][i]],
            hops: self.hops[i],
            ng: self.ng[i],
        }
    }
    pub fn push(&mut self, h: HotLabel) {
        self.key.push(h.key);
        self.cost.push(h.cost);
        self.additional_cost.push(h.additional_cost);
        self.resources[0].push(h.resources[0]);
        self.resources[1].push(h.resources[1]);
        self.hops.push(h.hops);
        self.ng.push(h.ng);
    }
    fn insert(&mut self, i: usize, h: HotLabel) {
        self.key.insert(i, h.key);
        self.cost.insert(i, h.cost);
        self.additional_cost.insert(i, h.additional_cost);
        self.resources[0].insert(i, h.resources[0]);
        self.resources[1].insert(i, h.resources[1]);
        self.hops.insert(i, h.hops);
        self.ng.insert(i, h.ng);
    }
    /// Moves entries `from` down to start at `to`.
    #[inline]
    fn move_run(&mut self, from: std::ops::Range<usize>, to: usize) {
        self.key.copy_within(from.clone(), to);
        self.cost.copy_within(from.clone(), to);
        self.additional_cost.copy_within(from.clone(), to);
        self.resources[0].copy_within(from.clone(), to);
        self.resources[1].copy_within(from.clone(), to);
        self.hops.copy_within(from.clone(), to);
        self.ng.copy_within(from, to);
    }
    pub fn truncate(&mut self, n: usize) {
        self.key.truncate(n);
        self.cost.truncate(n);
        self.additional_cost.truncate(n);
        self.resources[0].truncate(n);
        self.resources[1].truncate(n);
        self.hops.truncate(n);
        self.ng.truncate(n);
    }
    /// First index at or after `from` that passes dominance's resource and
    /// NG tests against `c`, or `len` when none does. In the `PREFIX` scan the
    /// entries are the dominating side and a key above `threshold` also
    /// stops; otherwise `c` dominates and `threshold` is unused.
    #[inline(always)]
    fn next_candidate<
        const PREFIX: bool,
        const FORWARD: bool,
        const RESOURCES: usize,
        const NG: bool,
    >(
        &self,
        from: usize,
        threshold: f64,
        c: &HotLabel,
    ) -> usize {
        let n = self.key.len();
        // One label's test on values, so the 8-wide block below reads
        // fixed-size arrays without bounds checks and vectorizes.
        let test = |key: f64, r0: f64, r1: f64, ng: u16| -> bool {
            let mut pass = true;
            for (r, x) in [r0, r1].into_iter().enumerate().take(RESOURCES) {
                let (a, b) = if PREFIX { (x, c.resources[r]) } else { (c.resources[r], x) };
                pass &= if FORWARD { a <= b + EPSILON } else { b <= a + EPSILON };
            }
            if NG {
                let (a, b) = if PREFIX { (ng, c.ng) } else { (c.ng, ng) };
                pass &= a & !b == 0;
            }
            if PREFIX {
                (key > threshold) | pass
            } else {
                pass
            }
        };
        let hit = |i: usize| {
            test(self.key[i], self.resources[0][i], self.resources[1][i], self.ng[i])
        };
        let mut i = from;
        while i + 8 <= n {
            let key: &[f64; 8] = self.key[i..i + 8].try_into().unwrap();
            let r0: &[f64; 8] = self.resources[0][i..i + 8].try_into().unwrap();
            let r1: &[f64; 8] = self.resources[1][i..i + 8].try_into().unwrap();
            let ng: &[u16; 8] = self.ng[i..i + 8].try_into().unwrap();
            let mut mask = 0u32;
            for j in 0..8 {
                mask |= (test(key[j], r0[j], r1[j], ng[j]) as u32) << j;
            }
            if mask != 0 {
                return i + mask.trailing_zeros() as usize;
            }
            i += 8;
        }
        while i < n && !hit(i) {
            i += 1;
        }
        i
    }
}
#[derive(Clone, Debug, Default)]
pub struct LabelBucket {
    pub active: Vec<usize>,
    /// The `HotLabel` of each `active` entry, in the same order.
    pub hot: HotColumns,
    pub archived: Vec<usize>,
    pub next_id: u64,
    pub lower_cost: f64,
    /// Completion bound including native resource-knapsack dual costs.
    pub knapsack_lower_cost: f64,
}
impl LabelBucket {
    pub fn completion_bound(&self, knapsack: bool) -> f64 {
        if knapsack { self.knapsack_lower_cost } else { self.lower_cost }
    }
}
#[derive(Clone, Debug)]
pub struct LabelStorage {
    pub labels: Vec<StoredLabel>,
    /// `HotLabel::of(&labels[i])`, kept contiguous so bucket scans stay in cache.
    pub hot: Vec<HotLabel>,
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
                        knapsack_lower_cost: 1e12,
                        ..Default::default()
                    })
                    .collect()
            })
            .collect();
        let crossing = vec![vec![]; buckets.len()];
        Self {
            labels: vec![],
            hot: vec![],
            buckets,
            crossing,
            root: None,
        }
    }
    /// `*self = Self::new(counts)`, keeping the allocations when the bucket
    /// shape is unchanged; the arena and buckets regrow every pricing call.
    pub fn reset(&mut self, counts: impl IntoIterator<Item = usize>) {
        let counts: Vec<usize> = counts.into_iter().collect();
        if !counts.iter().copied().eq(self.buckets.iter().map(Vec::len)) {
            *self = Self::new(counts);
            return;
        }
        self.labels.clear();
        self.hot.clear();
        for bucket in self.buckets.iter_mut().flatten() {
            bucket.active.clear();
            bucket.hot.truncate(0);
            bucket.archived.clear();
            bucket.next_id = 0;
            bucket.lower_cost = 1e12;
            bucket.knapsack_lower_cost = 1e12;
        }
        self.crossing.iter_mut().for_each(Vec::clear);
        self.root = None;
    }
    pub fn retain(&mut self, mut label: StoredLabel) -> usize {
        let id = self.labels.len();
        label.id = id;
        self.hot.push(HotLabel::of(&label));
        self.labels.push(label);
        id
    }
    #[inline]
    pub fn dominates(
        g: &Graph,
        a: &StoredLabel,
        b: &StoredLabel,
        forward: bool,
        test: InsertionTest,
        hop_limit: bool,
        cuts: &[DominanceDescriptor],
    ) -> bool {
        a.branch_state == b.branch_state && Self::dominates_hot(
            g,
            &HotLabel::of(a),
            &HotLabel::of(b),
            forward,
            test,
            hop_limit,
            cuts,
            || (&a.cut_words, &b.cut_words),
        )
    }
    /// `dominates` on the packed fields; `words` loads the cut words only
    /// when the full test reaches them.
    #[inline(always)]
    pub fn dominates_hot<'w>(
        g: &Graph,
        a: &HotLabel,
        b: &HotLabel,
        forward: bool,
        test: InsertionTest,
        hop_limit: bool,
        cuts: &[DominanceDescriptor],
        words: impl FnOnce() -> (&'w [u32; LIMITED_WORDS], &'w [u32; LIMITED_WORDS]),
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
                let (wa, wb) = words();
                packed_dominate(cuts, wa, wb, &mut gap)
            }
            InsertionTest::ResourcesAndCost => a.cost - b.cost < EPSILON,
            _ => true,
        }
    }
    /// `HotColumns::next_candidate` for this graph's resources; without a
    /// specialization every index is a candidate.
    #[inline]
    fn next_candidate<const PREFIX: bool>(
        g: &Graph,
        cols: &HotColumns,
        from: usize,
        threshold: f64,
        c: &HotLabel,
        forward: bool,
        ng: bool,
    ) -> usize {
        let disposable = g.resources.iter().all(|k| matches!(k, ResourceKind::Disposable));
        macro_rules! scan {
            ($f:literal, $r:literal, $n:literal) => {
                cols.next_candidate::<PREFIX, $f, $r, $n>(from, threshold, c)
            };
        }
        match (disposable, g.resources.len(), forward, ng) {
            (true, 1, true, true) => scan!(true, 1, true),
            (true, 1, true, false) => scan!(true, 1, false),
            (true, 1, false, true) => scan!(false, 1, true),
            (true, 1, false, false) => scan!(false, 1, false),
            (true, 2, true, true) => scan!(true, 2, true),
            (true, 2, true, false) => scan!(true, 2, false),
            (true, 2, false, true) => scan!(false, 2, true),
            (true, 2, false, false) => scan!(false, 2, false),
            _ => from,
        }
    }
    pub fn insert(
        &mut self,
        g: &Graph,
        label: StoredLabel,
        forward: bool,
        test: InsertionTest,
        capacity: usize,
        hop_limit: bool,
        cuts: &[DominanceDescriptor],
        stats: &mut Statistics,
    ) -> bool {
        let candidate = HotLabel::of(&label);
        self.insert_ref(g, &label, candidate, forward, test, capacity, hop_limit, cuts, stats)
    }
    /// `insert` that copies `label` into the arena only when it is kept.
    /// `candidate` is `HotLabel::of(label)`, built by the caller while the
    /// fields are still in registers: reading them back from a label just
    /// written stalls on store forwarding. Inlined for the same reason: passed
    /// through memory, `candidate` was reloaded as 16-byte pairs.
    #[inline(always)]
    pub fn insert_ref(
        &mut self,
        g: &Graph,
        label: &StoredLabel,
        candidate: HotLabel,
        forward: bool,
        test: InsertionTest,
        capacity: usize,
        hop_limit: bool,
        cuts: &[DominanceDescriptor],
        stats: &mut Statistics,
    ) -> bool {
        let b = label.bucket;
        let labels = &self.labels;
        let bucket = &mut self.buckets[b.vertex][b.bucket];
        // Native single-label phases retain the cheapest prefix, regardless of
        // resource/NG dominance, and notify the SCC scheduler on replacement.
        if capacity == 1 {
            if let Some(&previous) = bucket.active.first() {
                if !(bucket.hot.key[0] - EPSILON > candidate.key) {
                    return false;
                }
                if labels[previous].expanded {
                    bucket.archived.push(previous);
                }
            }
            let local_id = bucket.next_id;
            bucket.next_id += 1;
            bucket.active.clear();
            bucket.hot.truncate(0);
            let id = self.retain(StoredLabel { local_id, ..label.clone() });
            let bucket = &mut self.buckets[b.vertex][b.bucket];
            bucket.active.push(id);
            bucket.hot.push(candidate);
            stats.generated += 1;
            return true;
        }
        let threshold = EPSILON + candidate.key;
        let ng = matches!(test, InsertionTest::Full | InsertionTest::ResourcesAndNg);
        let mut position = 0;
        loop {
            // Labels skipped here fail dominance's resource or NG test, but
            // still count as checked.
            let next =
                Self::next_candidate::<true>(g, &bucket.hot, position, threshold, &candidate, forward, ng);
            stats.dominance_checks += next - position;
            position = next;
            if position == bucket.active.len() || bucket.hot.key[position] > threshold {
                break;
            }
            stats.dominance_checks += 1;
            let previous = bucket.active[position];
            let hot = bucket.hot.get(position);
            if labels[previous].branch_state == label.branch_state && Self::dominates_hot(g, &hot, &candidate, forward, test, hop_limit, cuts, || {
                (&labels[previous].cut_words, &label.cut_words)
            }) {
                stats.dominated += 1;
                return false;
            }
            position += 1;
        }
        if position == bucket.active.len() && capacity <= bucket.active.len() {
            stats.capped += 1;
            return false;
        }
        let local_id = bucket.next_id;
        bucket.next_id += 1;
        let inserted = self.retain(StoredLabel { local_id, ..label.clone() });
        let labels = &self.labels;
        let bucket = &mut self.buckets[b.vertex][b.bucket];
        bucket.active.insert(position, inserted);
        bucket.hot.insert(position, candidate);
        // Compact the later labels the candidate does not dominate; every
        // one of them is checked.
        let len = bucket.active.len();
        let tail = len - (position + 1);
        stats.dominance_checks += tail;
        stats.later_label_checks += tail.saturating_sub(1);
        let mut kept = position + 1;
        let mut read = position + 1;
        loop {
            let next =
                Self::next_candidate::<false>(g, &bucket.hot, read, 0., &candidate, forward, ng);
            if kept != read {
                bucket.active.copy_within(read..next, kept);
                bucket.hot.move_run(read..next, kept);
            }
            kept += next - read;
            if next == len {
                break;
            }
            read = next + 1;
            let id = bucket.active[next];
            let hot = bucket.hot.get(next);
            if labels[inserted].branch_state == labels[id].branch_state && Self::dominates_hot(g, &candidate, &hot, forward, test, hop_limit, cuts, || {
                (&labels[inserted].cut_words, &labels[id].cut_words)
            }) {
                stats.dominated += 1;
                // archives(): only the resources(+cost) tests keep dominated
                // expanded labels reachable for solution recording.
                if labels[id].expanded
                    && matches!(test, InsertionTest::ResourcesAndCost | InsertionTest::Resources)
                {
                    bucket.archived.push(id);
                }
            } else {
                bucket.active[kept] = id;
                bucket.hot.move_run(next..next + 1, kept);
                kept += 1;
            }
        }
        // Native insertion keeps the newly inserted candidate even if the last
        // carried label cannot fit, but reports no new expansion in that case.
        let accepted = kept <= capacity;
        if !accepted {
            stats.capped += 1;
        }
        let len = kept.min(capacity);
        bucket.active.truncate(len);
        bucket.hot.truncate(len);
        stats.generated += 1;
        accepted
    }
}
