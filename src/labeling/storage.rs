//! Per-bucket native insertion with immutable ancestry in an arena. Removing a
//! dominated active label never destroys a predecessor needed for reconstruction.
pub use super::arena::LabelArena;
use super::arena::ArenaRecord;
/// Keep the plain borrowed path a pointer-sized argument, while moving owned
/// labels with heap state into the arena after acceptance.
pub(crate) trait LabelInput {
    fn label(&self) -> &StoredLabel;
    fn into_label(self) -> StoredLabel;
}
impl LabelInput for StoredLabel {
    #[inline(always)] fn label(&self) -> &StoredLabel { self }
    #[inline(always)] fn into_label(self) -> StoredLabel { self }
}
impl LabelInput for &StoredLabel {
    #[inline(always)] fn label(&self) -> &StoredLabel { self }
    #[inline(always)] fn into_label(self) -> StoredLabel { self.clone() }
}
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
/// avoid loading the full record. The scalar cache is 56 bytes on 64-bit targets.
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
        candidate_r0: f64,
        candidate_r1: f64,
        candidate_ng: u16,
    ) -> usize {
        let n = self.key.len();
        // One label's test on values, so the 8-wide block below reads
        // fixed-size arrays without bounds checks and vectorizes.
        let test = |key: f64, r0: f64, r1: f64, ng: u16| -> bool {
            let mut pass = true;
            for (r, x) in [r0, r1].into_iter().enumerate().take(RESOURCES) {
                let candidate = if r == 0 { candidate_r0 } else { candidate_r1 };
                let (a, b) = if PREFIX { (x, candidate) } else { (candidate, x) };
                pass &= if FORWARD { a <= b + EPSILON } else { b <= a + EPSILON };
            }
            if NG {
                let (a, b) = if PREFIX { (ng, candidate_ng) } else { (candidate_ng, ng) };
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
    pub labels: LabelArena,
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
            labels: LabelArena::default(),
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
    #[inline(never)]
    pub(crate) fn retain_for<R: ArenaRecord>(&mut self, mut label: StoredLabel) -> usize {
        let id = self.labels.len(); label.id = id;
        self.hot.push(HotLabel::of(&label)); self.labels.push_for::<R>(label); id
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
    /// when full dominance reaches a nonempty cut list.
    #[inline(always)]
    pub fn dominates_hot<'w>(
        g: &Graph, a: &HotLabel, b: &HotLabel, forward: bool, test: InsertionTest,
        hop_limit: bool, cuts: &[DominanceDescriptor],
        words: impl FnOnce() -> (&'w [u32; LIMITED_WORDS], &'w [u32; LIMITED_WORDS]),
    ) -> bool {
        Self::dominates_hot_slices(g, a, b, forward, test, hop_limit, cuts,
            || { let (a,b) = words(); (&a[..], &b[..]) })
    }
    #[inline(always)]
    pub(crate) fn dominates_hot_slices<'w>(
        g: &Graph,
        a: &HotLabel,
        b: &HotLabel,
        forward: bool,
        test: InsertionTest,
        hop_limit: bool,
        cuts: &[DominanceDescriptor],
        words: impl FnOnce() -> (&'w [u32], &'w [u32]),
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
                if cuts.is_empty() {
                    return true;
                }
                let (wa, wb) = words();
                packed_dominate(cuts, wa, wb, &mut gap)
            }
            InsertionTest::ResourcesAndCost => a.cost - b.cost < EPSILON,
            _ => true,
        }
    }
    /// `HotColumns::next_candidate` for this graph's resources; without a
    /// specialization every index is a candidate. Pass the candidate resources
    /// as scalars instead of reloading a freshly assembled `HotLabel`.
    #[inline]
    fn next_candidate<const PREFIX: bool>(
        g: &Graph,
        cols: &HotColumns,
        from: usize,
        threshold: f64,
        candidate_r0: f64,
        candidate_r1: f64,
        candidate_ng: u16,
        forward: bool,
        ng: bool,
    ) -> usize {
        let disposable = g.resources.iter().all(|k| matches!(k, ResourceKind::Disposable));
        macro_rules! scan {
            ($f:literal, $r:literal, $n:literal) => {
                cols.next_candidate::<PREFIX, $f, $r, $n>(
                    from, threshold, candidate_r0, candidate_r1, candidate_ng,
                )
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
        self.insert_label::<true, StoredLabel, _>(g, label, candidate, forward, test, capacity, hop_limit, cuts, stats)
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
        self.insert_label::<true, StoredLabel, _>(
            g, label, candidate, forward, test, capacity, hop_limit, cuts, stats,
        )
    }
    /// The branch-free specialization is only used by a sweep with no active
    /// branch resources; the public insertion path checks branch state.
    #[inline(always)]
    pub(crate) fn insert_label<const BRANCHING: bool, R: ArenaRecord, L: LabelInput>(
        &mut self,
        g: &Graph,
        input: L,
        candidate: HotLabel,
        forward: bool,
        test: InsertionTest,
        capacity: usize,
        hop_limit: bool,
        cuts: &[DominanceDescriptor],
        stats: &mut Statistics,
    ) -> bool {
        let label = input.label();
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
                if labels.view(previous).expanded() {
                    bucket.archived.push(previous);
                }
            }
            let local_id = bucket.next_id;
            bucket.next_id += 1;
            bucket.active.clear();
            bucket.hot.truncate(0);
            let mut kept_label = input.into_label();
            kept_label.local_id = local_id;
            let id = self.retain_for::<R>(kept_label);
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
                Self::next_candidate::<true>(g, &bucket.hot, position, threshold,
                    candidate.resources[0], candidate.resources[1], candidate.ng, forward, ng);
            stats.dominance_checks += next - position;
            position = next;
            if position == bucket.active.len() || bucket.hot.key[position] > threshold {
                break;
            }
            stats.dominance_checks += 1;
            let previous = bucket.active[position];
            let hot = bucket.hot.get(position);
            if (!BRANCHING || labels.branch_for::<R>(previous) == &label.branch_state) && Self::dominates_hot_slices(g, &hot, &candidate, forward, test, hop_limit, cuts, || {
                (labels.words_for::<R>(previous), &label.cut_words)
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
        let mut kept_label = input.into_label();
            kept_label.local_id = local_id;
            let inserted = self.retain_for::<R>(kept_label);
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
                Self::next_candidate::<false>(g, &bucket.hot, read, 0.,
                    candidate.resources[0], candidate.resources[1], candidate.ng, forward, ng);
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
            if (!BRANCHING || labels.branch_for::<R>(inserted) == labels.branch_for::<R>(id)) && Self::dominates_hot_slices(g, &candidate, &hot, forward, test, hop_limit, cuts, || {
                (labels.words_for::<R>(inserted), labels.words_for::<R>(id))
            }) {
                stats.dominated += 1;
                // archives(): only the resources(+cost) tests keep dominated
                // expanded labels reachable for solution recording.
                if labels.view(id).expanded()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn check_scan<
        const PREFIX: bool,
        const FORWARD: bool,
        const RESOURCES: usize,
        const NG: bool,
    >() {
        // A unique qualifying label at every position exercises short scans,
        // vector block boundaries, and resuming after a previous candidate.
        const LEN: usize = 25;
        for use_ng in [false, true] {
            if use_ng && !NG {
                continue;
            }
            for matching in 0..=LEN {
                let mut columns = HotColumns::default();
                for i in 0..LEN {
                    let failing = if PREFIX == FORWARD { 2. } else { 0. };
                    let resources = if i == matching || use_ng {
                        [1.; 2]
                    } else {
                        [failing; 2]
                    };
                    columns.push(HotLabel {
                        key: 0.,
                        cost: 0.,
                        additional_cost: 0.,
                        resources,
                        hops: 1,
                        ng: if !use_ng || i == matching {
                            1
                        } else if PREFIX {
                            2
                        } else {
                            0
                        },
                    });
                }
                for from in [0, 1, 7, 8, 9, 15, 16, 24, LEN] {
                    let expected = if matching >= from { matching } else { LEN };
                    assert_eq!(
                        columns.next_candidate::<PREFIX, FORWARD, RESOURCES, NG>(from, 1., 1., 1., 1),
                        expected,
                        "prefix={PREFIX} forward={FORWARD} resources={RESOURCES} ng={NG} use_ng={use_ng} matching={matching} from={from}"
                    );
                }
            }
        }
        if PREFIX {
            let mut columns = HotColumns::default();
            for _ in 0..LEN {
                columns.push(HotLabel {
                    key: 2.,
                    cost: 0.,
                    additional_cost: 0.,
                    resources: [3.; 2],
                    hops: 1,
                    ng: 2,
                });
            }
            // A key above the prefix threshold stops the scan independently
            // of resource and NG dominance.
            assert_eq!(
                columns.next_candidate::<PREFIX, FORWARD, RESOURCES, NG>(0, 1., 1., 1., 1),
                0
            );
        }
    }

    #[test]
    fn dominance_scan_preserves_first_candidate_across_block_boundaries() {
        macro_rules! check {
            ($p:literal, $f:literal) => {
                check_scan::<$p, $f, 1, false>();
                check_scan::<$p, $f, 1, true>();
                check_scan::<$p, $f, 2, false>();
                check_scan::<$p, $f, 2, true>();
            };
        }
        check!(true, true);
        check!(true, false);
        check!(false, true);
        check!(false, false);
    }
    #[test]
    fn empty_cut_dominance_does_not_load_cold_words() {
        let g = Graph::new(3, 1).unwrap();
        let mut a = HotLabel { key: 0., cost: -2., additional_cost: 0.,
            resources: [0.; 2], hops: 1, ng: 0 };
        let b = HotLabel { cost: -1., ..a };
        for cost in [-2., -1., -1. + EPSILON, f64::NAN] {
            a.cost = cost;
            let mut gap = (a.cost - b.cost) + (a.additional_cost - b.additional_cost);
            let expected = if gap >= EPSILON { false } else {
                packed_dominate(&[], &[0; LIMITED_WORDS], &[0; LIMITED_WORDS], &mut gap)
            };
            assert_eq!(LabelStorage::dominates_hot(&g, &a, &b, true,
                InsertionTest::Full, false, &[], || panic!("empty cuts accessed cold state")), expected);
        }
        a.cost = -2.;
        let descriptors = [DominanceDescriptor { mask: 1, word: 0,
            cost: 2., ..Default::default() }];
        let mut left = [0; LIMITED_WORDS]; left[0] = 1;
        let right = [0; LIMITED_WORDS];
        let called = std::cell::Cell::new(false);
        assert!(!LabelStorage::dominates_hot(&g, &a, &b, true,
            InsertionTest::Full, false, &descriptors, || { called.set(true); (&left, &right) }));
        assert!(called.get());
    }

    #[test]
    fn owned_insertion_moves_branch_state_and_borrowed_insertion_clones_on_acceptance() {
        let graph = Graph::new(3,1).unwrap();
        let make = || StoredLabel { id:0, local_id:0, bucket:BucketId {vertex:1,bucket:0}, vertex:1,
            cost:-1., original_cost:2., resources:[0.;2], ng:0,
            branch_state:Some(Box::new([5;16])), parent:None, arc:None, hops:1, expanded:false,
            cut_words:[0;LIMITED_WORDS], additional_cost:0., half_potential:0. };
        let mut owned = LabelStorage::new([1,1,1]);
        owned.labels.set_word_limit(2); owned.labels.set_mode(false,false);
        let label = make(); let pointer = label.branch_state.as_deref().unwrap().as_ptr();
        assert!(owned.insert(&graph,label,true,InsertionTest::Full,10,false,&[],&mut Statistics::default()));
        assert_eq!(owned.labels.view(0).branch_state().as_deref().unwrap().as_ptr(), pointer);
        let mut borrowed = LabelStorage::new([1,1,1]);
        borrowed.labels.set_word_limit(2); borrowed.labels.set_mode(false,false);
        let label = make(); let pointer = label.branch_state.as_deref().unwrap().as_ptr();
        assert!(borrowed.insert_ref(&graph,&label,HotLabel::of(&label),true,InsertionTest::Full,10,false,&[],&mut Statistics::default()));
        let state = borrowed.labels.view(0).branch_state();
        assert_eq!(state, &label.branch_state);
        assert_ne!(state.as_deref().unwrap().as_ptr(), pointer);
        assert_eq!(label.branch_state.as_deref().unwrap().as_ptr(), pointer);
    }

}
