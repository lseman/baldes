use super::*;

impl<const D: usize> LabelStorage<D> {
    pub fn insert(
        &mut self,
        g: &Graph<D>,
        label: StoredLabel<D>,
        forward: bool,
        test: InsertionTest,
        capacity: usize,
        hop_limit: bool,
        cuts: &[DominanceDescriptor],
        stats: &mut Statistics,
    ) -> bool {
        let candidate = HotLabel::of(&label);
        self.insert_label::<true, StoredLabel<D>, _>(
            g, label, candidate, forward, test, capacity, hop_limit, cuts, stats,
        )
    }

    /// `insert` that copies `label` into the arena only when it is kept.
    /// `candidate` is `HotLabel::of(label)`, built by the caller while the
    /// fields are still in registers: reading them back from a label just
    /// written stalls on store forwarding. Inlined for the same reason: passed
    /// through memory, `candidate` was reloaded as 16-byte pairs.
    #[inline(always)]
    pub fn insert_ref(
        &mut self,
        g: &Graph<D>,
        label: &StoredLabel<D>,
        candidate: HotLabel<D>,
        forward: bool,
        test: InsertionTest,
        capacity: usize,
        hop_limit: bool,
        cuts: &[DominanceDescriptor],
        stats: &mut Statistics,
    ) -> bool {
        self.insert_label::<true, StoredLabel<D>, _>(
            g, label, candidate, forward, test, capacity, hop_limit, cuts, stats,
        )
    }

    /// The branch-free specialization is only used by a sweep with no active
    /// branch resources; the public insertion path checks branch state.
    #[inline(always)]
    pub(crate) fn insert_label<const BRANCHING: bool, R: ArenaRecord<D>, L: LabelInput<D>>(
        &mut self,
        g: &Graph<D>,
        input: L,
        candidate: HotLabel<D>,
        forward: bool,
        test: InsertionTest,
        capacity: usize,
        hop_limit: bool,
        cuts: &[DominanceDescriptor],
        stats: &mut Statistics,
    ) -> bool {
        let label = input.label();
        let b = label.bucket;
        // Bucket IDs use the internal ordering; continuation metadata uses
        // physical graph vertices, retained independently by the label.
        let physical_vertex = label.vertex;
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
            let id = input.retain_for::<R>(self, local_id);
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
            let next = Self::next_candidate::<true>(
                g,
                &bucket.hot,
                position,
                threshold,
                candidate.resources.first().copied().unwrap_or(0.),
                candidate.resources.get(1).copied().unwrap_or(0.),
                candidate.ng,
                forward,
                ng,
            );
            stats.dominance_checks += next - position;
            position = next;
            if position == bucket.active.len() || bucket.hot.key[position] > threshold {
                break;
            }
            stats.dominance_checks += 1;
            let previous = bucket.active[position];
            let hot = bucket.hot.get(position);
            if labels.presence_for::<R>(previous) == &label.presence_state
                && (!BRANCHING || labels.branch_for::<R>(previous) == &label.branch_state)
                && Self::protected_resources_equal(&self.protected_resources, &hot, &candidate)
                && Self::dominates_coupled_slices(
                    g,
                    &hot,
                    &candidate,
                    forward,
                    physical_vertex,
                    test,
                    hop_limit,
                    cuts,
                    labels.view(previous).coupled_state(),
                    label.coupled_state.as_deref(),
                    || (labels.words_for::<R>(previous), &label.cut_words),
                )
            {
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
        let inserted = input.retain_for::<R>(self, local_id);
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
            let next = Self::next_candidate::<false>(
                g,
                &bucket.hot,
                read,
                0.,
                candidate.resources.first().copied().unwrap_or(0.),
                candidate.resources.get(1).copied().unwrap_or(0.),
                candidate.ng,
                forward,
                ng,
            );
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
            if labels.presence_for::<R>(inserted) == labels.presence_for::<R>(id)
                && (!BRANCHING || labels.branch_for::<R>(inserted) == labels.branch_for::<R>(id))
                && Self::protected_resources_equal(&self.protected_resources, &candidate, &hot)
                && Self::dominates_coupled_slices(
                    g,
                    &candidate,
                    &hot,
                    forward,
                    physical_vertex,
                    test,
                    hop_limit,
                    cuts,
                    labels.view(inserted).coupled_state(),
                    labels.view(id).coupled_state(),
                    || (labels.words_for::<R>(inserted), labels.words_for::<R>(id)),
                )
            {
                stats.dominated += 1;
                // archives(): only the resources(+cost) tests keep dominated
                // expanded labels reachable for solution recording.
                if labels.view(id).expanded()
                    && matches!(
                        test,
                        InsertionTest::ResourcesAndCost | InsertionTest::Resources
                    )
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
