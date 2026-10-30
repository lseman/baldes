use super::*;

/// Scratch belongs to one direction's sweep. No labels or search outcomes are
/// cached across calls; only allocation capacity is reused between buckets.
#[derive(Default)]
pub(super) struct FilterScratch {
    retained: Vec<usize>,
    work: Vec<(usize, usize, i32, bool)>,
}

#[derive(Default)]
pub(super) struct SweepScratch<const D: usize> {
    pub(super) filter: FilterScratch,
    pub(super) coupled: crate::graph::coupled::CoupledWorkspace,
    pub(super) labels: Vec<usize>,
    pub(super) pending: Vec<(StoredLabel<D>, HotLabel<D>)>,
    pub(super) terminals: Vec<(usize, f64, usize)>,
}
impl<const D: usize> SweepScratch<D> {
    pub(super) fn reset(&mut self) {
        self.filter.retained.clear();
        self.filter.work.clear();
        self.labels.clear();
        self.pending.clear();
    }
}
impl<const D: usize> std::fmt::Debug for SweepScratch<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SweepScratch")
            .field("label_capacity", &self.labels.capacity())
            .field("pending_capacity", &self.pending.capacity())
            .finish_non_exhaustive()
    }
}
/// One direction's labeling sweep: the shared read-only context, that
/// direction's storage, and private counters merged afterwards. Terminal pool
/// candidates are recorded and replayed in sweep order, so a forward and a
/// backward sweep can run in parallel with the result of running them in turn.
pub(super) struct Sweep<'a, const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    pub(super) ctx: Context<'a, D>,
    pub(super) storage: &'a mut LabelStorage<D>,
    pub(super) statistics: &'a mut Statistics,
    pub(super) extension_checks: &'a mut [i32],
    pub(super) filter_checks: &'a mut [i32],
    pub(super) filter_scratch: &'a mut FilterScratch,
    /// Owned by this direction; scratch never escapes into retained labels.
    pub(super) coupled_workspace: &'a mut crate::graph::coupled::CoupledWorkspace,
    pub(super) label_scratch: &'a mut Vec<usize>,
    pub(super) pending: &'a mut Vec<(StoredLabel<D>, HotLabel<D>)>,
    /// (group, cost, label) of each terminal candidate, in insertion order.
    pub(super) terminals: &'a mut Vec<(usize, f64, usize)>,
    /// This direction's and the other direction's label counts, for the
    /// combined label limit.
    pub(super) labels: (&'a AtomicUsize, &'a AtomicUsize),
}
impl<'a, const D: usize> Sweep<'a, D> {
    #[inline(always)]
    pub(super) fn insert<const BRANCHING: bool, R: ArenaRecord<D>, L: LabelInput<D>>(
        &mut self,
        g: &Graph<D>,
        forward: bool,
        input: L,
        hot: HotLabel<D>,
        controls: &Controls,
    ) -> bool {
        let label = input.label();
        let test = match (
            self.ctx.phase.check_ng,
            self.ctx.controls.dominance_mode == 1,
        ) {
            (true, false) => InsertionTest::Full,
            (true, true) => InsertionTest::ResourcesAndNg,
            (false, false) => InsertionTest::ResourcesAndCost,
            (false, true) => InsertionTest::Resources,
        };
        let capacity = self.ctx.phase.labels_per_bucket as usize;
        let vertex = label.bucket.vertex;
        let before = self.statistics.dominance_checks;
        let descriptors = self
            .ctx
            .cuts
            .info
            .vertices
            .get(label.vertex)
            .map_or(&[][..], |v| v.dominance.as_slice());
        let stats = &mut *self.statistics;
        let storage = &mut *self.storage;
        let inserted = storage.insert_label::<BRANCHING, R, L>(
            g,
            input,
            hot,
            forward,
            test,
            capacity,
            controls.max_route_arcs.is_some(),
            descriptors,
            stats,
        );
        self.extension_checks[vertex] = self.extension_checks[vertex]
            .wrapping_add((self.statistics.dominance_checks - before) as i32);
        inserted
    }
    pub(super) fn filter<const BRANCHING: bool>(
        &mut self,
        g: &Graph<D>,
        forward: bool,
        id: BucketId,
        controls: &Controls,
    ) {
        if !self.ctx.controls.filter_other_buckets {
            return;
        }
        let direction = self.ctx.grid.direction(forward);
        let graph = &direction.buckets[id.vertex][id.bucket];
        let depth = direction.depth[id.vertex];
        if depth == 0 {
            return;
        }
        let component = graph.component;
        // Only unexpanded labels are filtered; the others stay as they are.
        let storage = &*self.storage;
        let bucket = &storage.buckets[id.vertex][id.bucket];
        if bucket
            .active
            .iter()
            .all(|&i| storage.labels.view(i).expanded())
        {
            return;
        }
        let active = &bucket.active;
        let FilterScratch { retained, work } = &mut *self.filter_scratch;
        retained.clear();
        retained.reserve(active.len());
        let before = self.statistics.other_dominance_checks;
        let hop_limit = controls.max_route_arcs.is_some();
        for (position, &candidate) in active.iter().enumerate() {
            let candidate_label = storage.labels.view(candidate);
            if candidate_label.expanded() {
                retained.push(candidate);
                continue;
            }
            let candidate_hot = &bucket.hot.get(position);
            let threshold = EPSILON + candidate_hot.key;
            let descriptors = self
                .ctx
                .cuts
                .info
                .vertices
                .get(candidate_label.vertex())
                .map_or(&[][..], |v| v.dominance.as_slice());
            let candidate_presence = candidate_label.presence_state();
            let candidate_branch = candidate_label.branch_state();
            let candidate_coupled = candidate_label.coupled_state();
            work.clear();
            work.push((id.bucket, 0usize, depth, false));
            let mut dominated = false;
            while let Some((k, min, remaining, check)) = work.pop() {
                self.statistics.other_bucket_searches += 1;
                if check {
                    let bucket = &storage.buckets[id.vertex][k];
                    for (j, &other) in bucket.active.iter().enumerate() {
                        let hot = &bucket.hot.get(j);
                        if hot.key >= threshold {
                            break;
                        }
                        self.statistics.other_dominance_checks += 1;
                        let other_label = storage.labels.view(other);
                        if other_label.presence_state() == candidate_presence
                            && (!BRANCHING || other_label.branch_state() == candidate_branch)
                            && LabelStorage::protected_resources_equal(
                                &storage.protected_resources,
                                hot,
                                candidate_hot,
                            )
                            && LabelStorage::dominates_coupled_slices(
                                g,
                                hot,
                                candidate_hot,
                                forward,
                                candidate_label.vertex(),
                                InsertionTest::Full,
                                hop_limit,
                                descriptors,
                                other_label.coupled_state(),
                                candidate_coupled,
                                || (other_label.cut_words(), candidate_label.cut_words()),
                            )
                        {
                            dominated = true;
                            break;
                        }
                    }
                }
                if dominated {
                    break;
                }
                if remaining == 0 {
                    continue;
                }
                // The predecessors before the first one under `min`, pushed
                // last first so that they are searched in order.
                let predecessors = &direction.buckets[id.vertex][k].predecessors;
                let count = predecessors
                    .iter()
                    .take_while(|p| p.bucket.bucket >= min)
                    .count();
                for p in predecessors[..count].iter().rev() {
                    let previous = &direction.buckets[id.vertex][p.bucket.bucket];
                    let lower = storage.buckets[id.vertex][p.bucket.bucket].lower_cost;
                    if threshold > lower || previous.component == component {
                        work.push((p.bucket.bucket, min.max(p.minimum), remaining - 1, true));
                    }
                }
            }
            if dominated {
                self.statistics.dominated += 1;
                self.statistics.other_dominated += 1;
            } else {
                retained.push(candidate);
            }
        }
        self.filter_checks[id.vertex] = self.filter_checks[id.vertex]
            .wrapping_add((self.statistics.other_dominance_checks - before) as i32);
        if retained.len() == active.len() {
            return;
        }
        let storage = &mut *self.storage;
        let bucket = &mut storage.buckets[id.vertex][id.bucket];
        // Compact the existing hot columns in the same order as the IDs.
        bucket.hot.retain_subsequence(&bucket.active, retained);
        // Hand the filtered vector to the bucket and reuse its old allocation
        // as the next bucket's scratch. Surviving labels keep their order.
        std::mem::swap(&mut bucket.active, retained);
        retained.clear();
    }
    pub(super) fn run_direction(
        &mut self,
        g: &Graph<D>,
        forward: bool,
        controls: &Controls,
        started: Instant,
    ) -> Result<()> {
        // Decide once per sweep: keep direction constant through resource
        // propagation and bucket lookup, and let root dominance stay on the
        // hot columns without loading branch state from full label records.
        match (self.ctx.branching.resources != 0, forward) {
            (false, true) => self.run_direction_with_branching::<false, true>(g, controls, started),
            (false, false) => {
                self.run_direction_with_branching::<false, false>(g, controls, started)
            }
            (true, true) => self.run_direction_with_branching::<true, true>(g, controls, started),
            (true, false) => self.run_direction_with_branching::<true, false>(g, controls, started),
        }
    }
    pub(super) fn run_direction_with_branching<const BRANCHING: bool, const FORWARD: bool>(
        &mut self,
        g: &Graph<D>,
        controls: &Controls,
        started: Instant,
    ) -> Result<()> {
        if self.ctx.has_cuts() {
            self.run_cut_mode::<BRANCHING, FORWARD, true>(g, controls, started)
        } else {
            self.run_cut_mode::<BRANCHING, FORWARD, false>(g, controls, started)
        }
    }
    pub(super) fn run_cut_mode<const BRANCHING: bool, const FORWARD: bool, const CUTS: bool>(
        &mut self,
        g: &Graph<D>,
        controls: &Controls,
        started: Instant,
    ) -> Result<()> {
        match self.storage.labels.kind() {
            RecordKind::Ordinary => {
                self.run_records::<BRANCHING, FORWARD, CUTS, OrdinaryLabel<D>>(g, controls, started)
            }
            RecordKind::Extended => self
                .run_records::<BRANCHING, FORWARD, CUTS, ExtendedLabel<LIMITED_WORDS, D>>(
                    g, controls, started,
                ),
            RecordKind::Branched => {
                self.run_records::<BRANCHING, FORWARD, CUTS, BranchedLabel<D>>(g, controls, started)
            }
            RecordKind::Compact2 => self
                .run_records::<BRANCHING, FORWARD, CUTS, ExtendedLabel<2, D>>(g, controls, started),
            RecordKind::Compact4 => self
                .run_records::<BRANCHING, FORWARD, CUTS, ExtendedLabel<4, D>>(g, controls, started),
            RecordKind::Compact8 => self
                .run_records::<BRANCHING, FORWARD, CUTS, ExtendedLabel<8, D>>(g, controls, started),
            RecordKind::Auxiliary2 => self
                .run_records::<BRANCHING, FORWARD, CUTS, AuxiliaryLabel<2, D>>(
                    g, controls, started,
                ),
            RecordKind::Auxiliary4 => self
                .run_records::<BRANCHING, FORWARD, CUTS, AuxiliaryLabel<4, D>>(
                    g, controls, started,
                ),
            RecordKind::Auxiliary8 => self
                .run_records::<BRANCHING, FORWARD, CUTS, AuxiliaryLabel<8, D>>(
                    g, controls, started,
                ),
            RecordKind::AuxiliaryFull => self
                .run_records::<BRANCHING, FORWARD, CUTS, AuxiliaryLabel<LIMITED_WORDS, D>>(
                    g, controls, started,
                ),
            RecordKind::Full => {
                self.run_records::<BRANCHING, FORWARD, CUTS, StoredLabel<D>>(g, controls, started)
            }
        }
    }
    pub(super) fn run_records<
        const BRANCHING: bool,
        const FORWARD: bool,
        const CUTS: bool,
        R: ArenaRecord<D>,
    >(
        &mut self,
        g: &Graph<D>,
        controls: &Controls,
        started: Instant,
    ) -> Result<()> {
        let forward = FORWARD;
        let ctx = self.ctx;
        let components = &ctx.grid.direction(forward).components;
        let flags = if self.ctx.runtime_bidirectional() {
            self.ctx.controls.direction_flags
        } else {
            3
        };
        // Reading the clock per extension cost ~15% of pricing; sample it.
        let mut clock = 0u32;
        // Keep the snapshot and deferred same-bucket insertions in sweep-local
        // scratch so successive buckets reuse their allocations. The snapshot
        // preserves the expansion order while insertions mutate storage.
        let prefilter_labels = controls.max_labels == usize::MAX;
        for component in components {
            loop {
                let mut repeat = false;
                for &id in component {
                    self.filter::<BRANCHING>(g, forward, id, controls);
                    // Empty buckets have no extensions or labels to expand.
                    // Keep filtering above and component lower bounds below.
                    if self.storage.buckets[id.vertex][id.bucket].active.is_empty() {
                        continue;
                    }
                    // The bucket graph is fixed during the sweep: copy each arc
                    // as it is used instead of cloning the bucket's arc list.
                    let arc_count = self.ctx.grid.direction(forward).buckets[id.vertex][id.bucket]
                        .arcs
                        .len();
                    // Same-bucket extensions go to `pending`, so this bucket's
                    // active list is fixed for the whole arc loop.
                    self.label_scratch.clear();
                    if prefilter_labels {
                        // Eligibility cannot change during the arc loop: same-bucket
                        // insertions are deferred and expansion happens afterwards.
                        // Filter once instead of testing every label on every arc.
                        for &index in &self.storage.buckets[id.vertex][id.bucket].active {
                            let eligible = if let Some(from) = R::get(&self.storage.labels, index) {
                                !from.expanded()
                                    && controls
                                        .max_route_arcs
                                        .is_none_or(|limit| from.hops() < limit)
                            } else {
                                let from = self.storage.labels.wide_record(index);
                                !from.expanded
                                    && controls
                                        .max_route_arcs
                                        .is_none_or(|limit| from.hops < limit)
                            };
                            if eligible {
                                self.label_scratch.push(index);
                            }
                        }
                    } else {
                        // Preserve per-record budget checks before eligibility.
                        self.label_scratch
                            .extend_from_slice(&self.storage.buckets[id.vertex][id.bucket].active);
                    }
                    for k in 0..arc_count {
                        let arc =
                            self.ctx.grid.direction(forward).buckets[id.vertex][id.bucket].arcs[k];
                        if (!self.ctx.phase.jump_arcs && arc.jump.is_some())
                            || arc.flags & flags == 0
                        {
                            continue;
                        }
                        for position in 0..self.label_scratch.len() {
                            let index = self.label_scratch[position];
                            clock = clock.wrapping_add(1);
                            if clock % 64 == 0 && started.elapsed() >= controls.time_limit {
                                return Err(incomplete("pricing hard time limit exceeded"));
                            }
                            // usize::MAX denotes an unlimited label budget. In that
                            // case neither direction needs to publish or read counts.
                            if !prefilter_labels {
                                let own = self.storage.labels.len();
                                self.labels.0.store(own, Ordering::Relaxed);
                                if own + self.labels.1.load(Ordering::Relaxed)
                                    >= controls.max_labels
                                {
                                    return Err(incomplete("pricing label limit exceeded"));
                                }
                            }
                            let (from_vertex, extension) = if let Some(from) =
                                R::get(&self.storage.labels, index)
                            {
                                if !prefilter_labels
                                    && (from.expanded()
                                        || controls
                                            .max_route_arcs
                                            .is_some_and(|limit| from.hops() >= limit))
                                {
                                    continue;
                                }
                                (
                                    from.vertex(),
                                    self.ctx.extend_record_with_workspace::<BRANCHING, CUTS, _>(
                                        g,
                                        forward,
                                        &arc,
                                        from,
                                        from.ancestry_id(index),
                                        &mut self.coupled_workspace,
                                    ),
                                )
                            } else {
                                let from = self.storage.labels.wide_record(index);
                                if !prefilter_labels
                                    && (from.expanded
                                        || controls
                                            .max_route_arcs
                                            .is_some_and(|limit| from.hops >= limit))
                                {
                                    continue;
                                }
                                (
                                    from.vertex,
                                    self.ctx.extend_record_with_workspace::<BRANCHING, CUTS, _>(
                                        g,
                                        forward,
                                        &arc,
                                        from,
                                        from.id,
                                        &mut self.coupled_workspace,
                                    ),
                                )
                            };
                            let Some((mut label, resources)) = extension else {
                                continue;
                            };
                            if self.ctx.load_completion.is_some_and(|bound| {
                                bound.rejects(
                                    &label,
                                    forward,
                                    self.ctx.controls.load_completion_threshold,
                                )
                            }) {
                                self.statistics.completion_pruned += 1;
                                continue;
                            }
                            let terminal = label.vertex == if forward { g.sink } else { g.source };
                            if terminal {
                                if BRANCHING && !self.ctx.branching.accepts(&label.branch_state) {
                                    continue;
                                }
                                if !self.ctx.controls.sink_concatenation {
                                    continue;
                                }
                                let cost = self.ctx.terminal_cost(g, forward, &label);
                                let group = from_vertex;
                                let target = self.storage.retain_for::<R>(label);
                                self.terminals.push((group, cost, target));
                                continue;
                            }
                            let q = self.ctx.grid.coordinate(&resources);
                            let side = if forward {
                                if q[0] > self.ctx.grid.middle + EPSILON {
                                    2
                                } else {
                                    1
                                }
                            } else if q[0] <= self.ctx.grid.middle + EPSILON {
                                2
                            } else {
                                1
                            };
                            if flags & side == 0 {
                                if forward && flags & 2 == 0 {
                                    let v = label.vertex;
                                    let target = self.storage.retain_for::<R>(label);
                                    self.storage.crossing[v].push(target);
                                }
                                continue;
                            }
                            self.ctx.vertex_step_with_branching::<BRANCHING>(&mut label);
                            let hot = HotLabel {
                                resources: resources.0,
                                ..HotLabel::of(&label)
                            };
                            if label.bucket == id {
                                self.pending.push((label, hot));
                                continue;
                            }
                            let destination = label.bucket;
                            let inserted = if BRANCHING || R::OWNS_AUXILIARY {
                                self.insert::<BRANCHING, R, _>(g, forward, label, hot, controls)
                            } else {
                                self.insert::<BRANCHING, R, _>(g, forward, &label, hot, controls)
                            };
                            // Only accepted labels can require another component pass.
                            if inserted
                                && self.ctx.grid.direction(forward).buckets[destination.vertex]
                                    [destination.bucket]
                                    .component
                                    == self.ctx.grid.direction(forward).buckets[id.vertex]
                                        [id.bucket]
                                        .component
                            {
                                repeat = true;
                            }
                        }
                    }
                    self.statistics.expanded += self
                        .storage
                        .labels
                        .expand(&self.storage.buckets[id.vertex][id.bucket].active);
                    let mut pending = std::mem::take(self.pending);
                    for (label, hot) in pending.drain(..) {
                        let inserted = if BRANCHING || R::OWNS_AUXILIARY {
                            self.insert::<BRANCHING, R, _>(g, forward, label, hot, controls)
                        } else {
                            self.insert::<BRANCHING, R, _>(g, forward, &label, hot, controls)
                        };
                        if inserted {
                            repeat = true;
                        }
                    }
                    *self.pending = pending;
                }
                if !repeat {
                    break;
                }
            }
            for id in component {
                let b = &self.storage.buckets[id.vertex][id.bucket];
                let mut lower = b.active.first().map_or(NO_ROUTE_COST, |_| b.hot.key[0]);
                for p in
                    &self.ctx.grid.direction(forward).buckets[id.vertex][id.bucket].predecessors
                {
                    lower = lower.min(self.storage.buckets[id.vertex][p.bucket.bucket].lower_cost);
                }
                self.storage.buckets[id.vertex][id.bucket].lower_cost = lower;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/labeling/engine/sweep/presence_tests.rs"]
mod presence_tests;
