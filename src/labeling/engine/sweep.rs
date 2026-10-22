use super::*;
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
        let graph = &self.ctx.grid.direction(forward).buckets[id.vertex][id.bucket];
        let depth = self.ctx.grid.direction(forward).depth[id.vertex];
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
        let mut retained = Vec::with_capacity(active.len());
        let before = self.statistics.other_dominance_checks;
        let mut work = vec![];
        for &candidate in active {
            if self.storage.labels.view(candidate).expanded() {
                retained.push(candidate);
                continue;
            }
            let threshold = EPSILON + self.storage.hot[candidate].key;
            work.clear();
            work.push((id.bucket, 0usize, depth, false));
            let mut dominated = false;
            while let Some((k, min, remaining, check)) = work.pop() {
                self.statistics.other_bucket_searches += 1;
                if check {
                    let storage = &*self.storage;
                    let candidate_label = storage.labels.view(candidate);
                    let descriptors = self
                        .ctx
                        .cuts
                        .info
                        .vertices
                        .get(candidate_label.vertex())
                        .map_or(&[][..], |v| v.dominance.as_slice());
                    let candidate_hot = &storage.hot[candidate];
                    let bucket = &storage.buckets[id.vertex][k];
                    for (j, &other) in bucket.active.iter().enumerate() {
                        let hot = &bucket.hot.get(j);
                        if hot.key >= threshold {
                            break;
                        }
                        self.statistics.other_dominance_checks += 1;
                        if storage.labels.view(other).presence_state()
                            == candidate_label.presence_state()
                            && (!BRANCHING
                                || storage.labels.view(other).branch_state()
                                    == candidate_label.branch_state())
                            && LabelStorage::dominates_hot_slices(
                                g,
                                hot,
                                candidate_hot,
                                forward,
                                InsertionTest::Full,
                                controls.max_route_arcs.is_some(),
                                descriptors,
                                || {
                                    (
                                        storage.labels.view(other).cut_words(),
                                        candidate_label.cut_words(),
                                    )
                                },
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
                let predecessors =
                    &self.ctx.grid.direction(forward).buckets[id.vertex][k].predecessors;
                let count = predecessors
                    .iter()
                    .take_while(|p| p.bucket.bucket >= min)
                    .count();
                for p in predecessors[..count].iter().rev() {
                    let previous =
                        &self.ctx.grid.direction(forward).buckets[id.vertex][p.bucket.bucket];
                    let lower = self.storage.buckets[id.vertex][p.bucket.bucket].lower_cost;
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
        // Rebuild in retained order without discarding the column capacities.
        bucket.hot.truncate(0);
        for &i in &retained {
            bucket.hot.push(storage.hot[i]);
        }
        bucket.active = retained;
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
        let mut labels = Vec::new();
        let mut pending = Vec::new();
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
                    labels.clear();
                    labels.extend_from_slice(&self.storage.buckets[id.vertex][id.bucket].active);
                    for k in 0..arc_count {
                        let arc = self.ctx.grid.direction(forward).buckets[id.vertex][id.bucket]
                            .arcs[k]
                            .clone();
                        if (!self.ctx.phase.jump_arcs && arc.jump.is_some())
                            || arc.flags & flags == 0
                        {
                            continue;
                        }
                        for &index in &labels {
                            clock = clock.wrapping_add(1);
                            if clock % 64 == 0 && started.elapsed() >= controls.time_limit {
                                return Err(incomplete("pricing hard time limit exceeded"));
                            }
                            // usize::MAX denotes an unlimited label budget. In that
                            // case neither direction needs to publish or read counts.
                            if controls.max_labels != usize::MAX {
                                let own = self.storage.labels.len();
                                self.labels.0.store(own, Ordering::Relaxed);
                                if own + self.labels.1.load(Ordering::Relaxed)
                                    >= controls.max_labels
                                {
                                    return Err(incomplete("pricing label limit exceeded"));
                                }
                            }
                            let (from_vertex, extension) =
                                if let Some(from) = R::get(&self.storage.labels, index) {
                                    if from.expanded()
                                        || controls
                                            .max_route_arcs
                                            .is_some_and(|limit| from.hops() >= limit)
                                    {
                                        continue;
                                    }
                                    (
                                        from.vertex(),
                                        self.ctx.extend_record_with_cuts::<BRANCHING, CUTS, _>(
                                            g,
                                            forward,
                                            &arc,
                                            from,
                                            from.ancestry_id(index),
                                        ),
                                    )
                                } else {
                                    let from = self.storage.labels.wide_record(index);
                                    if from.expanded
                                        || controls
                                            .max_route_arcs
                                            .is_some_and(|limit| from.hops >= limit)
                                    {
                                        continue;
                                    }
                                    (
                                        from.vertex,
                                        self.ctx.extend_record_with_cuts::<BRANCHING, CUTS, _>(
                                            g, forward, &arc, from, from.id,
                                        ),
                                    )
                                };
                            let Some((mut label, resources)) = extension else {
                                continue;
                            };
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
                                pending.push((label, hot));
                                continue;
                            }
                            let same = self.ctx.grid.direction(forward).buckets
                                [label.bucket.vertex][label.bucket.bucket]
                                .component
                                == self.ctx.grid.direction(forward).buckets[id.vertex][id.bucket]
                                    .component;
                            let inserted = if BRANCHING {
                                self.insert::<BRANCHING, R, _>(g, forward, label, hot, controls)
                            } else {
                                self.insert::<BRANCHING, R, _>(g, forward, &label, hot, controls)
                            };
                            if inserted && same {
                                repeat = true;
                            }
                        }
                    }
                    self.statistics.expanded += self
                        .storage
                        .labels
                        .expand(&self.storage.buckets[id.vertex][id.bucket].active);
                    for (label, hot) in pending.drain(..) {
                        let inserted = if BRANCHING {
                            self.insert::<BRANCHING, R, _>(g, forward, label, hot, controls)
                        } else {
                            self.insert::<BRANCHING, R, _>(g, forward, &label, hot, controls)
                        };
                        if inserted {
                            repeat = true;
                        }
                    }
                }
                if !repeat {
                    break;
                }
            }
            for id in component {
                let b = &self.storage.buckets[id.vertex][id.bucket];
                let mut lower = b
                    .active
                    .first()
                    .map_or(NO_ROUTE_COST, |&i| self.storage.hot[i].key);
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
