//! Component-ordered forward/backward native no-cut labeling.
use super::pool::SolutionPool;
use super::storage::{HotLabel, InsertionTest, LabelStorage, StoredLabel};
use super::{extend_resources, Controls, PricingResult, Route, Statistics, EPSILON, NO_ROUTE_COST};
use crate::cuts::state::{
    self, CutResourceInfo, FullPreparation, LimitedInput, LimitedKPathInput, LimitedPreparation,
    LIMITED_WORDS,
};
use crate::graph::buckets::{BucketArc, BucketGraph, BucketId};
use crate::graph::ng::PreparedNg;
use crate::graph::{Graph, ResourceKind};
use crate::pricing::phases::Phase;
use crate::{incomplete, invalid, Result};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
#[derive(Clone, Debug)]
pub struct EngineControls {
    pub bidirectional: bool,
    pub bidirectional_mode: i32,
    pub dominance_mode: i32,
    pub steps: [f64; 2],
    pub target_buckets: i32,
    pub group_limit: i32,
    pub filter_other_buckets: bool,
    pub direction_flags: u8,
    pub sink_concatenation: bool,
    pub use_symmetric_execution: bool,
}
impl Default for EngineControls {
    fn default() -> Self {
        Self {
            bidirectional: true,
            bidirectional_mode: 2,
            dominance_mode: 0,
            steps: [0.; 2],
            target_buckets: 100,
            group_limit: 0,
            filter_other_buckets: true,
            direction_flags: 1,
            sink_concatenation: true,
            use_symmetric_execution: true,
        }
    }
}
/// Limited-memory cut state of the extended labels. Empty inputs select the
/// standard labels: no words, zero additional cost, no descriptors.
#[derive(Clone, Debug, Default)]
pub struct EngineCuts {
    pub inputs: Vec<LimitedInput>,
    pub kpath: Vec<LimitedKPathInput>,
    pub knapsack: crate::cuts::knapsack::KnapsackPricing,
    pub info: CutResourceInfo,
    pub limited: LimitedPreparation,
    /// Full-memory enumeration tables, prepared before an enumeration.
    pub full: Option<FullPreparation>,
}
#[derive(Debug)]
pub struct LabelingEngine {
    pub grid: BucketGraph,
    pub forward: LabelStorage,
    pub backward: LabelStorage,
    pub statistics: Statistics,
    pub controls: EngineControls,
    pub phase: Phase,
    pub ng: PreparedNg,
    pub half_ready: bool,
    pub full_ready: bool,
    last_costs: Vec<f64>,
    pending_middle: Option<bool>,
    pub consumption_joins: bool,
    pub reverse: Vec<usize>,
    extension_checks: Vec<i32>,
    filter_checks: Vec<i32>,
    pub cuts: EngineCuts,
}
/// Replays a sweep's terminal candidates into the pool, in recorded order.
fn insert_terminals(
    pool: &mut SolutionPool<(Option<usize>, Option<usize>)>,
    forward: bool,
    terminals: &[(usize, f64, usize)],
) {
    for &(group, cost, target) in terminals {
        pool.insert(group, cost, 0., if forward { (Some(target), None) } else { (None, Some(target)) });
    }
}
/// The read-only engine state a labeling sweep reads; both directions can
/// share it across threads.
#[derive(Clone, Copy)]
struct Context<'a> {
    grid: &'a BucketGraph,
    ng: &'a PreparedNg,
    cuts: &'a EngineCuts,
    controls: &'a EngineControls,
    phase: Phase,
    consumption_joins: bool,
}
impl<'a> Context<'a> {
    fn has_cuts(&self) -> bool {
        !self.cuts.inputs.is_empty() || !self.cuts.kpath.is_empty()
    }
    fn runtime_bidirectional(&self) -> bool {
        self.controls.bidirectional
            && (self.controls.bidirectional_mode == 1
                || (self.controls.bidirectional_mode == 2 && self.phase.exact()))
    }
    /// The meeting vertex's NG bits and cut supports.
    fn vertex_step(&self, label: &mut StoredLabel) {
        label.ng |= self.ng.vertex_bits[label.vertex];
        if let Some(v) = self.cuts.info.vertices.get(label.vertex) {
            state::apply_vertex_support(&mut label.cut_words, &mut label.additional_cost, &v.support);
        }
    }
    /// Cut state of a root label at its terminal vertex.
    fn root_cut_state(&self, vertex: usize) -> ([u32; LIMITED_WORDS], f64) {
        let mut words = [0; LIMITED_WORDS];
        let mut additional = 0.;
        if let Some(v) = self.cuts.info.vertices.get(vertex) {
            state::apply_vertex_support(&mut words, &mut additional, &v.support);
        }
        (words, additional)
    }
    /// Join of two labels at `vertex`: resources, then the
    /// cost (forward + backward reduced cost) + 0, then NG, then the cut join.
    /// The cost is 0 when resources reject and has no cut part when NG does.
    fn concatenate(&self, g: &Graph, vertex: usize, f: &StoredLabel, b: &StoredLabel) -> (bool, f64) {
        if !self.resources_join(g, f, b) {
            return (false, 0.);
        }
        let cost = (f.cost + b.cost) + self.cuts.knapsack.join_cost(g, &f.resources, &b.resources, self.consumption_joins);
        if f.ng & b.ng != 0 {
            return (false, cost);
        }
        if !self.has_cuts() {
            return (true, cost);
        }
        let cost = state::concatenation_cost(
            &self.cuts.info,
            vertex,
            (&f.cut_words, f.additional_cost),
            (&b.cut_words, b.additional_cost),
            cost,
        );
        (true, cost)
    }
    pub fn join_cost(&self, g: &Graph, vertex: usize, f: &StoredLabel, b: &StoredLabel) -> Option<f64> {
        match self.concatenate(g, vertex, f, b) {
            (true, cost) => Some(cost),
            _ => None,
        }
    }
    /// Join of a label reaching its end vertex with the opposite root (forward:
    /// the backward root, or the forward one for symmetric joins; backward: the
    /// forward root at the source). The join's result is not tested.
    fn terminal_cost(&self, g: &Graph, forward: bool, label: &StoredLabel) -> f64 {
        if !self.has_cuts() && !self.cuts.knapsack.active {
            return label.cost;
        }
        let root_forward = !forward || self.consumption_joins;
        let root_vertex = if root_forward { g.source } else { g.sink };
        let mut root = label.clone();
        root.vertex = root_vertex;
        root.cost = 0.;
        root.ng = self.ng.vertex_bits[root_vertex];
        for (r, _) in g.resources.iter().enumerate() {
            let w = g.vertices[root_vertex].bounds[r];
            root.resources[r] = if root_forward {
                g.initial[r].unwrap_or(w.lower)
            } else {
                w.upper
            };
        }
        (root.cut_words, root.additional_cost) = self.root_cut_state(root_vertex);
        if forward {
            self.concatenate(g, g.sink, label, &root).1
        } else {
            self.concatenate(g, g.source, &root, label).1
        }
    }
    #[inline(always)]
    fn extend(
        &self,
        g: &Graph,
        forward: bool,
        arc: &BucketArc,
        from: &StoredLabel,
    ) -> Option<StoredLabel> {
        self.extend_resources_kept(g, forward, arc, from).map(|(label, _)| label)
    }
    /// `extend`, also returning the label's resources as a value: reading
    /// them back from the new label (two 8-byte stores, one 16-byte load)
    /// stalls on store forwarding.
    #[inline(always)]
    fn extend_resources_kept(
        &self,
        g: &Graph,
        forward: bool,
        arc: &BucketArc,
        from: &StoredLabel,
    ) -> Option<(StoredLabel, [f64; 2])> {
        let physical = &g.arcs[arc.physical];
        if physical.reduced_cost >= NO_ROUTE_COST {
            return None;
        }
        // Scalars, not `resources[slot]`: see BucketGraph::coordinate.
        let [mut q0, mut q1] = from.resources;
        if self.phase.jump_arcs {
            if let Some(jump) = arc.jump {
                let point = self.grid.direction(forward).corners[arc.from.vertex][jump];
                for (r, &slot) in self.grid.resource_slots.iter().enumerate() {
                    let q = if slot == 0 { &mut q0 } else { &mut q1 };
                    *q = if forward { q.max(point[r]) } else { q.min(point[r]) };
                }
            }
        }
        let resources = extend_resources(g, physical, [q0, q1], forward)?;
        let vertex = if forward {
            physical.head
        } else {
            physical.tail
        };
        let transition = if forward {
            &self.ng.forward[arc.physical]
        } else {
            &self.ng.backward[arc.physical]
        };
        let ng = transition.extend(from.ng, 0)?;
        let internal = self.grid.internal[vertex];
        let bucket = self
            .grid
            .bucket_id(forward, internal, self.grid.coordinate(resources))?;
        let mut cut_words = from.cut_words;
        let mut additional_cost = from.additional_cost;
        if self.has_cuts() {
            let table = if forward {
                &self.cuts.info.forward
            } else {
                &self.cuts.info.backward
            };
            state::change_along_arc(&mut cut_words, &mut additional_cost, &table[arc.physical].ordinary);
        }
        Some((StoredLabel {
            id: 0,
            local_id: 0,
            bucket: BucketId {
                vertex: internal,
                bucket,
            },
            vertex,
            cost: physical.reduced_cost + from.cost,
            original_cost: physical.cost + from.original_cost,
            resources,
            ng,
            parent: Some(from.id),
            arc: Some(arc.physical),
            hops: from.hops + 1,
            expanded: false,
            cut_words,
            additional_cost,
            half_potential: self
                .cuts
                .info
                .vertex_cut_potential
                .get(vertex)
                .map_or(0.0, |&p| 0.5 * p),
        }, resources))
    }
    fn resources_join(&self, g: &Graph, f: &StoredLabel, b: &StoredLabel) -> bool {
        g.resources.iter().enumerate().all(|(r, k)| match k {
                ResourceKind::Disposable => {
                    if self.consumption_joins {
                        f.resources[r] + b.resources[r]
                            <= g.vertices[g.sink].bounds[r].upper + EPSILON
                    } else {
                        f.resources[r] <= b.resources[r] + EPSILON
                    }
                }
                ResourceKind::Equality => f.resources[r] == b.resources[r],
            })
    }
}

/// One direction's labeling sweep: the shared read-only context, that
/// direction's storage, and private counters merged afterwards. Terminal pool
/// candidates are recorded and replayed in sweep order, so a forward and a
/// backward sweep can run in parallel with the result of running them in turn.
struct Sweep<'a> {
    ctx: Context<'a>,
    storage: &'a mut LabelStorage,
    statistics: &'a mut Statistics,
    extension_checks: &'a mut [i32],
    filter_checks: &'a mut [i32],
    /// (group, cost, label) of each terminal candidate, in insertion order.
    terminals: &'a mut Vec<(usize, f64, usize)>,
    /// This direction's and the other direction's label counts, for the
    /// combined label limit.
    labels: (&'a AtomicUsize, &'a AtomicUsize),
}
impl<'a> Sweep<'a> {
    #[inline(always)]
    fn insert(
        &mut self,
        g: &Graph,
        forward: bool,
        label: &StoredLabel,
        hot: HotLabel,
        controls: &Controls,
    ) -> bool {
        let test = if self.ctx.phase.check_ng && self.ctx.controls.dominance_mode == 0 {
            InsertionTest::Full
        } else {
            InsertionTest::ResourcesAndCost
        };
        let capacity = self.ctx.phase.labels_per_bucket as usize;
        let vertex = label.bucket.vertex;
        let before = self.statistics.dominance_checks;
        let descriptors = self
            .ctx.cuts
            .info
            .vertices
            .get(label.vertex)
            .map_or(&[][..], |v| v.dominance.as_slice());
        let stats = &mut *self.statistics;
        let storage = &mut *self.storage;
        let inserted = storage.insert_ref(
            g,
            label,
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
    fn filter(&mut self, g: &Graph, forward: bool, id: BucketId, controls: &Controls) {
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
        if bucket.active.iter().all(|&i| storage.labels[i].expanded) {
            return;
        }
        let active = bucket.active.clone();
        let mut retained = Vec::with_capacity(active.len());
        let before = self.statistics.other_dominance_checks;
        let mut work = vec![];
        for &candidate in &active {
            if self.storage.labels[candidate].expanded {
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
                    let candidate_label = &storage.labels[candidate];
                    let descriptors = self
                        .ctx.cuts
                        .info
                        .vertices
                        .get(candidate_label.vertex)
                        .map_or(&[][..], |v| v.dominance.as_slice());
                    let candidate_hot = &storage.hot[candidate];
                    let bucket = &storage.buckets[id.vertex][k];
                    for (j, &other) in bucket.active.iter().enumerate() {
                        let hot = &bucket.hot.get(j);
                        if hot.key >= threshold {
                            break;
                        }
                        self.statistics.other_dominance_checks += 1;
                        if LabelStorage::dominates_hot(
                            g,
                            hot,
                            candidate_hot,
                            forward,
                            InsertionTest::Full,
                            controls.max_route_arcs.is_some(),
                            descriptors,
                            || (&storage.labels[other].cut_words, &candidate_label.cut_words),
                        ) {
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
                let predecessors = &self.ctx.grid.direction(forward).buckets[id.vertex][k].predecessors;
                let count = predecessors.iter().take_while(|p| p.bucket.bucket >= min).count();
                for p in predecessors[..count].iter().rev() {
                    let previous =
                        &self.ctx.grid.direction(forward).buckets[id.vertex][p.bucket.bucket];
                    let lower =
                        self.storage.buckets[id.vertex][p.bucket.bucket].lower_cost;
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
        let mut hot = super::storage::HotColumns::default();
        for &i in &retained {
            hot.push(storage.hot[i]);
        }
        let bucket = &mut storage.buckets[id.vertex][id.bucket];
        bucket.active = retained;
        bucket.hot = hot;
    }
    fn run_direction(
        &mut self,
        g: &Graph,
        forward: bool,
        controls: &Controls,
        started: Instant,
    ) -> Result<()> {
        let components = self.ctx.grid.direction(forward).components.clone();
        let flags = if self.ctx.runtime_bidirectional() {
            self.ctx.controls.direction_flags
        } else {
            3
        };
        // Reading the clock per extension cost ~15% of pricing; sample it.
        let mut clock = 0u32;
        for component in components {
            loop {
                let mut repeat = false;
                for &id in &component {
                    self.filter(g, forward, id, controls);
                    // The bucket graph is fixed during the sweep: copy each arc
                    // as it is used instead of cloning the bucket's arc list.
                    let arc_count = self.ctx.grid.direction(forward).buckets[id.vertex][id.bucket]
                        .arcs
                        .len();
                    let mut pending = vec![];
                    // Same-bucket extensions go to `pending`, so this bucket's
                    // active list is fixed for the whole arc loop.
                    let labels = self.storage.buckets[id.vertex][id.bucket]
                        .active
                        .clone();
                    for k in 0..arc_count {
                        let arc = self.ctx.grid.direction(forward).buckets[id.vertex][id.bucket].arcs[k].clone();
                        if (!self.ctx.phase.jump_arcs && arc.jump.is_some()) || arc.flags & flags == 0 {
                            continue;
                        }
                        for &index in &labels {
                            clock = clock.wrapping_add(1);
                            if clock % 64 == 0 && started.elapsed() >= controls.time_limit {
                                return Err(incomplete("pricing hard time limit exceeded"));
                            }
                            let own = self.storage.labels.len();
                            self.labels.0.store(own, Ordering::Relaxed);
                            if own + self.labels.1.load(Ordering::Relaxed) >= controls.max_labels {
                                return Err(incomplete("pricing label limit exceeded"));
                            }
                            let from = &self.storage.labels[index];
                            if from.expanded
                                || controls
                                    .max_route_arcs
                                    .is_some_and(|limit| from.hops >= limit)
                            {
                                continue;
                            }
                            let Some((mut label, resources)) =
                                self.ctx.extend_resources_kept(g, forward, &arc, from)
                            else {
                                continue;
                            };
                            let terminal = label.vertex == if forward { g.sink } else { g.source };
                            if terminal {
                                if !self.ctx.controls.sink_concatenation {
                                    continue;
                                }
                                let cost = self.ctx.terminal_cost(g, forward, &label);
                                let group = from.vertex;
                                let target = self.storage.retain(label);
                                self.terminals.push((group, cost, target));
                                continue;
                            }
                            let q = self.ctx.grid.coordinate(resources);
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
                                    let target = self.storage.retain(label);
                                    self.storage.crossing[v].push(target);
                                }
                                continue;
                            }
                            self.ctx.vertex_step(&mut label);
                            let hot = HotLabel { resources, ..HotLabel::of(&label) };
                            if label.bucket == id {
                                pending.push((label, hot));
                                continue;
                            }
                            let same = self.ctx.grid.direction(forward).buckets[label.bucket.vertex]
                                [label.bucket.bucket]
                                .component
                                == self.ctx.grid.direction(forward).buckets[id.vertex][id.bucket]
                                    .component;
                            if self.insert(g, forward, &label, hot, controls) && same {
                                repeat = true;
                            }
                        }
                    }
                    let active = self.storage.buckets[id.vertex][id.bucket]
                        .active
                        .clone();
                    for index in active {
                        if !self.storage.labels[index].expanded {
                            self.statistics.expanded += 1;
                        }
                        self.storage.labels[index].expanded = true;
                    }
                    for (label, hot) in pending {
                        if self.insert(g, forward, &label, hot, controls) {
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
                for p in &self.ctx.grid.direction(forward).buckets[id.vertex][id.bucket].predecessors {
                    lower = lower
                        .min(self.storage.buckets[id.vertex][p.bucket.bucket].lower_cost);
                }
                self.storage.buckets[id.vertex][id.bucket].lower_cost = lower;
            }
        }
        Ok(())
    }
}

impl LabelingEngine {
    fn ctx(&self) -> Context<'_> {
        Context {
            grid: &self.grid,
            ng: &self.ng,
            cuts: &self.cuts,
            controls: &self.controls,
            phase: self.phase,
            consumption_joins: self.consumption_joins,
        }
    }
    fn vertex_step(&self, label: &mut StoredLabel) {
        self.ctx().vertex_step(label)
    }
    fn root_cut_state(&self, vertex: usize) -> ([u32; LIMITED_WORDS], f64) {
        self.ctx().root_cut_state(vertex)
    }
    pub fn join_cost(&self, g: &Graph, vertex: usize, f: &StoredLabel, b: &StoredLabel) -> Option<f64> {
        self.ctx().join_cost(g, vertex, f, b)
    }
    fn extend(&self, g: &Graph, forward: bool, arc: &BucketArc, from: &StoredLabel) -> Option<StoredLabel> {
        self.ctx().extend(g, forward, arc, from)
    }
    /// A sweep of one direction on this engine's own counters; `other` holds
    /// the opposite direction's label count for the combined label limit.
    fn sweep<'a>(
        &'a mut self,
        forward: bool,
        terminals: &'a mut Vec<(usize, f64, usize)>,
        counts: &'a (AtomicUsize, AtomicUsize),
    ) -> Sweep<'a> {
        let ctx = Context {
            grid: &self.grid,
            ng: &self.ng,
            cuts: &self.cuts,
            controls: &self.controls,
            phase: self.phase,
            consumption_joins: self.consumption_joins,
        };
        let (storage, other) = if forward {
            (&mut self.forward, &self.backward)
        } else {
            (&mut self.backward, &self.forward)
        };
        counts.1.store(other.labels.len(), Ordering::Relaxed);
        Sweep {
            ctx,
            storage,
            statistics: &mut self.statistics,
            extension_checks: &mut self.extension_checks,
            filter_checks: &mut self.filter_checks,
            terminals,
            labels: (&counts.0, &counts.1),
        }
    }
    fn insert(&mut self, g: &Graph, forward: bool, label: &StoredLabel, hot: HotLabel, controls: &Controls) -> bool {
        let mut terminals = vec![];
        let counts = Default::default();
        self.sweep(forward, &mut terminals, &counts).insert(g, forward, label, hot, controls)
    }
    fn run_direction(
        &mut self,
        g: &Graph,
        forward: bool,
        controls: &Controls,
        started: Instant,
        pool: &mut SolutionPool<(Option<usize>, Option<usize>)>,
    ) -> Result<()> {
        let mut terminals = vec![];
        let counts = Default::default();
        let result = self.sweep(forward, &mut terminals, &counts).run_direction(g, forward, controls, started);
        insert_terminals(pool, forward, &terminals);
        if result.is_ok() { self.update_knapsack_bounds(g, forward); }
        result
    }
    /// The forward and backward sweeps of `solve` on two threads. The
    /// backward sweep counts into scratch statistics merged afterwards, and
    /// the pool receives the forward candidates, then the backward ones, as
    /// in sequential order. Only the combined label limit can trip at a
    /// different point; it ends pricing either way. Returns the sweep times.
    fn run_both_directions(
        &mut self,
        g: &Graph,
        controls: &Controls,
        started: Instant,
        pool: &mut SolutionPool<(Option<usize>, Option<usize>)>,
    ) -> Result<(f64, f64)> {
        let ctx = Context {
            grid: &self.grid,
            ng: &self.ng,
            cuts: &self.cuts,
            controls: &self.controls,
            phase: self.phase,
            consumption_joins: self.consumption_joins,
        };
        let vertices = self.extension_checks.len();
        let (mut statistics, mut extension_checks, mut filter_checks) =
            (Statistics::default(), vec![0; vertices], vec![0; vertices]);
        let (mut forward_terminals, mut backward_terminals) = (vec![], vec![]);
        let forward_count = AtomicUsize::new(self.forward.labels.len());
        let backward_count = AtomicUsize::new(self.backward.labels.len());
        let mut forward = Sweep {
            ctx,
            storage: &mut self.forward,
            statistics: &mut self.statistics,
            extension_checks: &mut self.extension_checks,
            filter_checks: &mut self.filter_checks,
            terminals: &mut forward_terminals,
            labels: (&forward_count, &backward_count),
        };
        let mut backward = Sweep {
            ctx,
            storage: &mut self.backward,
            statistics: &mut statistics,
            extension_checks: &mut extension_checks,
            filter_checks: &mut filter_checks,
            terminals: &mut backward_terminals,
            labels: (&backward_count, &forward_count),
        };
        let timed = |sweep: &mut Sweep, forward: bool| {
            let begin = Instant::now();
            let result = sweep.run_direction(g, forward, controls, started);
            (result, begin.elapsed().as_secs_f64())
        };
        let ((forward_result, forward_time), (backward_result, backward_time)) =
            std::thread::scope(|scope| {
                let handle = scope.spawn(|| timed(&mut backward, false));
                let forward_result = timed(&mut forward, true);
                (forward_result, handle.join().expect("backward labeling sweep panicked"))
            });
        self.statistics.absorb(&statistics);
        for (total, part) in self.extension_checks.iter_mut().zip(&extension_checks) {
            *total = total.wrapping_add(*part);
        }
        for (total, part) in self.filter_checks.iter_mut().zip(&filter_checks) {
            *total = total.wrapping_add(*part);
        }
        forward_result?;
        insert_terminals(pool, true, &forward_terminals);
        backward_result?;
        insert_terminals(pool, false, &backward_terminals);
        self.update_knapsack_bounds(g, true);
        self.update_knapsack_bounds(g, false);
        Ok((forward_time, backward_time))
    }
    pub fn prepare(g: &Graph, controls: EngineControls, phase: Phase) -> Result<Self> {
        phase.validate()?;
        let mut grid = BucketGraph::prepare(
            g,
            controls.steps,
            controls.target_buckets,
            controls.bidirectional,
        )?;
        let reverse = crate::graph::symmetry::reverse_arcs(g).unwrap_or_default();
        let consumption_joins = controls.bidirectional
            && controls.use_symmetric_execution
            && !reverse.is_empty()
            && crate::graph::symmetry::consumption_symmetric(g, &grid, &reverse);
        if consumption_joins {
            grid.bidirectional = false;
            grid.middle = (grid.lower[0][0] + grid.upper[grid.order.len() - 1][0]) * 0.5;
            for v in 0..grid.forward.buckets.len() {
                for k in 0..grid.forward.buckets[v].len() {
                    let arcs = grid.forward.buckets[v][k].arcs.clone();
                    let flags: Vec<_> = arcs.iter().map(|a| grid.flags(g, true, a)).collect();
                    for (a, flags) in grid.forward.buckets[v][k].arcs.iter_mut().zip(flags) {
                        a.flags = flags;
                    }
                }
            }
        }
        let forward = LabelStorage::new(grid.forward.buckets.iter().map(Vec::len));
        let backward = LabelStorage::new(grid.backward.buckets.iter().map(Vec::len));
        Ok(Self {
            grid,
            forward,
            backward,
            statistics: Statistics::default(),
            controls,
            phase,
            ng: PreparedNg::new(g)?,
            half_ready: false,
            full_ready: false,
            last_costs: vec![],
            pending_middle: None,
            consumption_joins,
            reverse,
            extension_checks: vec![0; g.vertices.len()],
            filter_checks: vec![0; g.vertices.len()],
            cuts: EngineCuts::default(),
        })
    }
    pub fn has_cuts(&self) -> bool {
        !self.cuts.inputs.is_empty() || !self.cuts.kpath.is_empty()
    }
    /// Limited-memory cut tables for both directions. False when a vertex
    /// would need more than twenty words. Labels are discarded.
    pub fn load_cuts(
        &mut self,
        g: &Graph,
        inputs: Vec<LimitedInput>,
        kpath: Vec<LimitedKPathInput>,
    ) -> bool {
        self.half_ready = false;
        self.full_ready = false;
        let mut cuts = EngineCuts {
            inputs,
            kpath,
            knapsack: self.cuts.knapsack.clone(),
            ..Default::default()
        };
        if !cuts.inputs.is_empty() || !cuts.kpath.is_empty() {
            if !state::prepare_vertices(
                &mut cuts.info,
                &mut cuts.limited,
                g,
                &cuts.inputs,
                &cuts.kpath,
            ) {
                return false;
            }
            state::prepare_arcs(
                &mut cuts.info,
                &mut cuts.limited,
                g,
                &cuts.inputs,
                &cuts.kpath,
                true,
            );
            state::prepare_arcs(
                &mut cuts.info,
                &mut cuts.limited,
                g,
                &cuts.inputs,
                &cuts.kpath,
                false,
            );
        }
        self.cuts = cuts;
        true
    }
    /// update_lim_mem_duals: same cut rows, new duals. Labels are discarded.
    pub fn update_cut_duals(&mut self, inputs: Vec<LimitedInput>, kpath: Vec<LimitedKPathInput>) {
        self.half_ready = false;
        self.full_ready = false;
        state::update_duals(
            &mut self.cuts.info,
            &self.cuts.limited,
            &inputs,
            &kpath,
            true,
            true,
        );
        self.cuts.inputs = inputs;
        self.cuts.kpath = kpath;
        self.cuts.full = None;
    }
    /// updateComplBoundsWithResConsKnapsackCutDuals: preserve the raw
    /// dominance bounds and rebuild the resource-cost completion bounds.
    fn update_knapsack_bounds(&mut self, g: &Graph, forward: bool) {
        if !self.cuts.knapsack.active { return; }
        let components = self.grid.direction(forward).components.clone();
        for component in components {
            for id in component {
                let bucket = &self.storage(forward).buckets[id.vertex][id.bucket];
                let mut lower = NO_ROUTE_COST;
                for &index in &bucket.active {
                    let label = &self.storage(forward).labels[index];
                    let cost = label.key() + self.cuts.knapsack.partial_cost(g, &label.resources, forward);
                    lower = lower.min(cost);
                }
                for predecessor in &self.grid.direction(forward).buckets[id.vertex][id.bucket].predecessors {
                    lower = lower.min(self.storage(forward).buckets[id.vertex][predecessor.bucket.bucket].knapsack_lower_cost);
                }
                self.storage_mut(forward).buckets[id.vertex][id.bucket].knapsack_lower_cost = lower;
            }
        }
    }
    fn storage(&self, forward: bool) -> &LabelStorage {
        if forward {
            &self.forward
        } else {
            &self.backward
        }
    }
    fn storage_mut(&mut self, forward: bool) -> &mut LabelStorage {
        if forward {
            &mut self.forward
        } else {
            &mut self.backward
        }
    }
    fn root(&mut self, g: &Graph, forward: bool) -> Result<()> {
        let vertex = if forward { g.source } else { g.sink };
        let mut resources = [0.; 2];
        for (r, kind) in g.resources.iter().enumerate() {
            let w = g.vertices[vertex].bounds[r];
            resources[r] = if forward {
                g.initial[r].unwrap_or(w.lower)
            } else {
                w.upper
            };
            if resources[r] < w.lower || resources[r] > w.upper {
                return Err(invalid("root resource outside vertex window"));
            }
            if *kind == ResourceKind::Equality && !forward && w.lower != w.upper {
                return Err(invalid(
                    "backward equality resource requires a fixed sink window",
                ));
            }
        }
        let internal = self.grid.internal[vertex];
        let bucket = self
            .grid
            .bucket_id(forward, internal, self.grid.coordinate(resources))
            .ok_or_else(|| invalid("root outside bucket grid"))?;
        let label = StoredLabel {
            id: 0,
            local_id: 0,
            bucket: BucketId {
                vertex: internal,
                bucket,
            },
            vertex,
            cost: 0.,
            original_cost: 0.,
            resources,
            ng: self.ng.vertex_bits[vertex],
            parent: None,
            arc: None,
            hops: 0,
            expanded: false,
            cut_words: [0; LIMITED_WORDS],
            additional_cost: 0.,
            half_potential: self
                .cuts
                .info
                .vertex_cut_potential
                .get(vertex)
                .map_or(0.0, |&p| 0.5 * p),
        };
        let mut label = label;
        (label.cut_words, label.additional_cost) = self.root_cut_state(vertex);
        let storage = self.storage_mut(forward);
        let id = storage.retain(label);
        storage.root = Some(id);
        let hot = storage.hot[id];
        storage.buckets[internal][bucket].active.push(id);
        storage.buckets[internal][bucket].hot.push(hot);
        storage.buckets[internal][bucket].next_id = 1;
        Ok(())
    }
    /// getPrevLabelPtr : every predecessor must still be active
    /// or archived in its bucket. A dominated expanded label that was not
    /// archived is gone, and candidates through it cannot be recorded.
    fn physical_path(&self, forward: bool, mut id: usize) -> Option<Vec<usize>> {
        let s = self.storage(forward);
        let mut arcs = vec![];
        loop {
            let l = &s.labels[id];
            if let Some(a) = l.arc {
                arcs.push(a);
            }
            let Some(parent) = l.parent else { break };
            let p = &s.labels[parent];
            let bucket = &s.buckets[p.bucket.vertex][p.bucket.bucket];
            if !bucket.active.contains(&parent) && !bucket.archived.contains(&parent) {
                return None;
            }
            id = parent;
        }
        if forward {
            arcs.reverse();
        }
        Some(arcs)
    }
    /// recordSolution ; `None` when a predecessor is missing.
    fn reconstruct(
        &self,
        g: &Graph,
        pair: (Option<usize>, Option<usize>),
        cost: f64,
    ) -> Result<Option<Route>> {
        let mut arcs = match pair.0 {
            Some(i) => match self.physical_path(true, i) {
                Some(arcs) => arcs,
                None => return Ok(None),
            },
            None => vec![],
        };
        if let Some(i) = pair.1 {
            let Some(suffix) = self.physical_path(self.consumption_joins, i) else {
                return Ok(None);
            };
            if self.consumption_joins {
                arcs.extend(suffix.into_iter().rev().map(|a| self.reverse[a]));
            } else {
                arcs.extend(suffix);
            }
        }
        let mut nodes = vec![g.source as i32];
        let mut q = [0.; 2];
        for r in 0..g.resources.len() {
            q[r] = g.initial[r].unwrap_or(g.vertices[g.source].bounds[r].lower);
        }
        let mut original_cost = 0.;
        for &index in &arcs {
            let a = &g.arcs[index];
            if nodes.last().copied() != Some(a.tail as i32) {
                return Err(invalid("broken labeling predecessor chain"));
            }
            q = extend_resources(g, a, q, true)
                .ok_or_else(|| invalid("reconstructed route violates resource windows"))?;
            nodes.push(a.head as i32);
            original_cost += a.cost;
        }
        if nodes.last().copied() != Some(g.sink as i32) {
            return Err(invalid("labeling predecessor chain does not reach sink"));
        }
        Ok(Some(Route {
            nodes,
            arcs: arcs.into_iter().map(|i| g.arcs[i].id).collect(),
            cost,
            original_cost,
            resources: q[..g.resources.len()].to_vec(),
        }))
    }
    pub fn solve(&mut self, g: &Graph, controls: &Controls) -> Result<PricingResult> {
        if controls.max_columns == 0 || controls.max_labels == 0 {
            return Err(invalid("positive pricing limits required"));
        }
        if let Some(direction) = self.pending_middle.take() {
            self.grid.move_middle(g, direction);
        }
        self.extension_checks.fill(0);
        self.filter_checks.fill(0);
        self.half_ready = false;
        self.full_ready = false;
        self.last_costs = g.arcs.iter().map(|a| a.reduced_cost).collect();
        self.forward.reset(self.grid.forward.buckets.iter().map(Vec::len));
        self.backward.reset(self.grid.backward.buckets.iter().map(Vec::len));
        self.statistics = Statistics::default();
        self.root(g, true)?;
        if self.runtime_bidirectional() && !self.consumption_joins {
            self.root(g, false)?;
        }
        let started = Instant::now();
        let mut pool = SolutionPool::new(
            g.vertices.len(),
            controls.max_columns,
            self.controls.group_limit,
        );
        let parallel = self.runtime_bidirectional() && !self.consumption_joins;
        if parallel {
            (self.statistics.forward_time, self.statistics.backward_time) =
                self.run_both_directions(g, controls, started, &mut pool)?;
        } else {
            self.run_direction(g, true, controls, started, &mut pool)?;
            self.statistics.forward_time = started.elapsed().as_secs_f64();
        }
        let mut concatenation_tests = 0;
        if self.runtime_bidirectional() {
            if !parallel {
                self.statistics.backward_time =
                    started.elapsed().as_secs_f64() - self.statistics.forward_time;
            }
            let concatenation_started = Instant::now();
            self.update_unreachable(self.consumption_joins);
            for internal in 0..self.grid.order.len() {
                let v = self.grid.order[internal];
                let mut cutoff = pool.cutoff(v);
                for &f in &self.forward.crossing[v] {
                    let label = &self.forward.labels[f];
                    let mut coordinate = self.grid.coordinate(label.resources);
                    if self.consumption_joins {
                        for r in 0..self.grid.dimensions {
                            coordinate[r] = (self.grid.upper[internal][r]
                                + self.grid.lower[internal][r])
                                - coordinate[r];
                        }
                    }
                    let Some(bucket) =
                        self.grid
                            .bucket_id(self.consumption_joins, internal, coordinate)
                    else {
                        continue;
                    };
                    let key = label.key() + self.cuts.knapsack.partial_cost(g, &label.resources, true);
                    let mut work = vec![(bucket, 0usize)];
                    while let Some((k, min)) = work.pop() {
                        let row = &self.storage(self.consumption_joins).buckets[internal][k];
                        if key + row.completion_bound(self.cuts.knapsack.active) >= cutoff {
                            continue;
                        }
                        for &b in &row.active {
                            let other = &self.storage(self.consumption_joins).labels[b];
                            if key + other.key() >= cutoff {
                                break;
                            }
                            concatenation_tests += 1;
                            if let Some(cost) = self.join_cost(g, v, label, other) {
                                if cost < cutoff
                                    && controls
                                        .max_route_arcs
                                        .is_none_or(|limit| label.hops + other.hops <= limit)
                                {
                                    pool.insert(v, cost, 0., (Some(f), Some(b)));
                                    cutoff = pool.cutoff(v);
                                }
                            }
                        }
                        for p in self.grid.direction(self.consumption_joins).buckets[internal][k]
                            .predecessors
                            .iter()
                            .rev()
                        {
                            if p.bucket.bucket >= min {
                                work.push((p.bucket.bucket, min.max(p.minimum)));
                            }
                        }
                    }
                }
                if started.elapsed() > controls.time_limit {
                    return Err(incomplete("pricing concatenation time limit exceeded"));
                }
            }
            self.statistics.concatenation_time = concatenation_started.elapsed().as_secs_f64();
            if self.phase.exact() && !self.consumption_joins {
                let f = self
                    .forward
                    .buckets
                    .iter()
                    .flatten()
                    .map(|b| b.active.len())
                    .sum::<usize>() as f64;
                let b = self
                    .backward
                    .buckets
                    .iter()
                    .flatten()
                    .map(|b| b.active.len())
                    .sum::<usize>() as f64;
                if f / b > 1.2 {
                    self.pending_middle = Some(false);
                } else if b / f > 1.2 {
                    self.pending_middle = Some(true);
                }
                self.half_ready = self.controls.direction_flags == 1;
            }
        }
        if self.phase.exact() && self.controls.filter_other_buckets {
            for v in 0..self.grid.order.len() {
                if self.grid.forward.depth[v] > 0
                    && self.filter_checks[v] > self.extension_checks[v].wrapping_mul(3)
                {
                    self.grid.forward.depth[v] =
                        (self.grid.forward.depth[v] as f64 * 0.9).floor() as i32;
                    if self.grid.bidirectional {
                        self.grid.backward.depth[v] =
                            (self.grid.backward.depth[v] as f64 * 0.9).floor() as i32;
                    }
                }
            }
        }
        self.statistics.concatenation_tests = concatenation_tests;
        self.statistics.forward_labels = self
            .forward
            .buckets
            .iter()
            .flatten()
            .map(|b| b.active.len())
            .sum();
        self.statistics.backward_labels = self
            .backward
            .buckets
            .iter()
            .flatten()
            .map(|b| b.active.len())
            .sum();
        self.statistics.elapsed = started.elapsed().as_secs_f64();
        // The minimum is the first candidate that can be recorded.
        let mut minimum = NO_ROUTE_COST;
        let mut recorded = false;
        let mut columns = vec![];
        for c in pool.into_sorted() {
            let Some(route) = self.reconstruct(g, c.value, c.cost)? else {
                continue;
            };
            if !recorded {
                minimum = c.cost;
                recorded = true;
            }
            if !controls.negative_only || c.cost < controls.zero_threshold - EPSILON {
                columns.push(route);
            }
        }
        Ok(PricingResult {
            columns,
            minimum_reduced_cost: minimum,
            exact: self.phase.exact(),
            statistics: self.statistics.clone(),
        })
    }
}
impl LabelingEngine {
    /// Fix only after both complete directional label sets are available at the
    /// same reduced costs. Cached associations are local to this fixing pass.
    pub fn fix_bucket_arcs(
        &mut self,
        g: &Graph,
        threshold: f64,
    ) -> Result<crate::pricing::fixing::FixingReport> {
        use crate::pricing::fixing::{contains, update, FixingReport, PairingCache};
        if !self.full_ready {
            return Err(invalid("fixing requires complete exact labeling"));
        }
        if !threshold.is_finite() {
            return Err(invalid("finite fixing threshold required"));
        }
        let mut report = FixingReport::default();
        for forward in [true, false] {
            if self.consumption_joins && !forward {
                continue;
            }
            let mut cache = PairingCache::default();
            let components = self.grid.direction(forward).components.clone();
            let mut removed = std::collections::BTreeSet::new();
            for component in components {
                for source in component {
                    let arcs = self.grid.direction(forward).buckets[source.vertex][source.bucket]
                        .arcs
                        .clone();
                    for arc in arcs {
                        if arc.jump.is_some() {
                            continue;
                        }
                        let physical = &g.arcs[arc.physical];
                        if physical.reduced_cost >= NO_ROUTE_COST {
                            continue;
                        }
                        report.checked += 1;
                        let target = self.grid.internal[if forward {
                            physical.head
                        } else {
                            physical.tail
                        }];
                        let point = self.grid.direction(forward).corners[source.vertex]
                            [arc.jump.unwrap_or(source.bucket)];
                        let arrival = self
                            .grid
                            .propagate(g, forward, arc.physical, point)
                            .and_then(|mut q| {
                                if self.consumption_joins {
                                    for r in 0..self.grid.dimensions {
                                        q[r] = self.grid.upper[target][r]
                                            + self.grid.lower[target][r]
                                            - q[r];
                                    }
                                }
                                self.grid
                                    .bucket_id(self.consumption_joins || !forward, target, q)
                            });
                        let Some(arrival) = arrival else {
                            removed.insert(arc.id);
                            continue;
                        };
                        let mut extended = vec![];
                        let mut minimum = NO_ROUTE_COST;
                        for &id in
                            &self.storage(forward).buckets[source.vertex][source.bucket].active
                        {
                            if let Some(l) =
                                self.extend(g, forward, &arc, &self.storage(forward).labels[id])
                            {
                                minimum = minimum.min(l.key() + self.cuts.knapsack.partial_cost(g, &l.resources, forward));
                                extended.push(l);
                            }
                        }
                        let mut frontier =
                            cache.initialize(&self.grid, forward, source, physical.id, target);
                        if !extended.is_empty() {
                            let mut work = vec![(arrival, 0usize)];
                            let mut seen = std::collections::BTreeSet::new();
                            while let Some((bucket, min)) = work.pop() {
                                if !seen.insert((bucket, min)) {
                                    continue;
                                }
                                let storage = self.storage(self.consumption_joins || !forward);
                                let storage_vertex = if self.consumption_joins
                                    && target + 1 == self.grid.order.len()
                                {
                                    0
                                } else {
                                    target
                                };
                                let opposite = &storage.buckets[storage_vertex][bucket];
                                if minimum + opposite.completion_bound(self.cuts.knapsack.active) >= threshold {
                                    continue;
                                }
                                if !contains(&self.grid, &frontier, target, bucket) {
                                    let meeting = if forward { physical.head } else { physical.tail };
                                    'pairs: for l in &extended {
                                        let key = l.key() + self.cuts.knapsack.partial_cost(g, &l.resources, forward);
                                        for &id in &opposite.active {
                                            let other = &storage.labels[id];
                                            if key + other.key() >= threshold {
                                                break;
                                            }
                                            let cost = if forward {
                                                self.join_cost(g, meeting, l, other)
                                            } else {
                                                self.join_cost(g, meeting, other, l)
                                            };
                                            if cost.is_some_and(|c| c < threshold) {
                                                update(&self.grid, &mut frontier, target, bucket);
                                                break 'pairs;
                                            }
                                        }
                                    }
                                }
                                for p in self
                                    .grid
                                    .direction(self.consumption_joins || !forward)
                                    .buckets[target][bucket]
                                    .predecessors
                                    .iter()
                                    .rev()
                                {
                                    if p.bucket.bucket >= min {
                                        work.push((p.bucket.bucket, min.max(p.minimum)));
                                    }
                                }
                            }
                        }
                        let eliminate = !contains(&self.grid, &frontier, target, arrival);
                        cache.associations.insert((source, physical.id), frontier);
                        if eliminate {
                            removed.insert(arc.id);
                        }
                    }
                }
            }
            let direction = if forward {
                &mut self.grid.forward
            } else {
                &mut self.grid.backward
            };
            for row in &mut direction.buckets {
                for b in row {
                    b.arcs.retain(|a| !removed.contains(&a.id));
                }
            }
            if forward {
                report.eliminated_forward = removed.len();
            } else {
                report.eliminated_backward = removed.len();
            }
        }
        if report.eliminated_forward + report.eliminated_backward > 0 {
            self.grid.regenerate_jumps(g, true);
            if !self.consumption_joins {
                self.grid.regenerate_jumps(g, false);
            }
            self.full_ready = false;
            self.half_ready = false;
        }
        Ok(report)
    }
}

impl LabelingEngine {
    fn update_unreachable(&mut self, forward: bool) {
        for v in 0..self.grid.order.len() {
            for k in 0..self.grid.direction(forward).buckets[v].len() {
                let b = &self.grid.direction(forward).buckets[v][k];
                if b.reachable {
                    continue;
                }
                let lower = b
                    .predecessors
                    .iter()
                    .map(|p| self.storage(forward).buckets[v][p.bucket.bucket].lower_cost)
                    .fold(NO_ROUTE_COST, f64::min);
                let knapsack_lower = b.predecessors.iter()
                    .map(|p| self.storage(forward).buckets[v][p.bucket.bucket].knapsack_lower_cost)
                    .fold(NO_ROUTE_COST, f64::min);
                self.storage_mut(forward).buckets[v][k].lower_cost = lower;
                self.storage_mut(forward).buckets[v][k].knapsack_lower_cost = knapsack_lower;
            }
        }
    }
    pub fn continue_after_halfway(&mut self, g: &Graph, controls: &Controls) -> Result<()> {
        if self.full_ready
            && self
                .last_costs
                .iter()
                .copied()
                .eq(g.arcs.iter().map(|a| a.reduced_cost))
        {
            return Ok(());
        }
        if !self.half_ready
            || !self
                .last_costs
                .iter()
                .copied()
                .eq(g.arcs.iter().map(|a| a.reduced_cost))
        {
            return Err(invalid("continuation requires fresh exact half-labels"));
        }
        self.half_ready = false;
        self.full_ready = false;
        let saved = self.controls.clone();
        let phase = self.phase;
        self.controls.direction_flags = 2;
        self.controls.sink_concatenation = false;
        self.phase.labels_per_bucket = i64::MAX;
        let result = (|| {
            for v in 0..self.forward.crossing.len() {
                let crossing = self.forward.crossing[v].clone();
                for id in crossing {
                    let mut label = self.forward.labels[id].clone();
                    self.vertex_step(&mut label);
                    let hot = HotLabel::of(&label);
                    self.insert(g, true, &label, hot, controls);
                }
            }
            for label in &mut self.forward.labels {
                if self.grid.coordinate(label.resources)[0] > self.grid.middle + EPSILON {
                    label.expanded = false;
                }
            }
            let started = Instant::now();
            let mut pool = SolutionPool::new(g.vertices.len(), 1, 0);
            self.run_direction(g, true, controls, started, &mut pool)?;
            for label in &mut self.backward.labels {
                label.expanded = false;
            }
            self.run_direction(g, false, controls, started, &mut pool)?;
            self.update_unreachable(true);
            self.update_unreachable(false);
            self.full_ready = true;
            Ok(())
        })();
        self.controls = saved;
        self.phase = phase;
        result
    }
}

impl LabelingEngine {
    pub fn costs_unchanged(&self, g: &Graph) -> bool {
        self.last_costs
            .iter()
            .copied()
            .eq(g.arcs.iter().map(|a| a.reduced_cost))
    }
}
impl LabelingEngine {
    /// Completion query shared by pricing, enumeration and fixing. The lower
    /// cost already includes dominance predecessors; scanning follows their
    /// native staircase and minimum-index bounds.
    pub fn completion_exists(
        &self,
        g: &Graph,
        forward: bool,
        label: &StoredLabel,
        threshold: f64,
        scan: bool,
    ) -> bool {
        let vertex = self.grid.internal[label.vertex];
        let mut q = self.grid.coordinate(label.resources);
        let opposite = self.consumption_joins || !forward;
        if self.consumption_joins && forward {
            for r in 0..self.grid.dimensions {
                q[r] = self.grid.upper[vertex][r] + self.grid.lower[vertex][r] - q[r];
            }
        }
        let Some(bucket) = self.grid.bucket_id(opposite, vertex, q) else {
            return false;
        };
        let storage = self.storage(opposite);
        let key = label.key() + self.cuts.knapsack.partial_cost(g, &label.resources, forward);
        let mut work = vec![(bucket, 0usize)];
        while let Some((k, min)) = work.pop() {
            let row = &storage.buckets[vertex][k];
            if key + row.completion_bound(self.cuts.knapsack.active) >= threshold {
                continue;
            }
            if !scan {
                return true;
            }
            for &id in &row.active {
                let other = &storage.labels[id];
                if key + other.key() >= threshold {
                    break;
                }
                let cost = if forward {
                    self.join_cost(g, label.vertex, label, other)
                } else {
                    self.join_cost(g, label.vertex, other, label)
                };
                if cost.is_some_and(|c| threshold > c) {
                    return true;
                }
            }
            for p in self.grid.direction(opposite).buckets[vertex][k]
                .predecessors
                .iter()
                .rev()
            {
                if p.bucket.bucket >= min {
                    work.push((p.bucket.bucket, min.max(p.minimum)));
                }
            }
        }
        false
    }
    pub fn enumeration_completion(
        &self,
        g: &Graph,
        forward: bool,
        label: &crate::enumeration::Label,
        threshold: f64,
        scan: bool,
    ) -> bool {
        let mut ng = 0;
        for (bit, &e) in self.ng.local_elements[label.vertex].iter().enumerate() {
            if label.visited[e / 64] & (1u64 << (e % 64)) != 0 {
                ng |= 1 << bit;
            }
        }
        let (cut_words, additional_cost) = match (&self.cuts.full, self.has_cuts()) {
            (Some(full), true) => (
                state::project_words(&self.cuts.limited, full, label.vertex, &label.cut_words),
                label.cut_cost,
            ),
            _ => ([0; LIMITED_WORDS], label.cut_cost),
        };
        let projected = StoredLabel {
            id: 0,
            local_id: 0,
            bucket: BucketId::default(),
            vertex: label.vertex,
            cost: label.reduced_cost,
            original_cost: label.original_cost,
            resources: label.resources,
            ng,
            parent: None,
            arc: None,
            hops: label.hops,
            expanded: false,
            cut_words,
            additional_cost,
            half_potential: self
                .cuts
                .info
                .vertex_cut_potential
                .get(label.vertex)
                .map_or(0.0, |&p| 0.5 * p),
        };
        self.completion_exists(g, forward, &projected, threshold, scan)
    }
}

impl LabelingEngine {
    fn runtime_bidirectional(&self) -> bool {
        self.controls.bidirectional
            && (self.controls.bidirectional_mode == 1
                || (self.controls.bidirectional_mode == 2 && self.phase.exact()))
    }
}

impl LabelingEngine {
    /// Completion passes rebuild both directional lower-cost tables. Ordinary
    /// pricing only needs the opposite table for joins and must not be used as
    /// a substitute: an unreachable forward cell can still inherit a valid
    /// completion bound from its dominance predecessors.
    pub fn prepare_full_labels(&mut self, g: &Graph, controls: &Controls) -> Result<()> {
        if self.costs_unchanged(g) {
            if self.full_ready {
                return Ok(());
            }
            if self.half_ready {
                return self.continue_after_halfway(g, controls);
            }
        }
        if !self.consumption_joins {
            self.grid.ensure_backward(g)?;
        }
        let saved = self.controls.clone();
        let phase = self.phase;
        self.controls.bidirectional = !self.consumption_joins;
        self.controls.bidirectional_mode = 1;
        self.controls.direction_flags = 3;
        self.controls.sink_concatenation = false;
        self.controls.dominance_mode = 0;
        self.phase = Phase {
            check_ng: true,
            jump_arcs: true,
            labels_per_bucket: i64::MAX,
        };
        self.half_ready = false;
        self.full_ready = false;
        self.last_costs = g.arcs.iter().map(|a| a.reduced_cost).collect();
        self.forward.reset(self.grid.forward.buckets.iter().map(Vec::len));
        self.backward.reset(self.grid.backward.buckets.iter().map(Vec::len));
        self.statistics = Statistics::default();
        self.extension_checks.fill(0);
        self.filter_checks.fill(0);
        // A fixing completion is not a new pricing call. In particular, retain
        // the pending border movement for the next actual pricing call.
        let result = (|| {
            self.root(g, true)?;
            if !self.consumption_joins {
                self.root(g, false)?;
            }
            let started = Instant::now();
            let mut pool = SolutionPool::new(g.vertices.len(), 1, 0);
            self.run_direction(g, true, controls, started, &mut pool)?;
            if !self.consumption_joins {
                self.run_direction(g, false, controls, started, &mut pool)?;
            }
            self.update_unreachable(true);
            if !self.consumption_joins {
                self.update_unreachable(false);
            }
            self.full_ready = true;
            Ok(())
        })();
        self.controls = saved;
        self.phase = phase;
        result
    }
}
