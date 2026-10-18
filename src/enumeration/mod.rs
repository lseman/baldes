//! Full-memory elementary enumeration without cut state. Equivalence classes
//! compare global element sets and original cost, independently of current duals.
use crate::cuts::state::{
    self, CutResourceInfo, FullPreparation, LimitedInput, LimitedKPathInput, FULL_WORDS,
};
use crate::graph::buckets::BucketGraph;
use crate::graph::{Graph, ResourceKind};
use crate::labeling::{extend_resources, Route, EPSILON};
use crate::{incomplete, invalid, Result};
use std::time::{Duration, Instant};
#[derive(Clone, Debug)]
pub struct Label {
    pub id: usize,
    pub vertex: usize,
    pub resources: [f64; 2],
    pub reduced_cost: f64,
    pub original_cost: f64,
    pub visited: [u64; 16],
    pub element_hash: u32,
    pub parent: Option<usize>,
    pub arc: Option<usize>,
    pub hops: usize,
    /// Full-memory cut state (State<FullMemSBCutsResources>).
    pub cut_words: [u32; FULL_WORDS],
    pub cut_cost: f64,
}
/// Full-memory cut tables of an enumeration with cuts.
#[derive(Clone, Copy, Debug)]
pub struct CutTables<'a> {
    pub info: &'a CutResourceInfo,
    pub full: &'a FullPreparation,
    pub knapsack: &'a crate::cuts::knapsack::KnapsackPricing,
}
impl<'a> CutTables<'a> {
    fn along_arc(&self, label: &mut Label, forward: bool, arc: usize) {
        let table = if forward {
            &self.info.forward
        } else {
            &self.info.backward
        };
        state::change_full_memory_along_arc(&mut label.cut_words, &mut label.cut_cost, &table[arc].full_memory);
    }
    fn to_vertex(&self, label: &mut Label) {
        state::apply_vertex_support(
            &mut label.cut_words,
            &mut label.cut_cost,
            &self.info.vertices[label.vertex].full_memory_support,
        );
    }
    /// enumeration_labels_can_be_concatenated cost part, at the forward vertex.
    fn join(&self, g: &Graph, symmetric: bool, f: &Label, b: &Label, cost: f64) -> f64 {
        let cost = cost + self.knapsack.join_cost(g, &f.resources, &b.resources, symmetric);
        state::full_memory_concatenation_cost(
            self.info,
            self.full,
            f.vertex,
            (&f.cut_words, f.cut_cost),
            (&b.cut_words, b.cut_cost),
            cost,
        )
    }
}
impl Label {
    /// Enumeration bucket key, ((0.5 potential + additional) + reduced) + 0.
    pub fn key(&self) -> f64 {
        ((0.0 + self.cut_cost) + self.reduced_cost) + 0.0
    }
    pub fn key_with(&self, potential: f64) -> f64 {
        ((0.5 * potential + self.cut_cost) + self.reduced_cost) + 0.0
    }
    pub fn add_elements(&mut self, elements: &[usize]) -> bool {
        for &e in elements {
            if e >= 1024 {
                return false;
            }
            let word = &mut self.visited[e / 64];
            let bit = 1u64 << (e % 64);
            if *word & bit != 0 {
                return false;
            }
            *word |= bit;
            let x = e as u32;
            self.element_hash = self
                .element_hash
                .wrapping_add(x.wrapping_mul(x).wrapping_mul(x));
        }
        true
    }
}
#[derive(Clone, Debug)]
pub struct Storage {
    pub partitions: Vec<Vec<usize>>,
    pub arena: Vec<Label>,
    pub live: Vec<bool>,
    cursor: usize,
    pub pending: usize,
    pub checks: u64,
}
impl Storage {
    pub fn new(partitions: usize) -> Result<Self> {
        if partitions == 0 {
            return Err(invalid("positive enumeration partitions required"));
        }
        Ok(Self {
            partitions: vec![vec![]; partitions],
            arena: vec![],
            live: vec![],
            cursor: 0,
            pending: 0,
            checks: 0,
        })
    }
    fn dominates(g: &Graph, a: &Label, b: &Label, forward: bool) -> bool {
        a.vertex == b.vertex
            && a.visited == b.visited
            && a.original_cost - b.original_cost < EPSILON
            && g.resources.iter().enumerate().all(|(r, k)| match k {
                ResourceKind::Disposable => {
                    if forward {
                        a.resources[r] <= b.resources[r] + EPSILON
                    } else {
                        b.resources[r] <= a.resources[r] + EPSILON
                    }
                }
                ResourceKind::Equality => (a.resources[r] - b.resources[r]).abs() <= EPSILON,
            })
    }
    pub fn insert(&mut self, g: &Graph, mut label: Label, forward: bool) -> bool {
        let partition = label.element_hash as usize % self.partitions.len();
        let old = self.partitions[partition].clone();
        for &id in &old {
            let previous = &self.arena[id];
            if label.original_cost + EPSILON < previous.original_cost {
                break;
            }
            self.checks += 1;
            if Self::dominates(g, previous, &label, forward) {
                return false;
            }
        }
        let mut records = vec![];
        for id in old {
            self.checks += 1;
            if Self::dominates(g, &label, &self.arena[id], forward) {
                self.live[id] = false;
                if id >= self.cursor {
                    self.pending -= 1;
                }
            } else {
                records.push(id);
            }
        }
        let at = records
            .partition_point(|&id| !(label.original_cost + EPSILON < self.arena[id].original_cost));
        let id = self.arena.len();
        label.id = id;
        self.arena.push(label);
        self.live.push(true);
        records.insert(at, id);
        self.partitions[partition] = records;
        self.pending += 1;
        true
    }
    pub fn pop(&mut self) -> Option<Label> {
        while self.cursor < self.arena.len() {
            let id = self.cursor;
            self.cursor += 1;
            if self.live[id] {
                self.pending -= 1;
                return Some(self.arena[id].clone());
            }
        }
        None
    }
}
#[derive(Clone, Debug)]
pub struct Path {
    pub reduced_cost: f64,
    pub original_cost: f64,
    pub visited: [u64; 16],
    pub element_hash: u32,
    pub arcs: Vec<usize>,
}
#[derive(Clone, Debug)]
pub struct PathStorage {
    pub partitions: Vec<Vec<Path>>,
    pub len: usize,
    pub checks: u64,
}
impl PathStorage {
    pub fn new(partitions: usize) -> Result<Self> {
        if partitions == 0 {
            return Err(invalid("positive enumeration path partitions required"));
        }
        Ok(Self {
            partitions: vec![vec![]; partitions],
            len: 0,
            checks: 0,
        })
    }
    pub fn insert(&mut self, path: Path) -> bool {
        let partition = path.element_hash as usize % self.partitions.len();
        let row = &mut self.partitions[partition];
        for previous in row.iter() {
            if previous.original_cost > path.original_cost + EPSILON {
                break;
            }
            self.checks += 1;
            if previous.visited == path.visited {
                return false;
            }
        }
        let before = row.len();
        row.retain(|p| {
            self.checks += 1;
            p.visited != path.visited
        });
        self.len -= before - row.len();
        let at = row.partition_point(|p| p.original_cost <= path.original_cost + EPSILON);
        row.insert(at, path);
        self.len += 1;
        true
    }
    /// Path repricing: physical arc costs, then rank-one costs from
    /// physical incidence (no memory).
    pub fn inspect(
        &mut self,
        g: &Graph,
        cutoff: f64,
        cuts: &[LimitedInput],
        kpath: &[LimitedKPathInput],
    ) -> Result<usize> {
        self.inspect_with_knapsack(g, cutoff, cuts, kpath, &Default::default())
    }
    pub fn inspect_with_knapsack(
        &mut self, g: &Graph, cutoff: f64, cuts: &[LimitedInput],
        kpath: &[LimitedKPathInput], knapsack: &crate::cuts::knapsack::KnapsackPricing,
    ) -> Result<usize> {
        if cutoff.is_nan() {
            return Err(invalid("invalid enumeration inspection cutoff"));
        }
        let before = self.len;
        for row in &mut self.partitions {
            for path in row.iter_mut() {
                path.reduced_cost = path.arcs.iter().map(|&a| g.arcs[a].reduced_cost).sum();
                if !cuts.is_empty() || !kpath.is_empty() {
                    let mut vertices = vec![g.source];
                    vertices.extend(path.arcs.iter().map(|&a| g.arcs[a].head));
                    let ids: Vec<i32> = path.arcs.iter().map(|&a| g.arcs[a].id).collect();
                    path.reduced_cost =
                        state::enumerated_path_cut_cost(cuts, kpath, &vertices, &ids, path.reduced_cost);
                }
                if knapsack.active {
                    let mut q = [0.; 2];
                    for r in 0..g.resources.len() { q[r] = g.initial[r].unwrap_or(g.vertices[g.source].bounds[r].lower); }
                    for &arc in &path.arcs {
                        q = extend_resources(g, &g.arcs[arc], q, true).ok_or_else(|| invalid("enumerated knapsack resource infeasible"))?;
                    }
                    path.reduced_cost += knapsack.cost(&q);
                }
            }
            row.retain(|p| p.reduced_cost < cutoff);
        }
        self.len = self.partitions.iter().map(Vec::len).sum();
        Ok(before - self.len)
    }
    pub fn routes(&self, g: &Graph) -> Result<Vec<Route>> {
        let mut result = vec![];
        for path in self.partitions.iter().flatten() {
            let mut q = [0.; 2];
            for r in 0..g.resources.len() {
                q[r] = g.initial[r].unwrap_or(g.vertices[g.source].bounds[r].lower);
            }
            let mut nodes = vec![g.source as i32];
            for &id in &path.arcs {
                let a = &g.arcs[id];
                q = extend_resources(g, a, q, true)
                    .ok_or_else(|| invalid("enumerated route is no longer resource feasible"))?;
                nodes.push(a.head as i32);
            }
            result.push(Route {
                nodes,
                arcs: path.arcs.iter().map(|&a| g.arcs[a].id).collect(),
                cost: path.reduced_cost,
                original_cost: path.original_cost,
                resources: q[..g.resources.len()].to_vec(),
            });
        }
        Ok(result)
    }
}
#[derive(Clone, Debug)]
pub struct Controls {
    pub label_limit: usize,
    pub path_limit: usize,
    pub partitions: usize,
    pub cutoff: f64,
    pub time_limit: Duration,
    pub max_route_arcs: Option<usize>,
}
impl Default for Controls {
    fn default() -> Self {
        Self {
            label_limit: 1_000_000,
            path_limit: 1_000_000,
            partitions: 1024,
            cutoff: 0.,
            time_limit: Duration::from_secs(1000),
            max_route_arcs: None,
        }
    }
}
#[derive(Debug)]
pub struct Enumeration {
    pub labels: Storage,
    pub paths: PathStorage,
    pub completed: bool,
    pub generated: usize,
}
impl Enumeration {
    /// Abort without publishing a partial universe. Existing successful path
    /// storage remains reusable for inspection at subsequent master duals.
    pub fn run(g: &Graph, grid: &BucketGraph, controls: &Controls, cuts: Option<CutTables>) -> Result<Self> {
        g.validate()?;
        if controls.label_limit == 0 || controls.path_limit == 0 || !controls.cutoff.is_finite() {
            return Err(invalid("invalid enumeration controls"));
        }
        let mut labels = Storage::new(controls.partitions)?;
        let mut paths = PathStorage::new(controls.partitions)?;
        let started = Instant::now();
        let mut root = Label {
            id: 0,
            vertex: g.source,
            resources: [0.; 2],
            reduced_cost: 0.,
            original_cost: 0.,
            visited: [0; 16],
            element_hash: 0,
            parent: None,
            arc: None,
            hops: 0,
            cut_words: [0; FULL_WORDS],
            cut_cost: 0.,
        };
        for r in 0..g.resources.len() {
            root.resources[r] = g.initial[r].unwrap_or(g.vertices[g.source].bounds[r].lower);
        }
        if !root.add_elements(&g.vertices[g.source].elements) {
            return Err(invalid("invalid source elementarity sets"));
        }
        if let Some(c) = &cuts {
            c.to_vertex(&mut root);
        }
        labels.insert(g, root, true);
        let step = controls.label_limit / 50;
        let mut checkpoint = 0;
        while let Some(from) = labels.pop() {
            if started.elapsed() > controls.time_limit {
                return Err(incomplete("enumeration time limit exceeded"));
            }
            let generated = labels.arena.len();
            if generated > controls.label_limit {
                return Err(incomplete("enumeration label budget exceeded"));
            }
            if generated - checkpoint >= step {
                let ratio = generated as f64 / controls.label_limit as f64;
                if 1. - ratio / 1.5 < labels.pending as f64 / generated as f64 {
                    return Err(incomplete("enumeration pending-label budget exceeded"));
                }
                checkpoint += step;
            }
            if controls
                .max_route_arcs
                .is_some_and(|limit| from.hops >= limit)
            {
                continue;
            }
            let v = grid.internal[from.vertex];
            let Some(bucket) = grid.bucket_id(true, v, grid.coordinate(from.resources)) else {
                continue;
            };
            for arc in &grid.forward.buckets[v][bucket].arcs {
                let physical = &g.arcs[arc.physical];
                let mut next = from.clone();
                next.parent = Some(from.id);
                next.arc = Some(arc.physical);
                next.vertex = physical.head;
                next.hops += 1;
                next.reduced_cost = physical.reduced_cost + from.reduced_cost;
                next.original_cost = from.original_cost + physical.cost;
                if !next.add_elements(&physical.elements) {
                    continue;
                }
                let mut q = next.resources;
                if let Some(jump) = arc.jump {
                    for (r, &slot) in grid.resource_slots.iter().enumerate() {
                        q[slot] = q[slot].max(grid.forward.corners[v][jump][r]);
                    }
                }
                let Some(q) = extend_resources(g, physical, q, true) else {
                    continue;
                };
                next.resources = q;
                if let Some(c) = &cuts {
                    c.along_arc(&mut next, true, arc.physical);
                }
                if !next.add_elements(&g.vertices[next.vertex].elements) {
                    continue;
                }
                if let Some(c) = &cuts {
                    c.to_vertex(&mut next);
                }
                if next.vertex == g.sink {
                    if let Some(c) = &cuts {
                        let mut empty = next.clone();
                        empty.cut_words = [0; FULL_WORDS];
                        empty.cut_cost = 0.;
                        for r in 0..g.resources.len() { empty.resources[r] = g.vertices[g.sink].bounds[r].upper; }
                        next.reduced_cost = c.join(g, false, &next, &empty, (next.reduced_cost + 0.0) + 0.0);
                    }
                    if next.reduced_cost > controls.cutoff - EPSILON {
                        continue;
                    }
                    let mut arcs = vec![arc.physical];
                    let mut parent = Some(from.id);
                    while let Some(id) = parent {
                        let label = &labels.arena[id];
                        if let Some(a) = label.arc {
                            arcs.push(a);
                        }
                        parent = label.parent;
                    }
                    arcs.reverse();
                    paths.insert(Path {
                        reduced_cost: next.reduced_cost,
                        original_cost: next.original_cost,
                        visited: next.visited,
                        element_hash: next.element_hash,
                        arcs,
                    });
                    if paths.len > controls.path_limit {
                        return Err(incomplete("enumeration path budget exceeded"));
                    }
                } else {
                    labels.insert(g, next, true);
                }
            }
        }
        let generated = labels.arena.len();
        Ok(Self {
            labels,
            paths,
            completed: true,
            generated,
        })
    }
}
impl Enumeration {
    /// Directional elementary enumeration with a shared path-equivalence store.
    /// Completion is supplied by the complete pricing stores at these same costs.
    pub fn run_directional<F: Fn(bool, &Label) -> bool>(
        g: &Graph,
        grid: &BucketGraph,
        controls: &Controls,
        completion: F,
    ) -> Result<Self> {
        Self::run_with_symmetry(g, grid, controls, &[], None, completion)
    }
    pub fn run_with_symmetry<F: Fn(bool, &Label) -> bool>(
        g: &Graph,
        grid: &BucketGraph,
        controls: &Controls,
        reverse: &[usize],
        cuts: Option<CutTables>,
        completion: F,
    ) -> Result<Self> {
        let consumption = !reverse.is_empty();
        if (!consumption && grid.backward.buckets.is_empty()) || grid.resource_slots.is_empty() {
            return Self::run(g, grid, controls, cuts);
        }
        g.validate()?;
        if controls.label_limit == 0 || controls.path_limit == 0 || !controls.cutoff.is_finite() {
            return Err(invalid("invalid enumeration controls"));
        }
        let started = Instant::now();
        let mut forward = Storage::new(controls.partitions)?;
        let mut backward = Storage::new(controls.partitions)?;
        let mut paths = PathStorage::new(controls.partitions)?;
        let mut crossing = vec![vec![]; g.vertices.len()];
        let mut roots = vec![];
        for direction in [true, false] {
            let vertex = if direction { g.source } else { g.sink };
            let mut root = Label {
                id: 0,
                vertex,
                resources: [0.; 2],
                reduced_cost: 0.,
                original_cost: 0.,
                visited: [0; 16],
                element_hash: 0,
                parent: None,
                arc: None,
                hops: 0,
                cut_words: [0; FULL_WORDS],
                cut_cost: 0.,
            };
            for r in 0..g.resources.len() {
                root.resources[r] = if direction {
                    g.initial[r].unwrap_or(g.vertices[vertex].bounds[r].lower)
                } else {
                    g.vertices[vertex].bounds[r].upper
                };
            }
            if !root.add_elements(&g.vertices[vertex].elements) {
                return Err(invalid("invalid enumeration root elements"));
            }
            if let Some(c) = &cuts {
                c.to_vertex(&mut root);
            }
            roots.push(root);
        }
        for direction in [true, false] {
            if consumption && !direction {
                continue;
            }
            let storage = if direction {
                &mut forward
            } else {
                &mut backward
            };
            let root = &roots[if direction { 0 } else { 1 }];
            let opposite_root = &roots[if direction && !consumption { 1 } else { 0 }];
            storage.insert(g, root.clone(), direction);
            let step = controls.label_limit / 50;
            let mut checkpoint = 0;
            while let Some(from) = storage.pop() {
                if started.elapsed() > controls.time_limit {
                    return Err(incomplete("enumeration time limit exceeded"));
                }
                let generated = storage.arena.len();
                if generated > controls.label_limit {
                    return Err(incomplete("enumeration label budget exceeded"));
                }
                if generated - checkpoint >= step {
                    let ratio = generated as f64 / controls.label_limit as f64;
                    if 1. - ratio / 1.5 < storage.pending as f64 / generated as f64 {
                        return Err(incomplete("enumeration pending-label budget exceeded"));
                    }
                    checkpoint += step;
                }
                if controls
                    .max_route_arcs
                    .is_some_and(|limit| from.hops >= limit)
                {
                    continue;
                }
                let v = grid.internal[from.vertex];
                let Some(bucket) = grid.bucket_id(direction, v, grid.coordinate(from.resources))
                else {
                    continue;
                };
                for arc in &grid.direction(direction).buckets[v][bucket].arcs {
                    let physical = &g.arcs[arc.physical];
                    let mut next = from.clone();
                    next.parent = Some(from.id);
                    next.arc = Some(arc.physical);
                    next.vertex = if direction {
                        physical.head
                    } else {
                        physical.tail
                    };
                    next.hops += 1;
                    next.reduced_cost = physical.reduced_cost + from.reduced_cost;
                    next.original_cost = from.original_cost + physical.cost;
                    if !next.add_elements(&physical.elements) {
                        continue;
                    }
                    let mut q = next.resources;
                    if let Some(jump) = arc.jump {
                        for (r, &slot) in grid.resource_slots.iter().enumerate() {
                            let value = grid.direction(direction).corners[v][jump][r];
                            q[slot] = if direction {
                                q[slot].max(value)
                            } else {
                                q[slot].min(value)
                            };
                        }
                    }
                    let Some(q) = extend_resources(g, physical, q, direction) else {
                        continue;
                    };
                    next.resources = q;
                    if let Some(c) = &cuts {
                        c.along_arc(&mut next, direction, arc.physical);
                    }
                    let terminal = next.vertex == if direction { g.sink } else { g.source };
                    if !terminal {
                        let before = grid.coordinate(q)[0] <= grid.middle + EPSILON;
                        if !direction && before {
                            continue;
                        }
                        if !completion(direction, &next) {
                            continue;
                        }
                        if direction && !before {
                            crossing[next.vertex].push(next);
                            continue;
                        }
                    }
                    if terminal {
                        if let Some(c) = &cuts {
                            // The opposite root owns the terminal's cut state.
                            next.reduced_cost = if direction {
                                c.join(g, consumption, &next, opposite_root, ((next.reduced_cost + 0.0) + 0.0) + 0.0)
                            } else {
                                c.join(g, consumption, opposite_root, &next, ((0.0 + next.reduced_cost) + 0.0) + 0.0)
                            };
                        }
                        if next
                            .visited
                            .iter()
                            .zip(&opposite_root.visited)
                            .any(|(a, b)| a & b != 0)
                            || next.reduced_cost > controls.cutoff - EPSILON
                        {
                            continue;
                        }
                        if !next.add_elements(&g.vertices[opposite_root.vertex].elements) {
                            continue;
                        }
                        let mut arcs = vec![arc.physical];
                        let mut parent = Some(from.id);
                        while let Some(id) = parent {
                            let label = &storage.arena[id];
                            if let Some(a) = label.arc {
                                arcs.push(a);
                            }
                            parent = label.parent;
                        }
                        if direction {
                            arcs.reverse();
                        }
                        paths.insert(Path {
                            reduced_cost: next.reduced_cost,
                            original_cost: next.original_cost,
                            visited: next.visited,
                            element_hash: next.element_hash,
                            arcs,
                        });
                        if paths.len > controls.path_limit {
                            return Err(incomplete("enumeration path budget exceeded"));
                        }
                    } else {
                        if !next.add_elements(&g.vertices[next.vertex].elements) {
                            continue;
                        }
                        if let Some(c) = &cuts {
                            c.to_vertex(&mut next);
                        }
                        storage.insert(g, next, direction);
                    }
                }
            }
        }
        // Bucket, partition and original-cost order determine which equivalent
        // elementary route survives. Preserve that order across crossing joins.
        let opposite = if consumption { &forward } else { &backward };
        let mut buckets: Vec<Vec<Vec<usize>>> = grid
            .direction(consumption)
            .buckets
            .iter()
            .map(|row| vec![vec![]; row.len()])
            .collect();
        for row in &opposite.partitions {
            for &id in row {
                let label = &opposite.arena[id];
                let vertex = grid.internal[label.vertex];
                if let Some(k) =
                    grid.bucket_id(consumption, vertex, grid.coordinate(label.resources))
                {
                    buckets[vertex][k].push(id);
                }
            }
        }
        let label_key = |l: &Label| {
            l.key_with(
                cuts.and_then(|c| c.info.vertex_cut_potential.get(l.vertex))
                    .copied()
                    .unwrap_or(0.0),
            )
        };
        for row in &mut buckets {
            for bucket in row {
                bucket.sort_by(|&a, &b| {
                    label_key(&opposite.arena[a])
                        .total_cmp(&label_key(&opposite.arena[b]))
                        .then(a.cmp(&b))
                });
            }
        }
        for vertex in 0..grid.order.len() {
            let v = grid.order[vertex];
            for f in &crossing[v] {
                let mut q = grid.coordinate(f.resources);
                if consumption {
                    for r in 0..grid.dimensions {
                        q[r] = grid.upper[vertex][r] + grid.lower[vertex][r] - q[r];
                    }
                }
                let Some(k) = grid.bucket_id(consumption, vertex, q) else {
                    continue;
                };
                let mut pending = vec![(k, 0usize)];
                while let Some((bucket, min)) = pending.pop() {
                    let key = label_key(f);
                    for &id in &buckets[vertex][bucket] {
                        let b = &opposite.arena[id];
                        if key + label_key(b) > controls.cutoff - EPSILON {
                            break;
                        }
                        let cost = match &cuts {
                            Some(c) => c.join(g, consumption, f, b, ((f.reduced_cost + b.reduced_cost) + 0.0) + 0.0),
                            None => f.reduced_cost + b.reduced_cost,
                        };
                        if cost > controls.cutoff - EPSILON {
                            continue;
                        }
                        if controls
                            .max_route_arcs
                            .is_some_and(|limit| f.hops + b.hops > limit)
                            || f.visited.iter().zip(&b.visited).any(|(x, y)| x & y != 0)
                        {
                            continue;
                        }
                        if !g.resources.iter().enumerate().all(|(r, kind)| match kind {
                            ResourceKind::Disposable => {
                                if consumption {
                                    f.resources[r] + b.resources[r]
                                        <= g.vertices[g.sink].bounds[r].upper + EPSILON
                                } else {
                                    f.resources[r] <= b.resources[r] + EPSILON
                                }
                            }
                            ResourceKind::Equality => f.resources[r] == b.resources[r],
                        }) {
                            continue;
                        }
                        let mut arcs = vec![];
                        let mut label = f;
                        loop {
                            if let Some(a) = label.arc {
                                arcs.push(a);
                            }
                            let Some(parent) = label.parent else {
                                break;
                            };
                            label = &forward.arena[parent];
                        }
                        arcs.reverse();
                        let mut suffix = vec![];
                        let mut label = b;
                        loop {
                            if let Some(a) = label.arc {
                                suffix.push(a);
                            }
                            let Some(parent) = label.parent else {
                                break;
                            };
                            label = &opposite.arena[parent];
                        }
                        if consumption {
                            arcs.extend(suffix.into_iter().map(|a| reverse[a]));
                        } else {
                            arcs.extend(suffix);
                        }
                        let mut visited = f.visited;
                        for (x, y) in visited.iter_mut().zip(&b.visited) {
                            *x |= *y;
                        }
                        paths.insert(Path {
                            reduced_cost: cost,
                            original_cost: f.original_cost + b.original_cost,
                            visited,
                            element_hash: f.element_hash.wrapping_add(b.element_hash),
                            arcs,
                        });
                        if paths.len > controls.path_limit {
                            return Err(incomplete("enumeration path budget exceeded"));
                        }
                    }
                    for p in grid.direction(consumption).buckets[vertex][bucket]
                        .predecessors
                        .iter()
                        .rev()
                    {
                        if p.bucket.bucket >= min {
                            pending.push((p.bucket.bucket, min.max(p.minimum)));
                        }
                    }
                }
                if started.elapsed() > controls.time_limit {
                    return Err(incomplete("enumeration concatenation time limit exceeded"));
                }
            }
        }
        let generated = forward.arena.len() + backward.arena.len();
        Ok(Self {
            labels: forward,
            paths,
            completed: true,
            generated,
        })
    }
}
