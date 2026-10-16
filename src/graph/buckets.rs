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
fn gcd(mut a: i32, mut b: i32) -> i32 {
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
impl BucketGraph {
    pub fn prepare(
        g: &Graph,
        steps: [f64; 2],
        target_buckets: i32,
        bidirectional: bool,
    ) -> Result<Self> {
        g.validate()?;
        if target_buckets <= 0 {
            return Err(invalid("positive bucket count required"));
        }
        let mut order = vec![g.source];
        order.extend(
            (0..g.vertices.len())
                .rev()
                .filter(|&v| v != g.source && v != g.sink),
        );
        order.push(g.sink);
        let mut internal = vec![0; order.len()];
        for (i, &v) in order.iter().enumerate() {
            internal[v] = i;
        }
        let main = g
            .main_resource
            .filter(|&r| r < g.resources.len())
            .or_else(|| {
                g.resources
                    .iter()
                    .position(|k| *k == super::ResourceKind::Disposable)
            })
            .unwrap_or(0);
        let resource_slots: Vec<_> = g
            .resources
            .iter()
            .enumerate()
            .filter_map(|(r, k)| {
                (r == main || (*k == super::ResourceKind::Disposable && steps[r] > 1e-6))
                    .then_some(r)
            })
            .collect();
        let mut grid_steps = [0.; 2];
        for (axis, &r) in resource_slots.iter().enumerate() {
            grid_steps[axis] = steps[r];
        }
        let dimensions = resource_slots.len().max(1);
        let n = order.len();
        let mut lower = vec![[0.; 2]; n];
        let mut upper = lower.clone();
        for (v, &id) in order.iter().enumerate() {
            for (r, &slot) in resource_slots.iter().enumerate() {
                lower[v][r] = g.vertices[id].bounds[slot].lower;
                upper[v][r] = g.vertices[id].bounds[slot].upper;
            }
        }
        for (r, &slot) in resource_slots.iter().enumerate() {
            if let Some(initial) = g.initial[slot] {
                lower[0][r] = lower[0][r].max(initial);
            }
        }
        let mut consumption = vec![[0.; 2]; g.arcs.len()];
        let mut arc_lower = consumption.clone();
        let mut arc_upper = consumption.clone();
        for (i, a) in g.arcs.iter().enumerate() {
            let h = internal[a.head];
            for (r, &slot) in resource_slots.iter().enumerate() {
                consumption[i][r] = a.consumption[slot];
                arc_lower[i][r] =
                    lower[h][r].max(a.bounds[slot].map_or(f64::NEG_INFINITY, |w| w.lower));
                arc_upper[i][r] =
                    upper[h][r].min(a.bounds[slot].map_or(f64::INFINITY, |w| w.upper));
            }
        }
        // Incoming arc windows tighten each
        // vertex; retain terminal upper seeds before widening source resources.
        for v in 0..n {
            for r in 0..resource_slots.len() {
                let mut incoming_lower = f64::INFINITY;
                let mut incoming_upper = f64::NEG_INFINITY;
                for (i, a) in g.arcs.iter().enumerate() {
                    if internal[a.head] == v {
                        incoming_lower = incoming_lower.min(arc_lower[i][r]);
                        incoming_upper = incoming_upper.max(arc_upper[i][r]);
                    }
                }
                if incoming_lower.is_finite() {
                    lower[v][r] = lower[v][r].max(incoming_lower);
                }
                if incoming_upper.is_finite() {
                    upper[v][r] = upper[v][r].min(incoming_upper);
                }
            }
        }
        for r in 0..resource_slots.len() {
            upper[0][r] = upper[n - 1][r];
        }
        let mut arc_order: Vec<_> = (0..g.arcs.len()).collect();
        arc_order.sort_by_key(|&a| {
            (
                internal[g.arcs[a].tail],
                internal[g.arcs[a].head],
                std::cmp::Reverse(a),
            )
        });
        let mut centers = 0.;
        for v in 0..n {
            centers = centers + upper[v][0] + lower[v][0];
        }
        let middle = (centers / (2 * n) as f64 * 10. + 0.5).floor() / 10.;
        let mut grid = Self {
            forward: Direction::default(),
            backward: Direction::default(),
            bidirectional,
            dimensions,
            lower,
            upper,
            stride: vec![[0; 2]; n],
            consumption,
            arc_lower,
            arc_upper,
            step: steps,
            middle,
            jump_arcs: true,
            version: 0,
            backward_version: 0,
            target_buckets,
            resource_slots,
            order,
            internal,
            arc_order,
        };
        grid.set_steps(grid_steps)?;
        grid.rebuild(g, true)?;
        if bidirectional {
            grid.rebuild(g, false)?;
        }
        grid.regenerate_jumps(g, true);
        if bidirectional {
            grid.regenerate_jumps(g, false);
        }
        Ok(grid)
    }
    pub fn set_steps(&mut self, steps: [f64; 2]) -> Result<()> {
        self.step = steps;
        if steps[..self.dimensions]
            .iter()
            .all(|s| s.is_finite() && *s >= 1e-6)
        {
            return Ok(());
        }
        let mut unit = [1_000_000_000i32; 2];
        for v in 0..self.order.len() {
            for r in 0..self.dimensions {
                for x in [self.lower[v][r], self.upper[v][r]] {
                    let scaled = (10000. * x).ceil();
                    if !scaled.is_finite() || scaled < i32::MIN as f64 || scaled > i32::MAX as f64 {
                        return Err(invalid("bucket grid unit overflows integer domain"));
                    }
                    unit[r] = gcd(unit[r], scaled as i32);
                }
            }
        }
        for a in &self.consumption {
            for r in 0..self.dimensions {
                let scaled = (10000. * a[r]).ceil();
                if !scaled.is_finite() || scaled < i32::MIN as f64 || scaled > i32::MAX as f64 {
                    return Err(invalid("arc grid unit overflows integer domain"));
                }
                unit[r] = gcd(unit[r], scaled as i32);
            }
        }
        let last = self.order.len() - 1;
        let range0 = self.upper[last][0] - self.lower[0][0];
        if range0 < 1e-6 {
            self.step = [1.; 2];
            return Ok(());
        }
        if unit[..self.dimensions].contains(&0) {
            return Err(invalid("zero bucket grid unit"));
        }
        if self.dimensions == 1 {
            let u = unit[0] as f64;
            self.step[0] = (range0 / self.target_buckets as f64 * 10000. / u).ceil() * u / 10000.;
        } else {
            let root = (self.target_buckets as f64).sqrt();
            let range1 = self.upper[last][1] - self.lower[0][1];
            if range1 < 1e-6 {
                self.step = [1.; 2];
            } else {
                let u0 = unit[0] as f64;
                let u1 = unit[1] as f64;
                self.step = [
                    u0 * (range0 / root * 10000. / u0).ceil() / 10000.,
                    (range1 / root * 10000. / u1).ceil() * u1 / 10000.,
                ];
            }
        }
        Ok(())
    }
    pub fn direction(&self, forward: bool) -> &Direction {
        if forward {
            &self.forward
        } else {
            &self.backward
        }
    }
    fn direction_mut(&mut self, forward: bool) -> &mut Direction {
        if forward {
            &mut self.forward
        } else {
            &mut self.backward
        }
    }
    pub fn coordinate(&self, q: [f64; 2]) -> [f64; 2] {
        let mut result = [0.; 2];
        for (i, &r) in self.resource_slots.iter().enumerate() {
            result[i] = q[r];
        }
        result
    }
    pub fn bucket_id(&self, forward: bool, vertex: usize, q: [f64; 2]) -> Option<usize> {
        let mut index = 0isize;
        for r in 0..self.dimensions {
            let x = if forward {
                q[r] - (self.lower[vertex][r] - 1e-6)
            } else {
                self.upper[vertex][r] + 1e-6 - q[r]
            };
            index += (x / self.step[r]).floor() as isize * self.stride[vertex][r] as isize;
        }
        usize::try_from(index).ok().filter(|&i| {
            self.direction(forward)
                .buckets
                .get(vertex)
                .is_some_and(|v| i < v.len())
        })
    }
    pub fn propagate(
        &self,
        g: &Graph,
        forward: bool,
        physical: usize,
        mut q: [f64; 2],
    ) -> Option<[f64; 2]> {
        let a = &g.arcs[physical];
        let tail = self.internal[a.tail];
        for r in 0..self.dimensions {
            if forward {
                q[r] += self.consumption[physical][r];
                if q[r] < self.arc_lower[physical][r] {
                    q[r] = self.arc_lower[physical][r];
                }
                if q[r] > self.arc_upper[physical][r] {
                    return None;
                }
            } else {
                if q[r] > self.arc_upper[physical][r] {
                    q[r] = self.arc_upper[physical][r];
                }
                if q[r] < self.arc_lower[physical][r] {
                    return None;
                }
                q[r] -= self.consumption[physical][r];
                if q[r] > self.upper[tail][r] {
                    q[r] = self.upper[tail][r];
                }
                if q[r] < self.lower[tail][r] {
                    return None;
                }
            }
        }
        Some(q)
    }
    pub fn flags(&self, g: &Graph, forward: bool, a: &BucketArc) -> u8 {
        let d = self.direction(forward);
        let mut q = d.corners[a.from.vertex][a.from.bucket];
        if let Some(jump) = a.jump {
            for r in 0..self.dimensions {
                let x = d.corners[a.from.vertex][jump][r];
                if forward {
                    q[r] = q[r].max(x);
                } else {
                    q[r] = q[r].min(x);
                }
            }
        }
        let Some(q) = self.propagate(g, forward, a.physical, q) else {
            return 0;
        };
        let border = self.middle + 1e-6;
        if forward {
            let c = d.corners[a.from.vertex][a.from.bucket][0];
            u8::from(c <= border) | if self.step[0] + c > border { 2 } else { 0 }
        } else {
            u8::from(q[0] > border) | if q[0] - self.step[0] <= border { 2 } else { 0 }
        }
    }
    pub fn move_middle(&mut self, g: &Graph, forward: bool) {
        let m = if forward {
            self.middle + (self.upper[self.order.len() - 1][0] - self.middle) * 0.05
        } else {
            self.middle - (self.middle - self.lower[0][0]) * 0.05
        };
        self.middle = (m * 10. + 0.5).floor() / 10.;
        for direction in [true, false] {
            if !direction && !self.bidirectional {
                continue;
            }
            let entries: Vec<_> = self
                .direction(direction)
                .buckets
                .iter()
                .flat_map(|row| row.iter().flat_map(|b| b.arcs.iter().cloned()))
                .map(|a| {
                    let flags = self.flags(g, direction, &a);
                    (a.from, a.id, flags)
                })
                .collect();
            for (b, id, flags) in entries {
                if let Some(a) = self.direction_mut(direction).buckets[b.vertex][b.bucket]
                    .arcs
                    .iter_mut()
                    .find(|a| a.id == id)
                {
                    a.flags = flags;
                }
            }
        }
    }
    fn regenerate(&mut self, forward: bool) -> Result<()> {
        if self.step[..self.dimensions]
            .iter()
            .any(|s| !s.is_finite() || *s <= 0.)
        {
            return Err(invalid("positive finite bucket step required"));
        }
        let mut rows = vec![];
        let mut corners = vec![];
        for v in 0..self.order.len() {
            let lo = self.lower[v];
            let up = self.upper[v];
            let mut total = 1usize;
            for r in 0..self.dimensions {
                if lo[r] > up[r] + 1e-6 {
                    return Err(invalid("bucket lower exceeds upper bound"));
                }
                let cells = (((up[r] + 1e-6) - lo[r]) / self.step[r]).floor() + 1.;
                if !cells.is_finite() || cells < 1. || cells > i32::MAX as f64 {
                    return Err(invalid("bucket count overflow"));
                }
                if forward {
                    self.stride[v][r] = total;
                }
                total = total
                    .checked_mul(cells as usize)
                    .ok_or_else(|| invalid("bucket count overflow"))?;
            }
            if total > i32::MAX as usize {
                return Err(invalid("bucket count overflow"));
            }
            let mut row = Vec::with_capacity(total);
            let mut points = Vec::with_capacity(total);
            let mut q = if forward { lo } else { up };
            for k in 0..total {
                row.push(Bucket {
                    id: BucketId {
                        vertex: v,
                        bucket: k,
                    },
                    component: None,
                    reachable: true,
                    predecessors: vec![],
                    arcs: vec![],
                });
                q[0] = round8(q[0]);
                if self.dimensions == 2 {
                    q[1] = round8(q[1]);
                }
                points.push(q);
                if forward {
                    q[0] += self.step[0];
                    if !(up[0] + 1e-6 > q[0]) {
                        q[0] = lo[0];
                        if self.dimensions == 2 {
                            q[1] += self.step[1];
                            if !(up[1] + 1e-6 > q[1]) {
                                q[1] = lo[1];
                            }
                        }
                    }
                } else {
                    q[0] -= self.step[0];
                    if q[0] <= lo[0] - 1e-6 {
                        q[0] = up[0];
                        if self.dimensions == 2 {
                            q[1] -= self.step[1];
                            if q[1] <= lo[1] - 1e-6 {
                                q[1] = up[1];
                            }
                        }
                    }
                }
            }
            rows.push(row);
            corners.push(points);
        }
        let d = self.direction_mut(forward);
        d.buckets = rows;
        d.corners = corners;
        Ok(())
    }
    pub fn rebuild(&mut self, g: &Graph, forward: bool) -> Result<()> {
        self.regenerate(forward)?;
        let mut counter = 0;
        for &index in &self.arc_order.clone() {
            let arc = &g.arcs[index];
            let v = self.internal[if forward { arc.tail } else { arc.head }];
            let count = self.direction(forward).buckets[v].len();
            for k in 0..count {
                let id = counter;
                counter += 1;
                let entry = BucketArc {
                    id,
                    from: BucketId {
                        vertex: v,
                        bucket: k,
                    },
                    physical: index,
                    jump: None,
                    flags: 0,
                };
                let flags = self.flags(g, forward, &entry);
                if flags != 0 {
                    self.direction_mut(forward).buckets[v][k]
                        .arcs
                        .push(BucketArc { flags, ..entry });
                }
            }
        }
        self.direction_mut(forward).arc_counter = counter;
        self.order_components(g, forward)?;
        self.refresh_counts(g, forward);
        Ok(())
    }
    fn refresh_counts(&mut self, g: &Graph, forward: bool) {
        let current = self
            .direction(forward)
            .buckets
            .iter()
            .flatten()
            .filter(|b| b.reachable)
            .map(|b| b.arcs.len())
            .sum();
        let mut maximum = 0;
        for (v, row) in self.direction(forward).buckets.iter().enumerate() {
            let original = self.order[v];
            let degree = g.arcs.iter().filter(|a| a.tail == original).count();
            maximum += row.iter().filter(|b| b.reachable).count() * degree;
        }
        let d = self.direction_mut(forward);
        d.current_arcs = current;
        d.maximum_arcs = maximum;
    }
    pub fn regenerate_jumps(&mut self, g: &Graph, forward: bool) {
        let mut identity = self.direction(forward).arc_counter;
        let mut attempts = 0;
        for v in 0..self.order.len() {
            let count = self.direction(forward).buckets[v].len();
            let mut later: BTreeMap<i32, (usize, VecDeque<usize>)> = BTreeMap::new();
            for k in (0..count).rev() {
                if !self.direction(forward).buckets[v][k].reachable {
                    continue;
                }
                self.direction_mut(forward).buckets[v][k]
                    .arcs
                    .retain(|a| a.jump.is_none());
                let ordinary: BTreeSet<_> = self.direction(forward).buckets[v][k]
                    .arcs
                    .iter()
                    .map(|a| g.arcs[a.physical].id)
                    .collect();
                let mut add = vec![];
                for (&physical, (index, targets)) in &mut later {
                    if ordinary.contains(&physical) {
                        if self.dimensions == 1 {
                            targets.pop_front();
                        }
                        targets.push_front(k);
                    } else if self.dimensions == 1 {
                        add.push((*index, targets[0]));
                    } else {
                        let stride = self.stride[v][1];
                        let mut row = k / stride;
                        let col = k % stride;
                        let mut last_col = stride;
                        for &target in targets.iter() {
                            let next_col = target % stride;
                            let next_row = target / stride;
                            if col <= next_col && next_col < last_col && row < next_row {
                                add.push((*index, target));
                                row = next_row;
                                last_col = next_col;
                                if col == next_col {
                                    break;
                                }
                            }
                        }
                    }
                }
                for (index, target) in add {
                    let mut a = BucketArc {
                        id: identity,
                        from: BucketId {
                            vertex: v,
                            bucket: k,
                        },
                        physical: index,
                        jump: Some(target),
                        flags: 0,
                    };
                    identity += 1;
                    attempts += 1;
                    a.flags = self.flags(g, forward, &a);
                    if a.flags != 0 {
                        self.direction_mut(forward).buckets[v][k].arcs.push(a);
                    }
                }
                for a in self.direction(forward).buckets[v][k]
                    .arcs
                    .iter()
                    .filter(|a| a.jump.is_none())
                {
                    later
                        .entry(g.arcs[a.physical].id)
                        .or_insert_with(|| (a.physical, VecDeque::from([k])));
                }
            }
        }
        self.direction_mut(forward).jump_attempts = attempts;
    }
    fn auxiliary(&self, g: &Graph, forward: bool) -> (Vec<BucketId>, Vec<usize>, Vec<Vec<usize>>) {
        let d = self.direction(forward);
        let mut ids = vec![];
        let mut offsets = vec![];
        for row in &d.buckets {
            offsets.push(ids.len());
            ids.extend(row.iter().map(|b| b.id));
        }
        let mut edges = vec![vec![]; ids.len()];
        for (source, id) in ids.iter().enumerate() {
            for a in &d.buckets[id.vertex][id.bucket].arcs {
                let physical = &g.arcs[a.physical];
                let target = self.internal[if forward {
                    physical.head
                } else {
                    physical.tail
                }];
                let corner = d.corners[id.vertex][id.bucket];
                let Some(mut high) = self.propagate(g, forward, a.physical, corner) else {
                    continue;
                };
                let mut low = high;
                let cons = self.consumption[a.physical];
                let lo = self.arc_lower[a.physical];
                let up = self.arc_upper[a.physical];
                if forward {
                    let x = (corner[0] + self.step[0] - 1e-6).min(self.upper[id.vertex][0]);
                    let max0 = lo[0].max(x + cons[0]);
                    if self.dimensions == 1 {
                        high[0] = up[0].min(max0);
                    } else {
                        let (selected, c1) = if up[0] < max0 {
                            (up[0], corner[1])
                        } else {
                            (if low[1] <= up[1] { max0 } else { up[0] }, low[1])
                        };
                        high[0] = lo[0].max(selected + cons[0]);
                        let t1 = (self.step[1] - 1e-6 + c1).min(self.upper[id.vertex][1]);
                        high[1] = up[1];
                        if high[0] <= up[0] {
                            let m = lo[1].max(t1 + cons[1]);
                            if m <= up[1] {
                                high[1] = m;
                            }
                        }
                    }
                } else {
                    low = corner;
                    low[0] = (corner[0] - (self.step[0] - 1e-6)).max(self.lower[id.vertex][0]);
                    let value = low[0].min(up[0]);
                    let mut feasible = lo[0] <= value;
                    if feasible {
                        low[0] = (value - cons[0]).min(self.upper[target][0]);
                        feasible = self.lower[target][0] <= low[0];
                    }
                    if !feasible {
                        low[0] = self.lower[target][0];
                    }
                    if self.dimensions == 2 {
                        if feasible {
                            low[1] = high[1];
                        }
                        low[1] = (low[1] - (self.step[1] - 1e-6)).max(self.lower[id.vertex][1]);
                        low[0] = low[0].min(up[0]);
                        feasible = lo[0] <= low[0];
                        if feasible {
                            low[0] = (low[0] - cons[0]).min(self.upper[target][0]);
                            feasible = self.lower[target][0] <= low[0];
                        }
                        if feasible {
                            let value = low[1].min(up[1]);
                            feasible = lo[1] <= value;
                            if feasible {
                                low[1] = (value - cons[1]).min(self.upper[target][1]);
                                feasible = self.lower[target][1] <= low[1];
                            }
                        }
                        if !feasible {
                            low[1] = self.lower[target][1];
                        }
                    }
                }
                let Some(start) = self.bucket_id(forward, target, if forward { low } else { high })
                else {
                    continue;
                };
                let corners = &d.corners[target];
                let mut row = start;
                let mut k = start;
                loop {
                    edges[source].push(offsets[target] + k);
                    k += 1;
                    let next_row = if k >= corners.len() {
                        if self.dimensions != 2 {
                            break;
                        }
                        true
                    } else if forward {
                        let end = corners[k][0] - 1e-7;
                        if self.dimensions == 2 {
                            high[0] <= end || corners[row][1] != corners[k][1]
                        } else {
                            if high[0] <= end {
                                break;
                            }
                            false
                        }
                    } else {
                        let end = corners[k][0] + 1e-7;
                        if self.dimensions == 2 {
                            end <= low[0] || corners[k - 1][1] != corners[k][1]
                        } else {
                            if end <= low[0] {
                                break;
                            }
                            false
                        }
                    };
                    if next_row {
                        row += self.stride[target][1];
                        if row < corners.len()
                            && if forward {
                                corners[row][1] - 1e-7 < high[1]
                            } else {
                                low[1] < corners[row][1] + 1e-7
                            }
                        {
                            k = row;
                        } else {
                            break;
                        }
                    }
                }
            }
        }
        (ids, offsets, edges)
    }
    fn order_components(&mut self, g: &Graph, forward: bool) -> Result<()> {
        let (ids, offsets, mut edges) = self.auxiliary(g, forward);
        if ids.is_empty() {
            return Err(invalid("empty bucket graph"));
        }
        let root = if forward {
            0
        } else {
            offsets[self.order.len() - 1]
        };
        let mut reached = vec![false; ids.len()];
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            if reached[i] {
                continue;
            }
            reached[i] = true;
            stack.extend(edges[i].iter().rev().copied());
        }
        for (i, id) in ids.iter().enumerate() {
            let b = &mut self.direction_mut(forward).buckets[id.vertex][id.bucket];
            b.reachable = reached[i];
            b.component = None;
            b.predecessors.clear();
            if !reached[i] {
                b.arcs.clear();
            }
        }
        for (i, id) in ids.iter().enumerate() {
            let v = id.vertex;
            let idx = id.bucket;
            let mut preds = vec![];
            if self.dimensions == 1 {
                if let Some(k) = (0..idx).rev().find(|&k| reached[offsets[v] + k]) {
                    preds.push(Predecessor {
                        bucket: BucketId {
                            vertex: v,
                            bucket: k,
                        },
                        minimum: 0,
                    });
                }
            } else {
                let stride = self.stride[v][1] as isize;
                let mut row = idx as isize / stride;
                let col = idx as isize % stride;
                let mut last = -1;
                let mut j = col - 1;
                let mut base = stride * row;
                loop {
                    if last < j {
                        let mut jj = j;
                        while jj > last && !reached[offsets[v] + (base + jj) as usize] {
                            jj -= 1;
                        }
                        if jj > last {
                            preds.push(Predecessor {
                                bucket: BucketId {
                                    vertex: v,
                                    bucket: (base + jj) as usize,
                                },
                                minimum: 0,
                            });
                            last = jj;
                        }
                        j = jj;
                    }
                    if col != j {
                        if let Some(p) = preds.last_mut() {
                            p.minimum = base as usize;
                        }
                    }
                    row -= 1;
                    base -= stride;
                    if last >= col || row < 0 {
                        break;
                    }
                    j = col;
                }
            }
            if reached[i] {
                for p in &preds {
                    edges[offsets[v] + p.bucket.bucket].push(i);
                }
            }
            self.direction_mut(forward).buckets[v][idx].predecessors = preds;
        }
        // Iterative Tarjan retains native adjacency visitation order without a
        // recursive call stack proportional to the number of resource cells.
        let n = ids.len();
        let mut index = vec![None; n];
        let mut low = vec![0; n];
        let mut on = vec![false; n];
        let mut active = vec![];
        let mut component = vec![usize::MAX; n];
        let mut serial = 0;
        let mut count = 0;
        index[root] = Some(serial);
        low[root] = serial;
        serial += 1;
        active.push(root);
        on[root] = true;
        let mut frames = vec![(root, 0usize)];
        while let Some((v, next)) = frames.last_mut() {
            let v = *v;
            if *next < edges[v].len() {
                let w = edges[v][*next];
                *next += 1;
                if !reached[w] {
                    continue;
                }
                if index[w].is_none() {
                    index[w] = Some(serial);
                    low[w] = serial;
                    serial += 1;
                    active.push(w);
                    on[w] = true;
                    frames.push((w, 0));
                } else if on[w] {
                    low[v] = low[v].min(index[w].unwrap());
                }
            } else {
                frames.pop();
                if low[v] == index[v].unwrap() {
                    loop {
                        let w = active.pop().unwrap();
                        on[w] = false;
                        component[w] = count;
                        if w == v {
                            break;
                        }
                    }
                    count += 1;
                }
                if let Some((parent, _)) = frames.last() {
                    low[*parent] = low[*parent].min(low[v]);
                }
            }
        }
        let mut components = vec![vec![]; count];
        for (i, &id) in ids.iter().enumerate() {
            if reached[i] {
                let c = count - 1 - component[i];
                self.direction_mut(forward).buckets[id.vertex][id.bucket].component = Some(c);
                components[c].push(id);
            }
        }
        for row in &mut components {
            row.sort_by_key(|id| (id.bucket, id.vertex));
        }
        let mut depth = vec![0; self.order.len()];
        for v in 0..depth.len() {
            let Some(last) = self.direction(forward).buckets[v].last() else {
                continue;
            };
            let mut work = vec![(last.id.bucket, 0usize, 0i32)];
            while let Some((k, minimum, level)) = work.pop() {
                depth[v] = depth[v].max(level);
                for p in &self.direction(forward).buckets[v][k].predecessors {
                    if p.bucket.bucket >= minimum {
                        work.push((p.bucket.bucket, minimum.max(p.minimum), level + 1));
                    }
                }
            }
        }
        let d = self.direction_mut(forward);
        d.components = components;
        for (v, old) in d.depth.iter().enumerate() {
            depth[v] = depth[v].max(*old);
        }
        d.depth = depth;
        Ok(())
    }
}
impl BucketGraph {
    /// Refine cells while transferring only surviving physical arcs. Rebuilding
    /// from the physical graph here would silently undo reduced-cost fixing.
    pub fn decrease_steps(&mut self, g: &Graph) -> Result<()> {
        let old_steps = self.step;
        let old_forward = self.forward.clone();
        let old_backward = self.backward.clone();
        for r in 0..self.dimensions {
            self.step[r] = if self.dimensions == 1 {
                (old_steps[r] * 5. + 0.5).floor() / 10.
            } else {
                (old_steps[r] * 10. / 1.4 + 0.5).floor() / 10.
            };
        }
        if self.step[..self.dimensions].iter().any(|&s| s <= 0.) {
            self.step = old_steps;
            return Err(invalid("bucket step cannot be refined further"));
        }
        for forward in [true, false] {
            if !forward && !self.bidirectional {
                continue;
            }
            let old = if forward { &old_forward } else { &old_backward };
            self.regenerate(forward)?;
            let mut identity = 0;
            for v in 0..self.order.len() {
                let count = self.direction(forward).buckets[v].len();
                for k in 0..count {
                    let point = self.direction(forward).corners[v][k];
                    let mut selected = BTreeSet::new();
                    let old_row = (((self.upper[v][0] - self.lower[v][0]) / old_steps[0]).floor()
                        as usize)
                        + 1;
                    let origin = if forward {
                        point[0] - (self.lower[v][0] - 1e-6)
                    } else {
                        self.upper[v][0] + 1e-6 - point[0]
                    };
                    let mut index = (origin / old_steps[0]).floor() as usize;
                    if self.dimensions == 2 {
                        let origin = if forward {
                            point[1] - (self.lower[v][1] - 1e-6)
                        } else {
                            self.upper[v][1] + 1e-6 - point[1]
                        };
                        index += ((origin / old_steps[1]).floor() as usize) * old_row;
                    }
                    let bound0 = if forward {
                        (point[0] + self.step[0] - 1e-6).min(self.upper[v][0])
                    } else {
                        (point[0] - self.step[0] + 1e-6).max(self.lower[v][0])
                    };
                    let bound1 = if forward {
                        (point[1] + self.step[1] - 1e-6).min(self.upper[v][1])
                    } else {
                        (point[1] - self.step[1] + 1e-6).max(self.lower[v][1])
                    };
                    let within = if self.dimensions == 1 {
                        index
                    } else {
                        index % old_row
                    };
                    let origin0 = if forward {
                        self.lower[v][0] + within as f64 * old_steps[0]
                    } else {
                        self.upper[v][0] - within as f64 * old_steps[0]
                    };
                    let mut run1 = if forward {
                        self.lower[v][1] + (index / old_row) as f64 * old_steps[1]
                    } else {
                        self.upper[v][1] - (index / old_row) as f64 * old_steps[1]
                    };
                    let mut base = index;
                    loop {
                        let mut coord = origin0;
                        let mut scan = base;
                        loop {
                            if let Some(bucket) = old.buckets[v].get(scan) {
                                selected.extend(
                                    bucket
                                        .arcs
                                        .iter()
                                        .filter(|a| a.jump.is_none())
                                        .map(|a| a.physical),
                                );
                            }
                            coord = if forward {
                                coord + old_steps[0]
                            } else {
                                coord - old_steps[0]
                            };
                            if !coord.is_finite()
                                || if forward {
                                    bound0 <= coord - 1e-7
                                } else {
                                    coord + 1e-7 <= bound0
                                }
                            {
                                break;
                            }
                            scan += 1;
                        }
                        if self.dimensions != 2 {
                            break;
                        }
                        run1 = if forward {
                            run1 + old_steps[1]
                        } else {
                            run1 - old_steps[1]
                        };
                        base += old_row;
                        if base >= old.buckets[v].len()
                            || !run1.is_finite()
                            || if forward {
                                bound1 <= run1 - 1e-7
                            } else {
                                run1 + 1e-7 <= bound1
                            }
                        {
                            break;
                        }
                    }
                    let order: Vec<_> = self
                        .arc_order
                        .iter()
                        .copied()
                        .filter(|a| selected.contains(a))
                        .collect();
                    for physical in order {
                        let mut arc = BucketArc {
                            id: identity,
                            from: BucketId {
                                vertex: v,
                                bucket: k,
                            },
                            physical,
                            jump: None,
                            flags: 0,
                        };
                        identity += 1;
                        arc.flags = self.flags(g, forward, &arc);
                        if arc.flags != 0 {
                            self.direction_mut(forward).buckets[v][k].arcs.push(arc);
                        }
                    }
                }
            }
            self.direction_mut(forward).arc_counter = identity;
            self.order_components(g, forward)?;
            self.refresh_counts(g, forward);
            self.regenerate_jumps(g, forward);
            let factor = if self.dimensions == 1 { 2. } else { 1.4 };
            for depth in &mut self.direction_mut(forward).depth {
                *depth = if *depth < 1 {
                    1
                } else {
                    (*depth as f64 * factor) as i32
                };
            }
        }
        self.version += 1;
        if self.bidirectional {
            self.backward_version += 1;
        }
        Ok(())
    }
}
impl BucketGraph {
    pub fn ensure_backward(&mut self, g: &Graph) -> Result<()> {
        if self.backward.buckets.is_empty() {
            self.rebuild(g, false)?;
            self.regenerate_jumps(g, false);
        }
        Ok(())
    }
}
