use super::*;

impl BucketGraph {
    pub fn prepare<const D: usize>(
        g: &Graph<D>,
        steps: impl AsRef<[f64]>,
        target_buckets: i32,
        bidirectional: bool,
    ) -> Result<Self> {
        g.validate()?;
        let steps = steps.as_ref();
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
                    .position(|k| *k == crate::graph::ResourceKind::Disposable)
            })
            .unwrap_or(0);
        let mut resource_slots: Vec<_> = g
            .resources
            .iter()
            .enumerate()
            .filter_map(|(r, k)| {
                (r == main
                    || (*k == crate::graph::ResourceKind::Disposable
                        && steps.get(r).copied().unwrap_or(0.) > 1e-6))
                    .then_some(r)
            })
            .collect();
        // Resource dimensions and indexing axes are independent. Always retain
        // the designated main resource, including when its index exceeds one.
        if g.resources.len() > 2 {
            if let Some(axis) = resource_slots.iter().position(|&r| r == main) {
                resource_slots.swap(0, axis);
            }
        }
        resource_slots.truncate(2);
        let mut grid_steps = [0.; 2];
        for (axis, &r) in resource_slots.iter().enumerate() {
            grid_steps[axis] = steps.get(r).copied().unwrap_or(0.);
        }
        let dimensions = resource_slots.len().max(1);
        let n = order.len();
        let mut lower = vec![[0.; 2]; n];
        let mut upper = lower.clone();
        for (v, &id) in order.iter().enumerate() {
            for (r, &slot) in resource_slots.iter().enumerate() {
                let bounds = g.vertices[id].bounds[slot];
                lower[v][r] = g
                    .project_resource(id, slot, bounds, bounds.lower, true)
                    .unwrap_or(bounds.lower);
                upper[v][r] = g
                    .project_resource(id, slot, bounds, bounds.upper, false)
                    .unwrap_or(bounds.upper);
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
        if g.has_time_windows() {
            // Incoming arc hulls can end inside a gap. Reproject both grid
            // endpoints so the first terminal bucket contains its root.
            // If the hull intersection is empty, retain the vertex domain;
            // exact transitions will reject every incoming arc instead.
            for (v, &id) in order.iter().enumerate() {
                for (axis, &slot) in resource_slots.iter().enumerate() {
                    let effective = crate::graph::Window {
                        lower: lower[v][axis],
                        upper: upper[v][axis],
                    };
                    let first = g.project_resource(id, slot, effective, effective.lower, true);
                    let last = g.project_resource(id, slot, effective, effective.upper, false);
                    if let (Some(first), Some(last)) = (first, last) {
                        lower[v][axis] = first;
                        upper[v][axis] = last;
                    } else {
                        let original = g.vertices[id].bounds[slot];
                        lower[v][axis] = g
                            .project_resource(id, slot, original, original.lower, true)
                            .unwrap_or(original.lower);
                        upper[v][axis] = g
                            .project_resource(id, slot, original, original.upper, false)
                            .unwrap_or(original.upper);
                    }
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
            step: grid_steps,
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
        let mut unit = [1_000_000_000i64; 2];
        // Keep scaled integers exact in both the conversion and the later
        // conversion back to f64. Large resource windows need not fit i32.
        const MAX_EXACT_INTEGER: f64 = 9_007_199_254_740_992.;
        for v in 0..self.order.len() {
            for r in 0..self.dimensions {
                for x in [self.lower[v][r], self.upper[v][r]] {
                    let scaled = (10000. * x).ceil();
                    if !scaled.is_finite() || scaled.abs() > MAX_EXACT_INTEGER {
                        return Err(invalid("bucket grid unit overflows integer domain"));
                    }
                    unit[r] = gcd(unit[r], scaled as i64);
                }
            }
        }
        for a in &self.consumption {
            for r in 0..self.dimensions {
                let scaled = (10000. * a[r]).ceil();
                if !scaled.is_finite() || scaled.abs() > MAX_EXACT_INTEGER {
                    return Err(invalid("arc grid unit overflows integer domain"));
                }
                unit[r] = gcd(unit[r], scaled as i64);
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

    // Both take `q` element by element through selects: indexing it with a
    // runtime slot keeps the array in memory, and reloading it right after
    // its two scalar stores stalls on store forwarding in the labeling loop.
    #[inline]
    pub fn coordinate(&self, q: impl AsRef<[f64]>) -> [f64; 2] {
        let q = q.as_ref();
        let pick = |r: usize| {
            if r == 0 {
                q[0]
            } else if r == 1 {
                q[1]
            } else {
                q[r]
            }
        };
        let slots = &self.resource_slots;
        [
            slots.first().map_or(0., |&r| pick(r)),
            slots.get(1).map_or(0., |&r| pick(r)),
        ]
    }

    #[inline(always)]
    pub fn bucket_id(&self, forward: bool, vertex: usize, q: [f64; 2]) -> Option<usize> {
        let term = |r: usize, q: f64| {
            let x = if forward {
                q - (self.lower[vertex][r] - 1e-6)
            } else {
                self.upper[vertex][r] + 1e-6 - q
            };
            floor_isize(x / self.step[r]) * self.stride[vertex][r] as isize
        };
        let mut index = 0isize;
        if self.dimensions > 0 {
            index += term(0, q[0]);
        }
        if self.dimensions > 1 {
            index += term(1, q[1]);
        }
        usize::try_from(index).ok().filter(|&i| {
            self.direction(forward)
                .buckets
                .get(vertex)
                .is_some_and(|v| i < v.len())
        })
    }
}
