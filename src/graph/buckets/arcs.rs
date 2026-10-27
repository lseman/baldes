use super::*;

impl BucketGraph {
    pub(super) fn validate_bucket_geometry(&self, steps: [f64; 2]) -> Result<()> {
        if steps[..self.dimensions]
            .iter()
            .any(|s| !s.is_finite() || *s <= 0.)
        {
            return Err(invalid("positive finite bucket step required"));
        }
        for v in 0..self.order.len() {
            let mut total = 1usize;
            for (r, &step) in steps.iter().enumerate().take(self.dimensions) {
                if self.lower[v][r] > self.upper[v][r] + 1e-6 {
                    return Err(invalid("bucket lower exceeds upper bound"));
                }
                let cells = (((self.upper[v][r] + 1e-6) - self.lower[v][r]) / step).floor() + 1.;
                if !cells.is_finite() || cells < 1. || cells > i32::MAX as f64 {
                    return Err(invalid("bucket count overflow"));
                }
                total = total
                    .checked_mul(cells as usize)
                    .ok_or_else(|| invalid("bucket count overflow"))?;
            }
            if total > i32::MAX as usize {
                return Err(invalid("bucket count overflow"));
            }
        }
        Ok(())
    }

    pub fn propagate<const D: usize>(
        &self,
        g: &Graph<D>,
        forward: bool,
        physical: usize,
        mut q: [f64; 2],
    ) -> Option<[f64; 2]> {
        let a = &g.arcs[physical];
        let tail = self.internal[a.tail];
        for r in 0..self.dimensions {
            if self
                .resource_slots
                .get(r)
                .is_some_and(|&slot| g.resource_has_time_windows(slot))
            {
                q[r] =
                    crate::labeling::extend_resource(g, a, self.resource_slots[r], q[r], forward)?;
                continue;
            }
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

    pub fn flags<const D: usize>(&self, g: &Graph<D>, forward: bool, a: &BucketArc) -> u8 {
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
        // A backward projection can cross an arbitrarily large window gap
        // inside one input cell. `q - step` is not a lower image bound;
        // retain both phase flags and let actual label resources choose.
        if !forward && g.has_time_windows() {
            return 3;
        }
        let border = self.middle + 1e-6;
        if forward {
            let c = d.corners[a.from.vertex][a.from.bucket][0];
            u8::from(c <= border) | if self.step[0] + c > border { 2 } else { 0 }
        } else {
            u8::from(q[0] > border) | if q[0] - self.step[0] <= border { 2 } else { 0 }
        }
    }

    pub fn move_middle<const D: usize>(&mut self, g: &Graph<D>, forward: bool) {
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
            let flags: Vec<_> = self
                .direction(direction)
                .buckets
                .iter()
                .flat_map(|row| row.iter().flat_map(|b| b.arcs.iter()))
                .map(|a| self.flags(g, direction, a))
                .collect();
            // No arcs are inserted or removed while flags are computed, so
            // the same traversal can assign them without per-bucket ID scans.
            for (arc, flags) in self
                .direction_mut(direction)
                .buckets
                .iter_mut()
                .flat_map(|row| row.iter_mut().flat_map(|b| b.arcs.iter_mut()))
                .zip(flags)
            {
                arc.flags = flags;
            }
        }
    }

    pub(super) fn regenerate(&mut self, forward: bool) -> Result<()> {
        self.regenerate_retaining_buckets(forward).map(drop)
    }

    // Commit newly allocated rows only after all dimensions are valid. The
    // caller may retain the replaced rows to transfer reduced-cost fixings.
    pub(super) fn regenerate_retaining_buckets(
        &mut self,
        forward: bool,
    ) -> Result<Vec<Vec<Bucket>>> {
        self.validate_bucket_geometry(self.step)?;
        let mut rows = vec![];
        let mut corners = vec![];
        for v in 0..self.order.len() {
            let lo = self.lower[v];
            let up = self.upper[v];
            let mut total = 1usize;
            for r in 0..self.dimensions {
                let cells = (((up[r] + 1e-6) - lo[r]) / self.step[r]).floor() + 1.;
                if forward {
                    self.stride[v][r] = total;
                }
                total *= cells as usize;
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
        let previous = std::mem::replace(&mut d.buckets, rows);
        d.corners = corners;
        Ok(previous)
    }

    pub fn rebuild<const D: usize>(&mut self, g: &Graph<D>, forward: bool) -> Result<()> {
        self.regenerate(forward)?;
        let mut counter = 0;
        for position in 0..self.arc_order.len() {
            let index = self.arc_order[position];
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

    pub(super) fn refresh_counts<const D: usize>(&mut self, g: &Graph<D>, forward: bool) {
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
    pub fn ensure_backward<const D: usize>(&mut self, g: &Graph<D>) -> Result<()> {
        if self.backward.buckets.is_empty() {
            self.rebuild(g, false)?;
            self.regenerate_jumps(g, false);
        }
        Ok(())
    }
}
