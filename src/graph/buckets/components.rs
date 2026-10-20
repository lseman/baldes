use super::*;

impl BucketGraph {
    pub(super) fn auxiliary<const D: usize>(
        &self,
        g: &Graph<D>,
        forward: bool,
    ) -> (Vec<BucketId>, Vec<usize>, Vec<Vec<usize>>) {
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

    pub(super) fn order_components<const D: usize>(
        &mut self,
        g: &Graph<D>,
        forward: bool,
    ) -> Result<()> {
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
