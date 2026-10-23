use super::*;

impl<const D: usize> LabelingEngine<D> {
    /// Fix after exact completion at the same reduced costs. Symmetric search
    /// pairs forward prefixes; ordinary search uses both directions. Cached
    /// associations are local to this fixing pass.
    pub fn fix_bucket_arcs(
        &mut self,
        g: &Graph<D>,
        threshold: f64,
    ) -> Result<crate::pricing::fixing::FixingReport> {
        use crate::pricing::fixing::{contains, update, FixingReport, PairingCache};
        if !self.full_ready {
            return Err(invalid("fixing requires complete exact labeling"));
        }
        if !threshold.is_finite() {
            return Err(invalid("finite fixing threshold required"));
        }
        if g.coupled_kernel()
            .is_some_and(|kernel| kernel.has_slope_fronts())
        {
            return Ok(FixingReport::default());
        }
        // Both roles must be disproved before fixing a dual-orientation arc.
        if self.consumption_joins
            && g.coupled_kernel()
                .is_some_and(|k| k.reversal_union_required())
        {
            return Ok(FixingReport::default());
        }
        let mut report = FixingReport::default();
        let mut concatenation_tests = 0;
        let mut removals = Vec::new();
        for forward in [true, false] {
            if self.consumption_joins && !forward {
                continue;
            }
            let mut cache = PairingCache::default();
            let components = &self.grid.direction(forward).components;
            let mut removed = std::collections::BTreeSet::new();
            for component in components {
                for &source in component {
                    let arcs = self.grid.direction(forward).buckets[source.vertex][source.bucket]
                        .arcs
                        .as_slice();
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
                        if self.consumption_joins && physical.head == g.sink {
                            // The physical terminal has no suffix. Dominated
                            // prefixes still inherit useful terminal arcs from
                            // predecessor buckets, even without active labels.
                            let inherited =
                                cache.initialize(&self.grid, true, source, physical.id, target);
                            let useful = !inherited.is_empty()
                                || self.forward.buckets[source.vertex][source.bucket]
                                    .active
                                    .iter()
                                    .any(|&id| {
                                        self.extend(g, true, arc, self.forward.labels.view(id))
                                            .is_some_and(|label| {
                                                self.ctx().terminal_cost(g, true, &label)
                                                    < threshold
                                            })
                                    });
                            cache.associations.insert(
                                (source, physical.id),
                                if useful { vec![0] } else { vec![] },
                            );
                            if !useful {
                                removed.insert(arc.id);
                            }
                            continue;
                        }
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
                                self.extend(g, forward, arc, self.storage(forward).labels.view(id))
                            {
                                let key = l.key()
                                    + self.cuts.knapsack.partial_cost(g, &l.resources, forward);
                                minimum = minimum.min(self.ctx().resource_completion_key(
                                    g,
                                    forward,
                                    l.coupled_state.as_deref(),
                                    &l.presence_state,
                                    key,
                                ));
                                extended.push(l);
                            }
                        }
                        let mut frontier =
                            cache.initialize(&self.grid, forward, source, physical.id, target);
                        if !extended.is_empty() {
                            let mut work = vec![(arrival, 0usize)];
                            while let Some((bucket, min)) = work.pop() {
                                let storage = self.storage(self.consumption_joins || !forward);
                                let storage_vertex = if self.consumption_joins
                                    && target + 1 == self.grid.order.len()
                                {
                                    0
                                } else {
                                    target
                                };
                                let opposite = &storage.buckets[storage_vertex][bucket];
                                if self.ctx().completion_sum(
                                    minimum,
                                    opposite.completion_bound(self.cuts.knapsack.active),
                                ) >= threshold
                                {
                                    continue;
                                }
                                if !contains(&self.grid, &frontier, target, bucket) {
                                    let meeting = if forward {
                                        physical.head
                                    } else {
                                        physical.tail
                                    };
                                    'pairs: for l in &extended {
                                        let key = l.key()
                                            + self.cuts.knapsack.partial_cost(
                                                g,
                                                &l.resources,
                                                forward,
                                            );
                                        let Some(key) = self.ctx().bucket_completion(
                                            g,
                                            forward,
                                            l.into(),
                                            key,
                                            BucketId {
                                                vertex: storage_vertex,
                                                bucket,
                                            },
                                            opposite.hot.key.first().copied(),
                                            threshold,
                                        ) else {
                                            continue 'pairs;
                                        };
                                        for &id in &opposite.active {
                                            let other = storage.labels.view(id);
                                            if self.ctx().completion_sum(key, other.key())
                                                >= threshold
                                            {
                                                break;
                                            }
                                            concatenation_tests += 1;
                                            let cost = if forward {
                                                self.ctx().join_cost(g, meeting, l.into(), other)
                                            } else {
                                                self.ctx().join_cost(g, meeting, other, l.into())
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
            if self.resource_error.load(Ordering::Relaxed) {
                return Err(incomplete("coupled resource fixing arithmetic failure"));
            }
            if forward {
                report.eliminated_forward = removed.len();
            } else {
                report.eliminated_backward = removed.len();
            }
            removals.push((forward, removed));
        }
        for (forward, removed) in removals {
            let direction = if forward {
                &mut self.grid.forward
            } else {
                &mut self.grid.backward
            };
            for row in &mut direction.buckets {
                for bucket in row {
                    bucket.arcs.retain(|a| !removed.contains(&a.id));
                }
            }
        }
        self.statistics.concatenation_tests += concatenation_tests;
        if report.eliminated_forward + report.eliminated_backward > 0 {
            self.grid.regenerate_jumps(g, true);
            if !self.consumption_joins {
                self.grid.regenerate_jumps(g, false);
            }
            self.full_ready = false;
            self.coupled_completion = std::array::from_fn(|_| Default::default());
            self.half_ready = false;
        }
        Ok(report)
    }
}
