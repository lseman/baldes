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
        let has_full_cuts = self
            .cuts
            .full
            .as_ref()
            .is_some_and(|full| !full.fields.is_empty());
        self.fix_bucket_arcs_with_full_cut_state(g, threshold, has_full_cuts)
    }

    pub(crate) fn fix_bucket_arcs_with_full_cut_state(
        &mut self,
        g: &Graph<D>,
        threshold: f64,
        has_full_cuts: bool,
    ) -> Result<crate::pricing::fixing::FixingReport> {
        use crate::pricing::fixing::{contains, update, FixingReport, PairingCache};
        if !self.full_ready {
            return Err(invalid("fixing requires complete exact labeling"));
        }
        if !self.labels_match_graph(g) {
            return Err(invalid(
                "fixing requires labels at current reduced costs and resource configuration",
            ));
        }
        if !threshold.is_finite() {
            return Err(invalid("finite fixing threshold required"));
        }
        // Packed pricing cuts can forget information retained by full-memory
        // enumeration. In that case scan only for feasibility; pricing's cut
        // costs cannot establish that an otherwise compatible route is costly.
        let feasibility_only = g.has_coupled_resources() && (self.has_cuts() || has_full_cuts);
        let threshold = if feasibility_only {
            f64::INFINITY
        } else {
            threshold
        };
        // A global complete-join knapsack lower bound can be added to the
        // resource lower bound without assuming matching minimizing controls.
        let scan_all = feasibility_only;
        // Paired fixing scans use the literal label-key sums before testing
        // concatenation. Pricing completion corrections change that scan's
        // cutoff decisions, including terminal arcs with signed increments.
        let paired_keys = g
            .coupled_kernel()
            .is_some_and(|kernel| kernel.machine_paired_join_complete());
        let knapsack_lower = self
            .cuts
            .knapsack
            .global_cost_lower_bound()
            .unwrap_or(f64::NEG_INFINITY);
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
                            .map(|mut q| {
                                if self.consumption_joins {
                                    for r in 0..self.grid.dimensions {
                                        q[r] = self.grid.upper[target][r]
                                            + self.grid.lower[target][r]
                                            - q[r];
                                    }
                                }
                                self.grid.bucket_id_checked(
                                    self.consumption_joins || !forward,
                                    target,
                                    q,
                                )
                            })
                            .transpose()
                            .map_err(|()| incomplete("fixing bucket index overflow"))?
                            .flatten();
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
                                let key = crate::core::rounding::add_down(l.key(), knapsack_lower)
                                    .unwrap_or(f64::NEG_INFINITY);
                                minimum = if scan_all {
                                    f64::NEG_INFINITY
                                } else if paired_keys {
                                    minimum.min(key)
                                } else {
                                    minimum.min(self.ctx().resource_completion_key(
                                        g,
                                        forward,
                                        l.coupled_state.as_deref(),
                                        &l.presence_state,
                                        key,
                                    ))
                                };
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
                                let bound = if paired_keys {
                                    minimum + opposite.completion_bound(false)
                                } else {
                                    self.ctx()
                                        .completion_sum(minimum, opposite.completion_bound(false))
                                };
                                if bound >= threshold {
                                    continue;
                                }
                                if !contains(&self.grid, &frontier, target, bucket) {
                                    let meeting = if forward {
                                        physical.head
                                    } else {
                                        physical.tail
                                    };
                                    'pairs: for l in &extended {
                                        let key = crate::core::rounding::add_down(
                                            l.key(),
                                            knapsack_lower,
                                        )
                                        .unwrap_or(f64::NEG_INFINITY);
                                        let key = if scan_all {
                                            if !self.ctx().bucket_completion_feasible(
                                                g,
                                                forward,
                                                l.coupled_state.as_deref(),
                                                storage_vertex,
                                                bucket,
                                            ) {
                                                continue 'pairs;
                                            }
                                            f64::NEG_INFINITY
                                        } else if paired_keys {
                                            key
                                        } else {
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
                                            key
                                        };
                                        for &id in &opposite.active {
                                            let other = storage.labels.view(id);
                                            let bound = if paired_keys {
                                                key + other.key()
                                            } else {
                                                self.ctx().completion_sum(key, other.key())
                                            };
                                            if bound >= threshold {
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
                return Err(incomplete("resource or bucket fixing arithmetic failure"));
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
