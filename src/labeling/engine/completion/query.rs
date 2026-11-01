//! Shared completion query for retained and in-flight labels.
use super::*;

impl<const D: usize> Context<'_, D> {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::labeling::engine) fn completion_exists_in<L: LabelData>(
        &self,
        storage: &LabelStorage<D>,
        g: &Graph<D>,
        forward: bool,
        label: &L,
        own: BucketId,
        threshold: f64,
        scan: bool,
        complete_labels: bool,
        feasibility_only: bool,
        timer: Option<(Instant, std::time::Duration)>,
    ) -> bool {
        if threshold.is_nan() {
            return true;
        }
        if feasibility_only && (!complete_labels || !scan) {
            return true;
        }
        let threshold = if feasibility_only {
            f64::INFINITY
        } else {
            threshold
        };
        // A partial opposite-label store cannot certify that a coupled
        // resource prefix has no completion. This applies to numeric and
        // piecewise states as well as paired fronts.
        if !complete_labels && g.has_coupled_resources() {
            return true;
        }
        // The retained prefix may only be feasible when used in reverse.
        // An ordinary forward completion witness cannot delete that role.
        if self.consumption_joins
            && g.coupled_kernel()
                .is_some_and(|k| k.reversal_union_required())
        {
            return true;
        }
        if self.resource_error.load(Ordering::Relaxed) {
            return true;
        }
        if let Some(kernel) = g.coupled_kernel() {
            if label
                .coupled_state()
                .is_none_or(|state| !kernel.completion_state_matches(state))
            {
                return true;
            }
        }
        // Invalid incidence cannot justify deleting a caller-supplied label;
        // signal the owner so it does not publish the partial enumeration.
        if self.cuts.presence.active()
            && self
                .cuts
                .presence
                .join_correction_lower_bound(label.presence_state())
                .is_err()
        {
            self.resource_error.store(true, Ordering::Relaxed);
            return true;
        }
        let vertex = self.grid.internal[label.vertex()];
        let Some(mut q) = self.label_coordinates(g, forward, label) else {
            self.resource_error.store(true, Ordering::Relaxed);
            return true;
        };
        // Invalid coordinates do not prove that a completion is impossible.
        // In particular, bucket indexing casts floating-point values to signed
        // integers and may otherwise turn a NaN into an ordinary bucket.
        if q.iter().take(self.grid.dimensions).any(|x| !x.is_finite()) {
            return true;
        }
        let opposite = self.consumption_joins || !forward;
        if self.consumption_joins && forward {
            for r in 0..self.grid.dimensions {
                q[r] = self.grid.upper[vertex][r] + self.grid.lower[vertex][r] - q[r];
            }
        }
        if q.iter().take(self.grid.dimensions).any(|x| !x.is_finite()) {
            return true;
        }
        let bucket = match self.grid.bucket_id_checked(opposite, vertex, q) {
            Ok(Some(bucket)) => bucket,
            Ok(None) => return false,
            Err(()) => return true,
        };

        let knapsack_lower = self
            .cuts
            .knapsack
            .global_cost_lower_bound()
            .unwrap_or(f64::NEG_INFINITY);
        let key = crate::core::rounding::add_down(label.key(), knapsack_lower)
            .unwrap_or(f64::NEG_INFINITY);
        let base_key = key;
        // The complete knapsack charge has its own global lower bound above;
        // it no longer relies on combining unrelated partial minima.
        let scan_all = feasibility_only;
        let key = if scan_all {
            f64::NEG_INFINITY
        } else {
            self.resource_completion_key(
                g,
                forward,
                label.coupled_state(),
                label.presence_state(),
                key,
            )
        };
        let paired_sum = !scan_all
            && !self.cuts.presence.active()
            && !self.cuts.knapsack.active
            && label
                .coupled_state()
                .and_then(|state| state.paired_completion_key(0., 0.))
                .is_some()
            && key.to_bits() == label.key().to_bits();
        let completion_sum = |left: f64, right: f64| {
            if paired_sum {
                left + right
            } else {
                self.completion_sum(left, right)
            }
        };
        // The initial bucket needs no heap storage. Bounds-only queries and
        // leaf buckets finish here without allocating a predecessor stack.
        let mut initial = Some((bucket, 0usize));
        let mut work = Vec::new();
        let mut examined = 0usize;
        while let Some((k, min)) = work.pop().or_else(|| initial.take()) {
            examined = examined.wrapping_add(1);
            if examined & 255 == 0 && timer.is_some_and(|(start, limit)| start.elapsed() > limit) {
                return true;
            }
            let row = &storage.buckets[vertex][k];
            let row_bound = completion_sum(key, row.completion_bound(false));
            if row_bound >= threshold {
                continue;
            }
            if !scan {
                return true;
            }
            let scan_key = if scan_all {
                self.bucket_completion_feasible(g, forward, label.coupled_state(), vertex, k)
                    .then_some(f64::NEG_INFINITY)
            } else {
                self.bucket_completion_data(
                    g,
                    forward,
                    label,
                    own,
                    base_key,
                    BucketId { vertex, bucket: k },
                    row.hot.key.first().copied(),
                    threshold,
                )
            };
            if let Some(scan_key) = scan_key {
                for (position, &id) in row.active.iter().enumerate() {
                    examined = examined.wrapping_add(1);
                    if examined & 255 == 0
                        && timer.is_some_and(|(start, limit)| start.elapsed() > limit)
                    {
                        return true;
                    }
                    let other = storage.labels.view(id);
                    let candidate_key = if position == 0 {
                        other.key()
                    } else {
                        let half_potential = self
                            .cuts
                            .info
                            .vertex_cut_potential
                            .get(label.vertex())
                            .map_or(0., |&value| 0.5 * value);
                        other
                            .coupled_state()
                            .and_then(|state| {
                                state.paired_completion_scan_key(
                                    half_potential,
                                    other.additional_cost(),
                                )
                            })
                            .unwrap_or_else(|| other.key())
                    };
                    let candidate_bound = completion_sum(scan_key, candidate_key);
                    if if paired_sum {
                        !(candidate_bound < threshold)
                    } else {
                        candidate_bound >= threshold
                    } {
                        break;
                    }
                    let (feasible, cost) = if forward {
                        self.concatenate_records(g, label.vertex(), label, &other)
                    } else {
                        self.concatenate_records(g, label.vertex(), &other, label)
                    };
                    if feasible && threshold > cost {
                        return true;
                    }
                }
            }
            let predecessors = &self.grid.direction(opposite).buckets[vertex][k].predecessors;
            let mut end = 0;
            while end < predecessors.len() && predecessors[end].bucket.bucket >= min {
                examined = examined.wrapping_add(1);
                if examined & 255 == 0
                    && timer.is_some_and(|(start, limit)| start.elapsed() > limit)
                {
                    return true;
                }
                end += 1;
            }
            for p in predecessors[..end].iter().rev() {
                examined = examined.wrapping_add(1);
                if examined & 255 == 0
                    && timer.is_some_and(|(start, limit)| start.elapsed() > limit)
                {
                    return true;
                }
                work.push((p.bucket.bucket, min.max(p.minimum)));
            }
        }
        self.resource_error.load(Ordering::Relaxed)
    }
}
