use super::*;
mod query;

/// Enumeration's limited-memory projection borrows expensive resource payloads.
struct EnumerationProjection<'a, const D: usize> {
    label: &'a crate::enumeration::Label<D>,
    ng: NgMask,
    cut_words: [u32; LIMITED_WORDS],
    half_potential: f64,
}

impl<const D: usize> LabelData for EnumerationProjection<'_, D> {
    fn vertex(&self) -> usize {
        self.label.vertex
    }
    fn cost(&self) -> f64 {
        self.label.reduced_cost
    }
    fn original_cost(&self) -> f64 {
        self.label.original_cost
    }
    fn resources(&self) -> &[f64] {
        &self.label.resources
    }
    fn ng(&self) -> NgMask {
        self.ng
    }
    fn branch_state(&self) -> &crate::pricing::restrictions::BranchState {
        self.label.branch_state.as_ref().unwrap_or(&None)
    }
    fn presence_state(&self) -> &crate::cuts::soft_conflicts::LabelState {
        &self.label.presence_state
    }
    fn coupled_state(&self) -> Option<&crate::graph::coupled::CoupledLabelState> {
        self.label.coupled_state.as_deref()
    }
    fn hops(&self) -> usize {
        self.label.hops
    }
    fn cut_words(&self) -> &[u32] {
        &self.cut_words
    }
    fn additional_cost(&self) -> f64 {
        self.label.cut_cost
    }
    fn key(&self) -> f64 {
        if let Some(key) =
            self.label.coupled_state.as_deref().and_then(|state| {
                state.paired_completion_key(self.half_potential, self.label.cut_cost)
            })
        {
            return key;
        }
        (self.half_potential + self.label.cut_cost) + (0.0 + self.label.reduced_cost)
    }
}

impl<const D: usize> LabelingEngine<D> {
    pub(crate) fn completion_resource_failed(&self) -> bool {
        self.resource_error.load(Ordering::Relaxed)
    }

    pub(super) fn refresh_coupled_completion(&mut self, g: &Graph<D>) {
        self.coupled_completion = std::array::from_fn(|_| Default::default());
        let Some(kernel) = g.coupled_kernel().filter(|_| self.coupled_present) else {
            return;
        };
        // Bound both preparation work and summary storage. Oversized rows retain
        // the existing per-query checks instead of allocating a quadratic table.
        let mut remaining_pairs = 65_536usize;
        for (index, forward) in [true, false].into_iter().enumerate() {
            let source = forward || self.consumption_joins;
            let reversed = self.consumption_joins && !forward;
            let mut global = crate::graph::coupled::CoupledStateBounds::default();
            let buckets: Vec<Vec<_>> = self
                .storage(source)
                .buckets
                .iter()
                .map(|vertex| {
                    vertex
                        .iter()
                        .map(|bucket| {
                            let mut bounds = crate::graph::coupled::CoupledStateBounds::default();
                            for &id in &bucket.active {
                                let state = self.storage(source).labels.view(id).coupled_state();
                                if reversed {
                                    bounds.include_reversed(state);
                                } else {
                                    bounds.include(state);
                                }
                            }
                            global.include_hull(&bounds);
                            bounds.prepare(kernel);
                            bounds
                        })
                        .collect()
                })
                .collect();
            global.prepare(kernel);
            // A flat/sparse vertex already has a cheap point-conditioned
            // bucket bound. Only prepare pair gates where several populated
            // opposite buckets can amortize the extra query hulls and table.
            let eligible: Vec<_> = buckets
                .iter()
                .map(|row| row.iter().filter(|b| b.label_count() >= 8).take(2).count() == 2)
                .collect();
            let own = self.storage(!forward);
            let mut queries: Vec<Vec<_>> = own
                .buckets
                .iter()
                .enumerate()
                .map(|(v, row)| {
                    if eligible.get(v).copied().unwrap_or(false) {
                        vec![crate::graph::coupled::CoupledStateBounds::default(); row.len()]
                    } else {
                        vec![]
                    }
                })
                .collect();
            if eligible.iter().any(|&yes| yes) {
                for id in 0..own.labels.len() {
                    let label = own.labels.view(id);
                    let bucket = label.bucket();
                    if !eligible.get(bucket.vertex).copied().unwrap_or(false) {
                        continue;
                    }
                    if let Some(bounds) = queries
                        .get_mut(bucket.vertex)
                        .and_then(|row| row.get_mut(bucket.bucket))
                    {
                        bounds.include(label.coupled_state());
                    }
                }
            }
            let pairs = queries
                .iter()
                .enumerate()
                .map(|(vertex, row)| {
                    row.iter()
                        .map(|query| {
                            let opposite = buckets.get(vertex)?;
                            if !query.usable()
                                || opposite.is_empty()
                                || opposite.len() > remaining_pairs
                            {
                                return None;
                            }
                            remaining_pairs -= opposite.len();
                            Some(
                                opposite
                                    .iter()
                                    .map(|bounds| kernel.pairing_summary(query, bounds))
                                    .collect(),
                            )
                        })
                        .collect()
                })
                .collect();
            self.coupled_completion[index] = CoupledCompletion {
                global,
                buckets,
                queries,
                pairs,
                identity: Some(kernel.cache_identity().clone()),
            };
        }
    }
    /// Completion query shared by pricing, enumeration and fixing. The lower
    /// cost already includes dominance predecessors; scanning follows their
    /// native staircase and minimum-index bounds. The label must be at the
    /// stage before entering the meeting vertex, whose elements and cut
    /// supports are already represented by the opposite stored label.
    pub fn completion_exists(
        &self,
        g: &Graph<D>,
        forward: bool,
        label: &StoredLabel<D>,
        threshold: f64,
        scan: bool,
    ) -> bool {
        if !self.labels_match_graph(g) {
            return true;
        }
        self.completion_exists_data(
            g,
            forward,
            label,
            label.bucket,
            threshold,
            scan,
            self.full_ready,
            false,
            None,
        )
    }

    fn completion_exists_data<L: LabelData>(
        &self,
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
        self.ctx().completion_exists_in(
            self.storage(self.consumption_joins || !forward),
            g,
            forward,
            label,
            own,
            threshold,
            scan,
            complete_labels,
            feasibility_only,
            timer,
        )
    }

    pub fn enumeration_completion(
        &self,
        g: &Graph<D>,
        forward: bool,
        label: &crate::enumeration::Label<D>,
        threshold: f64,
        scan: bool,
    ) -> bool {
        if !self.labels_match_graph(g) {
            return true;
        }
        self.enumeration_completion_with_timer(g, forward, label, threshold, scan, None)
    }

    pub(crate) fn enumeration_completion_with_timer(
        &self,
        g: &Graph<D>,
        forward: bool,
        label: &crate::enumeration::Label<D>,
        threshold: f64,
        scan: bool,
        timer: Option<(Instant, std::time::Duration)>,
    ) -> bool {
        if timer.is_some_and(|(start, limit)| start.elapsed() > limit) {
            return true;
        }
        if label.vertex >= self.ng.local_elements.len() || label.vertex >= self.grid.internal.len()
        {
            return true;
        }
        if self.branching.resources != 0 && label.branch_state.is_none() {
            return true;
        }
        // These full-memory charges can improve on pricing's cut bound.
        // In that case only a complete feasibility scan can reject a prefix.
        let feasibility_only = self.cuts.full.as_ref().is_some_and(|full| {
            full.fields
                .iter()
                .any(|field| field.cost < 0. || field.covering)
        });
        let mut ng = 0;
        for (bit, &e) in self.ng.local_elements[label.vertex].iter().enumerate() {
            let Some(word) = label.visited.get(e / 64) else {
                return true;
            };
            if word & (1u64 << (e % 64)) != 0 {
                ng |= 1 << bit;
            }
        }
        let cut_words = match (&self.cuts.full, self.has_cuts()) {
            (Some(full), true) => {
                let Some(words) = state::project_words_mapped(
                    &self.cuts.limited,
                    full,
                    label.vertex,
                    &label.cut_words,
                ) else {
                    return true;
                };
                words
            }
            _ => [0; LIMITED_WORDS],
        };
        let projected = EnumerationProjection {
            label,
            ng,
            cut_words,
            half_potential: self
                .cuts
                .info
                .vertex_cut_potential
                .get(label.vertex)
                .map_or(0.0, |&p| 0.5 * p),
        };
        self.completion_exists_data(
            g,
            forward,
            &projected,
            BucketId::default(),
            threshold,
            scan,
            self.full_ready,
            feasibility_only,
            timer,
        )
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/labeling/engine/completion_tests.rs"]
mod tests;
