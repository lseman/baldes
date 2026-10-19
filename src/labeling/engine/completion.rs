use super::*;

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
        let mut q = self.grid.coordinate(*label.resources());
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
        let key = label.key()
            + self
                .cuts
                .knapsack
                .partial_cost(g, label.resources(), forward);
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
                let other = storage.labels.view(id);
                if key + other.key() >= threshold {
                    break;
                }
                let cost = if forward {
                    self.ctx().join_cost(g, label.vertex, label.into(), other)
                } else {
                    self.ctx().join_cost(g, label.vertex, other, label.into())
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
        if self.branching.resources != 0 {
            return true;
        }
        // Pricing completion bounds use limited-memory states. Negative
        // full-memory charges or covering join corrections can improve a
        // continuation beyond those bounds. Until a bound on that improvement
        // is available, retain the label.
        if self.cuts.full.as_ref().is_some_and(|full| {
            full.fields
                .iter()
                .any(|field| field.cost < 0. || field.covering)
        }) {
            return true;
        }
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
            branch_state: None,
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
