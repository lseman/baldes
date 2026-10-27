use super::*;

impl<const D: usize> LabelingEngine<D> {
    /// getPrevLabelPtr: every predecessor must still be active
    /// or archived in its bucket. A dominated expanded label that was not
    /// archived is gone, and candidates through it cannot be recorded.
    pub(super) fn physical_path(&self, forward: bool, mut id: usize) -> Option<Vec<usize>> {
        let s = self.storage(forward);
        let mut arcs = vec![];
        loop {
            let l = s.labels.view(id);
            if let Some(a) = l.arc() {
                arcs.push(a);
            }
            let Some(parent) = l.parent() else { break };
            let p = s.labels.view(parent);
            let bucket = &s.buckets[p.bucket().vertex][p.bucket().bucket];
            if !bucket.active.contains(&parent) && !bucket.archived.contains(&parent) {
                return None;
            }
            id = parent;
        }
        if forward {
            arcs.reverse();
        }
        Some(arcs)
    }

    /// recordSolution; `None` when a predecessor is missing.
    pub(super) fn reconstruct(
        &self,
        g: &Graph<D>,
        pair: (Option<usize>, Option<usize>),
        cost: f64,
    ) -> Result<Option<Route>> {
        let mut arcs = match pair.0 {
            Some(i) => match self.physical_path(true, i) {
                Some(arcs) => arcs,
                None => return Ok(None),
            },
            None => vec![],
        };
        if let Some(i) = pair.1 {
            let Some(suffix) = self.physical_path(self.consumption_joins, i) else {
                return Ok(None);
            };
            if self.consumption_joins {
                arcs.extend(suffix.into_iter().rev().map(|a| self.reverse[a]));
            } else {
                arcs.extend(suffix);
            }
        }
        let mut nodes = vec![g.source as i32];
        let mut q = crate::graph::ResourceValues::filled(g.resources.len(), 0.);
        for r in 0..g.resources.len() {
            q[r] = g
                .root_resource(r, true)
                .ok_or_else(|| invalid("initial resource outside source windows"))?;
        }
        let mut prefix_states = vec![];
        let mut current = pair.0;
        while let Some(id) = current {
            let label = self.forward.labels.view(id);
            prefix_states.push(crate::graph::ResourceValues::from_slice(label.resources()));
            current = label.parent();
        }
        prefix_states.reverse();
        let mut resource_profile = vec![q[..g.resources.len()].to_vec()];
        let mut original_cost = 0.;
        for (position, &index) in arcs.iter().enumerate() {
            let a = &g.arcs[index];
            if nodes.last().copied() != Some(a.tail as i32) {
                return Err(invalid("broken labeling predecessor chain"));
            }
            let extended = extend_resources(g, a, &q, true)
                .ok_or_else(|| incomplete(RESOURCE_REPLAY_MISMATCH))?;
            // Forward labels retain waiting imposed by jump arcs. Replaying
            // their arcs at earliest arrival would collapse distinct schedules.
            // Continue the backward suffix from the actual forward state.
            q = prefix_states.get(position + 1).cloned().unwrap_or(extended);
            resource_profile.push(q[..g.resources.len()].to_vec());
            nodes.push(a.head as i32);
            original_cost += a.cost;
        }
        if nodes.last().copied() != Some(g.sink as i32) {
            return Err(invalid("labeling predecessor chain does not reach sink"));
        }
        original_cost += g
            .coupled_path_cost(&arcs)?
            .ok_or_else(|| incomplete(RESOURCE_REPLAY_MISMATCH))?;
        Ok(Some(Route {
            coupled_controls: Vec::new(),
            coupled_values: g.coupled_paired_values(&arcs)?,
            nodes,
            arcs: arcs.into_iter().map(|i| g.arcs[i].id).collect(),
            cost,
            original_cost,
            resources: q[..g.resources.len()].to_vec(),
            resource_profile,
        }))
    }
}
