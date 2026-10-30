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
        let retained = if !self.consumption_joins {
            if let Some(kernel) = g
                .coupled_kernel()
                .filter(|kernel| kernel.machine_paired_join_complete())
            {
                use crate::graph::coupled::descriptor::{FrontierBudget, FrontierOutcome};
                let forward_label = pair.0.map(|id| self.forward.labels.view(id));
                let backward_label = pair.1.map(|id| self.backward.labels.view(id));
                let forward_root = pair.0.is_none().then(|| kernel.root(true));
                let backward_root = pair.1.is_none().then(|| kernel.root(false));
                let forward = forward_label
                    .as_ref()
                    .and_then(|label| label.coupled_state())
                    .or(forward_root.as_deref())
                    .ok_or_else(|| incomplete(RESOURCE_REPLAY_MISMATCH))?;
                let backward = backward_label
                    .as_ref()
                    .and_then(|label| label.coupled_state())
                    .or(backward_root.as_deref())
                    .ok_or_else(|| incomplete(RESOURCE_REPLAY_MISMATCH))?;
                let resource = kernel
                    .retained_paired_machine_resource()
                    .ok_or_else(|| incomplete(RESOURCE_REPLAY_MISMATCH))?;
                let selection = match kernel.paired_machine_resource_join(
                    resource,
                    forward,
                    backward,
                    FrontierBudget::default(),
                ) {
                    FrontierOutcome::Complete(value) => value,
                    FrontierOutcome::Empty => return Err(incomplete(RESOURCE_REPLAY_MISMATCH)),
                    FrontierOutcome::Unknown(reason) => {
                        return Err(incomplete(format!(
                            "paired retained join unavailable: {reason:?}"
                        )))
                    }
                };
                let (_, coupled) = kernel
                    .join_objective_costs(forward, backward)?
                    .ok_or_else(|| incomplete(RESOURCE_REPLAY_MISMATCH))?;
                let left = forward
                    .scalar_original_cost()
                    .ok_or_else(|| incomplete(RESOURCE_REPLAY_MISMATCH))?;
                let right = backward
                    .scalar_original_cost()
                    .ok_or_else(|| incomplete(RESOURCE_REPLAY_MISMATCH))?;
                Some((
                    (left + right) + coupled,
                    selection
                        .dependent
                        .map(|value| vec![(resource, value)])
                        .unwrap_or_default(),
                ))
            } else {
                None
            }
        } else {
            None
        };
        let paired_values = if let Some((retained_cost, values)) = retained {
            original_cost = retained_cost;
            values
        } else {
            original_cost += g
                .coupled_route_cost(&arcs)?
                .ok_or_else(|| incomplete(RESOURCE_REPLAY_MISMATCH))?;
            if g.coupled_kernel()
                .is_some_and(|kernel| kernel.uses_zero_paired_inspection_cost())
            {
                Vec::new()
            } else {
                g.coupled_paired_values(&arcs)?
            }
        };
        if !original_cost.is_finite() {
            return Err(incomplete("reconstructed route objective is nonfinite"));
        }
        Ok(Some(Route {
            coupled_values: paired_values,
            nodes,
            arcs: arcs.into_iter().map(|i| g.arcs[i].id).collect(),
            cost,
            original_cost,
            resources: q[..g.resources.len()].to_vec(),
            resource_profile,
        }))
    }
}
