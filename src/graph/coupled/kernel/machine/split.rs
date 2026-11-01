//! Paired pricing pieces retain physical path identity and scalar cost sums.
use super::*;

pub(crate) struct PairedLabelPiece {
    pub(crate) state: Box<CoupledLabelState>,
    pub(crate) coordinate: f64,
    pub(crate) cost: f64,
    pub(crate) original_cost: f64,
}

impl CoupledKernel {
    /// Directional arrival of a paired bucket corner. Forward mapping and
    /// backward inversion use the same literal vector lookup as transitions.
    pub(crate) fn paired_bucket_arrival(
        &self,
        physical: usize,
        forward: bool,
        coordinate: f64,
        tail_bounds: [f64; 2],
        head_bounds: [f64; 2],
    ) -> Result<Option<f64>> {
        if !self.machine_paired_join_complete() {
            return Err(invalid(
                "paired bucket arrival requires one prepared paired resource",
            ));
        }
        let input = self.slope_paired[0]
            .machine_arc_input(physical)
            .ok_or_else(|| invalid("missing paired bucket arc mapping"))?;
        let lower = head_bounds[0].max(input.waiting_lower);
        let upper = head_bounds[1].min(input.waiting_upper);
        let lookup = |function: &descriptor::PreparedSlopeFunction, argument| {
            function
                .machine_vector_value_and_segment(argument)
                // Out-of-domain lookup supplies the sentinel before directional
                // clipping. A backward waiting interval can extend beyond the
                // inverse vector while still having a feasible tail endpoint.
                .map(|value| {
                    value
                        .map(|(value, _)| value)
                        .unwrap_or(crate::labeling::NO_ROUTE_COST)
                })
                .map_err(|error| {
                    crate::incomplete(format!("paired bucket mapping failed: {error:?}"))
                })
        };
        if forward {
            let mut value = lookup(&input.forward_mapping, coordinate)?;
            if value < lower {
                value = lower;
            }
            Ok((value <= upper).then_some(value))
        } else {
            let value = if coordinate <= upper {
                coordinate
            } else {
                upper
            };
            if value < lower {
                return Ok(None);
            }
            let mut value = lookup(&input.inverse, value)?;
            if tail_bounds[1] < value {
                value = tail_bounds[1];
            }
            Ok((tail_bounds[0] <= value).then_some(value))
        }
    }
    /// The paired manager's own base intervals. A missing interval is permitted
    /// only for a nonterminal vertex with no outgoing arcs; it cannot occur in
    /// a complete source-to-sink path.
    pub(crate) fn paired_bucket_domains<const D: usize>(
        &self,
        graph: &Graph<D>,
    ) -> Result<Option<Vec<Option<Window>>>> {
        if !self.machine_paired_join_complete() || !self.pricing_activities.is_empty() {
            return Ok(None);
        }
        if let Some(slot) = self.bound_paired_base_resource(graph) {
            return Ok(Some(
                graph
                    .vertices
                    .iter()
                    .map(|v| Some(v.bounds[slot]))
                    .collect(),
            ));
        }
        let Some(bounds) = self.slope_paired[0].machine_vertex_bounds() else {
            return Ok(None);
        };
        if bounds.len() != graph.vertices.len()
            || bounds.iter().enumerate().any(|(v, bound)| {
                bound.is_none()
                    && (v == graph.source
                        || v == graph.sink
                        || graph.arcs.iter().any(|a| a.tail == v))
            })
        {
            return Err(invalid("paired bucket vertex domains are incomplete"));
        }
        Ok(Some(bounds))
    }

    pub(crate) fn paired_bucket_arc_bounds(&self, physical: usize) -> Option<Window> {
        let input = self.slope_paired.first()?.machine_arc_input(physical)?;
        Some(Window {
            lower: input.waiting_lower,
            upper: input.waiting_upper,
        })
    }

    pub(crate) fn paired_split_base_resource<const D: usize>(
        &self,
        graph: &Graph<D>,
    ) -> Result<Option<usize>> {
        if !self.machine_paired_join_complete() || !self.pricing_activities.is_empty() {
            return Err(invalid("paired splitting requires one prepared paired resource without independent pricing activity fronts"));
        }
        let Some(bounds) = self.paired_bucket_domains(graph)? else {
            return Err(invalid(
                "paired splitting requires installed vertex base domains",
            ));
        };
        if bounds.iter().any(Option::is_none) {
            return Err(invalid("paired splitting requires explicit base bounds for every vertex, including isolated vertices"));
        }
        Ok(self.bound_paired_base_resource(graph))
    }

    pub(crate) fn paired_state_coordinate(
        &self,
        state: &CoupledLabelState,
        forward: bool,
    ) -> Option<f64> {
        let [label] = state.slope_relations.as_slice() else {
            return None;
        };
        let history = label.machine_history.as_ref()?;
        let [function] = history.last().front().components() else {
            return None;
        };
        if forward {
            function.pairs().first().map(|p| p.1)
        } else {
            function.pairs().last().map(|p| p.1)
        }
    }

    pub(crate) fn split_paired_label(
        &self,
        state: Box<CoupledLabelState>,
        strategy: i32,
        halfway: f64,
        bucket_ends: &[f64],
        bucket_grid: [f64; 3],
    ) -> Result<Vec<PairedLabelPiece>> {
        if !self.machine_paired_join_complete()
            || state
                .pricing_activities
                .as_ref()
                .is_some_and(|s| !s.is_empty())
        {
            return Err(invalid(
                "paired split state does not match the prepared paired domain",
            ));
        }
        let forward = state.suffix.is_none();
        let label = state
            .slope_relations
            .first()
            .ok_or_else(|| invalid("missing paired label state"))?;
        let history = label
            .machine_history
            .as_ref()
            .ok_or_else(|| invalid("missing paired label history"))?;
        let prepared = &self.slope_paired[0];
        let histories = super::super::slope_interval::outcome(history.split_last(
            strategy,
            halfway,
            prepared.machine_mode(),
            bucket_ends,
            bucket_grid,
            descriptor::FrontierBudget::default(),
        ))?
        .ok_or_else(|| crate::incomplete("paired split unexpectedly returned an empty domain"))?;
        if histories.len() <= 1 {
            return Ok(vec![self.paired_label_piece(state, forward)?]);
        }
        let mut output = Vec::with_capacity(histories.len());
        for history in histories {
            let mut next = state.clone();
            let key = match history.last() {
                descriptor::MachinePathSnapshot::Root(root) => root.key(),
                descriptor::MachinePathSnapshot::Forward { state, .. } => state.key(),
                descriptor::MachinePathSnapshot::Backward { key, .. } => *key,
            };
            let label = &mut next.slope_relations[0];
            label.minimum = prepared.final_objective(key, self.cost_scale)?;
            label.front = forward.then(|| history.last().front().clone());
            label.machine_history = Some(std::sync::Arc::new(history));
            next.resource_objective = self.final_resource_objective(&next)?;
            output.push(self.paired_label_piece(next, forward)?);
        }
        Ok(output)
    }

    fn paired_label_piece(
        &self,
        state: Box<CoupledLabelState>,
        forward: bool,
    ) -> Result<PairedLabelPiece> {
        let cost = finite(
            state
                .scalar_reduced_cost()
                .ok_or_else(|| invalid("missing split scalar cost"))?
                + state.pricing_objective()?,
        )?;
        let original_cost = finite(
            state
                .scalar_original_cost()
                .ok_or_else(|| invalid("missing split original cost"))?
                + state.resource_objective,
        )?;
        let coordinate = self
            .paired_state_coordinate(&state, forward)
            .ok_or_else(|| invalid("missing split paired coordinate"))?;
        Ok(PairedLabelPiece {
            state,
            coordinate,
            cost,
            original_cost,
        })
    }
}

impl<const D: usize> Graph<D> {
    /// Complete the paired manager's vertex domains, including isolated
    /// vertices that are absent from per-arc function inputs. Supplied bounds
    /// must preserve every already registered source, terminal and tail window.
    pub fn set_paired_bucket_vertex_bounds(
        &mut self,
        resource: usize,
        bounds: &[Window],
    ) -> Result<()> {
        if bounds.len() != self.vertices.len()
            || bounds
                .iter()
                .any(|w| Window::new(w.lower, w.upper).is_err())
        {
            return Err(invalid(
                "finite paired base bounds matching every vertex required",
            ));
        }
        let kernel = self
            .coupled
            .as_mut()
            .ok_or_else(|| invalid("no coupled resources"))?;
        if !matches!(
            kernel.definitions.get(resource),
            Some(CoupledResource::PairedFront { .. })
        ) {
            return Err(invalid("paired bucket bounds require a paired resource"));
        }
        let index = kernel.definitions[..resource]
            .iter()
            .filter(|r| matches!(r, CoupledResource::PairedFront { .. }))
            .count();
        let paired = &mut kernel.slope_paired[index];
        let existing = paired
            .machine_vertex_bounds()
            .ok_or_else(|| invalid("paired bucket bounds require an installed front program"))?;
        if existing.len() != bounds.len()
            || existing.iter().zip(bounds).any(|(old, new)| {
                old.is_some_and(|old| {
                    old.lower.to_bits() != new.lower.to_bits()
                        || old.upper.to_bits() != new.upper.to_bits()
                })
            })
        {
            return Err(invalid(
                "paired bucket bounds disagree with registered program windows",
            ));
        }
        let supplied: Vec<_> = bounds.iter().copied().map(Some).collect();
        if paired.set_machine_vertex_bounds(&supplied) {
            kernel.identity = std::sync::Arc::new(());
            kernel.machine_identity = std::sync::Arc::new(());
        }
        Ok(())
    }
}
