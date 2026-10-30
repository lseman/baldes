//! Raw paired-base graph preprocessing, before explicit machine installation.
use super::*;
use descriptor::{FrontierBudget, FrontierOutcome, FrontierUnknown, MachineRootInput};

/// Derived raw paired-base domains. Unlike prepared machine inputs, this is a
/// graph-preprocessing artifact and must not be applied a second time.
#[derive(Clone, Debug)]
pub struct PairedBaseDomains {
    vertex_bounds: Vec<Window>,
    arc_bounds: Vec<Window>,
    removed_arcs: Vec<usize>,
    resource: usize,
    base_resource: usize,
    source: usize,
    sink: usize,
    topology: Vec<(i32, usize, usize)>,
    raw: Vec<(Window, Window, PreparedSlopeFunction)>,
    raw_vertex_bounds: Vec<Window>,
    raw_arc_bounds: Vec<Option<Window>>,
    raw_consumption: Vec<u64>,
}
fn changed(a: f64, b: f64) -> bool {
    let difference = (a - b).abs();
    difference >= 0.0001 && difference >= 2.220446049250313e-10 * (a + b).abs()
}
fn unsupported<T>(message: &'static str) -> FrontierOutcome<T> {
    FrontierOutcome::Unknown(FrontierUnknown::Unsupported(message))
}
impl PairedBaseDomains {
    pub fn vertex_bounds(&self) -> &[Window] {
        &self.vertex_bounds
    }
    pub fn arc_bounds(&self) -> &[Window] {
        &self.arc_bounds
    }
    pub fn removed_arcs(&self) -> &[usize] {
        &self.removed_arcs
    }
    pub fn base_resource(&self) -> usize {
        self.base_resource
    }

    /// Install the derived scalar domains, explicit binding, and raw front
    /// program atomically. Existing prepared-installation APIs remain literal.
    /// Removed physical arcs currently require a separate topology/exclusion
    /// installation; this method explicitly declines that case.
    pub fn install_front_program<const D: usize>(
        &self,
        graph: &mut Graph<D>,
        inputs: &[MachineArcFunctionInput<'_>],
        root: MachineRootInput<'_>,
        budget: FrontierBudget,
    ) -> FrontierOutcome<()> {
        let started = std::time::Instant::now();
        let Some(work) = graph
            .vertices
            .len()
            .checked_mul(graph.resources.len().saturating_add(2))
            .and_then(|v| {
                graph
                    .arcs
                    .len()
                    .checked_mul(graph.resources.len().saturating_add(14))
                    .and_then(|a| v.checked_add(a))
            })
        else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        let Some(remaining_work) = budget.max_operations.checked_sub(work) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        let remaining = || -> std::result::Result<FrontierBudget, FrontierUnknown> {
            let time = budget
                .max_elapsed
                .map(|limit| limit.saturating_sub(started.elapsed()));
            if time.is_some_and(|t| t.is_zero()) {
                return Err(FrontierUnknown::Deadline);
            }
            Ok(FrontierBudget {
                max_operations: remaining_work,
                max_elapsed: time,
                ..budget
            })
        };
        if let Err(e) = remaining() {
            return FrontierOutcome::Unknown(e);
        }
        if graph.validate_structure().is_err() {
            return unsupported("paired base installation requires a valid graph structure");
        }
        let Some(binding) = graph.coupled_kernel().and_then(|kernel| {
            if !matches!(
                kernel.definitions.get(self.resource),
                Some(CoupledResource::PairedFront { .. })
            ) {
                return None;
            }
            let index = kernel.definitions[..self.resource]
                .iter()
                .filter(|r| matches!(r, CoupledResource::PairedFront { .. }))
                .count();
            kernel.slope_paired[index].base_window_snapshot()
        }) else {
            return unsupported("paired base domains require the original explicit scalar binding");
        };
        if graph.resources.get(self.base_resource) != Some(&crate::graph::ResourceKind::Disposable)
            || binding.resource != self.base_resource
            || graph
                .vertices
                .iter()
                .map(|v| v.bounds.get(self.base_resource).copied())
                .ne(self.raw_vertex_bounds.iter().copied().map(Some))
            || graph
                .arcs
                .iter()
                .map(|a| a.bounds.get(self.base_resource).copied())
                .ne(self.raw_arc_bounds.iter().copied().map(Some))
            || graph
                .arcs
                .iter()
                .map(|a| a.consumption.get(self.base_resource).map(|v| v.to_bits()))
                .ne(self.raw_consumption.iter().copied().map(Some))
            || binding.vertices.iter().copied().ne(self
                .raw_vertex_bounds
                .iter()
                .map(|v| (v.lower.to_bits(), v.upper.to_bits())))
            || binding.arcs.iter().copied().ne(self
                .raw_arc_bounds
                .iter()
                .map(|v| v.map(|w| (w.lower.to_bits(), w.upper.to_bits()))))
        {
            return unsupported(
                "paired base domains scalar registration changed after preprocessing",
            );
        }
        if graph.source != self.source
            || graph.sink != self.sink
            || graph.vertices.len() != self.vertex_bounds.len()
            || graph
                .arcs
                .iter()
                .map(|a| (a.id, a.tail, a.head))
                .ne(self.topology.iter().copied())
            || inputs.len() != self.raw.len()
            || inputs.iter().zip(&self.raw).enumerate().any(
                |(i, (input, (tail, waiting, base)))| {
                    input.physical_arc != i
                        || input.tail_bounds != *tail
                        || input.waiting_bounds != *waiting
                        || input.base != base
                },
            )
        {
            return unsupported("paired base domains do not match raw inputs or graph topology");
        }
        if !self.removed_arcs.is_empty() {
            return unsupported("paired base preprocessing removed arcs require explicit topology exclusion installation");
        }
        if let Err(e) = remaining() {
            return FrontierOutcome::Unknown(e);
        }
        let mut staged = graph.clone();
        if let Err(e) = remaining() {
            return FrontierOutcome::Unknown(e);
        }
        let rates = staged.coupled_kernel().and_then(|kernel| {
            let index = kernel.definitions[..self.resource]
                .iter()
                .filter(|r| matches!(r, CoupledResource::PairedFront { .. }))
                .count();
            kernel.slope_paired[index]
                .machine_waiting_rates()
                .map(|v| v.to_vec())
        });
        for (vertex, bounds) in self.vertex_bounds.iter().enumerate() {
            if staged
                .set_bounds(vertex, self.base_resource, bounds.lower, bounds.upper)
                .is_err()
            {
                return unsupported("paired base scalar domain installation failed");
            }
        }
        for (arc, bounds) in staged.arcs.iter_mut().zip(&self.arc_bounds) {
            arc.bounds[self.base_resource] = Some(*bounds);
        }
        if staged
            .set_paired_waiting_windows_from_resource(self.resource, self.base_resource)
            .is_err()
        {
            return unsupported("paired base resource binding installation failed");
        }
        let prepared: Vec<_> = inputs
            .iter()
            .zip(&staged.arcs)
            .zip(&self.arc_bounds)
            .map(|((input, arc), waiting)| MachineArcFunctionInput {
                tail_bounds: self.vertex_bounds[arc.tail],
                waiting_bounds: *waiting,
                ..*input
            })
            .collect();
        let remaining_budget = match remaining() {
            Ok(v) => v,
            Err(e) => return FrontierOutcome::Unknown(e),
        };
        let result = if let Some(rates) = rates {
            staged.prepare_paired_front_program_with_rates(
                self.resource,
                &prepared,
                root,
                self.vertex_bounds[self.source],
                self.vertex_bounds[self.sink],
                &rates,
                remaining_budget,
            )
        } else {
            staged.prepare_paired_front_program(
                self.resource,
                &prepared,
                root,
                self.vertex_bounds[self.source],
                self.vertex_bounds[self.sink],
                remaining_budget,
            )
        };
        if matches!(result, FrontierOutcome::Complete(())) {
            if let Err(e) = remaining() {
                return FrontierOutcome::Unknown(e);
            }
            *graph = staged;
        }
        result
    }
}
impl<const D: usize> Graph<D> {
    /// Preprocess an explicitly bound raw paired base graph. This does not
    /// mutate the graph or reinterpret an already prepared machine program.
    /// The scalar resource must be disposable and match the raw arrival maps.
    pub fn preprocess_paired_base_domains(
        &self,
        resource: usize,
        inputs: &[MachineArcFunctionInput<'_>],
        budget: FrontierBudget,
    ) -> FrontierOutcome<PairedBaseDomains> {
        let started = std::time::Instant::now();
        if self.validate_structure().is_err() {
            return unsupported("paired base preprocessing requires a valid raw graph");
        }
        let Some(kernel) = self.coupled_kernel() else {
            return unsupported("paired base preprocessing requires a resource binding");
        };
        if !matches!(
            kernel.definitions.get(resource),
            Some(CoupledResource::PairedFront { .. })
        ) {
            return unsupported("paired base preprocessing requires a paired descriptor");
        }
        let index = kernel.definitions[..resource]
            .iter()
            .filter(|r| matches!(r, CoupledResource::PairedFront { .. }))
            .count();
        let Some(binding) = kernel.slope_paired[index].base_window_snapshot() else {
            return unsupported(
                "paired base preprocessing requires an explicit scalar resource binding",
            );
        };
        let slot = binding.resource;
        if self.resources.get(slot) != Some(&crate::graph::ResourceKind::Disposable)
            || self.resource_has_time_windows(slot)
            || self.initial[slot].is_some_and(|value| {
                value.to_bits() != self.vertices[self.source].bounds[slot].lower.to_bits()
            })
            || inputs.len() != self.arcs.len()
            || binding.vertices.len() != self.vertices.len()
            || binding.arcs.len() != self.arcs.len()
            || self
                .vertices
                .iter()
                .zip(&binding.vertices)
                .any(|(v, &bits)| {
                    (
                        v.bounds[slot].lower.to_bits(),
                        v.bounds[slot].upper.to_bits(),
                    ) != bits
                })
            || self
                .arcs
                .iter()
                .zip(&binding.arcs)
                .any(|(a, b)| a.bounds[slot].map(|w| (w.lower.to_bits(), w.upper.to_bits())) != *b)
        {
            return unsupported("paired base preprocessing scalar binding is stale or unsupported");
        }
        let mut remaining = budget.max_operations;
        let mut charge = |count: usize| -> std::result::Result<(), FrontierUnknown> {
            remaining = remaining
                .checked_sub(count)
                .ok_or(FrontierUnknown::WorkLimit)?;
            if budget
                .max_elapsed
                .is_some_and(|limit| started.elapsed() >= limit)
            {
                return Err(FrontierUnknown::Deadline);
            }
            Ok(())
        };
        let result = (|| -> std::result::Result<PairedBaseDomains, FrontierUnknown> {
            let fail = |m| FrontierUnknown::Unsupported(m);
            charge(self.vertices.len() + self.arcs.len())?;
            let mut vertices: Vec<_> = self.vertices.iter().map(|v| v.bounds[slot]).collect();
            let mut arcs = Vec::with_capacity(inputs.len());
            let mut forward = Vec::with_capacity(inputs.len());
            let mut inverse = Vec::with_capacity(inputs.len());
            for (i, (arc, input)) in self.arcs.iter().zip(inputs).enumerate() {
                charge(
                    input
                        .base
                        .pairs()
                        .len()
                        .saturating_mul(16)
                        .saturating_add(32),
                )?;
                if input.physical_arc != i
                    || input.tail_bounds != vertices[arc.tail]
                    || input.base.pairs().len() > budget.max_pairs
                {
                    return Err(fail(
                        "paired base raw input bounds/order do not match scalar registration",
                    ));
                }
                let expected = arc.bounds[slot].map_or(vertices[arc.head], |w| Window {
                    lower: w.lower.max(vertices[arc.head].lower),
                    upper: w.upper.min(vertices[arc.head].upper),
                });
                if input.waiting_bounds != expected {
                    return Err(fail(
                        "paired base raw arc window does not match scalar registration",
                    ));
                }
                let truncated = input
                    .base
                    .machine_truncate(input.tail_bounds.lower, input.tail_bounds.upper)
                    .map_err(FrontierUnknown::Arithmetic)?;
                let q = truncated
                    .machine_forward_from_truncated()
                    .map_err(FrontierUnknown::Arithmetic)?;
                if q.pairs()[0].0.to_bits() != (q.pairs()[0].1 + arc.consumption[slot]).to_bits()
                    || q.pairs()[1..].iter().any(|&(s, _)| s != 1.)
                {
                    return Err(fail(
                        "paired base scalar consumption does not match raw arrival mapping",
                    ));
                }
                let inv = match descriptor::machine_inverse(&q, budget) {
                    FrontierOutcome::Complete(v) => v,
                    FrontierOutcome::Unknown(e) => return Err(e),
                    FrontierOutcome::Empty => return Err(fail("paired base inverse unavailable")),
                };
                arcs.push(expected);
                forward.push(q);
                inverse.push(inv);
            }
            let lookup = |f: &PreparedSlopeFunction,
                          x: f64|
             -> std::result::Result<f64, FrontierUnknown> {
                f.machine_vector_value_and_segment(x)
                    .map_err(FrontierUnknown::Arithmetic)?
                    .map(|v| v.0)
                    .filter(|&v| v != 1e12)
                    .ok_or_else(|| fail("paired base preprocessing lookup returned the sentinel"))
            };
            // Forward worklist: temporary vertex and arc lower bounds feed
            // later rounds. The sink vertex itself is deliberately excluded.
            loop {
                let mut any = false;
                for (i, arc) in self.arcs.iter().enumerate() {
                    if arc.tail == self.sink {
                        continue;
                    }
                    charge(forward[i].pairs().len() + 1)?;
                    let value = lookup(&forward[i], vertices[arc.tail].lower)?;
                    if value > arcs[i].lower && changed(value, arcs[i].lower) {
                        arcs[i].lower = value;
                        any = true;
                    }
                }
                for (vertex, bounds) in vertices.iter_mut().enumerate() {
                    if vertex == self.sink {
                        continue;
                    }
                    charge(self.arcs.len() + 1)?;
                    if let Some(value) = self
                        .arcs
                        .iter()
                        .enumerate()
                        .filter(|(_, a)| a.head == vertex)
                        .map(|(i, _)| arcs[i].lower)
                        .reduce(f64::min)
                    {
                        if value > bounds.lower && changed(value, bounds.lower) {
                            bounds.lower = value;
                            any = true;
                        }
                    }
                }
                if !any {
                    break;
                }
            }
            let mut removed: Vec<_> = arcs
                .iter()
                .enumerate()
                .filter(|(_, w)| w.lower > w.upper && changed(w.lower, w.upper))
                .map(|(i, _)| i)
                .collect();
            // Native backward lookup reads original stored arc upper values,
            // even while updated head caps are staged separately. Do not turn
            // this into a stronger inverse fixed point.
            let original_upper: Vec<_> = arcs.iter().map(|w| w.upper).collect();
            for (vertex, bounds) in vertices.iter_mut().enumerate() {
                if vertex == self.sink {
                    continue;
                }
                let mut maximum = None::<f64>;
                for (i, _arc) in self
                    .arcs
                    .iter()
                    .enumerate()
                    .filter(|(_, a)| a.tail == vertex)
                {
                    if removed.contains(&i) {
                        continue;
                    }
                    charge(inverse[i].pairs().len() + 1)?;
                    let value = lookup(&inverse[i], original_upper[i])?;
                    maximum = Some(maximum.map_or(value, |old| old.max(value)));
                }
                if let Some(value) = maximum {
                    if value < bounds.upper && changed(value, bounds.upper) {
                        bounds.upper = value;
                    }
                }
            }
            for (i, arc) in self.arcs.iter().enumerate() {
                let value = vertices[arc.head].upper;
                if value < arcs[i].upper && changed(value, arcs[i].upper) {
                    arcs[i].upper = value;
                }
                if arcs[i].lower > arcs[i].upper
                    && changed(arcs[i].lower, arcs[i].upper)
                    && !removed.contains(&i)
                {
                    removed.push(i);
                }
            }
            Ok(PairedBaseDomains {
                vertex_bounds: vertices,
                arc_bounds: arcs,
                removed_arcs: removed,
                resource,
                base_resource: slot,
                source: self.source,
                sink: self.sink,
                topology: self.arcs.iter().map(|a| (a.id, a.tail, a.head)).collect(),
                raw_vertex_bounds: self.vertices.iter().map(|v| v.bounds[slot]).collect(),
                raw_arc_bounds: self.arcs.iter().map(|a| a.bounds[slot]).collect(),
                raw_consumption: self
                    .arcs
                    .iter()
                    .map(|a| a.consumption[slot].to_bits())
                    .collect(),
                raw: inputs
                    .iter()
                    .map(|i| (i.tail_bounds, i.waiting_bounds, i.base.clone()))
                    .collect(),
            })
        })();
        match result {
            Ok(v) => FrontierOutcome::Complete(v),
            Err(e) => FrontierOutcome::Unknown(e),
        }
    }
}

impl CoupledKernel {
    pub(crate) fn bound_paired_base_resource<const D: usize>(
        &self,
        graph: &Graph<D>,
    ) -> Option<usize> {
        self.slope_paired.iter().find_map(|paired| {
            if !paired.machine_program_inputs_complete() || !paired.machine_topology_matches(graph)
            {
                return None;
            }
            paired.base_window_snapshot().and_then(|snapshot| {
                (graph.resources.get(snapshot.resource)
                    == Some(&crate::graph::ResourceKind::Disposable))
                .then_some(snapshot.resource)
            })
        })
    }
}

#[cfg(test)]
#[path = "../../../../../tests/unit/graph/coupled/kernel/base_domains_tests.rs"]
mod tests;
