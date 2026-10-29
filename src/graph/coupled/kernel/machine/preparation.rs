use super::*;

impl<const D: usize> Graph<D> {
    /// Install exact prepared machine operands in physical arc order for a
    /// slope-paired resource. This is an explicit data sidecar; it does not
    /// select a feasibility domain or change pricing by itself. Pass `None`
    /// to remove it. Replacing its contents invalidates prepared graph state.
    pub fn set_paired_machine_inputs(
        &mut self,
        resource: usize,
        inputs: Option<&[PreparedMachineArc]>,
    ) -> Result<()> {
        let topology = inputs.map(|_| slope_paired::MachineTopology {
            source: self.source,
            sink: self.sink,
            vertex_count: self.vertices.len(),
            arcs: self
                .arcs
                .iter()
                .map(|arc| (arc.id, arc.tail, arc.head))
                .collect(),
        });
        let kernel = self
            .coupled
            .as_mut()
            .ok_or_else(|| invalid("no coupled resources"))?;
        if !matches!(
            kernel.definitions.get(resource),
            Some(CoupledResource::SlopePaired { .. })
        ) {
            return Err(invalid("machine inputs require a paired slope resource"));
        }
        let index = kernel.definitions[..resource]
            .iter()
            .filter(|r| matches!(r, CoupledResource::SlopePaired { .. }))
            .count();
        if kernel.slope_paired[index].set_machine_inputs(inputs, topology)? {
            kernel.identity = std::sync::Arc::new(());
        }
        Ok(())
    }

    /// Set the already prepared dependent vector copied into the machine root.
    ///
    /// The caller must prepare this vector using the intended machine input
    /// and bounds. It is separate from the source model's initial frontier.
    /// Replacing or removing it invalidates prepared graph state.
    pub fn set_paired_machine_root(
        &mut self,
        resource: usize,
        root: Option<&PreparedSlopeFunction>,
    ) -> Result<()> {
        let kernel = self
            .coupled
            .as_mut()
            .ok_or_else(|| invalid("no coupled resources"))?;
        if !matches!(
            kernel.definitions.get(resource),
            Some(CoupledResource::SlopePaired { .. })
        ) {
            return Err(invalid("machine root requires a paired slope resource"));
        }
        let index = kernel.definitions[..resource]
            .iter()
            .filter(|r| matches!(r, CoupledResource::SlopePaired { .. }))
            .count();
        if kernel.slope_paired[index].set_machine_root_input(root)? {
            kernel.identity = std::sync::Arc::new(());
        }
        Ok(())
    }

    /// Prepare and install explicit per-arc machine operands atomically.
    ///
    /// `inputs` must be in physical arc-vector order. Its tail windows are
    /// caller-supplied reference inputs, not inferred from source resources;
    /// arcs with the same tail must supply identical window bits. The base
    /// vectors must already be truncated for those windows, and the dependent
    /// vectors and waiting windows must already be prepared. The caller must
    /// reinstall after any graph or model changes affecting those inputs.
    ///
    /// This operation does not select machine pricing or certify a source
    /// model. Unsupported arithmetic, invalid input, and interrupted preparation
    /// return unknown without changing the installed sidecar. Work is divided
    /// conservatively across arcs and all stages share one elapsed deadline.
    pub fn prepare_paired_machine_inputs(
        &mut self,
        resource: usize,
        inputs: &[MachineArcPreparationInput<'_>],
        budget: descriptor::FrontierBudget,
    ) -> descriptor::FrontierOutcome<()> {
        use descriptor::{FrontierOutcome, FrontierUnknown};
        let started = std::time::Instant::now();
        let Some(kernel) = self.coupled.as_ref() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("no coupled resources"));
        };
        if !matches!(
            kernel.definitions.get(resource),
            Some(CoupledResource::SlopePaired {
                policy: FrontOperationPolicy::FrontAlgebra,
                ..
            })
        ) {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine preparation requires a front-algebra paired resource",
            ));
        }
        if inputs.len() != self.arcs.len() || self.source >= self.vertices.len() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine preparation graph source or arc count",
            ));
        }
        let index = kernel.definitions[..resource]
            .iter()
            .filter(|definition| matches!(definition, CoupledResource::SlopePaired { .. }))
            .count();
        let Some(scans) = inputs.len().checked_add(self.vertices.len()) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        let Some(remaining_work) = budget.max_operations.checked_sub(scans) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        // Reserve half of the remaining allowance for sidecar validation,
        // comparison and construction before its atomic publication.
        let preparation_work = remaining_work / 2;
        let per_arc = if inputs.is_empty() {
            preparation_work
        } else {
            preparation_work / inputs.len()
        };
        let installation_budget = remaining_work - per_arc * inputs.len();
        let mut installation_work = 0usize;
        let mut tails: Vec<Option<Window>> = vec![None; self.vertices.len()];
        let mut prepared = Vec::with_capacity(inputs.len());
        for (position, input) in inputs.iter().enumerate() {
            let remaining_time = budget
                .max_elapsed
                .map(|limit| limit.saturating_sub(started.elapsed()));
            if remaining_time.is_some_and(|remaining| remaining.is_zero()) {
                return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
            }
            let arc = &self.arcs[position];
            if input.dependent_lower.pairs().len() > budget.max_pairs
                || input.dependent_upper.pairs().len() > budget.max_pairs
            {
                return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
            }
            if input.physical_arc != position
                || arc.tail >= self.vertices.len()
                || arc.head >= self.vertices.len()
                || Window::new(input.tail_bounds.lower, input.tail_bounds.upper).is_err()
                || Window::new(input.waiting_bounds.lower, input.waiting_bounds.upper).is_err()
            {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine preparation arc order or bounds",
                ));
            }
            if let Some(previous) = tails[arc.tail] {
                if previous.lower.to_bits() != input.tail_bounds.lower.to_bits()
                    || previous.upper.to_bits() != input.tail_bounds.upper.to_bits()
                {
                    return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                        "machine preparation inconsistent tail bounds",
                    ));
                }
            } else {
                tails[arc.tail] = Some(input.tail_bounds);
            }
            let arc_budget = descriptor::FrontierBudget {
                max_operations: per_arc,
                max_elapsed: remaining_time,
                ..budget
            };
            let result = descriptor::prepare_machine_arc(
                input.truncated_base,
                input.dependent,
                input.tail_bounds.lower,
                input.tail_bounds.upper,
                input.waiting_bounds.upper,
                arc_budget,
            );
            let value = match result {
                FrontierOutcome::Complete(value) => value,
                FrontierOutcome::Empty => {
                    return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                        "machine preparation unavailable",
                    ));
                }
                FrontierOutcome::Unknown(reason) => return FrontierOutcome::Unknown(reason),
            };
            // Charge cloning, validation and potential bitwise comparison
            // before copying the caller's dependent bound functions.
            let arc_work = [
                &value.inputs.forward,
                &value.inputs.inverse,
                &value.inputs.sum,
                input.dependent_lower,
                input.dependent_upper,
                input.dependent,
            ]
            .iter()
            .try_fold(6usize, |total, function| {
                function
                    .pairs()
                    .len()
                    .checked_mul(4)
                    .and_then(|pairs| total.checked_add(pairs))
            });
            let Some(next_work) = arc_work.and_then(|work| installation_work.checked_add(work))
            else {
                return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
            };
            if next_work > installation_budget {
                return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
            }
            installation_work = next_work;
            prepared.push(PreparedMachineArc {
                tail_limit: value.limits.tail_limit,
                head_limit: input.waiting_bounds.upper,
                waiting_lower: input.waiting_bounds.lower,
                waiting_upper: input.waiting_bounds.upper,
                waiting_rate: input.waiting_rate,
                forward_mapping: value.inputs.forward,
                inverse: value.inputs.inverse,
                sum_vector: value.inputs.sum,
                dependent_lower: input.dependent_lower.clone(),
                dependent_upper: input.dependent_upper.clone(),
                backward: Some(PreparedMachineBackwardInputs {
                    tail_lower: input.tail_bounds.lower,
                    tail_upper: input.tail_bounds.upper,
                    forward_at_tail_lower: value.limits.forward_at_tail_lower,
                    mapped_upper: value.limits.mapped_upper,
                    dependent: input.dependent.clone(),
                }),
            });
        }
        if budget
            .max_elapsed
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        // Stage all allocations and validation before publication so a
        // deadline or failure cannot leave a partially installed sidecar.
        let topology = slope_paired::MachineTopology {
            source: self.source,
            sink: self.sink,
            vertex_count: self.vertices.len(),
            arcs: self
                .arcs
                .iter()
                .map(|arc| (arc.id, arc.tail, arc.head))
                .collect(),
        };
        let mut staged = kernel.slope_paired[index].clone();
        let changed = match staged.set_machine_inputs(Some(&prepared), Some(topology)) {
            Ok(changed) => changed,
            Err(_) => {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine preparation could not install inputs",
                ))
            }
        };
        if budget
            .max_elapsed
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        if changed {
            let kernel = self.coupled.as_mut().expect("kernel was present");
            kernel.slope_paired[index] = staged;
            kernel.identity = std::sync::Arc::new(());
        }
        FrontierOutcome::Complete(())
    }

    /// Prepare a backward machine root from installed sink-arc upper vectors.
    ///
    /// The supplied terminal bounds must be the intended machine bounds. The
    /// installed operands and graph are explicit inputs, not a source-model
    /// certificate. Unsupported or stale data returns unknown.
    pub fn prepare_paired_machine_backward_root(
        &self,
        resource: usize,
        terminal_bounds: Window,
        budget: descriptor::FrontierBudget,
    ) -> descriptor::FrontierOutcome<descriptor::MachineRootState> {
        use descriptor::{FrontierBudget, FrontierOutcome, FrontierUnknown, MachineRootState};
        let started = std::time::Instant::now();
        let Some(kernel) = self.coupled.as_ref() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("no coupled resources"));
        };
        if budget
            .max_elapsed
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        let Some(validation_work) = self
            .arcs
            .len()
            .checked_mul(2)
            .and_then(|work| work.checked_add(resource))
        else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        let Some(remaining_work) = budget.max_operations.checked_sub(validation_work) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        if !matches!(
            kernel.definitions.get(resource),
            Some(CoupledResource::SlopePaired { .. })
        ) {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine backward root requires a paired slope resource",
            ));
        }
        let index = kernel.definitions[..resource]
            .iter()
            .filter(|item| matches!(item, CoupledResource::SlopePaired { .. }))
            .count();
        let paired = &kernel.slope_paired[index];
        let Some(inputs) = paired.machine_input_handle() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "paired machine inputs unavailable",
            ));
        };
        if self.sink >= self.vertices.len() || !paired.machine_topology_matches(self) {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "paired machine graph topology changed",
            ));
        }
        let mut upper_vectors = Vec::new();
        for (arc, input) in self.arcs.iter().zip(inputs.iter()) {
            if budget
                .max_elapsed
                .is_some_and(|limit| started.elapsed() >= limit)
            {
                return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
            }
            if arc.head == self.sink {
                upper_vectors.push(&input.dependent_upper);
            }
        }
        let remaining_time = budget
            .max_elapsed
            .map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining_time.is_some_and(|limit| limit.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        MachineRootState::from_backward_inputs(
            &upper_vectors,
            terminal_bounds.lower,
            terminal_bounds.upper,
            FrontierBudget {
                max_operations: remaining_work,
                max_elapsed: remaining_time,
                ..budget
            },
        )
    }

    /// Install the prepared backward root for machine label creation.
    /// A later arc-input replacement clears the root and requires reinstalling.
    pub fn install_paired_machine_backward_root(
        &mut self,
        resource: usize,
        terminal_bounds: Window,
        budget: descriptor::FrontierBudget,
    ) -> descriptor::FrontierOutcome<()> {
        use descriptor::{FrontierOutcome, FrontierUnknown};
        let started = std::time::Instant::now();
        let root_work = budget.max_operations / 2;
        let root = match self.prepare_paired_machine_backward_root(
            resource,
            terminal_bounds,
            descriptor::FrontierBudget {
                max_operations: root_work,
                ..budget
            },
        ) {
            FrontierOutcome::Complete(root) => root,
            FrontierOutcome::Empty => return FrontierOutcome::Empty,
            FrontierOutcome::Unknown(reason) => return FrontierOutcome::Unknown(reason),
        };
        let Some(kernel) = self.coupled.as_mut() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("no coupled resources"));
        };
        let index = kernel.definitions[..resource]
            .iter()
            .filter(|item| matches!(item, CoupledResource::SlopePaired { .. }))
            .count();
        let remaining_time = budget
            .max_elapsed
            .map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining_time.is_some_and(|limit| limit.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        match kernel.slope_paired[index].set_machine_backward_root(
            root,
            descriptor::FrontierBudget {
                max_operations: budget.max_operations - root_work,
                max_elapsed: remaining_time,
                ..budget
            },
        ) {
            FrontierOutcome::Complete(()) => {}
            FrontierOutcome::Empty => return FrontierOutcome::Empty,
            FrontierOutcome::Unknown(reason) => return FrontierOutcome::Unknown(reason),
        }
        kernel.identity = std::sync::Arc::new(());
        FrontierOutcome::Complete(())
    }
}
