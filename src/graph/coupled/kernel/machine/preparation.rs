use super::*;

impl<const D: usize> Graph<D> {
    /// Retain original per-arc waiting coefficients for machine solution
    /// recording. Their integer truncations must match installed transitions.
    pub fn set_paired_machine_waiting_rates(&mut self, resource: usize, rates: &[f64]) -> Result<()> {
        let kernel = self.coupled.as_mut().ok_or_else(|| invalid("no coupled resources"))?;
        if !matches!(kernel.definitions.get(resource), Some(CoupledResource::SlopePaired { .. })) {
            return Err(invalid("machine waiting rates require a paired slope resource"));
        }
        let index = kernel.definitions[..resource].iter()
            .filter(|item| matches!(item, CoupledResource::SlopePaired { .. })).count();
        if kernel.slope_paired[index].machine_mode() == 6 && rates.len() > 1 && rates[0] < 0. {
            return Err(invalid("bounded paired machine cost requires a nonnegative first waiting rate"));
        }
        if kernel.slope_paired[index].set_machine_waiting_rates(rates)? {
            kernel.identity = std::sync::Arc::new(());
            kernel.machine_identity = std::sync::Arc::new(());
        }
        Ok(())
    }

    /// Prepare a paired program and its original waiting coefficients as a
    /// single installation. Invalid coefficients leave the previous program
    /// intact.
    pub fn prepare_paired_front_program_with_rates(
        &mut self,
        resource: usize,
        inputs: &[MachineArcFunctionInput<'_>],
        root_input: descriptor::MachineRootInput<'_>,
        source_bounds: Window,
        terminal_bounds: Window,
        waiting_rates: &[f64],
        budget: descriptor::FrontierBudget,
    ) -> descriptor::FrontierOutcome<()> {
        use descriptor::{FrontierOutcome, FrontierUnknown};
        let started = std::time::Instant::now();
        if waiting_rates.len() != inputs.len() || waiting_rates.iter().zip(inputs).any(|(rate, input)| {
            !rate.is_finite() || rate.trunc() != f64::from(input.waiting_rate)
        }) {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("paired machine waiting rate mismatch"));
        }
        let original = self.coupled.clone();
        let remaining = budget.max_elapsed.map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining.is_some_and(|time| time.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        let result = self.prepare_paired_front_program(resource, inputs, root_input, source_bounds, terminal_bounds, descriptor::FrontierBudget {
            max_elapsed: remaining,
            ..budget
        });
        if !matches!(result, FrontierOutcome::Complete(())) {
            return result;
        }
        if budget.max_elapsed.is_some_and(|limit| started.elapsed() >= limit) {
            self.coupled = original;
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        if self.set_paired_machine_waiting_rates(resource, waiting_rates).is_err() {
            self.coupled = original;
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("paired machine waiting rate installation"));
        }
        if budget.max_elapsed.is_some_and(|limit| started.elapsed() >= limit) {
            self.coupled = original;
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        FrontierOutcome::Complete(())
    }

    /// Prepare arc functions and both roots as one front-program installation.
    /// Windows and encodings are explicit program inputs; this does not
    /// establish a correspondence to a source-model objective.
    pub fn prepare_paired_front_program(
        &mut self,
        resource: usize,
        inputs: &[MachineArcFunctionInput<'_>],
        root_input: descriptor::MachineRootInput<'_>,
        source_bounds: Window,
        terminal_bounds: Window,
        budget: descriptor::FrontierBudget,
    ) -> descriptor::FrontierOutcome<()> {
        use descriptor::{FrontierBudget, FrontierOutcome, FrontierUnknown, MachineRootState};
        let started = std::time::Instant::now();
        if budget.max_elapsed.is_some_and(|limit| limit.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        if inputs.len() != self.arcs.len() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine preparation arc count",
            ));
        }
        let same_window = |a: Window, b: Window| {
            a.lower.to_bits() == b.lower.to_bits() && a.upper.to_bits() == b.upper.to_bits()
        };
        let mut vertex_bounds = vec![None; self.vertices.len()];
        for (vertex, bounds) in [(self.source, source_bounds), (self.sink, terminal_bounds)] {
            let Some(slot) = vertex_bounds.get_mut(vertex) else {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine root vertex unavailable",
                ));
            };
            if slot.is_some_and(|previous| !same_window(previous, bounds)) {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine root bounds disagree with vertex bounds",
                ));
            }
            *slot = Some(bounds);
        }
        for (arc, input) in self.arcs.iter().zip(inputs) {
            if budget.max_elapsed.is_some_and(|limit| started.elapsed() >= limit) {
                return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
            }
            let Some(slot) = vertex_bounds.get_mut(arc.tail) else {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine arc tail vertex unavailable",
                ));
            };
            if slot.is_some_and(|previous| !same_window(previous, input.tail_bounds)) {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine arc bounds disagree with vertex bounds",
                ));
            }
            *slot = Some(input.tail_bounds);
        }
        // Reserve for the two roots by their bounded scan sizes, rather than
        // assigning fixed fractions of the whole program budget to them.
        // Truncation cannot add pairs to a valid source vector.
        let forward_work = match root_input {
            descriptor::MachineRootInput::Affine(_) => Some(10usize),
            descriptor::MachineRootInput::Vector(vector) => vector.pairs().len()
                .checked_mul(17).and_then(|work| work.checked_add(34)),
        };
        let backward_work = self.arcs.iter().zip(inputs).filter(|(arc, _)| arc.head == self.sink)
            .try_fold(0usize, |work, (_, input)| {
                let pairs = match input.dependent_upper {
                    descriptor::MachineRootInput::Affine(_) => 2,
                    descriptor::MachineRootInput::Vector(vector) => vector.pairs().len(),
                };
                pairs.checked_mul(4).and_then(|scan| work.checked_add(scan))
            })
            .and_then(|work| self.arcs.len().checked_mul(2).and_then(|scan| work.checked_add(scan)))
            .and_then(|work| work.checked_add(resource))
            .and_then(|work| work.checked_add(5));
        let (Some(forward_work), Some(backward_work)) = (forward_work, backward_work) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        // If a conservative bound exceeds the old quarter allocation, keep
        // that allocation: truncated roots may need less than their input
        // size, and supported low-budget calls must not become narrower.
        let forward_work = forward_work.min(budget.max_operations / 4);
        let backward_work = backward_work.min(budget.max_operations / 4);
        let Some(arc_work) = budget.max_operations.checked_sub(forward_work)
            .and_then(|work| work.checked_sub(backward_work)) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        let root_time = budget.max_elapsed.map(|limit| limit.saturating_sub(started.elapsed()));
        if root_time.is_some_and(|time| time.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        let root = match MachineRootState::from_input(
            root_input,
            source_bounds.lower,
            source_bounds.upper,
            FrontierBudget {
                max_operations: forward_work,
                max_elapsed: root_time,
                ..budget
            },
        ) {
            FrontierOutcome::Complete(root) => root,
            FrontierOutcome::Empty => return FrontierOutcome::Empty,
            FrontierOutcome::Unknown(reason) => return FrontierOutcome::Unknown(reason),
        };
        // The native initializer checks the prepared domain only when the
        // original root encoding contains multiple pairs.
        if matches!(root_input, descriptor::MachineRootInput::Vector(_)) {
            match root.front().components()[0]
                .check_machine_domain(source_bounds.lower, source_bounds.upper)
            {
                Ok(true) => {}
                Ok(false) => return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine initial root domain check failed",
                )),
                Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
            }
        }
        let Some(original) = self.coupled.clone() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("no coupled resources"));
        };
        let remaining = |started: std::time::Instant| {
            budget
                .max_elapsed
                .map(|limit| limit.saturating_sub(started.elapsed()))
        };
        let result = (|| {
            let time = remaining(started);
            if time.is_some_and(|duration| duration.is_zero()) {
                return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
            }
            match self.prepare_paired_machine_functions(
                resource,
                inputs,
                FrontierBudget {
                    max_operations: arc_work,
                    max_elapsed: time,
                    ..budget
                },
            ) {
                FrontierOutcome::Complete(()) => {}
                other => return other,
            }
            // The recovered cost-resource initializer first checks the
            // initial arc's prepared lower and upper vectors for the literal
            // two-pair constant encoding. Its later-arc loop reuses that
            // first record, so do not infer an all-arcs machine check here.
            let Some(kernel) = self.coupled.as_ref() else {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("no coupled resources"));
            };
            let index = kernel.definitions[..resource].iter()
                .filter(|item| matches!(item, CoupledResource::SlopePaired { .. })).count();
            let paired = &kernel.slope_paired[index];
            if paired.machine_mode() != 4 {
                let Some(inputs) = paired.machine_input_handle() else {
                    return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("paired machine inputs unavailable"));
                };
                let Some(first) = inputs.first() else {
                    return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("paired machine cost needs an arc"));
                };
                for function in [&first.dependent_lower, &first.dependent_upper] {
                    let pairs = function.pairs();
                    if pairs.len() != 2 || pairs[1].0 != 0. {
                        return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("paired machine cost bound is not constant"));
                    }
                }
                if paired.machine_mode() == 6 {
                    let mut value = root.front().components()[0].pairs()[0].0;
                    let mut minimum = value;
                    for pair in root.front().components()[0].pairs().windows(2) {
                        value += (pair[1].1 - pair[0].1) * pair[1].0;
                        if !value.is_finite() {
                            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("paired machine initial minimum unavailable"));
                        }
                        if value < minimum { minimum = value; }
                    }
                    if minimum < first.dependent_lower.pairs()[0].0 {
                        return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("paired machine initial minimum below lower bound"));
                    }
                    if inputs.len() > 1 {
                        if first.waiting_rate < 0 {
                            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("bounded paired machine cost has negative waiting"));
                        }
                        let Some(backward) = first.backward.as_ref() else {
                            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("paired machine dependent input unavailable"));
                        };
                        let pairs = backward.dependent.pairs();
                        let mut value = pairs[0].0;
                        for pair in pairs.windows(2) {
                            value += (pair[1].1 - pair[0].1) * pair[1].0;
                            if !value.is_finite() || value < 0. {
                                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("bounded paired machine increment check failed"));
                            }
                        }
                    }
                }
            }
            if self
                .set_paired_machine_root(resource, Some(&root.front().components()[0]))
                .is_err()
            {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine forward root installation",
                ));
            }
            let time = remaining(started);
            if time.is_some_and(|duration| duration.is_zero()) {
                return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
            }
            self.install_paired_machine_backward_root(
                resource,
                terminal_bounds,
                FrontierBudget {
                    max_operations: backward_work,
                    max_elapsed: time,
                    ..budget
                },
            )
        })();
        let result = if matches!(result, FrontierOutcome::Complete(()))
            && budget.max_elapsed.is_some_and(|limit| started.elapsed() >= limit)
        {
            FrontierOutcome::Unknown(FrontierUnknown::Deadline)
        } else {
            result
        };
        if !matches!(result, FrontierOutcome::Complete(())) {
            self.coupled = Some(original);
        }
        result
    }

    /// Truncate base functions and prepare dependent arc functions over the
    /// front program's tail and waiting windows, then install the resulting
    /// arc operands atomically.
    /// These inputs are front-program descriptors, not source certificates.
    pub fn prepare_paired_machine_functions(
        &mut self,
        resource: usize,
        inputs: &[MachineArcFunctionInput<'_>],
        budget: descriptor::FrontierBudget,
    ) -> descriptor::FrontierOutcome<()> {
        use descriptor::{FrontierBudget, FrontierOutcome, FrontierUnknown, MachineRootState};
        let started = std::time::Instant::now();
        if inputs.len() != self.arcs.len() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine preparation arc count",
            ));
        }
        let mut remaining_work = budget.max_operations;
        let mut functions = Vec::with_capacity(inputs.len());
        for input in inputs {
            let Some(base_work) = input
                .base
                .pairs()
                .len()
                .checked_mul(16)
                .and_then(|n| n.checked_add(32))
            else {
                return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
            };
            let Some(after_base) = remaining_work.checked_sub(base_work) else {
                return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
            };
            remaining_work = after_base;
            if input.base.pairs().len() > budget.max_pairs {
                return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
            }
            if budget
                .max_elapsed
                .is_some_and(|limit| started.elapsed() >= limit)
            {
                return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
            }
            let base = match input
                .base
                .machine_truncate(input.tail_bounds.lower, input.tail_bounds.upper)
            {
                Ok(base) => base,
                Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
            };
            let mut values = Vec::with_capacity(4);
            values.push(base);
            for (function, window) in [
                (input.dependent, input.tail_bounds),
                (input.dependent_lower, input.waiting_bounds),
                (input.dependent_upper, input.waiting_bounds),
            ] {
                let stage_work = match function {
                    descriptor::MachineRootInput::Affine(_) => Some(10),
                    descriptor::MachineRootInput::Vector(vector) => vector
                        .pairs()
                        .len()
                        .checked_mul(17)
                        .and_then(|n| n.checked_add(34)),
                };
                let Some(stage_work) = stage_work else {
                    return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
                };
                let Some(after_stage) = remaining_work.checked_sub(stage_work) else {
                    return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
                };
                remaining_work = after_stage;
                let remaining = budget
                    .max_elapsed
                    .map(|limit| limit.saturating_sub(started.elapsed()));
                if remaining.is_some_and(|time| time.is_zero()) {
                    return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
                }
                match MachineRootState::from_input(
                    function,
                    window.lower,
                    window.upper,
                    FrontierBudget {
                        max_operations: stage_work,
                        max_elapsed: remaining,
                        ..budget
                    },
                ) {
                    FrontierOutcome::Complete(root) => {
                        values.push(root.front().components()[0].clone())
                    }
                    FrontierOutcome::Empty => return FrontierOutcome::Empty,
                    FrontierOutcome::Unknown(reason) => return FrontierOutcome::Unknown(reason),
                }
            }
            functions.push(values);
        }
        let prepared: Vec<_> = inputs
            .iter()
            .zip(&functions)
            .map(|(input, values)| MachineArcPreparationInput {
                physical_arc: input.physical_arc,
                tail_bounds: input.tail_bounds,
                waiting_bounds: input.waiting_bounds,
                waiting_rate: input.waiting_rate,
                truncated_base: &values[0],
                dependent: &values[1],
                dependent_lower: &values[2],
                dependent_upper: &values[3],
            })
            .collect();
        let remaining = budget
            .max_elapsed
            .map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining.is_some_and(|time| time.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        self.prepare_paired_machine_inputs(
            resource,
            &prepared,
            FrontierBudget {
                max_operations: remaining_work,
                max_elapsed: remaining,
                ..budget
            },
        )
    }

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
            kernel.machine_identity = std::sync::Arc::new(());
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
            kernel.machine_identity = std::sync::Arc::new(());
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
            kernel.machine_identity = std::sync::Arc::new(());
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
        // A backward root always has two stored pairs, so retaining its
        // history consumes exactly three operations. Reserve that amount and
        // give the remaining budget to preparation, whose scan grows with
        // the number of sink arcs and their upper-vector lengths.
        let Some(root_work) = budget.max_operations.checked_sub(3) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
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
                max_operations: 3,
                max_elapsed: remaining_time,
                ..budget
            },
        ) {
            FrontierOutcome::Complete(()) => {}
            FrontierOutcome::Empty => return FrontierOutcome::Empty,
            FrontierOutcome::Unknown(reason) => return FrontierOutcome::Unknown(reason),
        }
        kernel.identity = std::sync::Arc::new(());
        kernel.machine_identity = std::sync::Arc::new(());
        FrontierOutcome::Complete(())
    }
}
