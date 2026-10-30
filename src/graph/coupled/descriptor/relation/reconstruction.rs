//! Prepared relation reconstruction operations.
use super::*;

impl PreparedSlopeRelationArc {
    /// Port of the backward-entry solution-recording loop. Steps are supplied
    /// in physical traversal order, opposite to backward history construction.
    /// The initial dependent value belongs to the joined forward prefix.
    pub fn machine_reconstruct_backward_path(
        history: &MachinePathHistory,
        steps: &[(usize, MachineBackwardInputs<'_>)],
        initial_base: f64,
        initial_dependent: f64,
        mode: i32,
        budget: FrontierBudget,
    ) -> FrontierOutcome<MachinePathReconstruction> {
        Self::machine_reconstruct_backward_path_with_rates(
            history, steps, initial_base, initial_dependent, mode, None, budget,
        )
    }

    /// Reconstruct with the original per-arc waiting coefficients. Label
    /// extension uses each input's integer rate; recording uses these raw
    /// coefficients when supplied, as in the paired solution recorder.
    pub fn machine_reconstruct_backward_path_with_rates(
        history: &MachinePathHistory,
        steps: &[(usize, MachineBackwardInputs<'_>)],
        initial_base: f64,
        initial_dependent: f64,
        mode: i32,
        waiting_rates: Option<&[f64]>,
        budget: FrontierBudget,
    ) -> FrontierOutcome<MachinePathReconstruction> {
        Self::machine_reconstruct_backward_path_with_rates_and_debug(
            history, steps, initial_base, initial_dependent, mode, waiting_rates, 0, budget,
        )
    }

    /// Reconstruct with original waiting rates and the resource diagnostic level.
    pub fn machine_reconstruct_backward_path_with_rates_and_debug(
        history: &MachinePathHistory,
        steps: &[(usize, MachineBackwardInputs<'_>)],
        initial_base: f64,
        initial_dependent: f64,
        mode: i32,
        waiting_rates: Option<&[f64]>,
        debug_level: i32,
        budget: FrontierBudget,
    ) -> FrontierOutcome<MachinePathReconstruction> {
        let started = Instant::now();
        if !initial_base.is_finite()
            || !initial_dependent.is_finite()
            || history.len().checked_sub(1) != Some(steps.len())
            || waiting_rates.is_some_and(|rates| {
                rates.len() != steps.len() || rates.iter().zip(steps).any(|(rate, (_, input))| {
                    !rate.is_finite() || rate.trunc() != f64::from(input.waiting_rate)
                })
            })
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine suffix reconstruction shape",
            ));
        }
        let snapshots = match history.snapshots(budget) {
            FrontierOutcome::Complete(value) => value,
            FrontierOutcome::Empty => {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine suffix history empty",
                ))
            }
            FrontierOutcome::Unknown(reason) => return FrontierOutcome::Unknown(reason),
        };
        if !matches!(snapshots.first(), Some(MachinePathSnapshot::Root(_))) {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine suffix terminal root",
            ));
        }
        let mut work = history.len();
        for snapshot in &snapshots {
            if snapshot.front().components().len() != 1 {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine suffix component count",
                ));
            }
            let count = snapshot.front().components()[0].pairs().len();
            if count > budget.max_pairs {
                return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
            }
            work = work.saturating_add(count.saturating_mul(8));
        }
        for (_, input) in steps {
            for function in [input.mapping, input.dependent] {
                if function.pairs().len() > budget.max_pairs {
                    return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
                }
                work = work.saturating_add(function.pairs().len().saturating_mul(8));
            }
            work = work.saturating_add(16);
        }
        let Some(mut remaining_work) = budget.max_operations.checked_sub(work) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        let mut compute = || -> Result<MachinePathReconstruction, FrontierUnknown> {
            let deadline = || -> Result<Option<std::time::Duration>, FrontierUnknown> {
                let remaining = budget
                    .max_elapsed
                    .map(|limit| limit.saturating_sub(started.elapsed()));
                if remaining.is_some_and(|limit| limit.is_zero()) {
                    return Err(FrontierUnknown::Deadline);
                }
                Ok(remaining)
            };
            let lookup =
                |function: &PreparedSlopeFunction, coordinate| -> Result<f64, FrontierUnknown> {
                    function
                        .machine_vector_value_and_segment(coordinate)
                        .map_err(FrontierUnknown::Arithmetic)?
                        .map(|(value, _)| value)
                        .filter(|value| *value != 1e12)
                        .ok_or(FrontierUnknown::Unsupported(
                            "machine suffix lookup sentinel",
                        ))
                };
            deadline()?;
            let mut result = MachinePathReconstruction {
                base: Vec::with_capacity(history.len()),
                dependent: Vec::with_capacity(history.len()),
                waiting: Vec::with_capacity(steps.len()),
            };
            let mut base = initial_base;
            let mut dependent = initial_dependent;
            result.base.push(base);
            result.dependent.push(dependent);
            for (position, (arc, input)) in steps.iter().enumerate() {
                deadline()?;
                let current = snapshots[snapshots.len() - position - 1];
                let MachinePathSnapshot::Backward {
                    arc: stored_arc,
                    front,
                    waiting_bits,
                    ..
                } = current
                else {
                    return Err(FrontierUnknown::Unsupported(
                        "machine suffix history direction",
                    ));
                };
                if stored_arc != arc {
                    return Err(FrontierUnknown::Unsupported(
                        "machine suffix history arc mismatch",
                    ));
                }
                let (value, segment) = front.components()[0]
                    .machine_value_and_segment(base)
                    .map_err(FrontierUnknown::Arithmetic)?
                    .filter(|(value, _)| *value != 1e12)
                    .ok_or(FrontierUnknown::Unsupported(
                        "machine suffix front lookup sentinel",
                    ))?;
                let arrival = lookup(input.mapping, base)?;
                let increment = lookup(input.dependent, base)?;
                let waiting = if segment < 64 && (*waiting_bits >> segment) & 1 != 0 {
                    let next = &snapshots[snapshots.len() - position - 2]
                        .front()
                        .components()[0];
                    let target = finite(increment + value)?;
                    // Reserve only for waiting scans actually reached. A
                    // clipped point may be expanded to two machine pairs.
                    let count = next.pairs().len();
                    let wait_work = count.saturating_mul(16).saturating_add(160)
                        .saturating_add(count.max(2).saturating_add(input.dependent_upper.pairs().len()).saturating_mul(64));
                    remaining_work = remaining_work.checked_sub(wait_work)
                        .ok_or(FrontierUnknown::WorkLimit)?;
                    match Self::machine_backward_waiting_with_debug(
                        next,
                        input,
                        arrival,
                        target,
                        mode,
                        debug_level,
                        FrontierBudget {
                            max_operations: wait_work,
                            max_elapsed: deadline()?,
                            ..budget
                        },
                    ) {
                        FrontierOutcome::Complete(waiting) => waiting,
                        FrontierOutcome::Empty => {
                            return Err(FrontierUnknown::Unsupported(
                                "machine suffix waiting empty",
                            ))
                        }
                        FrontierOutcome::Unknown(reason) => return Err(reason),
                    }
                } else {
                    0.
                };
                base = finite(arrival + waiting)?;
                let waiting_rate = waiting_rates.map_or(f64::from(input.waiting_rate), |rates| rates[position]);
                dependent = finite(
                    finite(dependent + increment)?
                        + finite(waiting * waiting_rate)?,
                )?;
                result.waiting.push(waiting);
                result.base.push(base);
                result.dependent.push(dependent);
            }
            deadline()?;
            Ok(result)
        };
        match compute() {
            Ok(value) => FrontierOutcome::Complete(value),
            Err(reason) => FrontierOutcome::Unknown(reason),
        }
    }

    /// Walk backward from an explicitly selected terminal coordinate through
    /// a retained machine history. `inverses` contains
    /// `(physical_arc, prepared_inverse)` in root-to-terminal order, one for
    /// each forward snapshot. The inverse for the later snapshot is used to
    /// obtain the coordinate at its predecessor. The returned coordinates are
    /// in root-to-terminal order and are not a source or route certificate.
    /// The caller is responsible for supplying the exact arc inverses; arc
    /// identity is checked against the history but not against a graph.
    /// This treats the explicit root as a reconstruction entry; whether it
    /// corresponds to the reference label-path entry zero is not established.
    pub fn machine_backtrack_coordinates(
        history: &MachinePathHistory,
        terminal: f64,
        inverses: &[(usize, &PreparedSlopeFunction)],
        budget: FrontierBudget,
    ) -> FrontierOutcome<Vec<f64>> {
        match Self::machine_reconstruct_forward_path(history, terminal, inverses, budget) {
            FrontierOutcome::Complete((path, _)) => FrontierOutcome::Complete(path.base),
            FrontierOutcome::Empty => FrontierOutcome::Empty,
            FrontierOutcome::Unknown(reason) => FrontierOutcome::Unknown(reason),
        }
    }

    /// Recompute an explicit machine path, preserving each state before the
    /// following arc's reduction. Arc identities and prepared inputs are
    /// supplied by the caller and are not validated against a graph here.
    /// A recovered machine preflight rejection returns `Empty`. This does not
    /// certify a pricing-label history, a source-control route, or feasibility.
    pub fn machine_replay_path(
        root: &MachineRootState,
        steps: &[(
            usize,
            &PreparedSlopeRelationArc,
            MachineContinuationInputs<'_>,
        )],
        mode: i32,
        max_states: usize,
        budget: FrontierBudget,
    ) -> FrontierOutcome<MachinePathHistory> {
        let started = Instant::now();
        let stages = match steps.len().checked_mul(2).and_then(|n| n.checked_add(1)) {
            Some(stages) => stages,
            None => return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit),
        };
        let per_stage = budget.max_operations / stages;
        if per_stage == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        }
        let stage_budget = || -> std::result::Result<FrontierBudget, FrontierUnknown> {
            let remaining = budget
                .max_elapsed
                .map(|limit| limit.saturating_sub(started.elapsed()));
            if remaining.is_some_and(|time| time.is_zero()) {
                return Err(FrontierUnknown::Deadline);
            }
            Ok(FrontierBudget {
                max_operations: per_stage,
                max_elapsed: remaining,
                ..budget
            })
        };
        macro_rules! run_stage {
            ($expression:expr) => {
                match $expression {
                    FrontierOutcome::Complete(value) => value,
                    FrontierOutcome::Empty => {
                        return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                            "machine path replay stage rejected",
                        ));
                    }
                    FrontierOutcome::Unknown(reason) => return FrontierOutcome::Unknown(reason),
                }
            };
        }
        macro_rules! next_budget {
            () => {
                match stage_budget() {
                    Ok(value) => value,
                    Err(reason) => return FrontierOutcome::Unknown(reason),
                }
            };
        }
        let mut history = run_stage!(MachinePathHistory::from_root(
            root,
            max_states,
            next_budget!()
        ));
        for (arc, transition, inputs) in steps {
            let (reduced, next) = match transition.machine_transition(
                history.last().front(),
                inputs,
                mode,
                next_budget!(),
            ) {
                FrontierOutcome::Complete(value) => value,
                FrontierOutcome::Empty => return FrontierOutcome::Empty,
                FrontierOutcome::Unknown(reason) => return FrontierOutcome::Unknown(reason),
            };
            history = run_stage!(history.append(*arc, &reduced, &next, next_budget!()));
        }
        if budget
            .max_elapsed
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        FrontierOutcome::Complete(history)
    }
}
