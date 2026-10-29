//! Ordered paired machine concatenation, separate from source certification.
use super::*;
use std::time::Instant;

/// Selected machine join. Feasibility mode chooses a coordinate without
/// writing a dependent cost; cost modes retain their literal joined value.
#[derive(Clone, Copy, Debug)]
pub struct MachineConcatenation {
    pub coordinate: f64,
    pub dependent: Option<f64>,
}

/// Raw label-wrapper outputs. An NG overlap rejects the join after the cost
/// outputs have been computed. Neither output certifies a source objective.
#[derive(Clone, Copy, Debug)]
pub struct MachineLabelConcatenation {
    pub compatible: bool,
    pub reduced_cost: f64,
    pub dependent_cost: f64,
}

impl MachineConcatenation {
    /// Combine a selected paired join with physical label costs and NG memory.
    /// The paired contribution and additional cost are added before the two
    /// physical label costs are combined with them. This order is observable
    /// in binary64 arithmetic. Coefficients and source normalization must have
    /// been handled by the caller; this method does not infer either one.
    pub fn join_labels(
        &self,
        forward_cost: f64,
        backward_cost: f64,
        additional_cost: f64,
        forward_memory: u64,
        backward_memory: u64,
    ) -> FrontierOutcome<MachineLabelConcatenation> {
        let compute = || -> Result<MachineLabelConcatenation, FrontierUnknown> {
            let dependent_cost = finite(self.dependent.unwrap_or(0.))?;
            let additional = finite(dependent_cost + finite(additional_cost)?)?;
            let physical = finite(finite(forward_cost)? + finite(backward_cost)?)?;
            Ok(MachineLabelConcatenation {
                compatible: forward_memory & backward_memory == 0,
                reduced_cost: finite(physical + additional)?,
                dependent_cost,
            })
        };
        match compute() {
            Ok(value) => FrontierOutcome::Complete(value),
            Err(reason) => FrontierOutcome::Unknown(reason),
        }
    }
}

/// Reconstructed machine consumptions at each vertex and waiting at each arc's
/// arrival. The root's own waiting is retained separately. These values do not
/// establish source feasibility or objective normalization.
#[derive(Clone, Debug)]
pub struct MachineJoinedPathReconstruction {
    pub selection: MachineConcatenation,
    pub path: MachinePathReconstruction,
    pub initial_waiting: f64,
}

#[cfg(test)]
#[path = "../../../../tests/unit/graph/coupled/descriptor/concatenation_tests.rs"]
mod tests;

// The solution recorder uses this same bit scan for the root and every
// forward state, including near knots and zero-width machine segments.
fn forward_entry(
    snapshot: &MachinePathSnapshot,
    coordinate: f64,
) -> Result<(f64, f64, f64), FrontierUnknown> {
    let (bits, stored_predecessor) = match snapshot {
        MachinePathSnapshot::Root(root) => (root.waiting_bits(), Some(root.predecessor())),
        MachinePathSnapshot::Forward { state, .. } => (state.waiting_bits(), state.predecessor()),
        MachinePathSnapshot::Backward { .. } => {
            return Err(FrontierUnknown::Unsupported(
                "machine forward recorder direction",
            ))
        }
    };
    let [function] = snapshot.front().components() else {
        return Err(FrontierUnknown::Unsupported(
            "machine forward recorder components",
        ));
    };
    let (value, mut segment) = function
        .machine_value_and_segment(coordinate)
        .map_err(FrontierUnknown::Arithmetic)?
        .filter(|(value, _)| *value != 1e12)
        .ok_or(FrontierUnknown::Unsupported(
            "machine forward recorder lookup sentinel",
        ))?;
    let pairs = function.pairs();
    if segment != 1 {
        let previous = pairs
            .get(segment.saturating_sub(1))
            .ok_or(FrontierUnknown::Unsupported(
                "machine forward recorder segment",
            ))?;
        if near(previous.1, coordinate)? {
            segment -= 1;
        }
    }
    if segment >= 64 {
        return Err(FrontierUnknown::Unsupported(
            "machine forward recorder waiting bit",
        ));
    }
    if (bits >> segment) & 1 == 0 {
        return Ok((value, coordinate, 0.));
    }
    segment = segment.checked_sub(1).ok_or(FrontierUnknown::Unsupported(
        "machine forward recorder waiting segment",
    ))?;
    while segment > 0 && (bits >> segment) & 1 != 0 {
        segment -= 1;
    }
    let predecessor = if segment == 0 {
        stored_predecessor.ok_or(FrontierUnknown::Unsupported(
            "machine forward recorder predecessor unavailable",
        ))?
    } else {
        pairs
            .get(segment)
            .ok_or(FrontierUnknown::Unsupported(
                "machine forward recorder predecessor segment",
            ))?
            .1
    };
    if predecessor == 1e12 || predecessor < 0. || !predecessor.is_finite() {
        return Err(FrontierUnknown::Unsupported(
            "machine forward recorder invalid predecessor",
        ));
    }
    Ok((value, predecessor, finite(coordinate - predecessor)?))
}

fn finite(value: f64) -> Result<f64, FrontierUnknown> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(FrontierUnknown::Arithmetic(
            RawFunctionError::ArithmeticOverflow("machine concatenation"),
        ))
    }
}
fn near(a: f64, b: f64) -> Result<bool, FrontierUnknown> {
    let difference = finite(a - b)?.abs();
    Ok(difference < 0.0001 || difference < finite(finite(a + b)?.abs() * 2.220446049250313e-10)?)
}

impl PreparedSlopeRelationArc {
    /// Port the machine concatenation scan and its mode-specific endpoint
    /// selection. This does not normalize a cost or certify a source route.
    pub fn machine_concatenation(
        forward: &PreparedSlopeFunction,
        backward: &PreparedSlopeFunction,
        ceiling: f64,
        mode: i32,
        budget: FrontierBudget,
    ) -> FrontierOutcome<MachineConcatenation> {
        let started = Instant::now();
        let a = forward.pairs();
        let b = backward.pairs();
        if budget.max_components == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        if a.len() > budget.max_pairs || b.len() > budget.max_pairs {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        if a.is_empty()
            || b.is_empty()
            || a.len() > 12
            || b.len() > 12
            || !ceiling.is_finite()
            || !matches!(mode, 4..=6)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine concatenation inputs",
            ));
        }
        if budget.max_operations < a.len().saturating_add(b.len()).saturating_mul(64) {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        }
        let compute = || -> Result<MachineConcatenation, FrontierUnknown> {
            let deadline = || -> Result<(), FrontierUnknown> {
                if budget
                    .max_elapsed
                    .is_some_and(|limit| started.elapsed() >= limit)
                {
                    return Err(FrontierUnknown::Deadline);
                }
                Ok(())
            };
            deadline()?;
            if (a[0].1 > b.last().unwrap().1 && !near(a[0].1, b.last().unwrap().1)?)
                || (b[0].1 > a.last().unwrap().1 && !near(b[0].1, a.last().unwrap().1)?)
            {
                return Err(FrontierUnknown::Unsupported(
                    "machine concatenation disjoint domains",
                ));
            }
            // Ordered max/min select the forward coordinate on a tie.
            let mut coordinate = if b[0].1 <= a[0].1 { a[0].1 } else { b[0].1 };
            let upper = if a.last().unwrap().1 <= b.last().unwrap().1 {
                a.last().unwrap().1
            } else {
                b.last().unwrap().1
            };
            let (mut av, mut i) = forward
                .machine_value_and_segment(coordinate)
                .map_err(FrontierUnknown::Arithmetic)?
                .filter(|(value, _)| *value != 1e12)
                .ok_or(FrontierUnknown::Unsupported(
                    "machine concatenation forward lookup",
                ))?;
            let (mut bv, mut j) = backward
                .machine_value_and_segment(coordinate)
                .map_err(FrontierUnknown::Arithmetic)?
                .filter(|(value, _)| *value != 1e12)
                .ok_or(FrontierUnknown::Unsupported(
                    "machine concatenation backward lookup",
                ))?;
            let allowed = |av, bv| -> Result<bool, FrontierUnknown> {
                Ok(mode == 5 || av <= bv || near(av, bv)?)
            };
            let mut selected = None;
            let mut minimum = 1e12;
            if allowed(av, bv)? {
                if mode == 4 {
                    return Ok(MachineConcatenation {
                        coordinate,
                        dependent: None,
                    });
                }
                let value = finite(finite(ceiling - bv)? + av)?;
                if value < minimum {
                    minimum = value;
                    selected = Some(coordinate);
                }
            }
            while i < a.len() && j < b.len() {
                deadline()?;
                let (sa, ae) = a[i];
                let (sb, be) = b[j];
                let first_end = if ae <= be { ae } else { be };
                if upper < first_end {
                    break;
                }
                let previous = coordinate;
                if near(ae, be)? {
                    // Each near endpoint uses its own span, then the forward
                    // coordinate becomes the common anchor for the next step.
                    bv = finite(bv + finite(sb * finite(be - previous)?)?)?;
                    av = finite(av + finite(sa * finite(ae - previous)?)?)?;
                    coordinate = ae;
                    i += 1;
                    j += 1;
                } else if be <= ae {
                    bv = finite(bv + finite(sb * finite(be - previous)?)?)?;
                    av = finite(av + finite(sa * finite(be - previous)?)?)?;
                    coordinate = be;
                    j += 1;
                } else {
                    av = finite(av + finite(sa * finite(ae - previous)?)?)?;
                    bv = finite(bv + finite(sb * finite(ae - previous)?)?)?;
                    coordinate = ae;
                    i += 1;
                }
                if !allowed(av, bv)? {
                    continue;
                }
                if mode == 4 {
                    let numerator =
                        finite(finite(finite(bv - av)? + finite(sa * ae)?)? - finite(sb * be)?)?;
                    let coordinate = finite(numerator / finite(sa - sb)?)?;
                    return Ok(MachineConcatenation {
                        coordinate,
                        dependent: None,
                    });
                }
                let value = finite(finite(ceiling - bv)? + av)?;
                if value < minimum {
                    minimum = value;
                    selected = Some(coordinate);
                }
            }
            if !near(coordinate, upper)? {
                // Retain the final branch's original segment-anchor spans;
                // reassociation around the last synthetic split changes it.
                let &(sa, _) = a.get(i).ok_or(FrontierUnknown::Unsupported(
                    "machine concatenation exhausted forward stream",
                ))?;
                let &(sb, _) = b.get(j).ok_or(FrontierUnknown::Unsupported(
                    "machine concatenation exhausted backward stream",
                ))?;
                av = finite(av + finite(finite(upper - a[i - 1].1)? * sa)?)?;
                bv = finite(bv + finite(finite(upper - b[j - 1].1)? * sb)?)?;
                if allowed(av, bv)? {
                    if mode == 4 {
                        let numerator = finite(
                            finite(finite(bv - av)? + finite(sa * upper)?)? - finite(upper * sb)?,
                        )?;
                        return Ok(MachineConcatenation {
                            coordinate: finite(numerator / finite(sa - sb)?)?,
                            dependent: None,
                        });
                    }
                    let value = finite(finite(ceiling - bv)? + av)?;
                    if value < minimum {
                        minimum = value;
                        selected = Some(upper);
                    }
                }
            }
            deadline()?;
            let coordinate = selected.ok_or(FrontierUnknown::Unsupported(
                "machine concatenation has no selected endpoint",
            ))?;
            Ok(MachineConcatenation {
                coordinate,
                dependent: Some(minimum),
            })
        };
        match compute() {
            Ok(_)
                if budget
                    .max_elapsed
                    .is_some_and(|limit| started.elapsed() >= limit) =>
            {
                FrontierOutcome::Unknown(FrontierUnknown::Deadline)
            }
            Ok(value) => FrontierOutcome::Complete(value),
            Err(reason) => FrontierOutcome::Unknown(reason),
        }
    }

    /// Recover the forward recorder loop using the original per-arc inverses.
    /// The tuple's second element is waiting at the root before any arc.
    pub fn machine_reconstruct_forward_path(
        history: &MachinePathHistory,
        terminal: f64,
        inverses: &[(usize, &PreparedSlopeFunction)],
        budget: FrontierBudget,
    ) -> FrontierOutcome<(MachinePathReconstruction, f64)> {
        let started = Instant::now();
        let compute = || -> Result<(MachinePathReconstruction, f64), FrontierUnknown> {
            let deadline = || -> Result<(), FrontierUnknown> {
                if budget
                    .max_elapsed
                    .is_some_and(|limit| started.elapsed() >= limit)
                {
                    return Err(FrontierUnknown::Deadline);
                }
                Ok(())
            };
            deadline()?;
            if !terminal.is_finite() || history.len().checked_sub(1) != Some(inverses.len()) {
                return Err(FrontierUnknown::Unsupported(
                    "machine forward recorder path shape",
                ));
            }
            if history.len() > budget.max_components {
                return Err(FrontierUnknown::ComponentLimit);
            }
            // Reserve the retained-history walk before allocating snapshots.
            if history.len() > budget.max_operations {
                return Err(FrontierUnknown::WorkLimit);
            }
            let snapshots = match history.snapshots(budget) {
                FrontierOutcome::Complete(value) => value,
                FrontierOutcome::Empty => {
                    return Err(FrontierUnknown::Unsupported(
                        "machine forward recorder empty history",
                    ))
                }
                FrontierOutcome::Unknown(reason) => return Err(reason),
            };
            if !matches!(snapshots.first(), Some(MachinePathSnapshot::Root(_))) {
                return Err(FrontierUnknown::Unsupported(
                    "machine forward recorder root",
                ));
            }
            let mut work = history.len();
            for snapshot in &snapshots {
                deadline()?;
                let [function] = snapshot.front().components() else {
                    return Err(FrontierUnknown::Unsupported(
                        "machine forward recorder components",
                    ));
                };
                if function.pairs().len() > budget.max_pairs {
                    return Err(FrontierUnknown::PairLimit);
                }
                if function.pairs().len() > 12 {
                    return Err(FrontierUnknown::Unsupported(
                        "machine forward recorder array size",
                    ));
                }
                work = work
                    .saturating_add(function.pairs().len().saturating_mul(16))
                    .saturating_add(16);
            }
            for (_, inverse) in inverses {
                deadline()?;
                if inverse.pairs().len() > budget.max_pairs {
                    return Err(FrontierUnknown::PairLimit);
                }
                work = work.saturating_add(inverse.pairs().len().saturating_mul(8));
            }
            if work > budget.max_operations {
                return Err(FrontierUnknown::WorkLimit);
            }
            let mut result = MachinePathReconstruction {
                base: vec![0.; snapshots.len()],
                dependent: vec![0.; snapshots.len()],
                waiting: vec![0.; inverses.len()],
            };
            let mut coordinate = terminal;
            let mut initial_waiting = 0.;
            for position in (0..snapshots.len()).rev() {
                deadline()?;
                let (value, predecessor, waiting) = forward_entry(snapshots[position], coordinate)?;
                result.base[position] = coordinate;
                result.dependent[position] = value;
                if position == 0 {
                    initial_waiting = waiting;
                    break;
                }
                let MachinePathSnapshot::Forward { arc, .. } = snapshots[position] else {
                    return Err(FrontierUnknown::Unsupported(
                        "machine forward recorder history ordering",
                    ));
                };
                let (inverse_arc, inverse) = inverses[position - 1];
                if *arc != inverse_arc {
                    return Err(FrontierUnknown::Unsupported(
                        "machine forward recorder inverse arc",
                    ));
                }
                result.waiting[position - 1] = waiting;
                coordinate = inverse
                    .evaluate(predecessor, EvaluationPolicy::EndpointTolerance)
                    .map_err(FrontierUnknown::Arithmetic)?
                    .filter(|value| *value != 1e12)
                    .ok_or(FrontierUnknown::Unsupported(
                        "machine forward recorder inverse sentinel",
                    ))?;
            }
            deadline()?;
            Ok((result, initial_waiting))
        };
        match compute() {
            Ok(value) => FrontierOutcome::Complete(value),
            Err(reason) => FrontierOutcome::Unknown(reason),
        }
    }

    /// Select a join and record both halves using their retained fronts and
    /// original machine operands. No machine cost is converted into a source
    /// objective, and a recorder failure is unknown rather than infeasibility.
    pub fn machine_reconstruct_joined_path(
        forward: &MachinePathHistory,
        backward: &MachinePathHistory,
        inverses: &[(usize, &PreparedSlopeFunction)],
        suffix: &[(usize, MachineBackwardInputs<'_>)],
        mode: i32,
        budget: FrontierBudget,
    ) -> FrontierOutcome<MachineJoinedPathReconstruction> {
        let started = Instant::now();
        let compute = || -> Result<MachineJoinedPathReconstruction, FrontierUnknown> {
            let remaining = |work| -> Result<FrontierBudget, FrontierUnknown> {
                let max_elapsed = budget
                    .max_elapsed
                    .map(|limit| limit.saturating_sub(started.elapsed()));
                if max_elapsed.is_some_and(|limit| limit.is_zero()) {
                    return Err(FrontierUnknown::Deadline);
                }
                Ok(FrontierBudget {
                    max_operations: work,
                    max_elapsed,
                    ..budget
                })
            };
            let mut work = budget.max_operations;
            let traversal = forward.len();
            work = work
                .checked_sub(traversal)
                .ok_or(FrontierUnknown::WorkLimit)?;
            let fs = match forward.snapshots(remaining(forward.len())?) {
                FrontierOutcome::Complete(value) => value,
                FrontierOutcome::Empty => {
                    return Err(FrontierUnknown::Unsupported(
                        "machine joined forward history empty",
                    ))
                }
                FrontierOutcome::Unknown(reason) => return Err(reason),
            };
            let ([a], [b]) = (
                forward.last().front().components(),
                backward.last().front().components(),
            ) else {
                return Err(FrontierUnknown::Unsupported(
                    "machine joined front components",
                ));
            };
            let ceiling = backward
                .backward_ceiling()
                .ok_or(FrontierUnknown::Unsupported(
                    "machine joined terminal ceiling unavailable",
                ))?;
            let concat_work = a
                .pairs()
                .len()
                .saturating_add(b.pairs().len())
                .saturating_mul(64);
            work = work
                .checked_sub(concat_work)
                .ok_or(FrontierUnknown::WorkLimit)?;
            let selection =
                match Self::machine_concatenation(a, b, ceiling, mode, remaining(concat_work)?) {
                    FrontierOutcome::Complete(value) => value,
                    FrontierOutcome::Empty => {
                        return Err(FrontierUnknown::Unsupported(
                            "machine joined selection empty",
                        ))
                    }
                    FrontierOutcome::Unknown(reason) => return Err(reason),
                };
            // Reserve the actual bounded forward-recorder scans; leave all
            // remaining work for the variable-length backward waiting scans.
            let mut forward_work = forward.len();
            for snapshot in fs {
                remaining(work)?;
                let [function] = snapshot.front().components() else {
                    return Err(FrontierUnknown::Unsupported(
                        "machine joined forward components",
                    ));
                };
                forward_work = forward_work
                    .saturating_add(function.pairs().len().saturating_mul(16))
                    .saturating_add(16);
            }
            for (_, inverse) in inverses {
                remaining(work)?;
                forward_work = forward_work.saturating_add(inverse.pairs().len().saturating_mul(8));
            }
            let backward_work = work
                .checked_sub(forward_work)
                .ok_or(FrontierUnknown::WorkLimit)?;
            let (mut path, initial_waiting) = match Self::machine_reconstruct_forward_path(
                forward,
                selection.coordinate,
                inverses,
                remaining(forward_work)?,
            ) {
                FrontierOutcome::Complete(value) => value,
                FrontierOutcome::Empty => {
                    return Err(FrontierUnknown::Unsupported("machine joined prefix empty"))
                }
                FrontierOutcome::Unknown(reason) => return Err(reason),
            };
            let tail = match Self::machine_reconstruct_backward_path(
                backward,
                suffix,
                selection.coordinate,
                *path.dependent.last().unwrap(),
                mode,
                remaining(backward_work)?,
            ) {
                FrontierOutcome::Complete(value) => value,
                FrontierOutcome::Empty => {
                    return Err(FrontierUnknown::Unsupported("machine joined suffix empty"))
                }
                FrontierOutcome::Unknown(reason) => return Err(reason),
            };
            if let Some(expected) = selection.dependent {
                if !near(expected, *tail.dependent.last().unwrap())? {
                    return Err(FrontierUnknown::Unsupported(
                        "machine joined recorded consumption differs from selection",
                    ));
                }
            }
            path.base.extend(tail.base.into_iter().skip(1));
            path.dependent.extend(tail.dependent.into_iter().skip(1));
            path.waiting.extend(tail.waiting);
            remaining(0)?;
            Ok(MachineJoinedPathReconstruction {
                selection,
                path,
                initial_waiting,
            })
        };
        match compute() {
            Ok(value) => FrontierOutcome::Complete(value),
            Err(reason) => FrontierOutcome::Unknown(reason),
        }
    }
}
