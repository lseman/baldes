//! Paired slope transitions. Unsupported arithmetic is never infeasibility.
use super::*;
use std::{sync::Arc, time::Instant};
mod backward_dominance;

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "clp", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "clp", serde(deny_unknown_fields))]
pub struct SlopeRelationArc {
    pub arrival: PreparedSlopeFunction,
    pub increment: PreparedSlopeFunction,
    pub lower: PreparedSlopeFunction,
    pub upper: PreparedSlopeFunction,
    pub waiting: i32,
}

/// Explicit inputs to the recovered machine-front continuation. The forward
/// lookup, inverse and sum vector are distinct prepared binary operands.
/// These inputs do not assert source-control feasibility.
pub struct MachineContinuationInputs<'a> {
    pub tail_limit: f64,
    pub head_limit: f64,
    pub forward_mapping: &'a PreparedSlopeFunction,
    pub inverse: &'a PreparedSlopeFunction,
    pub sum_vector: &'a PreparedSlopeFunction,
    pub waiting_lower: f64,
    pub waiting_upper: f64,
    pub waiting_rate: i32,
    pub dependent_lower: &'a PreparedSlopeFunction,
    pub dependent_upper: &'a PreparedSlopeFunction,
}

/// Prepared operands for a backward machine transition.
pub struct MachineBackwardInputs<'a> {
    pub waiting_lower: f64,
    pub waiting_upper: f64,
    pub forward_lower: f64,
    pub mapped_upper: f64,
    pub tail_lower: f64,
    pub tail_upper: f64,
    pub waiting_rate: i32,
    pub mapping: &'a PreparedSlopeFunction,
    pub inverse: &'a PreparedSlopeFunction,
    pub dependent: &'a PreparedSlopeFunction,
    pub dependent_lower: &'a PreparedSlopeFunction,
    pub dependent_upper: &'a PreparedSlopeFunction,
}

/// Machine-only reconstruction from a joined base/dependent value through a
/// backward suffix. Values retain the solution recorder's ordered arithmetic;
/// they do not certify a source model or a normalized objective.
#[derive(Clone, Debug)]
pub struct MachinePathReconstruction {
    pub base: Vec<f64>,
    pub dependent: Vec<f64>,
    /// One waiting amount per traversed arc, in physical traversal order.
    pub waiting: Vec<f64>,
}

/// Immutable preparation is shared across labels; raw functions are not rebuilt.
#[derive(Clone, Debug)]
pub struct PreparedSlopeRelationArc {
    source: Arc<SlopeRelationArc>,
    policy: FrontOperationPolicy,
    shape: Result<Translation, &'static str>,
    inverse: FrontierOutcome<Arc<PreparedSlopeFunction>>,
    composition: FrontierOutcome<PreparedCompositionMap>,
    increment_front: FrontierOutcome<SlopeFront>,
    bounds: PreparedClampFilter,
}
impl PartialEq for PreparedSlopeRelationArc {
    fn eq(&self, other: &Self) -> bool {
        self.source == other.source && self.policy == other.policy
    }
}
#[derive(Clone, Debug, PartialEq)]
struct Translation {
    delta: f64,
    phi: f64,
    tail_lo: f64,
    tail_hi: f64,
    head_lo: f64,
    head_hi: f64,
}
#[derive(Clone, Copy)]
struct Segment {
    lo: f64,
    hi: f64,
    y: f64,
    slope: f64,
}

impl PreparedSlopeRelationArc {
    /// Ordered waiting-aware composition, including short intervals, near
    /// knots and the two distinct secant-merge branches. Unsupported machine
    /// representations remain unknown rather than proving infeasibility.
    pub fn machine_backward_compose_streams(
        outer: &PreparedSlopeFunction,
        waiting_bits: u64,
        inner: &PreparedSlopeFunction,
        lower: f64,
        upper: f64,
        budget: FrontierBudget,
    ) -> FrontierOutcome<(PreparedSlopeFunction, u64)> {
        Self::machine_backward_compose_pairs(
            outer.pairs(),
            waiting_bits,
            inner,
            lower,
            upper,
            budget,
        )
    }

    fn machine_backward_compose_pairs(
        outer: &[(f64, f64)],
        waiting_bits: u64,
        inner: &PreparedSlopeFunction,
        lower: f64,
        upper: f64,
        budget: FrontierBudget,
    ) -> FrontierOutcome<(PreparedSlopeFunction, u64)> {
        let started = Instant::now();
        let compute = || -> Result<(PreparedSlopeFunction, u64), FrontierUnknown> {
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
            if budget.max_components == 0 {
                return Err(FrontierUnknown::ComponentLimit);
            }
            let a = outer;
            let b = inner.pairs();
            if a.len() > 12 || !lower.is_finite() || !upper.is_finite() || lower > upper {
                return Err(FrontierUnknown::Unsupported(
                    "machine waiting composition inputs",
                ));
            }
            if a.len() > budget.max_pairs || b.len() > budget.max_pairs {
                return Err(FrontierUnknown::PairLimit);
            }
            if budget.max_operations < a.len().saturating_add(b.len()).saturating_mul(64) {
                return Err(FrontierUnknown::WorkLimit);
            }
            if (lower > b.last().unwrap().1 && !near(lower, b.last().unwrap().1)?)
                || (upper < b[0].1 && !near(upper, b[0].1)?)
            {
                return Err(FrontierUnknown::Unsupported(
                    "machine waiting composition inner domain",
                ));
            }
            if near(lower, upper)? {
                let mapped = inner
                    .machine_value_and_segment(lower)
                    .map_err(FrontierUnknown::Arithmetic)?
                    .filter(|(value, _)| *value != 1e12)
                    .ok_or(FrontierUnknown::Unsupported(
                        "machine short waiting composition inner sentinel",
                    ))?
                    .0;
                let (value, segment) =
                    PreparedSlopeFunction::machine_array_value_and_segment(a, mapped)
                        .map_err(FrontierUnknown::Arithmetic)?
                        .filter(|(value, _)| *value != 1e12)
                        .ok_or(FrontierUnknown::Unsupported(
                            "machine short waiting composition outer sentinel",
                        ))?;
                if budget.max_pairs < 2 {
                    return Err(FrontierUnknown::PairLimit);
                }
                let result =
                    PreparedSlopeFunction::from_machine_pairs(vec![(value, lower), (0., upper)])
                        .map_err(FrontierUnknown::Arithmetic)?;
                deadline()?;
                return Ok((result, ((waiting_bits >> segment) & 1) << 1));
            }
            let (mut current_y, mut inner_index) = if near(lower, b[0].1)? {
                (b[0].0, 1usize)
            } else {
                inner
                    .machine_vector_value_and_segment(lower)
                    .map_err(FrontierUnknown::Arithmetic)?
                    .ok_or(FrontierUnknown::Unsupported(
                        "machine waiting composition start lookup",
                    ))?
            };
            // On the long path a literal inner value of 1e12 is passed to
            // the outer lookup; only the outer result is treated as sentinel.
            let (initial, mut outer_index) =
                PreparedSlopeFunction::machine_array_value_and_segment(a, current_y)
                    .map_err(FrontierUnknown::Arithmetic)?
                    .filter(|(value, _)| *value != 1e12)
                    .ok_or(FrontierUnknown::Unsupported(
                        "machine waiting composition outer sentinel",
                    ))?;
            let mut output: Vec<(f64, f64)> = vec![(initial, lower)];
            let mut output_bits = 0u64;
            let mut current_x = lower;
            let mut outer_coordinate = current_y;
            let mut outer_value = initial;
            let mut output_value = initial;
            // The knot-emission branch retains the prior slope on a merge;
            // the partial-inner-segment branch overwrites it with its secant.
            let emit = |output: &mut Vec<(f64, f64)>,
                        bits: &mut u64,
                        value: f64,
                        before_value: f64,
                        end: f64,
                        segment: usize,
                        overwrite_slope: bool|
             -> Result<(), FrontierUnknown> {
                let index = output.len();
                if index >= 12 || index >= budget.max_pairs {
                    return Err(FrontierUnknown::PairLimit);
                }
                let previous = index - 1;
                let slope =
                    finite(finite(value - before_value)? / finite(end - output[previous].1)?)?;
                let bit = (waiting_bits >> segment) & 1;
                *bits = (*bits & !(1u64 << index)) | (bit << index);
                if previous > 0
                    && ((*bits >> previous) & 1) == bit
                    && near(slope, output[previous].0)?
                {
                    if overwrite_slope {
                        output[previous].0 = slope;
                    }
                    output[previous].1 = end;
                } else {
                    output.push((slope, end));
                }
                Ok(())
            };
            while inner_index < b.len() {
                deadline()?;
                let end_x = if b[inner_index].1 <= upper {
                    b[inner_index].1
                } else {
                    upper
                };
                let end_y =
                    finite(finite(finite(end_x - current_x)? * b[inner_index].0)? + current_y)?;
                let mut emitted_value = output_value;
                while outer_index < a.len() && a[outer_index].1 <= end_y {
                    deadline()?;
                    let end = a[outer_index].1;
                    let coordinate = if near(end, end_y)? {
                        end_x
                    } else {
                        finite(finite(finite(end - current_y)? / b[inner_index].0)? + current_x)?
                    };
                    outer_value = finite(
                        outer_value + finite(finite(end - outer_coordinate)? * a[outer_index].0)?,
                    )?;
                    emit(
                        &mut output,
                        &mut output_bits,
                        outer_value,
                        output_value,
                        coordinate,
                        outer_index,
                        false,
                    )?;
                    outer_coordinate = end;
                    emitted_value = outer_value;
                    output_value = outer_value;
                    outer_index += 1;
                }
                if outer_index >= a.len() && !near(a[outer_index - 1].1, end_y)? {
                    return Err(FrontierUnknown::Unsupported(
                        "machine waiting composition exhausted outer domain",
                    ));
                }
                let previous_x = output.last().unwrap().1;
                if previous_x < end_x && !near(previous_x, end_x)? {
                    let &(slope, _) = a.get(outer_index).ok_or(FrontierUnknown::Unsupported(
                        "machine waiting composition exhausted partial segment",
                    ))?;
                    output_value =
                        finite(finite(finite(end_y - outer_coordinate)? * slope)? + outer_value)?;
                    emit(
                        &mut output,
                        &mut output_bits,
                        output_value,
                        emitted_value,
                        end_x,
                        outer_index,
                        true,
                    )?;
                } else {
                    output_value = emitted_value;
                }
                // Retain the original inner endpoint as the next anchor,
                // including the native upper-clipping and near-end branches.
                current_x = b[inner_index].1;
                inner_index += 1;
                if inner_index >= b.len() || near(upper, output.last().unwrap().1)? {
                    break;
                }
                current_y = end_y;
            }
            let result = PreparedSlopeFunction::from_machine_pairs(output)
                .map_err(FrontierUnknown::Arithmetic)?;
            deadline()?;
            Ok((result, output_bits))
        };
        match compute() {
            Ok(value) => FrontierOutcome::Complete(value),
            Err(reason) => FrontierOutcome::Unknown(reason),
        }
    }

    /// Run the recovered backward recurrence with shared work and deadline:
    /// reduction/truncation, dependent bounds, reverse waiting, composition,
    /// difference and the mode-specific machine key. No source proof follows.
    pub fn machine_backward_transition(
        front: &SlopeFront,
        input: &MachineBackwardInputs<'_>,
        terminal_ceiling: f64,
        mode: i32,
        budget: FrontierBudget,
    ) -> FrontierOutcome<(PreparedSlopeFunction, u64, f64)> {
        let started = Instant::now();
        if !input.tail_lower.is_finite() || !input.tail_upper.is_finite() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine backward tail bounds",
            ));
        }
        let initial_work = budget.max_operations / 2;
        let wait_work = (budget.max_operations - initial_work) / 2;
        let finish_work = budget.max_operations - initial_work - wait_work;
        let stage_budget = |work| -> Result<FrontierBudget, FrontierUnknown> {
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
        macro_rules! stage {
            ($call:expr) => {
                match $call {
                    FrontierOutcome::Complete(value) => value,
                    FrontierOutcome::Empty => return FrontierOutcome::Empty,
                    FrontierOutcome::Unknown(reason) => return FrontierOutcome::Unknown(reason),
                }
            };
        }
        macro_rules! remaining {
            ($work:expr) => {
                match stage_budget($work) {
                    Ok(value) => value,
                    Err(reason) => return FrontierOutcome::Unknown(reason),
                }
            };
        }
        let (bounded, limit) = stage!(Self::machine_backward_initial_bounds(
            front,
            input.waiting_lower,
            input.waiting_upper,
            input.forward_lower,
            input.mapped_upper,
            input.dependent_lower,
            input.dependent_upper,
            remaining!(initial_work),
        ));
        let start = if input.forward_lower <= input.waiting_lower {
            input.waiting_lower
        } else {
            input.forward_lower
        };
        // Retain the max operand's bits on equality, as the caller does when
        // restoring the original mapped lower bound before reverse waiting.
        let lower = if start <= input.forward_lower {
            start
        } else {
            input.forward_lower
        };
        let (envelope, bits) = stage!(Self::machine_backward_wait_pairs(
            &bounded,
            input.waiting_rate,
            lower,
            limit,
            remaining!(wait_work),
        ));
        let inverse_end = match input
            .inverse
            .machine_value_and_segment(envelope.last().unwrap().1)
        {
            Ok(Some((value, _))) if value != 1e12 => value,
            Ok(_) => {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine backward inverse endpoint sentinel",
                ))
            }
            Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
        };
        let upper = if input.tail_upper <= inverse_end {
            input.tail_upper
        } else {
            inverse_end
        };
        let result = Self::machine_backward_finish_pairs(
            &envelope,
            bits,
            input.mapping,
            input.dependent,
            input.tail_lower,
            upper,
            terminal_ceiling,
            mode,
            remaining!(finish_work),
        );
        if let Err(reason) = stage_budget(0) {
            return FrontierOutcome::Unknown(reason);
        }
        result
    }

    /// Reverse waiting scan from the complete filtered frontier with explicit
    /// mapped lower and upper limits. Entries and waiting bits are constructed
    /// right-to-left, then reversed together, exactly as in the machine caller.
    pub fn machine_backward_wait_window(
        front: &PreparedSlopeFunction,
        waiting_rate: i32,
        lower: f64,
        upper: f64,
        budget: FrontierBudget,
    ) -> FrontierOutcome<(PreparedSlopeFunction, u64)> {
        match Self::machine_backward_wait_pairs(front, waiting_rate, lower, upper, budget) {
            FrontierOutcome::Complete((pairs, bits)) => {
                match PreparedSlopeFunction::from_machine_pairs(pairs) {
                    Ok(function) => FrontierOutcome::Complete((function, bits)),
                    Err(error) => FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
                }
            }
            FrontierOutcome::Empty => FrontierOutcome::Empty,
            FrontierOutcome::Unknown(reason) => FrontierOutcome::Unknown(reason),
        }
    }

    fn machine_backward_wait_pairs(
        front: &PreparedSlopeFunction,
        waiting_rate: i32,
        lower: f64,
        upper: f64,
        budget: FrontierBudget,
    ) -> FrontierOutcome<(Vec<(f64, f64)>, u64)> {
        let started = Instant::now();
        let compute = || -> Result<(Vec<(f64, f64)>, u64), FrontierUnknown> {
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
            let pairs = front.pairs();
            if budget.max_components == 0 {
                return Err(FrontierUnknown::ComponentLimit);
            }
            if pairs.len() > budget.max_pairs {
                return Err(FrontierUnknown::PairLimit);
            }
            if !(2..=12).contains(&pairs.len()) || !lower.is_finite() || !upper.is_finite() {
                return Err(FrontierUnknown::Unsupported(
                    "machine reverse waiting inputs",
                ));
            }
            let needed = pairs
                .len()
                .saturating_mul(pairs.len())
                .saturating_mul(64)
                .saturating_add(64);
            if needed > budget.max_operations {
                return Err(FrontierUnknown::WorkLimit);
            }
            let rate = f64::from(waiting_rate);
            let mut k = pairs.len() - 1;
            let mut right = pairs[k].1;
            let mut value = front
                .machine_value_and_segment(right)
                .map_err(FrontierUnknown::Arithmetic)?
                .filter(|(value, _)| *value != 1e12)
                .ok_or(FrontierUnknown::Unsupported(
                    "machine reverse waiting terminal sentinel",
                ))?
                .0;
            let mut cap = upper;
            let mut reverse: Vec<(f64, f64)> = Vec::new();
            let mut reverse_bits = 0u64;
            let append = |reverse: &mut Vec<(f64, f64)>,
                          bits: &mut u64,
                          first: f64,
                          coordinate: f64,
                          waits: bool|
             -> Result<(), FrontierUnknown> {
                if reverse.len() >= 12 || reverse.len() >= budget.max_pairs {
                    return Err(FrontierUnknown::PairLimit);
                }
                let bit = 1u64 << reverse.len();
                if waits {
                    *bits |= bit;
                } else {
                    *bits &= !bit;
                }
                reverse.push((first, coordinate));
                Ok(())
            };
            loop {
                deadline()?;
                let slope = pairs[k].0;
                let previous = pairs[k - 1].1;
                if slope <= rate {
                    if previous < cap && !near(previous, cap)? {
                        let end = if right <= cap { right } else { cap };
                        append(&mut reverse, &mut reverse_bits, slope, end, false)?;
                        if previous < lower && !near(previous, lower)? {
                            let anchor = finite(value - finite(slope * finite(right - lower)?)?)?;
                            append(&mut reverse, &mut reverse_bits, anchor, lower, false)?;
                            break;
                        }
                    }
                    value = finite(value - finite(slope * finite(right - previous)?)?)?;
                    if k > 1 {
                        k -= 1;
                        right = pairs[k].1;
                        continue;
                    }
                    if lower < pairs[0].1 && !near(lower, pairs[0].1)? {
                        if pairs[0].1 <= cap {
                            cap = pairs[0].1;
                        }
                        append(&mut reverse, &mut reverse_bits, rate, cap, true)?;
                        value = finite(pairs[0].0 - finite(rate * finite(pairs[0].1 - lower)?)?)?;
                    }
                    append(&mut reverse, &mut reverse_bits, value, lower, false)?;
                    break;
                }
                let mut all_waiting = k == 1;
                let mut j = k.saturating_sub(2);
                let mut scanned = value;
                if !all_waiting {
                    scanned = finite(
                        finite(value - finite(slope * finite(right - previous)?)?)?
                            - finite(finite(previous - pairs[j].1)? * pairs[k - 1].0)?,
                    )?;
                    let mut line = finite(value - finite(finite(right - pairs[j].1)? * rate)?)?;
                    while line > scanned && !near(line, scanned)? {
                        deadline()?;
                        if j == 0 {
                            all_waiting = true;
                            break;
                        }
                        j -= 1;
                        scanned = finite(
                            scanned
                                - finite(finite(pairs[j + 1].1 - pairs[j].1)? * pairs[j + 1].0)?,
                        )?;
                        line = finite(value - finite(finite(right - pairs[j].1)? * rate)?)?;
                    }
                }
                if all_waiting {
                    if right <= cap {
                        cap = right;
                    }
                    append(&mut reverse, &mut reverse_bits, rate, cap, true)?;
                    let anchor = finite(value - finite(rate * finite(right - lower)?)?)?;
                    append(&mut reverse, &mut reverse_bits, anchor, lower, false)?;
                    break;
                }
                if pairs[j].1 >= cap || near(pairs[j].1, cap)? {
                    if j == 0 {
                        break;
                    }
                    // The skipped scan retains the value register while
                    // changing its copied coordinate and source index.
                    k = j;
                    right = pairs[j].1;
                    continue;
                }
                let next = j + 1;
                let next_slope = pairs[next].0;
                let crossing = finite(
                    finite(
                        finite(finite(scanned - value)? - finite(pairs[next].1 * next_slope)?)?
                            + finite(rate * right)?,
                    )? / finite(rate - next_slope)?,
                )?;
                if crossing < cap && !near(crossing, cap)? {
                    let end = if right <= cap { right } else { cap };
                    append(&mut reverse, &mut reverse_bits, rate, end, true)?;
                }
                if crossing <= lower || near(crossing, lower)? {
                    let anchor = finite(value - finite(rate * finite(right - lower)?)?)?;
                    append(&mut reverse, &mut reverse_bits, anchor, lower, false)?;
                    break;
                }
                let end = if crossing <= cap { crossing } else { cap };
                append(&mut reverse, &mut reverse_bits, next_slope, end, false)?;
                // This branch compares the original pair's first scalar,
                // rather than replacing it with a geometric knot coordinate.
                let scalar = pairs[j].0;
                if scalar < lower && !near(scalar, lower)? {
                    let anchor =
                        finite(scanned + finite(finite(lower - pairs[k - 1].1)? * pairs[k].0)?)?;
                    append(&mut reverse, &mut reverse_bits, anchor, lower, false)?;
                    break;
                }
                if next == 1 && lower < pairs[0].1 && !near(lower, pairs[0].1)? {
                    if pairs[0].1 <= cap {
                        cap = pairs[0].1;
                    }
                    append(&mut reverse, &mut reverse_bits, rate, cap, true)?;
                    let anchor = finite(pairs[0].0 - finite(rate * finite(pairs[0].1 - lower)?)?)?;
                    append(&mut reverse, &mut reverse_bits, anchor, lower, false)?;
                    break;
                }
                k -= 1;
                right = pairs[k].1;
            }
            if reverse.is_empty() {
                return Err(FrontierUnknown::Unsupported(
                    "machine reverse waiting invalid empty count",
                ));
            }
            let mut bits = 0;
            for index in 0..reverse.len() {
                bits |= ((reverse_bits >> index) & 1) << (reverse.len() - 1 - index);
            }
            reverse.reverse();
            deadline()?;
            Ok((reverse, bits))
        };
        match compute() {
            Ok(value) => FrontierOutcome::Complete(value),
            Err(reason) => FrontierOutcome::Unknown(reason),
        }
    }

    /// Reverse waiting over the full input domain, using the same scan as
    /// the explicitly capped/prefixed machine transition.
    pub fn machine_backward_wait_full_window(
        front: &PreparedSlopeFunction,
        waiting_rate: i32,
        budget: FrontierBudget,
    ) -> FrontierOutcome<(PreparedSlopeFunction, u64)> {
        let (lower, upper) = front.domain();
        Self::machine_backward_wait_window(front, waiting_rate, lower, upper, budget)
    }

    /// Run the recovered full-window waiting case and the subsequent
    /// composition, difference and key stages with one shared deadline.
    pub fn machine_backward_wait_finish(
        upper_front: &PreparedSlopeFunction,
        waiting_rate: i32,
        mapping: &PreparedSlopeFunction,
        increment: &PreparedSlopeFunction,
        lower: f64,
        upper: f64,
        terminal_ceiling: f64,
        mode: i32,
        budget: FrontierBudget,
    ) -> FrontierOutcome<(PreparedSlopeFunction, u64, f64)> {
        let started = Instant::now();
        let wait_work = budget.max_operations / 4;
        let (envelope, bits) = match Self::machine_backward_wait_full_window(
            upper_front,
            waiting_rate,
            FrontierBudget {
                max_operations: wait_work,
                ..budget
            },
        ) {
            FrontierOutcome::Complete(value) => value,
            FrontierOutcome::Empty => return FrontierOutcome::Empty,
            FrontierOutcome::Unknown(reason) => return FrontierOutcome::Unknown(reason),
        };
        let remaining_time = budget
            .max_elapsed
            .map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining_time.is_some_and(|limit| limit.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        Self::machine_backward_finish(
            &envelope,
            bits,
            mapping,
            increment,
            lower,
            upper,
            terminal_ceiling,
            mode,
            FrontierBudget {
                max_operations: budget.max_operations - wait_work,
                max_elapsed: remaining_time,
                ..budget
            },
        )
    }

    /// Compose an entire increasing mapping segment across a multi-segment
    /// outer frontier with identical endpoint coordinates. Waiting bits are
    /// copied per outer segment and participate in slope merging.
    pub fn machine_backward_compose_outer_stream(
        outer: &PreparedSlopeFunction,
        waiting_bits: u64,
        inner: &PreparedSlopeFunction,
        lower: f64,
        upper: f64,
        budget: FrontierBudget,
    ) -> FrontierOutcome<(PreparedSlopeFunction, u64)> {
        Self::machine_backward_compose_streams(outer, waiting_bits, inner, lower, upper, budget)
    }

    /// Compose a single-segment outer waiting envelope with a bounded
    /// increasing inner stream contained within its outer domain.
    /// The machine computes secants in stream order and merges near slopes.
    pub fn machine_backward_compose_stream(
        outer: &PreparedSlopeFunction,
        waiting_bits: u64,
        inner: &PreparedSlopeFunction,
        lower: f64,
        upper: f64,
        budget: FrontierBudget,
    ) -> FrontierOutcome<(PreparedSlopeFunction, u64)> {
        Self::machine_backward_compose_streams(outer, waiting_bits, inner, lower, upper, budget)
    }

    /// Complete the recovered composition, difference, and key stages from
    /// an already constructed backward waiting envelope. Unsupported stages
    /// remain unknown; the result does not certify a source route or bound.
    pub fn machine_backward_finish(
        envelope: &PreparedSlopeFunction,
        waiting_bits: u64,
        mapping: &PreparedSlopeFunction,
        increment: &PreparedSlopeFunction,
        lower: f64,
        upper: f64,
        terminal_ceiling: f64,
        mode: i32,
        budget: FrontierBudget,
    ) -> FrontierOutcome<(PreparedSlopeFunction, u64, f64)> {
        Self::machine_backward_finish_pairs(
            envelope.pairs(),
            waiting_bits,
            mapping,
            increment,
            lower,
            upper,
            terminal_ceiling,
            mode,
            budget,
        )
    }

    fn machine_backward_finish_pairs(
        envelope: &[(f64, f64)],
        waiting_bits: u64,
        mapping: &PreparedSlopeFunction,
        increment: &PreparedSlopeFunction,
        lower: f64,
        upper: f64,
        terminal_ceiling: f64,
        mode: i32,
        budget: FrontierBudget,
    ) -> FrontierOutcome<(PreparedSlopeFunction, u64, f64)> {
        let started = Instant::now();
        let stage_work = budget.max_operations / 3;
        let stage_budget = |work| -> Option<FrontierBudget> {
            let remaining = budget
                .max_elapsed
                .map(|limit| limit.saturating_sub(started.elapsed()));
            if remaining.is_some_and(|limit| limit.is_zero()) {
                return None;
            }
            Some(FrontierBudget {
                max_operations: work,
                max_elapsed: remaining,
                ..budget
            })
        };
        let Some(first_budget) = stage_budget(stage_work) else {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        };
        let composition = Self::machine_backward_compose_pairs(
            envelope,
            waiting_bits,
            mapping,
            lower,
            upper,
            first_budget,
        );
        let (composed, bits) = match composition {
            FrontierOutcome::Complete(value) => value,
            FrontierOutcome::Empty => return FrontierOutcome::Empty,
            FrontierOutcome::Unknown(reason) => return FrontierOutcome::Unknown(reason),
        };
        let Some(second_budget) = stage_budget(stage_work) else {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        };
        let (difference, bits) =
            match Self::machine_backward_difference(&composed, bits, increment, second_budget) {
                FrontierOutcome::Complete(value) => value,
                FrontierOutcome::Empty => return FrontierOutcome::Empty,
                FrontierOutcome::Unknown(reason) => return FrontierOutcome::Unknown(reason),
            };
        let Some(key_budget) = stage_budget(budget.max_operations - 2 * stage_work) else {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        };
        let key = match SlopeFront::from_prepared(difference.clone()).machine_backward_key(
            mode,
            terminal_ceiling,
            key_budget,
        ) {
            FrontierOutcome::Complete(value) => value,
            FrontierOutcome::Empty => return FrontierOutcome::Empty,
            FrontierOutcome::Unknown(reason) => return FrontierOutcome::Unknown(reason),
        };
        FrontierOutcome::Complete((difference, bits, key))
    }

    /// Waiting-aware composition for finite single-segment inputs whose
    /// requested interval and mapped values are contained in both domains.
    /// The secant and waiting bit follow the machine operation; this
    /// intermediate result does not certify a source route.
    pub fn machine_backward_compose_single_segment(
        outer: &PreparedSlopeFunction,
        waiting_bits: u64,
        inner: &PreparedSlopeFunction,
        lower: f64,
        upper: f64,
        budget: FrontierBudget,
    ) -> FrontierOutcome<(PreparedSlopeFunction, u64)> {
        Self::machine_backward_compose_streams(outer, waiting_bits, inner, lower, upper, budget)
    }

    /// Subtract overlapping machine vectors while carrying the
    /// first vector's segment waiting bits. Exceptional geometries remain
    /// unknown.
    /// The result is an intermediate machine vector, not a route certificate.
    pub fn machine_backward_difference(
        first: &PreparedSlopeFunction,
        waiting_bits: u64,
        second: &PreparedSlopeFunction,
        budget: FrontierBudget,
    ) -> FrontierOutcome<(PreparedSlopeFunction, u64)> {
        let started = Instant::now();
        let compute = || -> Result<(PreparedSlopeFunction, u64), FrontierUnknown> {
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
            if budget.max_components == 0 {
                return Err(FrontierUnknown::ComponentLimit);
            }
            let a = first.pairs();
            let b = second.pairs();
            if a.len() > 12 {
                return Err(FrontierUnknown::Unsupported(
                    "machine difference fixed array size",
                ));
            }
            if a.len() > budget.max_pairs || b.len() > budget.max_pairs {
                return Err(FrontierUnknown::PairLimit);
            }
            if budget.max_operations < a.len().saturating_add(b.len()).saturating_mul(32) {
                return Err(FrontierUnknown::WorkLimit);
            }
            if (a[0].1 > b.last().unwrap().1 && !near(a[0].1, b.last().unwrap().1)?)
                || (b[0].1 > a.last().unwrap().1 && !near(b[0].1, a.last().unwrap().1)?)
            {
                return Err(FrontierUnknown::Unsupported(
                    "machine difference disjoint domains",
                ));
            }
            let (mut i, mut j) = (1usize, 1usize);
            let (coordinate, anchor) = if near(a[0].1, b[0].1)? {
                (a[0].1, finite(a[0].0 - b[0].0)?)
            } else if b[0].1 <= a[0].1 {
                let (value, segment) = second
                    .machine_vector_value_and_segment(a[0].1)
                    .map_err(FrontierUnknown::Arithmetic)?
                    .ok_or(FrontierUnknown::Unsupported(
                        "machine difference vector start lookup",
                    ))?;
                j = segment;
                (a[0].1, finite(a[0].0 - value)?)
            } else {
                let (value, segment) = first
                    .machine_value_and_segment(b[0].1)
                    .map_err(FrontierUnknown::Arithmetic)?
                    .ok_or(FrontierUnknown::Unsupported(
                        "machine difference fixed start lookup",
                    ))?;
                i = segment;
                (b[0].1, finite(value - b[0].0)?)
            };
            let mut output: Vec<(f64, f64)> = vec![(anchor, coordinate)];
            let mut result_bits = 0u64;
            while i < a.len() && j < b.len() {
                deadline()?;
                // The native helper writes the candidate and its bit before
                // deciding whether to merge. Its next-write capacity check
                // therefore also applies to candidates that would merge.
                let index = output.len();
                if index >= 12 || index >= budget.max_pairs {
                    return Err(FrontierUnknown::PairLimit);
                }
                let bit = (waiting_bits >> i) & 1;
                result_bits = (result_bits & !(1u64 << index)) | (bit << index);
                let slope = finite(a[i].0 - b[j].0)?;
                let end = if near(a[i].1, b[j].1)? {
                    let end = a[i].1;
                    i += 1;
                    j += 1;
                    end
                } else if b[j].1 <= a[i].1 {
                    let end = b[j].1;
                    j += 1;
                    end
                } else {
                    let end = a[i].1;
                    i += 1;
                    end
                };
                let previous = output.len() - 1;
                if previous > 0
                    && (result_bits >> previous) & 1 == bit
                    && near(output[previous].0, slope)?
                {
                    output[previous].1 = end;
                    // Keep the scratch candidate's bit as the helper does;
                    // bits outside the emitted segment count are not silently
                    // normalized by this machine-only operation.
                } else {
                    output.push((slope, end));
                }
            }
            let result = PreparedSlopeFunction::from_machine_pairs(output)
                .map_err(FrontierUnknown::Arithmetic)?;
            deadline()?;
            Ok((result, result_bits))
        };
        match compute() {
            Ok(value) => FrontierOutcome::Complete(value),
            Err(reason) => FrontierOutcome::Unknown(reason),
        }
    }

    /// Run the bounded backward preflight, lower filter and upper minimum.
    /// The result is an intermediate machine vector, not a route certificate.
    pub fn machine_backward_initial_bounds(
        front: &SlopeFront,
        waiting_lower: f64,
        waiting_upper: f64,
        forward_lower: f64,
        mapped_upper: f64,
        lower: &PreparedSlopeFunction,
        upper: &PreparedSlopeFunction,
        budget: FrontierBudget,
    ) -> FrontierOutcome<(PreparedSlopeFunction, f64)> {
        let started = Instant::now();
        let first_work = budget.max_operations / 2;
        let (filtered, limit) = match Self::machine_backward_initial_filter(
            front,
            waiting_lower,
            waiting_upper,
            forward_lower,
            mapped_upper,
            lower,
            FrontierBudget {
                max_operations: first_work,
                ..budget
            },
        ) {
            FrontierOutcome::Complete(value) => value,
            FrontierOutcome::Empty => return FrontierOutcome::Empty,
            FrontierOutcome::Unknown(reason) => return FrontierOutcome::Unknown(reason),
        };
        let remaining_time = budget
            .max_elapsed
            .map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining_time.is_some_and(|limit| limit.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        let minimum_budget = FrontierBudget {
            max_operations: budget.max_operations - first_work,
            max_elapsed: remaining_time,
            ..budget
        };
        let minimum =
            Self::machine_minimum_streams(&filtered.components()[0], upper, minimum_budget);
        match minimum {
            FrontierOutcome::Complete(vector) => FrontierOutcome::Complete((vector, limit)),
            FrontierOutcome::Empty => FrontierOutcome::Empty,
            FrontierOutcome::Unknown(reason) => FrontierOutcome::Unknown(reason),
        }
    }

    /// Merge the ordered machine streams, carrying the selected operand and
    /// near-tie state. Losing-operand endpoints advance without manufacturing
    /// output pieces. This operation does not certify source-model feasibility.
    pub fn machine_minimum_streams(
        fixed: &PreparedSlopeFunction,
        vector: &PreparedSlopeFunction,
        budget: FrontierBudget,
    ) -> FrontierOutcome<PreparedSlopeFunction> {
        let started = Instant::now();
        let a = fixed.pairs();
        let b = vector.pairs();
        if budget.max_components == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        if a.len() > budget.max_pairs || b.len() > budget.max_pairs {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        if a.is_empty() || b.is_empty() || a.len() > 12 {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine minimum stream geometry",
            ));
        }
        if budget.max_operations < a.len().saturating_add(b.len()).saturating_mul(64) {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        }
        let compute = || -> Result<PreparedSlopeFunction, FrontierUnknown> {
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
            if (a[0].1 < b[0].1 && !near(a[0].1, b[0].1)?)
                || (a.last().unwrap().1 > b.last().unwrap().1
                    && !near(a.last().unwrap().1, b.last().unwrap().1)?)
            {
                return Err(FrontierUnknown::Unsupported(
                    "machine minimum stream domain",
                ));
            }
            let (mut i, mut j) = (1usize, 1usize);
            let (mut av, mut bv) = (a[0].0, b[0].0);
            let (mut ax, mut bx) = (a[0].1, b[0].1);
            let mut previous_order = 0i8;
            let b_start = if near(ax, bx)? {
                bv
            } else {
                // This lookup snaps to a near vector knot before returning
                // its value, unlike the ordinary point-evaluation helper.
                while j < b.len() && b[j].1 < ax {
                    deadline()?;
                    bv = finite(bv + finite(finite(b[j].1 - bx)? * b[j].0)?)?;
                    bx = b[j].1;
                    j += 1;
                }
                let &(slope, end) = b.get(j).ok_or(FrontierUnknown::Unsupported(
                    "machine minimum stream start lookup",
                ))?;
                if near(end, ax)? {
                    bv = finite(bv + finite(finite(end - bx)? * slope)?)?;
                    bx = end;
                    j += 1;
                    bv
                } else {
                    previous_order = -1;
                    finite(bv + finite(finite(ax - bx)? * slope)?)?
                }
            };
            let mut choose_a = av <= b_start;
            let mut previous_tie = near(av, b_start)?;
            let mut output = vec![(if choose_a { av } else { b_start }, ax)];
            let append = |output: &mut Vec<(f64, f64)>,
                          slope: f64,
                          end: f64|
             -> Result<(), FrontierUnknown> {
                if output.len() > 1 && near(output.last().unwrap().0, slope)? {
                    // Both slope and endpoint are overwritten on a merge.
                    *output.last_mut().unwrap() = (slope, end);
                } else {
                    if output.len() >= 12 || output.len() >= budget.max_pairs {
                        return Err(FrontierUnknown::PairLimit);
                    }
                    output.push((slope, end));
                }
                Ok(())
            };
            while i < a.len() && j < b.len() {
                deadline()?;
                let (ae, be) = (a[i].1, b[j].1);
                let aend = finite(av + finite(finite(ae - ax)? * a[i].0)?)?;
                let bend = finite(bv + finite(finite(be - bx)? * b[j].0)?)?;
                let order = if near(ae, be)? {
                    0
                } else if be <= ae {
                    1
                } else {
                    -1
                };
                let (x, ay, by) = match order {
                    1 => (be, finite(av + finite(finite(be - ax)? * a[i].0)?)?, bend),
                    -1 => (ae, aend, finite(bv + finite(finite(ae - bx)? * b[j].0)?)?),
                    _ => (ae, aend, bend),
                };
                let tie = near(ay, by)?;
                let next_a = if tie { choose_a } else { ay <= by };
                if choose_a != next_a {
                    // A tie at a jointly consumed previous endpoint changes
                    // selection without an artificial zero-width crossing.
                    if !(previous_tie && previous_order == 0) {
                        let ai = finite(aend - finite(ae * a[i].0)?)?;
                        let bi = finite(bend - finite(be * b[j].0)?)?;
                        let denominator = finite(a[i].0 - b[j].0)?;
                        if denominator == 0. {
                            return Err(FrontierUnknown::Unsupported(
                                "machine minimum stream parallel crossing",
                            ));
                        }
                        let crossing = finite(finite(bi - ai)? / denominator)?;
                        append(
                            &mut output,
                            if choose_a { a[i].0 } else { b[j].0 },
                            crossing,
                        )?;
                    }
                    append(&mut output, if next_a { a[i].0 } else { b[j].0 }, x)?;
                } else if (choose_a && order != 1) || (!choose_a && order != -1) {
                    append(&mut output, if choose_a { a[i].0 } else { b[j].0 }, x)?;
                }
                choose_a = next_a;
                previous_tie = tie;
                previous_order = order;
                if order != 1 {
                    av = aend;
                    ax = ae;
                    i += 1;
                }
                if order != -1 {
                    bv = bend;
                    bx = be;
                    j += 1;
                }
            }
            if output.len() == 1 {
                if budget.max_pairs < 2 {
                    return Err(FrontierUnknown::PairLimit);
                }
                output.push((0., output[0].1));
            }
            deadline()?;
            PreparedSlopeFunction::from_machine_pairs(output).map_err(FrontierUnknown::Arithmetic)
        };
        match compute() {
            Ok(value) => FrontierOutcome::Complete(value),
            Err(reason) => FrontierOutcome::Unknown(reason),
        }
    }

    /// Apply the recovered aligned segment minimum to a sequence of ordinary
    /// segments, retaining each input's ordered endpoint accumulation.
    pub fn machine_minimum_aligned_streams(
        fixed: &PreparedSlopeFunction,
        vector: &PreparedSlopeFunction,
        budget: FrontierBudget,
    ) -> FrontierOutcome<PreparedSlopeFunction> {
        let started = Instant::now();
        let a = fixed.pairs();
        let b = vector.pairs();
        if budget.max_components == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        if a.len() < 2
            || a.len() != b.len()
            || a.len() > 12
            || fixed.has_internal_zero_length()
            || vector.has_internal_zero_length()
            || fixed.is_machine_zero_length()
            || vector.is_machine_zero_length()
            || !a.iter().zip(b).all(|(x, y)| x.1.to_bits() == y.1.to_bits())
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine aligned minimum geometry",
            ));
        }
        if a.len() > budget.max_pairs {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        let required = a.len().saturating_mul(64);
        if budget.max_operations < required {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        }
        let compute = || -> Result<PreparedSlopeFunction, FrontierUnknown> {
            let mut output = Vec::with_capacity(a.len());
            let mut a_value = a[0].0;
            let mut b_value = b[0].0;
            for index in 1..a.len() {
                let start = a[index - 1].1;
                let end = a[index].1;
                if end <= start || near(start, end)? {
                    return Err(FrontierUnknown::Unsupported(
                        "machine aligned minimum short segment",
                    ));
                }
                let first =
                    PreparedSlopeFunction::from_prepared_pairs(vec![(a_value, start), a[index]])
                        .map_err(FrontierUnknown::Arithmetic)?;
                let second =
                    PreparedSlopeFunction::from_prepared_pairs(vec![(b_value, start), b[index]])
                        .map_err(FrontierUnknown::Arithmetic)?;
                let remaining = budget
                    .max_elapsed
                    .map(|limit| limit.saturating_sub(started.elapsed()));
                if remaining.is_some_and(|limit| limit.is_zero()) {
                    return Err(FrontierUnknown::Deadline);
                }
                let part = match Self::machine_minimum_separated_segments(
                    &first,
                    &second,
                    FrontierBudget {
                        max_operations: 64,
                        max_elapsed: remaining,
                        ..budget
                    },
                ) {
                    FrontierOutcome::Complete(value) => value,
                    FrontierOutcome::Empty => {
                        return Err(FrontierUnknown::Unsupported(
                            "machine aligned minimum empty segment",
                        ))
                    }
                    FrontierOutcome::Unknown(reason) => return Err(reason),
                };
                if output.is_empty() {
                    output.push(part.pairs()[0]);
                }
                for &segment in &part.pairs()[1..] {
                    if output.len() > 1 && near(output.last().unwrap().0, segment.0)? {
                        *output.last_mut().unwrap() = segment;
                        continue;
                    }
                    if output.len() >= 12 || output.len() >= budget.max_pairs {
                        return Err(FrontierUnknown::PairLimit);
                    }
                    output.push(segment);
                }
                let span = finite(end - start)?;
                a_value = finite(a_value + finite(span * a[index].0)?)?;
                b_value = finite(b_value + finite(span * b[index].0)?)?;
            }
            PreparedSlopeFunction::from_machine_pairs(output).map_err(FrontierUnknown::Arithmetic)
        };
        match compute() {
            Ok(result) => FrontierOutcome::Complete(result),
            Err(reason) => FrontierOutcome::Unknown(reason),
        }
    }

    /// Run the recovered backward preflight and constant lower-bound case.
    /// The returned front has not passed the later upper, waiting, composition,
    /// or increment stages and cannot certify a route.
    pub fn machine_backward_initial_filter(
        front: &SlopeFront,
        waiting_lower: f64,
        waiting_upper: f64,
        forward_lower: f64,
        mapped_upper: f64,
        lower: &PreparedSlopeFunction,
        budget: FrontierBudget,
    ) -> FrontierOutcome<(SlopeFront, f64)> {
        let started = Instant::now();
        let first_work = budget.max_operations / 2;
        let first_budget = FrontierBudget {
            max_operations: first_work,
            ..budget
        };
        let (front, upper) = match Self::machine_backward_cover_preflight(
            front,
            waiting_lower,
            waiting_upper,
            forward_lower,
            mapped_upper,
            first_budget,
        ) {
            FrontierOutcome::Complete(value) => value,
            FrontierOutcome::Empty => return FrontierOutcome::Empty,
            FrontierOutcome::Unknown(reason) => return FrontierOutcome::Unknown(reason),
        };
        let remaining_time = budget
            .max_elapsed
            .map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining_time.is_some_and(|limit| limit.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        match Self::machine_backward_constant_lower_pass(
            &front,
            lower,
            FrontierBudget {
                max_operations: budget.max_operations - first_work,
                max_elapsed: remaining_time,
                ..budget
            },
        ) {
            FrontierOutcome::Complete(front) => FrontierOutcome::Complete((front, upper)),
            FrontierOutcome::Empty => FrontierOutcome::Empty,
            FrontierOutcome::Unknown(reason) => FrontierOutcome::Unknown(reason),
        }
    }

    /// Minimum of an ordinary reduced front and a covering constant or
    /// aligned piecewise-linear vector when one stays strictly below the other.
    pub fn machine_minimum_constant_multisegment(
        fixed: &PreparedSlopeFunction,
        vector: &PreparedSlopeFunction,
        budget: FrontierBudget,
    ) -> FrontierOutcome<PreparedSlopeFunction> {
        let started = Instant::now();
        let f = fixed.pairs();
        let v = vector.pairs();
        if budget.max_components == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        if f.len() > budget.max_pairs || v.len() > budget.max_pairs {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        if budget.max_operations < f.len().saturating_mul(12) {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        }
        let constant = v.len() == 2 && v[1].0.to_bits() == 0.0f64.to_bits();
        let aligned =
            v.len() == f.len() && f.iter().zip(v).all(|(a, b)| a.1.to_bits() == b.1.to_bits());
        if f.len() < 2
            || f.len() > 12
            || v.len() < 2
            || fixed.has_internal_zero_length()
            || fixed.is_machine_zero_length()
            || vector.has_internal_zero_length()
            || vector.is_machine_zero_length()
            || (!constant && !aligned)
            || f[0].1 < v[0].1
            || f.last().unwrap().1 > v.last().unwrap().1
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine multi-segment minimum geometry",
            ));
        }
        if aligned
            && f.iter()
                .zip(v)
                .all(|(a, b)| a.0.to_bits() == b.0.to_bits() && a.1.to_bits() == b.1.to_bits())
        {
            let mut value = f[0].0;
            for index in 1..f.len() {
                if f[index].1 <= f[index - 1].1
                    || matches!(near(f[index].1, f[index - 1].1), Ok(true) | Err(_))
                    || (index > 1 && matches!(near(f[index].0, f[index - 1].0), Ok(true) | Err(_)))
                {
                    return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                        "machine multi-segment minimum equal geometry",
                    ));
                }
                let span = f[index].1 - f[index - 1].1;
                let product = f[index].0 * span;
                value += product;
                if !span.is_finite() || !product.is_finite() || !value.is_finite() {
                    return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                        "machine multi-segment minimum equal arithmetic",
                    ));
                }
            }
            if budget
                .max_elapsed
                .is_some_and(|limit| started.elapsed() >= limit)
            {
                return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
            }
            return FrontierOutcome::Complete(fixed.clone());
        }
        let calculate = || -> Result<(bool, f64), FrontierUnknown> {
            if (f[0].1.to_bits() != v[0].1.to_bits() && near(f[0].1, v[0].1)?)
                || (f.last().unwrap().1.to_bits() != v.last().unwrap().1.to_bits()
                    && near(f.last().unwrap().1, v.last().unwrap().1)?)
            {
                return Err(FrontierUnknown::Unsupported("machine minimum near domain"));
            }
            let mut bound = if f[0].1.to_bits() == v[0].1.to_bits() {
                v[0].0
            } else {
                finite(v[0].0 + finite(v[1].0 * finite(f[0].1 - v[0].1)?)?)?
            };
            let below = f[0].0 < bound;
            let initial_bound = bound;
            let mut value = f[0].0;
            let mut x = f[0].1;
            if near(value, bound)? {
                return Err(FrontierUnknown::Unsupported(
                    "machine multi-segment minimum near value",
                ));
            }
            for (index, &(slope, end)) in f.iter().enumerate().skip(1) {
                if end <= x || near(x, end)? || (index > 1 && near(f[index - 1].0, slope)?) {
                    return Err(FrontierUnknown::Unsupported(
                        "machine multi-segment minimum reduction",
                    ));
                }
                value = finite(value + finite(slope * finite(end - x)?)?)?;
                if !constant {
                    if index > 1 && near(v[index - 1].0, v[index].0)? {
                        return Err(FrontierUnknown::Unsupported(
                            "machine multi-segment minimum vector reduction",
                        ));
                    }
                    bound = finite(bound + finite(v[index].0 * finite(end - x)?)?)?;
                }
                if near(value, bound)? || (value < bound) != below {
                    return Err(FrontierUnknown::Unsupported(
                        "machine multi-segment minimum crossing",
                    ));
                }
                x = end;
            }
            Ok((below, initial_bound))
        };
        let (below, bound) = match calculate() {
            Ok(value) => value,
            Err(reason) => return FrontierOutcome::Unknown(reason),
        };
        if budget
            .max_elapsed
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        if below {
            FrontierOutcome::Complete(fixed.clone())
        } else if !constant {
            FrontierOutcome::Complete(vector.clone())
        } else {
            match PreparedSlopeFunction::from_prepared_pairs(vec![
                (bound, f[0].1),
                (v[1].0, f.last().unwrap().1),
            ]) {
                Ok(value) => FrontierOutcome::Complete(value),
                Err(error) => FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
            }
        }
    }

    /// Apply the machine lower-bound scan to one fixed front and a covering
    /// vector. Statuses without a valid retained machine front stay unknown.
    pub fn machine_backward_constant_lower_pass(
        front: &SlopeFront,
        lower: &PreparedSlopeFunction,
        budget: FrontierBudget,
    ) -> FrontierOutcome<SlopeFront> {
        Self::machine_backward_lower_streams(front, lower, budget)
    }

    /// Walk the original operand segments in order, including near starts,
    /// near endpoint comparisons and the lower filter's re-entry status.
    /// Machine tolerance is not a source-domain feasibility certificate.
    pub fn machine_backward_lower_streams(
        front: &SlopeFront,
        lower: &PreparedSlopeFunction,
        budget: FrontierBudget,
    ) -> FrontierOutcome<SlopeFront> {
        let started = Instant::now();
        let Some(fixed) = front
            .components()
            .first()
            .filter(|_| front.components().len() == 1)
        else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine lower stream components",
            ));
        };
        let a = fixed.pairs();
        let b = lower.pairs();
        if budget.max_components == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        if a.len() > budget.max_pairs || b.len() > budget.max_pairs {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        if a.is_empty() || b.is_empty() || a.len() > 12 {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine lower stream geometry",
            ));
        }
        if budget.max_operations < a.len().saturating_add(b.len()).saturating_mul(64) {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        }
        let compute = || -> Result<SlopeFront, FrontierUnknown> {
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
            if (a[0].1 < b[0].1 && !near(a[0].1, b[0].1)?)
                || (a.last().unwrap().1 > b.last().unwrap().1
                    && !near(a.last().unwrap().1, b.last().unwrap().1)?)
            {
                return Err(FrontierUnknown::Unsupported(
                    "machine lower stream domain status",
                ));
            }
            let (mut i, mut j) = (1usize, 1usize);
            let (mut av, mut bv) = (a[0].0, b[0].0);
            let (mut ax, mut bx) = (a[0].1, b[0].1);
            let b_start = if near(ax, bx)? {
                bv
            } else {
                while j < b.len() && b[j].1 < ax {
                    deadline()?;
                    bv = finite(bv + finite(finite(b[j].1 - bx)? * b[j].0)?)?;
                    bx = b[j].1;
                    j += 1;
                }
                let &(slope, end) = b.get(j).ok_or(FrontierUnknown::Unsupported(
                    "machine lower stream start lookup",
                ))?;
                if near(end, ax)? {
                    bv = finite(bv + finite(finite(end - bx)? * slope)?)?;
                    bx = end;
                    j += 1;
                    bv
                } else {
                    finite(bv + finite(finite(ax - bx)? * slope)?)?
                }
            };
            let mut active = b_start < av || near(av, b_start)?;
            let mut output = if active { vec![a[0]] } else { Vec::new() };
            let append =
                |output: &mut Vec<(f64, f64)>, pair: (f64, f64)| -> Result<(), FrontierUnknown> {
                    if output.len() >= 12 || output.len() >= budget.max_pairs {
                        return Err(FrontierUnknown::PairLimit);
                    }
                    output.push(pair);
                    Ok(())
                };
            while i < a.len() && j < b.len() {
                deadline()?;
                let (ae, be) = (a[i].1, b[j].1);
                let aend = finite(av + finite(finite(ae - ax)? * a[i].0)?)?;
                let bend = finite(bv + finite(finite(be - bx)? * b[j].0)?)?;
                let order = if near(ae, be)? {
                    0
                } else if be <= ae {
                    1
                } else {
                    -1
                };
                let (x, ay, by) = match order {
                    1 => (be, finite(av + finite(finite(be - ax)? * a[i].0)?)?, bend),
                    -1 => (ae, aend, finite(bv + finite(finite(ae - bx)? * b[j].0)?)?),
                    _ => (ae, aend, bend),
                };
                let tie = near(ay, by)?;
                if active {
                    if tie || by <= ay {
                        // The lower filter copies each common endpoint. It
                        // does not perform the minimum operator's slope merge.
                        append(&mut output, (a[i].0, x))?;
                    } else {
                        let ai = finite(aend - finite(ae * a[i].0)?)?;
                        let bi = finite(bend - finite(be * b[j].0)?)?;
                        let denominator = finite(a[i].0 - b[j].0)?;
                        if denominator == 0. {
                            return Err(FrontierUnknown::Unsupported(
                                "machine lower stream parallel crossing",
                            ));
                        }
                        let crossing = finite(finite(bi - ai)? / denominator)?;
                        if !near(crossing, output.last().unwrap().1)? {
                            append(&mut output, (a[i].0, crossing))?;
                        }
                        active = false;
                    }
                } else if tie {
                    if !output.is_empty() {
                        return Err(FrontierUnknown::Unsupported(
                            "machine lower stream reentry status",
                        ));
                    }
                    append(&mut output, (ay, x))?;
                    active = true;
                } else if by < ay {
                    if !output.is_empty() {
                        return Err(FrontierUnknown::Unsupported(
                            "machine lower stream reentry status",
                        ));
                    }
                    let ai = finite(aend - finite(ae * a[i].0)?)?;
                    let bi = finite(bend - finite(be * b[j].0)?)?;
                    let denominator = finite(a[i].0 - b[j].0)?;
                    if denominator == 0. {
                        return Err(FrontierUnknown::Unsupported(
                            "machine lower stream parallel crossing",
                        ));
                    }
                    let crossing = finite(finite(bi - ai)? / denominator)?;
                    let anchor = finite(finite(crossing * a[i].0)? + ai)?;
                    append(&mut output, (anchor, crossing))?;
                    // The entry branch stores the full fixed endpoint, even
                    // when this comparison exhausted only the vector segment.
                    append(&mut output, a[i])?;
                    active = true;
                }
                // Strictly rejected trailing segments emit nothing. A later
                // tie or return above the bound, rather than rejection alone,
                // is what produces the filter's unsupported re-entry status.
                if order != 1 {
                    av = aend;
                    ax = ae;
                    i += 1;
                }
                if order != -1 {
                    bv = bend;
                    bx = be;
                    j += 1;
                }
            }
            if output.is_empty() {
                return Err(FrontierUnknown::Unsupported(
                    "machine lower stream invalid empty count",
                ));
            }
            if output.len() == 1 {
                let coordinate = output[0].1;
                append(&mut output, (0., coordinate))?;
            }
            deadline()?;
            let result = PreparedSlopeFunction::from_machine_pairs(output)
                .map_err(FrontierUnknown::Arithmetic)?;
            Ok(SlopeFront::from_prepared(result))
        };
        match compute() {
            Ok(value) => FrontierOutcome::Complete(value),
            Err(reason) => FrontierOutcome::Unknown(reason),
        }
    }

    /// First backward recurrence stage, including recovered bounded
    /// truncation cases. This does not establish feasibility.
    pub fn machine_backward_cover_preflight(
        front: &SlopeFront,
        waiting_lower: f64,
        waiting_upper: f64,
        forward_lower: f64,
        mapped_upper: f64,
        budget: FrontierBudget,
    ) -> FrontierOutcome<(SlopeFront, f64)> {
        let started = Instant::now();
        if [waiting_lower, waiting_upper, forward_lower, mapped_upper]
            .iter()
            .any(|value| !value.is_finite())
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine backward limits",
            ));
        }
        let reduced = match front.machine_reduce_for_continuation(budget) {
            FrontierOutcome::Complete(front) => front,
            FrontierOutcome::Empty => {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine backward reduction empty",
                ))
            }
            FrontierOutcome::Unknown(reason) => return FrontierOutcome::Unknown(reason),
        };
        let components = reduced.components();
        if components.len() != 1 || components[0].pairs().len() < 2 {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine backward cover geometry",
            ));
        }
        let pairs = components[0].pairs();
        // MAXSD selects its second operand on ties, including signed zero.
        let lower = if forward_lower > waiting_lower {
            forward_lower
        } else {
            waiting_lower
        };
        if lower > pairs.last().unwrap().1 {
            match near(lower, pairs.last().unwrap().1) {
                Ok(false) => return FrontierOutcome::Empty,
                Ok(true) => {}
                Err(reason) => return FrontierOutcome::Unknown(reason),
            }
        }
        // Reduction and the entire fixed-array truncation share this budget.
        let original_pairs = front.components()[0].pairs().len();
        let needed = original_pairs
            .saturating_mul(8)
            .saturating_add(4)
            .saturating_add(pairs.len().saturating_mul(24).saturating_add(4));
        if budget.max_operations < needed {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        }
        let reduced = match components[0].machine_truncate_fixed(lower, waiting_upper) {
            Ok(value) => SlopeFront::from_prepared(value),
            Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
        };
        let last = reduced.components()[0].pairs().last().unwrap().1;
        let after_head = if last <= waiting_upper {
            last
        } else {
            waiting_upper
        };
        let after_mapped = if mapped_upper <= after_head {
            mapped_upper
        } else {
            after_head
        };
        if budget
            .max_elapsed
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        FrontierOutcome::Complete((reduced, after_mapped))
    }

    /// Mode-4 meeting coordinate for two ordinary, common-end one-segment
    /// machine fronts. This is not a source-domain feasibility certificate.
    /// Unrecovered and exceptional cases return unknown.
    pub fn machine_mode_four_single_segment_meeting(
        forward: &PreparedSlopeFunction,
        backward: &PreparedSlopeFunction,
        budget: FrontierBudget,
    ) -> FrontierOutcome<f64> {
        let started = Instant::now();
        if budget.max_components == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        if budget.max_pairs < 2 {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        if budget.max_operations < 24 {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        }
        let f = forward.pairs();
        let b = backward.pairs();
        if forward.is_machine_zero_length()
            || backward.is_machine_zero_length()
            || f.len() != 2
            || b.len() != 2
            || f[1].1.to_bits() != b[1].1.to_bits()
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine meeting geometry",
            ));
        }
        let calculate = || -> Result<f64, FrontierUnknown> {
            let start = if b[0].1 > f[0].1 { b[0].1 } else { f[0].1 };
            if start >= f[1].1 || near(start, f[1].1)? {
                return Err(FrontierUnknown::Unsupported(
                    "machine meeting short interval",
                ));
            }
            if f[0].1 != b[0].1 && near(f[0].1, b[0].1)? {
                return Err(FrontierUnknown::Unsupported("machine meeting near starts"));
            }
            let lookup = |function: &PreparedSlopeFunction| match function
                .evaluate(start, EvaluationPolicy::EndpointTolerance)
            {
                Ok(Some(value)) if value != 1e12 => Ok(value),
                Ok(_) => Err(FrontierUnknown::Unsupported("machine meeting sentinel")),
                Err(error) => Err(FrontierUnknown::Arithmetic(error)),
            };
            let f_start = lookup(forward)?;
            let b_start = lookup(backward)?;
            if f_start == 1e12 || b_start == 1e12 {
                return Err(FrontierUnknown::Unsupported("machine meeting sentinel"));
            }
            if near(f_start, b_start)? || f_start <= b_start {
                return Ok(start);
            }
            let span = finite(f[1].1 - start)?;
            let f_end = finite(f_start + finite(f[1].0 * span)?)?;
            let b_end = finite(b_start + finite(b[1].0 * span)?)?;
            if f_end > b_end && !near(f_end, b_end)? {
                return Err(FrontierUnknown::Unsupported(
                    "machine meeting requires later scan",
                ));
            }
            let denominator = finite(f[1].0 - b[1].0)?;
            if denominator == 0. {
                return Err(FrontierUnknown::Unsupported("machine meeting equal slopes"));
            }
            let numerator = finite(
                finite(finite(b_end - f_end)? + finite(f[1].1 * f[1].0)?)?
                    - finite(b[1].1 * b[1].0)?,
            )?;
            finite(numerator / denominator)
        };
        let result = match calculate() {
            Ok(result) => result,
            Err(reason) => return FrontierOutcome::Unknown(reason),
        };
        if budget
            .max_elapsed
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        FrontierOutcome::Complete(result)
    }

    /// Recover the bounded minimum of two aligned, separated constant
    /// machine fronts. This is not a source-domain dominance certificate.
    /// Input endpoints must be bit-identical, the slopes positive zero, and
    /// the constants outside the machine tolerance band. Endpoints must not
    /// themselves be near. This is not a source-control certificate.
    pub fn machine_minimum_separated_constants(
        fixed: &PreparedSlopeFunction,
        vector: &PreparedSlopeFunction,
        budget: FrontierBudget,
    ) -> FrontierOutcome<PreparedSlopeFunction> {
        let started = Instant::now();
        if budget.max_components == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        if budget.max_pairs < 2 {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        if budget.max_operations < 12 {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        }
        if fixed.is_machine_zero_length()
            || vector.is_machine_zero_length()
            || fixed.pairs().len() != 2
            || vector.pairs().len() != 2
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine minimum constant geometry",
            ));
        }
        let a = fixed.pairs();
        let b = vector.pairs();
        if a[0].1.to_bits() != b[0].1.to_bits()
            || a[1].0.to_bits() != 0.0f64.to_bits()
            || b[1].0.to_bits() != 0.0f64.to_bits()
            || a[1].1.to_bits() != b[1].1.to_bits()
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine minimum constant bounds",
            ));
        }
        match near(a[0].1, a[1].1) {
            Ok(false) => {}
            Ok(true) => {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine minimum short interval",
                ));
            }
            Err(reason) => return FrontierOutcome::Unknown(reason),
        }
        if a.iter().zip(b).all(|(left, right)| {
            left.0.to_bits() == right.0.to_bits() && left.1.to_bits() == right.1.to_bits()
        }) {
            if budget
                .max_elapsed
                .is_some_and(|limit| started.elapsed() >= limit)
            {
                return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
            }
            return FrontierOutcome::Complete(fixed.clone());
        }
        match near(a[0].0, b[0].0) {
            Ok(false) => {}
            Ok(true) => {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine minimum near constants",
                ));
            }
            Err(reason) => return FrontierOutcome::Unknown(reason),
        }
        if budget
            .max_elapsed
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        // The smaller anchor wins; the endpoint is copied from the fixed
        // input. Both recovered slopes are positive zero.
        let value = if a[0].0 < b[0].0 { a[0].0 } else { b[0].0 };
        let result = match PreparedSlopeFunction::from_prepared_pairs(vec![
            (value, a[0].1),
            (0.0, a[1].1),
        ]) {
            Ok(result) => result,
            Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
        };
        if budget
            .max_elapsed
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        FrontierOutcome::Complete(result)
    }

    /// Recover the first scan point of the machine backward-waiting helper.
    ///
    /// The caller supplies the finite anchor `(value, coordinate)` of an
    /// already computed minimum front, the initial forward lookup, target,
    /// and the reference's already converted integer waiting rate. This does
    /// not compute or validate that minimum, inspect later scan points, or
    /// certify a source route. Sentinel ambiguity and unmatched anchors are
    /// unknown rather than infeasibility.
    pub fn machine_backward_wait_first_point(
        anchor: (f64, f64),
        initial: f64,
        target: f64,
        waiting_rate: i32,
        budget: FrontierBudget,
    ) -> FrontierOutcome<f64> {
        let started = Instant::now();
        if budget.max_components == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        if budget.max_pairs == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        if budget.max_operations < 8 {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        }
        if budget
            .max_elapsed
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        let (value, coordinate) = anchor;
        if [value, coordinate, initial, target]
            .iter()
            .any(|v| !v.is_finite() || *v == 1e12)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine backward waiting first-point inputs",
            ));
        }
        let (position, adjusted) = if coordinate > initial {
            let adjusted = finite(coordinate - initial)
                .and_then(|span| finite(f64::from(waiting_rate) * span))
                .and_then(|product| finite(target + product));
            match adjusted {
                Ok(value) => (coordinate, value),
                Err(reason) => return FrontierOutcome::Unknown(reason),
            }
        } else {
            (initial, target)
        };
        match near(value, adjusted) {
            Ok(true) => {}
            Ok(false) => {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "unrecovered machine backward waiting later scan",
                ));
            }
            Err(reason) => return FrontierOutcome::Unknown(reason),
        }
        let waiting = match finite(position - initial) {
            Ok(value) => value,
            Err(reason) => return FrontierOutcome::Unknown(reason),
        };
        if budget
            .max_elapsed
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        FrontierOutcome::Complete(waiting)
    }

    /// Recover waiting from the next backward entry: clip its copied front,
    /// take the machine minimum, then scan its stored endpoints in order.
    /// Diagnostic/no-intersection branches are unknown, never infeasibility.
    pub fn machine_backward_waiting(
        next: &PreparedSlopeFunction,
        input: &MachineBackwardInputs<'_>,
        initial: f64,
        target: f64,
        mode: i32,
        budget: FrontierBudget,
    ) -> FrontierOutcome<f64> {
        let started = Instant::now();
        let count = next.pairs().len();
        if budget.max_components == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        if count > budget.max_pairs || input.dependent_upper.pairs().len() > budget.max_pairs {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        if count == 0
            || count > 12
            || !initial.is_finite()
            || !target.is_finite()
            || !input.waiting_lower.is_finite()
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine backward waiting inputs",
            ));
        }
        let reserved = count.saturating_mul(16).saturating_add(160);
        let Some(minimum_work) = budget.max_operations.checked_sub(reserved) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        let compute = || -> Result<f64, FrontierUnknown> {
            let deadline = || -> Result<Option<std::time::Duration>, FrontierUnknown> {
                let remaining = budget
                    .max_elapsed
                    .map(|limit| limit.saturating_sub(started.elapsed()));
                if remaining.is_some_and(|limit| limit.is_zero()) {
                    return Err(FrontierUnknown::Deadline);
                }
                Ok(remaining)
            };
            deadline()?;
            let pairs = next.pairs();
            if mode > 1
                && initial > pairs.last().unwrap().1
                && !near(initial, pairs.last().unwrap().1)?
            {
                return Err(FrontierUnknown::Unsupported(
                    "machine backward waiting initial diagnostic",
                ));
            }
            let mut clipped = None;
            let lower = input.waiting_lower;
            if lower > pairs[0].1 && !near(lower, pairs[0].1)? {
                let mut value = pairs[0].0;
                let mut previous = pairs[0].1;
                let mut first = 1;
                while let Some(&(slope, end)) = pairs.get(first) {
                    deadline()?;
                    if lower < end && !near(end, lower)? {
                        break;
                    }
                    value = finite(value + finite(slope * finite(end - previous)?)?)?;
                    previous = end;
                    first += 1;
                }
                if first == count {
                    if near(pairs.last().unwrap().1, lower)? {
                        clipped = Some(
                            PreparedSlopeFunction::machine_zero_length(value, 0., lower, lower)
                                .map_err(FrontierUnknown::Arithmetic)?,
                        );
                    }
                } else {
                    let anchor =
                        finite(value + finite(finite(lower - previous)? * pairs[first].0)?)?;
                    let mut output = Vec::with_capacity(count - first + 1);
                    output.push((anchor, lower));
                    output.extend_from_slice(&pairs[first..]);
                    clipped = Some(
                        PreparedSlopeFunction::from_machine_pairs(output)
                            .map_err(FrontierUnknown::Arithmetic)?,
                    );
                }
            }
            let minimum = match Self::machine_minimum_streams(
                clipped.as_ref().unwrap_or(next),
                input.dependent_upper,
                FrontierBudget {
                    max_operations: minimum_work,
                    max_elapsed: deadline()?,
                    ..budget
                },
            ) {
                FrontierOutcome::Complete(value) => value,
                FrontierOutcome::Empty => {
                    return Err(FrontierUnknown::Unsupported(
                        "machine backward waiting minimum empty",
                    ))
                }
                FrontierOutcome::Unknown(reason) => return Err(reason),
            };
            let mut position = initial;
            let mut adjusted = target;
            let mut value = minimum.pairs()[0].0;
            let mut previous = minimum.pairs()[0].1;
            for (index, &(slope, coordinate)) in minimum.pairs().iter().enumerate() {
                deadline()?;
                if position < coordinate {
                    let span = finite(coordinate - position)?;
                    adjusted = finite(adjusted + finite(f64::from(input.waiting_rate) * span)?)?;
                    position = coordinate;
                }
                if index != 0 {
                    value = finite(value + finite(finite(coordinate - previous)? * slope)?)?;
                }
                if near(value, adjusted)? {
                    return finite(position - initial);
                }
                previous = coordinate;
            }
            Err(FrontierUnknown::Unsupported(
                "machine backward waiting no intersection",
            ))
        };
        match compute() {
            Ok(value) => FrontierOutcome::Complete(value),
            Err(reason) => FrontierOutcome::Unknown(reason),
        }
    }

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
        let started = Instant::now();
        if !initial_base.is_finite()
            || !initial_dependent.is_finite()
            || history.len().checked_sub(1) != Some(steps.len())
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
        let Some(remaining_work) = budget.max_operations.checked_sub(work) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        let per_wait = remaining_work / steps.len().max(1);
        let compute = || -> Result<MachinePathReconstruction, FrontierUnknown> {
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
                        .evaluate(coordinate, EvaluationPolicy::EndpointTolerance)
                        .map_err(FrontierUnknown::Arithmetic)?
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
                    match Self::machine_backward_waiting(
                        next,
                        input,
                        arrival,
                        target,
                        mode,
                        FrontierBudget {
                            max_operations: per_wait,
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
                dependent = finite(
                    finite(dependent + increment)?
                        + finite(waiting * f64::from(input.waiting_rate))?,
                )?;
                result.waiting.push(waiting);
                result.base.push(base);
                result.dependent.push(dependent);
            }
            // Check that the reconstructed coordinate can be read at the
            // retained terminal entry. This is still a machine-domain check.
            snapshots[0].front().components()[0]
                .machine_value_and_segment(base)
                .map_err(FrontierUnknown::Arithmetic)?
                .filter(|(value, _)| *value != 1e12)
                .ok_or(FrontierUnknown::Unsupported(
                    "machine suffix terminal lookup sentinel",
                ))?;
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
    /// This models the bounded recomputation path; it does not certify a
    /// pricing-label history, a source-control route, or feasibility.
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
            let (reduced, next) = run_stage!(transition.machine_transition(
                history.last().front(),
                inputs,
                mode,
                next_budget!(),
            ));
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

    pub fn new(
        source: &SlopeRelationArc,
        policy: FrontOperationPolicy,
    ) -> Result<Self, RawFunctionError> {
        if [
            &source.arrival,
            &source.increment,
            &source.lower,
            &source.upper,
        ]
        .iter()
        .any(|function| function.is_machine_zero_length())
        {
            return Err(RawFunctionError::Unsupported(
                "machine zero-length segment as paired source",
            ));
        }
        if source.arrival.pairs().iter().skip(1).any(|p| p.0 <= 0.) {
            return Err(RawFunctionError::Unsupported(
                "arrival must be strictly increasing",
            ));
        }
        let mut previous = source.arrival.pairs()[0].1;
        let mut mapped = source.arrival.pairs()[0].0;
        if mapped < previous {
            return Err(RawFunctionError::Unsupported("arrival precedes departure"));
        }
        for &(slope, x) in &source.arrival.pairs()[1..] {
            mapped += slope * (x - previous);
            if !mapped.is_finite() {
                return Err(RawFunctionError::ArithmeticOverflow("arrival validation"));
            }
            if mapped < x {
                return Err(RawFunctionError::Unsupported("arrival precedes departure"));
            }
            previous = x;
        }
        let shape = (|| {
            let constant = |f: &PreparedSlopeFunction| {
                f.pairs()
                    .iter()
                    .skip(1)
                    .all(|p| p.0 == 0.)
                    .then_some(f.pairs()[0].0)
            };
            if source.arrival.pairs().len() > 2
                || !source.arrival.pairs().iter().skip(1).all(|p| p.0 == 1.)
                || source.arrival.pairs()[0].1 != 0.
            {
                return Err("general arrival composition");
            }
            let delta = source.arrival.pairs()[0].0;
            if delta < 0. {
                return Err("non-FIFO arrival translation");
            }
            let phi = constant(&source.increment).ok_or("nonconstant paired increment")?;
            let (al, ah) = source.arrival.domain();
            let (pl, ph) = source.increment.domain();
            let (ll, lh) = source.lower.domain();
            let (ul, uh) = source.upper.domain();
            Ok(Translation {
                delta,
                phi,
                tail_lo: al.max(pl),
                tail_hi: ah.min(ph),
                head_lo: ll.max(ul),
                head_hi: lh.min(uh),
            })
        })();
        let inverse = match inverse_increasing(&source.arrival, FrontierBudget::default()) {
            FrontierOutcome::Complete(f) => FrontierOutcome::Complete(Arc::new(f)),
            FrontierOutcome::Empty => FrontierOutcome::Empty,
            FrontierOutcome::Unknown(e) => FrontierOutcome::Unknown(e),
        };
        let composition = match &inverse {
            FrontierOutcome::Complete(f) => {
                PreparedCompositionMap::new(f, FrontierBudget::default())
            }
            FrontierOutcome::Empty => FrontierOutcome::Empty,
            FrontierOutcome::Unknown(e) => FrontierOutcome::Unknown(e.clone()),
        };
        // Native preparation constructs Phi = phi composed with q inverse.
        // Runtime composes F separately, then adds this cached Phi.
        let increment_front = match (&composition, &inverse) {
            (FrontierOutcome::Complete(map), FrontierOutcome::Complete(inverse)) => {
                let (lo, hi) = inverse.domain();
                match map.compose_clipped(
                    &source.increment,
                    lo,
                    hi,
                    policy,
                    FrontierBudget::default(),
                ) {
                    FrontierOutcome::Complete(f) => {
                        FrontierOutcome::Complete(SlopeFront::from_prepared(f))
                    }
                    FrontierOutcome::Empty => FrontierOutcome::Empty,
                    FrontierOutcome::Unknown(e) => FrontierOutcome::Unknown(e),
                }
            }
            (FrontierOutcome::Unknown(e), _) => FrontierOutcome::Unknown(e.clone()),
            (_, FrontierOutcome::Unknown(e)) => FrontierOutcome::Unknown(e.clone()),
            _ => FrontierOutcome::Empty,
        };
        Ok(Self {
            source: Arc::new(source.clone()),
            policy,
            shape,
            inverse,
            composition,
            increment_front,
            bounds: PreparedClampFilter::new(&source.lower, &source.upper)?,
        })
    }

    /// Ordered slope addition and monotone secant composition followed by a
    /// running-min waiting envelope, lower clamping and upper filtering.
    /// Unresolved numerical crossings return Unknown, never infeasibility.
    pub fn extend(
        &self,
        front: &SlopeFront,
        budget: FrontierBudget,
    ) -> FrontierOutcome<SlopeFront> {
        match self.extend_checked(front, budget, None) {
            Ok(Some(out)) => FrontierOutcome::Complete(out),
            Ok(None) => FrontierOutcome::Empty,
            Err(reason) => FrontierOutcome::Unknown(reason),
        }
    }

    /// Extend with the explicit machine waiting bounds supplied by the
    /// reference's per-arc accumulated-base record. These bounds are separate
    /// from the domains of the dependent bound functions. Only recovered
    /// zero-length waiting, maximum and upper-filter branches use them;
    /// unsupported geometry remains unknown and no source certificate follows.
    pub fn extend_with_machine_waiting_bounds(
        &self,
        front: &SlopeFront,
        lower: f64,
        upper: f64,
        budget: FrontierBudget,
    ) -> FrontierOutcome<SlopeFront> {
        if self.policy != FrontOperationPolicy::FrontAlgebra {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "explicit machine waiting bounds require front algebra",
            ));
        }
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine waiting needs finite ordered bounds",
            ));
        }
        match self.extend_checked(front, budget, Some((lower, upper))) {
            Ok(Some(out)) => FrontierOutcome::Complete(out),
            Ok(None) => FrontierOutcome::Empty,
            Err(reason) => FrontierOutcome::Unknown(reason),
        }
    }

    /// Apply the recovered finite preflight before machine-front composition.
    /// `tail_limit` and `head_limit` are respectively the first and third
    /// scalar arguments of the machine recurrence; `forward_mapping` is its
    /// prepared forward lookup vector, not the raw input or inverse. A
    /// complete result contains the reduced input and the
    /// ordered composition endpoints. `Empty` reports rejection by this
    /// machine preflight only; it is not a source-control infeasibility proof.
    /// Sentinel lookups and unsupported arithmetic remain unknown. This does
    /// not perform composition, extension, or produce a route witness.
    pub fn machine_continuation_preflight(
        &self,
        front: &SlopeFront,
        tail_limit: f64,
        head_limit: f64,
        forward_mapping: &PreparedSlopeFunction,
        budget: FrontierBudget,
    ) -> FrontierOutcome<(SlopeFront, f64, f64)> {
        if self.policy != FrontOperationPolicy::FrontAlgebra
            || !tail_limit.is_finite()
            || !head_limit.is_finite()
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine continuation preflight inputs",
            ));
        }
        let started = Instant::now();
        let reduced = match front.machine_reduce_for_continuation(budget) {
            FrontierOutcome::Complete(front) => front,
            FrontierOutcome::Empty => return FrontierOutcome::Empty,
            FrontierOutcome::Unknown(reason) => return FrontierOutcome::Unknown(reason),
        };
        // Reserve the complete reducer scan plus both possible vector lookups
        // before doing either lookup, under the same elapsed deadline.
        let reduction_work = front.components().first().map_or(0, |function| {
            function.pairs().len().saturating_mul(8).saturating_add(4)
        });
        let Some(required) = forward_mapping
            .pairs()
            .len()
            .checked_mul(2)
            .and_then(|n| n.checked_add(6))
            .and_then(|n| n.checked_add(reduction_work))
        else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        if budget.max_operations < required {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        }
        if budget.max_pairs < forward_mapping.pairs().len() {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        if budget
            .max_elapsed
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        let (first, last) = reduced.components()[0].domain();
        let first_near = match near(first, tail_limit) {
            Ok(value) => value,
            Err(reason) => return FrontierOutcome::Unknown(reason),
        };
        if first > tail_limit && !first_near {
            return FrontierOutcome::Empty;
        }
        let lookup = |coordinate| match forward_mapping.machine_value_and_segment(coordinate) {
            Ok(Some((value, _))) if value != 1e12 => Ok(value),
            Ok(_) => Err(FrontierUnknown::Unsupported(
                "machine continuation lookup sentinel",
            )),
            Err(error) => Err(FrontierUnknown::Arithmetic(error)),
        };
        let lower = match lookup(first) {
            Ok(value) => value,
            Err(reason) => return FrontierOutcome::Unknown(reason),
        };
        let lower_near = match near(lower, head_limit) {
            Ok(value) => value,
            Err(reason) => return FrontierOutcome::Unknown(reason),
        };
        if lower > head_limit && !lower_near {
            return FrontierOutcome::Empty;
        }
        let last_near = match near(tail_limit, last) {
            Ok(value) => value,
            Err(reason) => return FrontierOutcome::Unknown(reason),
        };
        let upper = if tail_limit > last && !last_near {
            match lookup(last) {
                Ok(value) => value,
                Err(reason) => return FrontierOutcome::Unknown(reason),
            }
        } else {
            head_limit
        };
        if budget
            .max_elapsed
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        FrontierOutcome::Complete((reduced, lower, upper))
    }

    /// Compose a preflight-reduced machine front with the separately prepared
    /// inverse map. The returned reduced input retains its provenance; the
    /// composed function is an intermediate machine value, not a source
    /// witness or a completed paired extension. Call this only with the
    /// result of `machine_continuation_preflight` for the same transition.
    /// Unsupported composition geometry remains unknown.
    pub fn machine_continuation_compose(
        &self,
        preflight: (SlopeFront, f64, f64),
        inverse: &PreparedSlopeFunction,
        budget: FrontierBudget,
    ) -> FrontierOutcome<(SlopeFront, PreparedSlopeFunction)> {
        let (reduced, lower, upper) = preflight;
        if self.policy != FrontOperationPolicy::FrontAlgebra
            || reduced.components().len() != 1
            || !matches!(
                reduced.provenance().as_ref(),
                FrontierProvenance::MachineReduction { .. }
            )
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine continuation composition inputs",
            ));
        }
        match compose_increasing(
            &reduced.components()[0],
            inverse,
            lower,
            upper,
            FrontOperationPolicy::FrontAlgebra,
            budget,
        ) {
            FrontierOutcome::Complete(output) => FrontierOutcome::Complete((reduced, output)),
            // A missing intersection is not yet a proved machine rejection
            // for every tolerance branch, nor a source infeasibility proof.
            FrontierOutcome::Empty => FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "unrecovered machine continuation composition failure",
            )),
            FrontierOutcome::Unknown(reason) => FrontierOutcome::Unknown(reason),
        }
    }

    /// Ordered fixed-array/vector sum. Near starts retain the fixed anchor;
    /// near segment ends consume both operands and retain the fixed endpoint.
    /// Merging near slopes extends the preceding segment without replacing
    /// its slope. This is machine arithmetic, not a source certificate.
    pub fn machine_sum_streams(
        fixed: &PreparedSlopeFunction,
        vector: &PreparedSlopeFunction,
        budget: FrontierBudget,
    ) -> FrontierOutcome<PreparedSlopeFunction> {
        let started = Instant::now();
        let compute = || -> Result<PreparedSlopeFunction, FrontierUnknown> {
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
            if budget.max_components == 0 {
                return Err(FrontierUnknown::ComponentLimit);
            }
            let a = fixed.pairs();
            let b = vector.pairs();
            if a.len() > 12 {
                return Err(FrontierUnknown::Unsupported("machine sum fixed array size"));
            }
            if a.len() > budget.max_pairs || b.len() > budget.max_pairs {
                return Err(FrontierUnknown::PairLimit);
            }
            if budget.max_operations < a.len().saturating_add(b.len()).saturating_mul(32) {
                return Err(FrontierUnknown::WorkLimit);
            }
            if (a[0].1 > b.last().unwrap().1 && !near(a[0].1, b.last().unwrap().1)?)
                || (b[0].1 > a.last().unwrap().1 && !near(b[0].1, a.last().unwrap().1)?)
            {
                return Err(FrontierUnknown::Unsupported("machine sum disjoint domains"));
            }
            let (mut i, mut j) = (1usize, 1usize);
            let (coordinate, anchor) = if near(a[0].1, b[0].1)? {
                (a[0].1, finite(a[0].0 + b[0].0)?)
            } else if b[0].1 <= a[0].1 {
                let (value, segment) = vector
                    .machine_vector_value_and_segment(a[0].1)
                    .map_err(FrontierUnknown::Arithmetic)?
                    .ok_or(FrontierUnknown::Unsupported(
                        "machine sum vector start lookup",
                    ))?;
                j = segment;
                (a[0].1, finite(a[0].0 + value)?)
            } else {
                let (value, segment) = fixed
                    .machine_value_and_segment(b[0].1)
                    .map_err(FrontierUnknown::Arithmetic)?
                    .ok_or(FrontierUnknown::Unsupported(
                        "machine sum fixed start lookup",
                    ))?;
                i = segment;
                (b[0].1, finite(value + b[0].0)?)
            };
            let mut output: Vec<(f64, f64)> = vec![(anchor, coordinate)];
            while i < a.len() && j < b.len() {
                deadline()?;
                // The fixed-array helper checks its next write position
                // before deciding whether that candidate would merge.
                if output.len() >= 12 || output.len() >= budget.max_pairs {
                    return Err(FrontierUnknown::PairLimit);
                }
                let slope = finite(a[i].0 + b[j].0)?;
                let end = if near(a[i].1, b[j].1)? {
                    let end = a[i].1;
                    i += 1;
                    j += 1;
                    end
                } else if b[j].1 <= a[i].1 {
                    let end = b[j].1;
                    j += 1;
                    end
                } else {
                    let end = a[i].1;
                    i += 1;
                    end
                };
                if output.len() > 1 && near(slope, output.last().unwrap().0)? {
                    output.last_mut().unwrap().1 = end;
                } else {
                    output.push((slope, end));
                }
            }
            if output.len() == 1 {
                if budget.max_pairs < 2 {
                    return Err(FrontierUnknown::PairLimit);
                }
                output.push((0., coordinate));
            }
            let result = PreparedSlopeFunction::from_machine_pairs(output)
                .map_err(FrontierUnknown::Arithmetic)?;
            deadline()?;
            Ok(result)
        };
        match compute() {
            Ok(value) => FrontierOutcome::Complete(value),
            Err(reason) => FrontierOutcome::Unknown(reason),
        }
    }

    /// Apply the recovered sum with the separately prepared vector operand,
    /// followed by the continuation caller's fixed-array validity check.
    pub fn machine_continuation_sum(
        &self,
        composition: (SlopeFront, PreparedSlopeFunction),
        vector: &PreparedSlopeFunction,
        budget: FrontierBudget,
    ) -> FrontierOutcome<(SlopeFront, PreparedSlopeFunction)> {
        let started = Instant::now();
        let (reduced, fixed) = composition;
        if self.policy != FrontOperationPolicy::FrontAlgebra
            || !matches!(
                reduced.provenance().as_ref(),
                FrontierProvenance::MachineReduction { .. }
            )
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine continuation sum inputs",
            ));
        }
        let Some(remaining_work) = budget.max_operations.checked_sub(12 * 6) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        let result = Self::machine_sum_streams(
            &fixed,
            vector,
            FrontierBudget {
                max_operations: remaining_work,
                ..budget
            },
        );
        let output = match result {
            FrontierOutcome::Complete(value) => value,
            FrontierOutcome::Empty => {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine continuation sum empty",
                ))
            }
            FrontierOutcome::Unknown(reason) => return FrontierOutcome::Unknown(reason),
        };
        let pairs = output.pairs();
        if !(2..=12).contains(&pairs.len()) || pairs[0].0 == 1e12 {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine continuation sum failed output check",
            ));
        }
        // The first segment may be a retained duplicate. Later coordinates
        // must increase strictly and must be outside the near-equality band.
        for index in 2..pairs.len() {
            match near(pairs[index].1, pairs[index - 1].1) {
                Ok(false) if pairs[index].1 > pairs[index - 1].1 => {}
                Ok(_) => {
                    return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                        "machine continuation sum consecutive coordinates",
                    ))
                }
                Err(reason) => return FrontierOutcome::Unknown(reason),
            }
        }
        if budget
            .max_elapsed
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        FrontierOutcome::Complete((reduced, output))
    }

    /// Execute the bounded recovered machine continuation with distinct
    /// caller-supplied operands. The returned reduced input retains machine
    /// provenance; the envelope retains waiting provenance. Neither is a
    /// source-control certificate or a reconstructed route. Unsupported and
    /// machine-only rejection cases return unknown.
    ///
    /// Six stages share the operation allowance conservatively and use the
    /// remaining elapsed time from one start instant. A stage may exhaust its
    /// fraction even when a different partition could have succeeded.
    pub fn machine_continue(
        &self,
        front: &SlopeFront,
        inputs: &MachineContinuationInputs<'_>,
        budget: FrontierBudget,
    ) -> FrontierOutcome<(SlopeFront, MachineWaitingEnvelope)> {
        let started = Instant::now();
        let per_stage = budget.max_operations / 6;
        if per_stage == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        }
        let stage_budget = || -> Result<FrontierBudget, FrontierUnknown> {
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
        macro_rules! stage {
            ($call:expr) => {
                match $call {
                    FrontierOutcome::Complete(value) => value,
                    FrontierOutcome::Empty => {
                        return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                            "machine continuation stage rejected",
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
        let preflight = stage!(self.machine_continuation_preflight(
            front,
            inputs.tail_limit,
            inputs.head_limit,
            inputs.forward_mapping,
            next_budget!()
        ));
        let composed =
            stage!(self.machine_continuation_compose(preflight, inputs.inverse, next_budget!()));
        let (reduced, sum) =
            stage!(self.machine_continuation_sum(composed, inputs.sum_vector, next_budget!()));
        let sum_front = SlopeFront::from_prepared(sum);
        let waiting = stage!(sum_front.machine_waiting(
            inputs.waiting_lower,
            inputs.waiting_upper,
            inputs.waiting_rate,
            next_budget!()
        ));
        let lower = stage!(waiting.apply_lower_bound(inputs.dependent_lower, next_budget!()));
        let upper = stage!(lower.apply_upper_bound(inputs.dependent_upper, next_budget!()));
        if budget
            .max_elapsed
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        FrontierOutcome::Complete((reduced, upper))
    }

    /// Execute a bounded machine transition and retain the reduced prior
    /// front alongside the resulting state. The raw state key and path
    /// provenance are not a source-control feasibility or pruning proof.
    /// The supplied inputs must describe this same physical arc.
    pub fn machine_transition(
        &self,
        front: &SlopeFront,
        inputs: &MachineContinuationInputs<'_>,
        mode: i32,
        budget: FrontierBudget,
    ) -> FrontierOutcome<(SlopeFront, MachineForwardState)> {
        if !matches!(
            front.provenance().as_ref(),
            FrontierProvenance::MachineRoot { .. }
                | FrontierProvenance::PairedMachineWaiting { .. }
        ) {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine transition requires a machine root or forward state",
            ));
        }
        let started = Instant::now();
        let stage_work = budget.max_operations / 7;
        if stage_work == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        }
        let continuation_budget = FrontierBudget {
            max_operations: stage_work * 6,
            ..budget
        };
        let (reduced, envelope) = match self.machine_continue(front, inputs, continuation_budget) {
            FrontierOutcome::Complete(value) => value,
            FrontierOutcome::Empty => {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine continuation rejected",
                ));
            }
            FrontierOutcome::Unknown(reason) => return FrontierOutcome::Unknown(reason),
        };
        let remaining = budget
            .max_elapsed
            .map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining.is_some_and(|limit| limit.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        let state_budget = FrontierBudget {
            max_operations: stage_work,
            max_elapsed: remaining,
            ..budget
        };
        match MachineForwardState::from_filtered_waiting(
            &reduced,
            &envelope,
            self.source.clone(),
            mode,
            state_budget,
        ) {
            FrontierOutcome::Complete(state) => {
                if budget
                    .max_elapsed
                    .is_some_and(|limit| started.elapsed() >= limit)
                {
                    return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
                }
                FrontierOutcome::Complete((reduced, state))
            }
            FrontierOutcome::Empty => FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine forward state unavailable",
            )),
            FrontierOutcome::Unknown(reason) => FrontierOutcome::Unknown(reason),
        }
    }

    /// Recover the machine minimum for aligned one-segment fronts with
    /// separated endpoint values, including an interior crossing. This
    /// intermediate operation is not a source-domain proof.
    pub fn machine_minimum_separated_segments(
        fixed: &PreparedSlopeFunction,
        vector: &PreparedSlopeFunction,
        budget: FrontierBudget,
    ) -> FrontierOutcome<PreparedSlopeFunction> {
        let started = Instant::now();
        if budget.max_components == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        if budget.max_pairs < 2 {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        if budget.max_operations < 16 {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        }
        if fixed.is_machine_zero_length()
            || vector.is_machine_zero_length()
            || fixed.pairs().len() != 2
            || vector.pairs().len() != 2
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine minimum segment geometry",
            ));
        }
        let a = fixed.pairs();
        let b = vector.pairs();
        if a[0].1.to_bits() != b[0].1.to_bits()
            || a[1].1.to_bits() != b[1].1.to_bits()
            || a[0].1 >= a[1].1
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine minimum segment alignment",
            ));
        }
        match near(a[0].1, a[1].1) {
            Ok(false) => {}
            Ok(true) => {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine minimum short interval",
                ));
            }
            Err(reason) => return FrontierOutcome::Unknown(reason),
        }
        if a.iter().zip(b).all(|(left, right)| {
            left.0.to_bits() == right.0.to_bits() && left.1.to_bits() == right.1.to_bits()
        }) {
            if budget
                .max_elapsed
                .is_some_and(|limit| started.elapsed() >= limit)
            {
                return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
            }
            return FrontierOutcome::Complete(fixed.clone());
        }
        match near(a[0].0, b[0].0) {
            Ok(false) => {}
            Ok(true) => {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine minimum near anchors",
                ))
            }
            Err(reason) => return FrontierOutcome::Unknown(reason),
        }
        let endpoints = (|| {
            let span_a = finite(a[1].1 - a[0].1)?;
            let span_b = finite(b[1].1 - b[0].1)?;
            let end_a = finite(a[0].0 + finite(span_a * a[1].0)?)?;
            let end_b = finite(b[0].0 + finite(span_b * b[1].0)?)?;
            Ok::<_, FrontierUnknown>((end_a, end_b))
        })();
        let (end_a, end_b) = match endpoints {
            Ok(values) => values,
            Err(reason) => return FrontierOutcome::Unknown(reason),
        };
        match near(end_a, end_b) {
            Ok(false) => {}
            Ok(true) => {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine minimum near endpoints",
                ))
            }
            Err(reason) => return FrontierOutcome::Unknown(reason),
        }
        let selected = if a[0].0 < b[0].0 && end_a < end_b {
            a
        } else if b[0].0 < a[0].0 && end_b < end_a {
            b
        } else {
            if budget.max_pairs < 3 {
                return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
            }
            let crossing = (|| -> Result<f64, FrontierUnknown> {
                // Preserve the reference's endpoint/intercept ordering.
                let b_intercept = finite(end_b - finite(b[1].1 * b[1].0)?)?;
                let a_intercept = finite(end_a - finite(a[1].1 * a[1].0)?)?;
                let numerator = finite(b_intercept - a_intercept)?;
                let denominator = finite(a[1].0 - b[1].0)?;
                if denominator == 0. {
                    return Err(FrontierUnknown::Unsupported(
                        "machine minimum parallel crossing",
                    ));
                }
                let coordinate = finite(numerator / denominator)?;
                if coordinate <= a[0].1
                    || coordinate >= a[1].1
                    || near(coordinate, a[0].1)?
                    || near(coordinate, a[1].1)?
                {
                    return Err(FrontierUnknown::Unsupported(
                        "machine minimum boundary crossing",
                    ));
                }
                Ok(coordinate)
            })();
            let coordinate = match crossing {
                Ok(value) => value,
                Err(reason) => return FrontierOutcome::Unknown(reason),
            };
            let (first, second) = if a[0].0 < b[0].0 { (a, b) } else { (b, a) };
            let result = PreparedSlopeFunction::from_prepared_pairs(vec![
                (first[0].0, a[0].1),
                (first[1].0, coordinate),
                (second[1].0, a[1].1),
            ]);
            if budget
                .max_elapsed
                .is_some_and(|limit| started.elapsed() >= limit)
            {
                return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
            }
            return match result {
                Ok(result) => FrontierOutcome::Complete(result),
                Err(error) => FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
            };
        };
        if budget
            .max_elapsed
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        match PreparedSlopeFunction::from_prepared_pairs(vec![
            (selected[0].0, a[0].1),
            (selected[1].0, a[1].1),
        ]) {
            Ok(result) => {
                if budget
                    .max_elapsed
                    .is_some_and(|limit| started.elapsed() >= limit)
                {
                    FrontierOutcome::Unknown(FrontierUnknown::Deadline)
                } else {
                    FrontierOutcome::Complete(result)
                }
            }
            Err(error) => FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
        }
    }

    fn extend_checked(
        &self,
        front: &SlopeFront,
        budget: FrontierBudget,
        machine_waiting_bounds: Option<(f64, f64)>,
    ) -> Result<Option<SlopeFront>, FrontierUnknown> {
        if front.has_machine_zero_length() {
            return Err(FrontierUnknown::Unsupported(
                "machine zero-length segment paired extension",
            ));
        }
        if front.components().len() > budget.max_components {
            return Err(FrontierUnknown::ComponentLimit);
        }
        let mut pairs = 0usize;
        for f in front.components() {
            pairs = pairs
                .checked_add(f.pairs().len())
                .ok_or(FrontierUnknown::PairLimit)?;
            if pairs > budget.max_pairs {
                return Err(FrontierUnknown::PairLimit);
            }
            if pairs > budget.max_operations / 2 {
                return Err(FrontierUnknown::WorkLimit);
            }
        }
        // Covers the preflight and exact-transport eligibility scan before
        // partitioning the remaining allowance between actual operators.
        let budget = FrontierBudget {
            max_operations: budget.max_operations - 2 * pairs,
            ..budget
        };
        match self.shape.as_ref() {
            Ok(t) if exact_translation(front, t) => self.extend_waiting(front, front, t, budget),
            _ => self.extend_composed(front, budget, machine_waiting_bounds),
        }
    }
    fn extend_waiting(
        &self,
        front: &SlopeFront,
        origin: &SlopeFront,
        t: &Translation,
        budget: FrontierBudget,
    ) -> Result<Option<SlopeFront>, FrontierUnknown> {
        let start = budget.max_elapsed.map(|_| Instant::now());
        let unsupported = |s| FrontierUnknown::Unsupported(s);
        let a = &self.source;
        // Independent bounded passes share one aggregate operation allowance:
        // input scan, envelope, geometry assembly, validation and bound filter.
        let budget = FrontierBudget {
            max_operations: budget.max_operations / 5,
            ..budget
        };
        let (delta, phi) = (t.delta, t.phi);
        let (tail_lo, tail_hi, head_lo, head_hi) = (t.tail_lo, t.tail_hi, t.head_lo, t.head_hi);
        if tail_lo > tail_hi || head_lo > head_hi {
            return Ok(None);
        }
        let mut input = Vec::new();
        let mut work = 0usize;
        for f in front.components() {
            work = work.checked_add(1).ok_or(FrontierUnknown::WorkLimit)?;
            if work > budget.max_operations || input.len() >= budget.max_pairs {
                return Err(FrontierUnknown::WorkLimit);
            }
            if start
                .zip(budget.max_elapsed)
                .is_some_and(|(s, d)| s.elapsed() > d)
            {
                return Err(FrontierUnknown::Deadline);
            }
            let mut x = f.pairs()[0].1;
            // Constant Phi is added once to the component anchor. Re-adding it
            // independently at each source knot changes ordered front algebra.
            let mut y = finite(f.pairs()[0].0 + phi)?;
            if f.pairs().len() == 1 && x >= tail_lo && x <= tail_hi {
                input.push(Segment {
                    lo: finite(x + delta)?,
                    hi: finite(x + delta)?,
                    y,
                    slope: 0.,
                });
            }
            for &(s, e) in &f.pairs()[1..] {
                work = work.checked_add(1).ok_or(FrontierUnknown::WorkLimit)?;
                if work > budget.max_operations || work > budget.max_pairs {
                    return Err(FrontierUnknown::WorkLimit);
                }
                let lo = x.max(tail_lo);
                let hi = e.min(tail_hi);
                if lo <= hi {
                    if self.policy == FrontOperationPolicy::FrontAlgebra
                        && ((lo != x && near(lo, x)?) || (hi != e && near(hi, e)?))
                    {
                        return Err(unsupported("unresolved native tail restriction alignment"));
                    }
                    if lo < hi && finite(lo + delta)? >= finite(hi + delta)? {
                        return Err(unsupported("collapsed arrival coordinates"));
                    }
                    input.push(Segment {
                        lo: finite(lo + delta)?,
                        hi: finite(hi + delta)?,
                        y: finite(y + finite(s * finite(lo - x)?)?)?,
                        slope: s,
                    });
                }
                y = finite(y + finite(s * finite(e - x)?)?)?;
                x = e;
            }
        }
        if input.is_empty() {
            return Ok(None);
        }
        // Running record at the current coordinate. A waiting ray bridges gaps.
        let k = f64::from(a.waiting);
        let mut output = Vec::new();
        let mut end = input[0].lo;
        let mut record = input[0].y;
        for (index, seg) in input.into_iter().enumerate() {
            if seg.lo > end {
                push(
                    &mut output,
                    Segment {
                        lo: end,
                        hi: seg.lo,
                        y: record,
                        slope: k,
                    },
                    budget,
                )?;
                record = value(record, k, seg.lo - end)?;
            }
            if seg.y < record {
                if index > 0 {
                    return Err(unsupported("downward discontinuity in waiting envelope"));
                }
                record = seg.y;
            }
            let difference = finite(seg.y - record)?;
            if seg.slope < k && difference > 0. {
                let crossing = finite(seg.lo + finite(difference / finite(k - seg.slope)?)?)?;
                if crossing < seg.hi {
                    // A rounded crossing must not silently create a false feasible
                    // interval. Unresolved branch equality is an explicit unknown.
                    if value(seg.y, seg.slope, crossing - seg.lo)?
                        != value(record, k, crossing - seg.lo)?
                    {
                        return Err(unsupported("rounded waiting crossing"));
                    }
                    push(
                        &mut output,
                        Segment {
                            lo: seg.lo,
                            hi: crossing,
                            y: record,
                            slope: k,
                        },
                        budget,
                    )?;
                    let cy = value(seg.y, seg.slope, crossing - seg.lo)?;
                    push(
                        &mut output,
                        Segment {
                            lo: crossing,
                            hi: seg.hi,
                            y: cy,
                            slope: seg.slope,
                        },
                        budget,
                    )?;
                    record = value(cy, seg.slope, seg.hi - crossing)?;
                } else {
                    push(
                        &mut output,
                        Segment {
                            lo: seg.lo,
                            hi: seg.hi,
                            y: record,
                            slope: k,
                        },
                        budget,
                    )?;
                    record = value(record, k, seg.hi - seg.lo)?;
                }
            } else {
                let slope = if difference == 0. {
                    seg.slope.min(k)
                } else {
                    k
                };
                push(
                    &mut output,
                    Segment {
                        lo: seg.lo,
                        hi: seg.hi,
                        y: record,
                        slope,
                    },
                    budget,
                )?;
                record = value(record, slope, seg.hi - seg.lo)?;
            }
            end = seg.hi;
        }
        if end < head_hi {
            push(
                &mut output,
                Segment {
                    lo: end,
                    hi: head_hi,
                    y: record,
                    slope: k,
                },
                budget,
            )?;
        }
        let mut functions = Vec::new();
        for s in output {
            work = work.checked_add(1).ok_or(FrontierUnknown::WorkLimit)?;
            if work > budget.max_operations {
                return Err(FrontierUnknown::WorkLimit);
            }
            if functions.len().saturating_add(1) > budget.max_pairs {
                return Err(FrontierUnknown::PairLimit);
            }
            if start
                .zip(budget.max_elapsed)
                .is_some_and(|(s, d)| s.elapsed() > d)
            {
                return Err(FrontierUnknown::Deadline);
            }
            let lo = s.lo.max(head_lo);
            let hi = s.hi.min(head_hi);
            if lo > hi {
                continue;
            }
            if self.policy == FrontOperationPolicy::FrontAlgebra
                && ((lo != s.lo && near(lo, s.lo)?) || (hi != s.hi && near(hi, s.hi)?))
            {
                return Err(unsupported(
                    "unresolved native waiting restriction alignment",
                ));
            }
            let y = value(s.y, s.slope, lo - s.lo)?;
            let function = if lo == hi {
                PreparedSlopeFunction::constant_point(y, lo)
            } else {
                PreparedSlopeFunction::from_prepared_pairs(vec![(y, lo), (s.slope, hi)])
            }
            .map_err(FrontierUnknown::Arithmetic)?;
            functions.push(function);
        }
        // Adjacent pieces are one component, retaining literal segment slopes.
        let mut components: Vec<PreparedSlopeFunction> = Vec::new();
        let mut current: Vec<(f64, f64)> = Vec::new();
        let mut current_y = 0.;
        for f in functions {
            if let Some(last) = current.last() {
                if f.domain().0 < last.1 {
                    if f.pairs().len() == 1 {
                        continue;
                    }
                    return Err(unsupported("overlapping waiting output"));
                }
                if last.1 == f.domain().0 {
                    if current_y != f.pairs()[0].0 {
                        return Err(unsupported("rounded waiting boundary"));
                    }
                    current_y = f
                        .evaluate(f.domain().1, EvaluationPolicy::StrictDomain)
                        .map_err(FrontierUnknown::Arithmetic)?
                        .ok_or_else(|| unsupported("missing waiting endpoint"))?;
                    current.extend_from_slice(&f.pairs()[1..]);
                    continue;
                }
                components.push(
                    PreparedSlopeFunction::from_prepared_pairs(std::mem::take(&mut current))
                        .map_err(FrontierUnknown::Arithmetic)?,
                );
            }
            current_y = f
                .evaluate(f.domain().1, EvaluationPolicy::StrictDomain)
                .map_err(FrontierUnknown::Arithmetic)?
                .ok_or_else(|| unsupported("missing waiting endpoint"))?;
            current.extend_from_slice(f.pairs());
        }
        if !current.is_empty() {
            components.push(
                PreparedSlopeFunction::from_prepared_pairs(current)
                    .map_err(FrontierUnknown::Arithmetic)?,
            );
        }
        match SlopeFront::from_components(components, budget) {
            FrontierOutcome::Complete(out) => match self.bounds.apply(
                &out,
                self.policy,
                FrontierBudget {
                    max_elapsed: start
                        .zip(budget.max_elapsed)
                        .map(|(s, d)| d.saturating_sub(s.elapsed())),
                    ..budget
                },
            ) {
                FrontierOutcome::Complete(out) => Ok(Some(out.paired_result(
                    origin,
                    self.source.clone(),
                    self.policy,
                    front.used_tolerant_alignment(),
                ))),
                FrontierOutcome::Empty => Ok(None),
                FrontierOutcome::Unknown(e) => Err(e),
            },
            FrontierOutcome::Empty => Ok(None),
            FrontierOutcome::Unknown(e) => Err(e),
        }
    }

    fn extend_composed(
        &self,
        front: &SlopeFront,
        budget: FrontierBudget,
        machine_waiting_bounds: Option<(f64, f64)>,
    ) -> Result<Option<SlopeFront>, FrontierUnknown> {
        let start = budget.max_elapsed.map(|_| Instant::now());
        let inverse = match &self.inverse {
            FrontierOutcome::Complete(f) => f,
            FrontierOutcome::Empty => return Ok(None),
            FrontierOutcome::Unknown(e) => return Err(e.clone()),
        };
        let composition = match &self.composition {
            FrontierOutcome::Complete(c) => c,
            FrontierOutcome::Empty => return Ok(None),
            FrontierOutcome::Unknown(e) => return Err(e.clone()),
        };
        let stages = front
            .components()
            .len()
            .checked_add(if machine_waiting_bounds.is_some() {
                6
            } else {
                3
            })
            .ok_or(FrontierUnknown::WorkLimit)?;
        let stage_budget = || FrontierBudget {
            max_operations: budget.max_operations / stages,
            max_elapsed: start
                .zip(budget.max_elapsed)
                .map(|(s, d)| d.saturating_sub(s.elapsed())),
            ..budget
        };
        let mut components = Vec::new();
        let mut pairs = 0usize;
        for f in front.components() {
            let (lo, hi) = inverse.domain();
            match composition.compose_clipped(f, lo, hi, self.policy, stage_budget()) {
                FrontierOutcome::Complete(f) => {
                    pairs = pairs
                        .checked_add(f.pairs().len())
                        .ok_or(FrontierUnknown::PairLimit)?;
                    if pairs > budget.max_pairs {
                        return Err(FrontierUnknown::PairLimit);
                    }
                    components.push(f);
                }
                FrontierOutcome::Empty => {}
                FrontierOutcome::Unknown(e) => return Err(e),
            }
        }
        let transformed = match SlopeFront::from_components(components, stage_budget()) {
            FrontierOutcome::Complete(f) => f,
            FrontierOutcome::Empty => return Ok(None),
            FrontierOutcome::Unknown(e) => return Err(e),
        };
        let phi = match &self.increment_front {
            FrontierOutcome::Complete(f) => f,
            FrontierOutcome::Empty => return Ok(None),
            FrontierOutcome::Unknown(e) => return Err(e.clone()),
        };
        let charged = match transformed.add(phi, self.policy, stage_budget()) {
            FrontierOutcome::Complete(f) => f,
            FrontierOutcome::Empty => return Ok(None),
            FrontierOutcome::Unknown(e) => return Err(e),
        };
        // The reference checks its fixed-array sum before constructing the
        // waiting envelope. A one-segment duplicate is accepted, but an
        // anchor equal to its 1e12 sentinel is rejected. Other numerical
        // validity checks are enforced by the prepared representation.
        if charged
            .components()
            .iter()
            .any(|component| component.is_machine_zero_length() && component.pairs()[0].0 == 1e12)
        {
            return Err(FrontierUnknown::Unsupported(
                "machine zero-length sum sentinel",
            ));
        }
        if charged.has_machine_zero_length() {
            let (lower, upper) = machine_waiting_bounds.ok_or(FrontierUnknown::Unsupported(
                "machine zero-length waiting bounds are unavailable",
            ))?;
            let waiting = match charged.machine_zero_length_waiting(
                lower,
                upper,
                self.source.waiting,
                stage_budget(),
            ) {
                FrontierOutcome::Complete(value) => value,
                FrontierOutcome::Empty => return Ok(None),
                FrontierOutcome::Unknown(reason) => return Err(reason),
            };
            let waiting = match waiting.pass_constant_lower(&self.source.lower, stage_budget()) {
                FrontierOutcome::Complete(value) => value,
                FrontierOutcome::Empty => return Ok(None),
                FrontierOutcome::Unknown(reason) => return Err(reason),
            };
            let waiting = match waiting.pass_constant_upper(&self.source.upper, stage_budget()) {
                FrontierOutcome::Complete(value) => value,
                FrontierOutcome::Empty => return Ok(None),
                FrontierOutcome::Unknown(reason) => return Err(reason),
            };
            return Ok(Some(
                SlopeFront::from_prepared(waiting.function().clone()).with_provenance(
                    FrontierProvenance::PairedMachineWaiting {
                        input: front.provenance().clone(),
                        arc: self.source.clone(),
                        predecessor: waiting.predecessor(),
                        waiting_bits: waiting.waiting_bits(),
                    },
                    true,
                ),
            ));
        }
        let (tail_lo, tail_hi) = inverse.domain();
        let (ll, lh) = self.source.lower.domain();
        let (ul, uh) = self.source.upper.domain();
        let shape = Translation {
            delta: 0.,
            phi: 0.,
            tail_lo,
            tail_hi,
            head_lo: ll.max(ul),
            head_hi: lh.min(uh),
        };
        self.extend_waiting(&charged, front, &shape, stage_budget())
    }
}
// Retaining slopes through translation agrees with recovered secant
// composition only when all intermediate endpoint arithmetic is exact.
// Other geometry uses the general prepared composition operation instead.
fn exact_translation(front: &SlopeFront, t: &Translation) -> bool {
    let integer = |x: f64| x.is_finite() && x.abs() <= ((1u64 << 40) as f64) && x.fract() == 0.;
    if !integer(t.delta) || !integer(t.phi) || !integer(t.tail_lo) || !integer(t.tail_hi) {
        return false;
    }
    for f in front.components() {
        let mut x = f.pairs()[0].1;
        let mut y = f.pairs()[0].0 + t.phi;
        if !integer(x) || !integer(x + t.delta) || !integer(f.pairs()[0].0) || !integer(y) {
            return false;
        }
        for &(s, end) in &f.pairs()[1..] {
            let product = s * (end - x);
            y += product;
            if !integer(s)
                || !integer(end)
                || !integer(end + t.delta)
                || !integer(product)
                || !integer(y)
            {
                return false;
            }
            x = end;
        }
    }
    true
}
fn finite(x: f64) -> Result<f64, FrontierUnknown> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(FrontierUnknown::Arithmetic(
            RawFunctionError::ArithmeticOverflow("paired slope operation"),
        ))
    }
}
fn near(a: f64, b: f64) -> Result<bool, FrontierUnknown> {
    let distance = finite(a - b)?.abs();
    Ok(distance < 0.0001 || distance < finite(finite(a + b)?.abs() * 2.220446049250313e-10)?)
}
fn value(y: f64, s: f64, dx: f64) -> Result<f64, FrontierUnknown> {
    finite(y + finite(s * dx)?)
}
fn push(out: &mut Vec<Segment>, s: Segment, b: FrontierBudget) -> Result<(), FrontierUnknown> {
    if out.len() >= b.max_pairs || out.len() >= b.max_operations {
        return Err(FrontierUnknown::PairLimit);
    }
    out.push(s);
    Ok(())
}

#[cfg(test)]
#[path = "../../../../tests/unit/graph/coupled/descriptor/relation/tests.rs"]
mod tests;
