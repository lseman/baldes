//! Prepared relation backward operations.
use super::*;

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

    pub(super) fn machine_backward_compose_pairs(
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

    pub(super) fn machine_backward_wait_pairs(
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

    pub(super) fn machine_backward_finish_pairs(
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
}
