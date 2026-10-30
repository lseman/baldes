//! Prepared relation minimum operations.
use super::*;

impl PreparedSlopeRelationArc {
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
            Err(FrontierUnknown::Unsupported("machine minimum stream domain")) => {
                FrontierOutcome::Empty
            }
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
            || fixed.has_machine_only_geometry()
            || vector.has_machine_only_geometry()
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
            // The lower-filter helper returns status 2 when the feasible
            // front re-enters after leaving. The backward caller rejects the
            // extension on that status; status 1 remains diagnostic.
            FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine lower stream reentry status",
            )) => FrontierOutcome::Empty,
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
            || fixed.has_machine_only_geometry()
            || vector.has_internal_zero_length()
            || vector.has_machine_only_geometry()
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
        if forward.has_machine_only_geometry()
            || backward.has_machine_only_geometry()
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
        if fixed.has_machine_only_geometry()
            || vector.has_machine_only_geometry()
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
        Self::machine_backward_waiting_with_debug(next, input, initial, target, mode, 0, budget)
    }

    /// Run the waiting scan with the resource's diagnostic level.
    pub fn machine_backward_waiting_with_debug(
        next: &PreparedSlopeFunction,
        input: &MachineBackwardInputs<'_>,
        initial: f64,
        target: f64,
        _mode: i32,
        debug_level: i32,
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
            if debug_level > 1
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
}
