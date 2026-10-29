//! MachineWaitingEnvelope operations.
use super::*;

impl MachineWaitingEnvelope {
    /// The intermediate slope function, before dependent bounds are applied.
    pub fn function(&self) -> &PreparedSlopeFunction {
        &self.function
    }
    /// Coordinate written by the recovered branch, or None if it leaves the
    /// predecessor field unchanged and its prior value is unavailable.
    pub fn predecessor(&self) -> Option<f64> {
        self.predecessor
    }
    /// Waiting provenance bits retained by the reference branch.
    pub fn waiting_bits(&self) -> u64 {
        self.waiting_bits
    }

    /// Compatibility entry point for the recovered lower-bound maximum.
    pub fn pass_constant_lower(
        &self,
        lower: &PreparedSlopeFunction,
        budget: FrontierBudget,
    ) -> FrontierOutcome<Self> {
        self.apply_lower_bound(lower, budget)
    }

    /// Ordered maximum with a covering dependent lower vector. Each emitted
    /// piece carries the active fixed segment's waiting bit, including pieces
    /// selected from the bound. Merging requires equal bits and near slopes.
    pub fn apply_lower_bound(
        &self,
        lower: &PreparedSlopeFunction,
        budget: FrontierBudget,
    ) -> FrontierOutcome<Self> {
        let started = Instant::now();
        let compute = || -> FrontResult<Self> {
            let deadline = || -> FrontResult<()> {
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
            let a = self.function.pairs();
            let b = lower.pairs();
            if a.len() > budget.max_pairs || b.len() > budget.max_pairs {
                return Err(FrontierUnknown::PairLimit);
            }
            if a.len() > 12 {
                return Err(FrontierUnknown::Unsupported(
                    "machine lower maximum array size",
                ));
            }
            if budget.max_operations < a.len().saturating_add(b.len()).saturating_mul(96) {
                return Err(FrontierUnknown::WorkLimit);
            }
            let close = |a, b| near(a, b, FrontOperationPolicy::FrontAlgebra);
            let finite = |value| arithmetic(value, "machine lower maximum");
            if (a[0].1 < b[0].1 && !close(a[0].1, b[0].1)?)
                || (a.last().unwrap().1 > b.last().unwrap().1
                    && !close(a.last().unwrap().1, b.last().unwrap().1)?)
            {
                return Err(FrontierUnknown::Unsupported("machine lower maximum domain"));
            }
            let (mut i, mut j) = (1usize, 1usize);
            let (mut av, mut bv) = (a[0].0, b[0].0);
            let (mut ax, mut bx) = (a[0].1, b[0].1);
            let b_start = if close(ax, bx)? {
                bv
            } else {
                while j < b.len() && b[j].1 < ax {
                    deadline()?;
                    bv = finite(bv + finite(finite(b[j].1 - bx)? * b[j].0)?)?;
                    bx = b[j].1;
                    j += 1;
                }
                let &(slope, end) = b.get(j).ok_or(FrontierUnknown::Unsupported(
                    "machine lower maximum start lookup",
                ))?;
                if close(end, ax)? {
                    bv = finite(bv + finite(finite(end - bx)? * slope)?)?;
                    bx = end;
                    j += 1;
                    bv
                } else {
                    finite(finite(slope * finite(ax - bx)?)? + bv)?
                }
            };
            let mut chosen_a = b_start <= av;
            let mut previous_tie = close(av, b_start)?;
            let mut output = vec![(if chosen_a { av } else { b_start }, ax)];
            let mut output_bits = 0u64;
            let append = |output: &mut Vec<(f64, f64)>,
                          bits: &mut u64,
                          slope: f64,
                          end: f64,
                          waits: bool|
             -> FrontResult<()> {
                let previous = output.len() - 1;
                let merge = previous > 0
                    && (((*bits >> previous) & 1 != 0) == waits)
                    && close(output[previous].0, slope)?;
                let index = if merge { previous } else { output.len() };
                if index >= 12 || index >= budget.max_pairs {
                    return Err(FrontierUnknown::PairLimit);
                }
                if merge {
                    output[index] = (slope, end);
                } else {
                    output.push((slope, end));
                }
                let bit = 1u64 << index;
                if waits {
                    *bits |= bit;
                } else {
                    *bits &= !bit;
                }
                Ok(())
            };
            while i < a.len() && j < b.len() {
                deadline()?;
                let (ae, be) = (a[i].1, b[j].1);
                let aend = finite(finite(finite(ae - ax)? * a[i].0)? + av)?;
                let bend = finite(finite(finite(be - bx)? * b[j].0)? + bv)?;
                let order = if close(ae, be)? {
                    0
                } else if be <= ae {
                    1
                } else {
                    -1
                };
                let (coordinate, ay, by) = match order {
                    1 => (be, finite(finite(finite(be - ax)? * a[i].0)? + av)?, bend),
                    -1 => (ae, aend, finite(finite(finite(ae - bx)? * b[j].0)? + bv)?),
                    _ => (ae, aend, bend),
                };
                let tie = close(ay, by)?;
                let next_a = if tie { chosen_a } else { by <= ay };
                let waits = (self.waiting_bits >> i) & 1 != 0;
                if chosen_a != next_a {
                    // A preceding near tie switches operands without an
                    // artificial zero-width crossing, irrespective of which
                    // original segment supplied that previous endpoint.
                    if !previous_tie {
                        let bi = finite(bend - finite(be * b[j].0)?)?;
                        let ai = finite(aend - finite(ae * a[i].0)?)?;
                        let crossing = finite(finite(bi - ai)? / finite(a[i].0 - b[j].0)?)?;
                        append(
                            &mut output,
                            &mut output_bits,
                            if chosen_a { a[i].0 } else { b[j].0 },
                            crossing,
                            waits,
                        )?;
                    }
                    append(
                        &mut output,
                        &mut output_bits,
                        if next_a { a[i].0 } else { b[j].0 },
                        coordinate,
                        waits,
                    )?;
                } else if (chosen_a && order != 1) || (!chosen_a && order != -1) {
                    append(
                        &mut output,
                        &mut output_bits,
                        if chosen_a { a[i].0 } else { b[j].0 },
                        coordinate,
                        waits,
                    )?;
                }
                chosen_a = next_a;
                previous_tie = tie;
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
            let function = PreparedSlopeFunction::from_machine_pairs(output)
                .map_err(FrontierUnknown::Arithmetic)?;
            deadline()?;
            Ok(Self {
                function,
                waiting_bits: output_bits,
                predecessor: self.predecessor,
            })
        };
        match compute() {
            Ok(value) => FrontierOutcome::Complete(value),
            Err(reason) => FrontierOutcome::Unknown(reason),
        }
    }

    /// Compatibility entry point for the recovered connected upper filter.
    pub fn pass_constant_upper(
        &self,
        upper: &PreparedSlopeFunction,
        budget: FrontierBudget,
    ) -> FrontierOutcome<Self> {
        self.apply_upper_bound(upper, budget)
    }

    /// Ordered connected-region upper filter. A later re-entry is the
    /// machine helper's status 2 and remains unknown. The reference caller
    /// retains its waiting bits and predecessor after this in-place filter.
    pub fn apply_upper_bound(
        &self,
        upper: &PreparedSlopeFunction,
        budget: FrontierBudget,
    ) -> FrontierOutcome<Self> {
        let started = Instant::now();
        let compute = || -> FrontResult<Self> {
            let deadline = || -> FrontResult<()> {
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
            let a = self.function.pairs();
            let b = upper.pairs();
            if a.len() > 12 {
                return Err(FrontierUnknown::Unsupported(
                    "machine upper filter array size",
                ));
            }
            if a.len() > budget.max_pairs || b.len() > budget.max_pairs {
                return Err(FrontierUnknown::PairLimit);
            }
            if budget.max_operations < a.len().saturating_add(b.len()).saturating_mul(64) {
                return Err(FrontierUnknown::WorkLimit);
            }
            let close = |a, b| near(a, b, FrontOperationPolicy::FrontAlgebra);
            let finite = |value| arithmetic(value, "machine upper filter");
            if (a[0].1 < b[0].1 && !close(a[0].1, b[0].1)?)
                || (a.last().unwrap().1 > b.last().unwrap().1
                    && !close(a.last().unwrap().1, b.last().unwrap().1)?)
            {
                return Err(FrontierUnknown::Unsupported(
                    "machine upper filter domain status",
                ));
            }
            let (mut i, mut j) = (1usize, 1usize);
            let (mut av, mut bv) = (a[0].0, b[0].0);
            let (mut ax, mut bx) = (a[0].1, b[0].1);
            let b_start = if close(ax, bx)? {
                bv
            } else {
                while j < b.len() && b[j].1 < ax {
                    deadline()?;
                    bv = finite(bv + finite(finite(b[j].1 - bx)? * b[j].0)?)?;
                    bx = b[j].1;
                    j += 1;
                }
                let &(slope, end) = b.get(j).ok_or(FrontierUnknown::Unsupported(
                    "machine upper filter start lookup",
                ))?;
                if close(end, ax)? {
                    bv = finite(bv + finite(finite(end - bx)? * slope)?)?;
                    bx = end;
                    j += 1;
                    bv
                } else {
                    finite(finite(finite(ax - bx)? * slope)? + bv)?
                }
            };
            let mut active = av < b_start || close(av, b_start)?;
            let mut output = if active { vec![a[0]] } else { Vec::new() };
            let append = |output: &mut Vec<(f64, f64)>, pair| -> FrontResult<()> {
                if output.len() >= 12 || output.len() >= budget.max_pairs {
                    return Err(FrontierUnknown::PairLimit);
                }
                output.push(pair);
                Ok(())
            };
            while i < a.len() && j < b.len() {
                deadline()?;
                let (ae, be) = (a[i].1, b[j].1);
                let aend = finite(finite(finite(ae - ax)? * a[i].0)? + av)?;
                let bend = finite(finite(finite(be - bx)? * b[j].0)? + bv)?;
                let order = if close(ae, be)? {
                    0
                } else if be <= ae {
                    1
                } else {
                    -1
                };
                let (coordinate, ay, by) = match order {
                    1 => (be, finite(finite(finite(be - ax)? * a[i].0)? + av)?, bend),
                    -1 => (ae, aend, finite(finite(finite(ae - bx)? * b[j].0)? + bv)?),
                    _ => (ae, aend, bend),
                };
                let tie = close(ay, by)?;
                if active {
                    if tie || ay <= by {
                        append(&mut output, (a[i].0, coordinate))?;
                    } else {
                        let ai = finite(aend - finite(ae * a[i].0)?)?;
                        let bi = finite(bend - finite(be * b[j].0)?)?;
                        let crossing = finite(finite(bi - ai)? / finite(a[i].0 - b[j].0)?)?;
                        if !close(crossing, output.last().unwrap().1)? {
                            append(&mut output, (a[i].0, crossing))?;
                        }
                        active = false;
                    }
                } else if tie {
                    if !output.is_empty() {
                        return Err(FrontierUnknown::Unsupported(
                            "machine upper filter reentry status",
                        ));
                    }
                    append(&mut output, (ay, coordinate))?;
                    active = true;
                } else if ay < by {
                    if !output.is_empty() {
                        return Err(FrontierUnknown::Unsupported(
                            "machine upper filter reentry status",
                        ));
                    }
                    let ai = finite(aend - finite(ae * a[i].0)?)?;
                    let bi = finite(bend - finite(be * b[j].0)?)?;
                    let crossing = finite(finite(bi - ai)? / finite(a[i].0 - b[j].0)?)?;
                    let anchor = finite(finite(crossing * a[i].0)? + ai)?;
                    append(&mut output, (anchor, crossing))?;
                    // The entry branch copies the original fixed endpoint,
                    // including when only the vector endpoint was consumed.
                    append(&mut output, a[i])?;
                    active = true;
                }
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
                    "machine upper filter invalid empty count",
                ));
            }
            if output.len() == 1 {
                let coordinate = output[0].1;
                append(&mut output, (0., coordinate))?;
            }
            let function = PreparedSlopeFunction::from_machine_pairs(output)
                .map_err(FrontierUnknown::Arithmetic)?;
            deadline()?;
            Ok(Self {
                function,
                waiting_bits: self.waiting_bits,
                predecessor: self.predecessor,
            })
        };
        match compute() {
            Ok(value) => FrontierOutcome::Complete(value),
            Err(reason) => FrontierOutcome::Unknown(reason),
        }
    }
}
