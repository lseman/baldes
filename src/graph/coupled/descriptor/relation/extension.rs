//! Prepared relation extension operations.
use super::*;

impl PreparedSlopeRelationArc {
    pub(super) fn extend_checked(
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
    pub(super) fn extend_waiting(
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

    pub(super) fn extend_composed(
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
        if charged.components().iter().any(|component| {
            component.has_machine_only_geometry() && component.pairs()[0].0 == 1e12
        }) {
            return Err(FrontierUnknown::Unsupported(
                "machine zero-length sum sentinel",
            ));
        }
        if charged.has_machine_zero_length() {
            let (lower, upper) = machine_waiting_bounds.ok_or(FrontierUnknown::Unsupported(
                "machine zero-length waiting bounds are unavailable",
            ))?;
            let waiting = match charged.machine_waiting(
                lower,
                upper,
                self.source.waiting,
                stage_budget(),
            ) {
                FrontierOutcome::Complete(value) => value,
                FrontierOutcome::Empty => return Ok(None),
                // The specialized scan has a smaller bounded-work cost for a
                // single zero-length segment. Use it when the general scan's
                // conservative reservation does not fit the caller's budget.
                FrontierOutcome::Unknown(FrontierUnknown::WorkLimit) => {
                    match charged.machine_zero_length_waiting(
                        lower, upper, self.source.waiting, stage_budget(),
                    ) {
                        FrontierOutcome::Complete(value) => value,
                        FrontierOutcome::Empty => return Ok(None),
                        FrontierOutcome::Unknown(reason) => return Err(reason),
                    }
                }
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
