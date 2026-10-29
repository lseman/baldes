//! Prepared relation continuation operations.
use super::*;

impl PreparedSlopeRelationArc {
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
}
