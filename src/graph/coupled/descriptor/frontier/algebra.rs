//! SlopeFront operations.
use super::*;

impl SlopeFront {
    /// Compatibility entry point for the recovered forward waiting scan.
    pub fn machine_single_segment_waiting(
        &self,
        lower: f64,
        upper: f64,
        waiting: i32,
        budget: FrontierBudget,
    ) -> FrontierOutcome<MachineWaitingEnvelope> {
        self.machine_waiting(lower, upper, waiting, budget)
    }

    /// Literal forward-extension waiting scan, retaining the original source
    /// value anchor across runs and the recorder's predecessor and waiting
    /// bits. This is an intermediate machine array, before dependent bounds.
    pub fn machine_waiting(
        &self,
        lower: f64,
        upper: f64,
        waiting: i32,
        budget: FrontierBudget,
    ) -> FrontierOutcome<MachineWaitingEnvelope> {
        let started = Instant::now();
        let compute = || -> FrontResult<MachineWaitingEnvelope> {
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
            if !lower.is_finite() || !upper.is_finite() || lower > upper {
                return Err(FrontierUnknown::Unsupported(
                    "machine waiting needs finite ordered bounds",
                ));
            }
            if budget.max_components == 0 {
                return Err(FrontierUnknown::ComponentLimit);
            }
            let [function] = self.components() else {
                return Err(FrontierUnknown::Unsupported(
                    "machine waiting requires one component",
                ));
            };
            let pairs = function.pairs();
            if !(2..=12).contains(&pairs.len()) || pairs[0].0 == 1e12 {
                return Err(FrontierUnknown::Unsupported("machine waiting input array"));
            }
            if pairs.len() > budget.max_pairs {
                return Err(FrontierUnknown::PairLimit);
            }
            let required = pairs
                .len()
                .saturating_mul(pairs.len())
                .saturating_mul(32)
                .saturating_add(32);
            if required > budget.max_operations {
                return Err(FrontierUnknown::WorkLimit);
            }
            let close = |a, b| near(a, b, FrontOperationPolicy::FrontAlgebra);
            let finite = |value| arithmetic(value, "machine forward waiting scan");
            fn emit(
                out: &mut Vec<(f64, f64)>,
                bits: &mut u64,
                slope: f64,
                end: f64,
                waits: bool,
                budget: FrontierBudget,
            ) -> FrontResult<()> {
                if out.len() >= 12 || out.len() >= budget.max_pairs {
                    return Err(FrontierUnknown::PairLimit);
                }
                let bit = 1u64 << out.len();
                if waits {
                    *bits |= bit;
                } else {
                    *bits &= !bit;
                }
                out.push((slope, end));
                Ok(())
            }
            // The wrapper resets bits and predecessor before extension.
            let mut output = vec![(0., 0.)];
            let mut bits = 0;
            let mut predecessor = None;
            let rate = f64::from(waiting);
            let mut value = pairs[0].0;
            let last = pairs.len() - 1;
            let mut i = 1;
            while i <= last {
                deadline()?;
                let (slope, end) = pairs[i];
                let previous = pairs[i - 1].1;
                if slope <= rate {
                    if lower < end && !close(lower, end)? {
                        if output.len() == 1 {
                            let start = if previous <= lower { lower } else { previous };
                            output[0] = (
                                finite(finite(finite(start - previous)? * slope)? + value)?,
                                start,
                            );
                        }
                        if upper <= end || close(upper, end)? {
                            emit(&mut output, &mut bits, slope, upper, false, budget)?;
                            break;
                        }
                        emit(&mut output, &mut bits, slope, end, false, budget)?;
                    }
                    value = finite(value + finite(slope * finite(end - previous)?)?)?;
                    if i < last {
                        i += 1;
                        continue;
                    }
                    if end < upper && !close(end, upper)? {
                        if output.len() == 1 {
                            output[0] =
                                (finite(value + finite(finite(lower - end)? * rate)?)?, lower);
                            predecessor = Some(end);
                        }
                        emit(&mut output, &mut bits, rate, upper, true, budget)?;
                    }
                    break;
                }
                // Follow a waiting line from the beginning of this run until
                // the scanned source endpoint reaches it or the source ends.
                let anchor_coordinate = previous;
                let mut scanned = finite(finite(finite(end - previous)? * slope)? + value)?;
                let mut j = i + 1;
                if j <= last {
                    let mut wait_value =
                        finite(finite(finite(pairs[j].1 - anchor_coordinate)? * rate)? + value)?;
                    scanned = finite(scanned + finite(pairs[j].0 * finite(pairs[j].1 - end)?)?)?;
                    while scanned > wait_value && !close(scanned, wait_value)? {
                        deadline()?;
                        j += 1;
                        if j > last {
                            break;
                        }
                        wait_value = finite(
                            finite(finite(pairs[j].1 - anchor_coordinate)? * rate)? + value,
                        )?;
                        scanned = finite(
                            scanned + finite(finite(pairs[j].1 - pairs[j - 1].1)? * pairs[j].0)?,
                        )?;
                    }
                }
                if j > last {
                    if output.len() == 1 {
                        let start = if anchor_coordinate <= lower {
                            lower
                        } else {
                            anchor_coordinate
                        };
                        // The exhausted branch uses the original run slope
                        // for its anchor, even though the emitted slope waits.
                        output[0] = (
                            finite(value + finite(slope * finite(start - anchor_coordinate)?)?)?,
                            start,
                        );
                        predecessor = Some(anchor_coordinate);
                    }
                    let finish = if pairs[last].1 <= upper {
                        upper
                    } else {
                        pairs[last].1
                    };
                    emit(&mut output, &mut bits, rate, finish, true, budget)?;
                    break;
                }
                let (next_slope, next_end) = pairs[j];
                if next_end <= lower || close(next_end, lower)? {
                    // Retain the value anchor, as the ordered recorder does;
                    // replacing it with the scanned endpoint changes later runs.
                    i = j + 1;
                    continue;
                }
                let before_next = pairs[j - 1].1;
                let crossing = finite(
                    finite(
                        finite(finite(scanned - value)? - finite(before_next * next_slope)?)?
                            + finite(anchor_coordinate * rate)?,
                    )? / finite(rate - next_slope)?,
                )?;
                if crossing <= lower || close(crossing, lower)? {
                    if output.len() == 1 {
                        let start = if anchor_coordinate <= lower {
                            lower
                        } else {
                            anchor_coordinate
                        };
                        output[0] = (
                            finite(scanned + finite(finite(start - before_next)? * next_slope)?)?,
                            start,
                        );
                    }
                } else {
                    if output.len() == 1 {
                        let start = if anchor_coordinate <= lower {
                            lower
                        } else {
                            anchor_coordinate
                        };
                        output[0] = (
                            finite(finite(finite(start - anchor_coordinate)? * rate)? + value)?,
                            start,
                        );
                        predecessor = Some(anchor_coordinate);
                    }
                    if upper <= crossing || close(upper, crossing)? {
                        emit(&mut output, &mut bits, rate, upper, true, budget)?;
                        break;
                    }
                    emit(&mut output, &mut bits, rate, crossing, true, budget)?;
                }
                if upper <= next_end || close(upper, next_end)? {
                    emit(&mut output, &mut bits, next_slope, upper, false, budget)?;
                    break;
                }
                emit(&mut output, &mut bits, next_slope, next_end, false, budget)?;
                if j == last {
                    if next_end < upper && !close(next_end, upper)? {
                        emit(&mut output, &mut bits, rate, upper, true, budget)?;
                    }
                    break;
                }
                i = j + 1;
            }
            if output.len() == 1 {
                return Err(FrontierUnknown::Unsupported(
                    "machine waiting produced no segments",
                ));
            }
            let function = PreparedSlopeFunction::from_machine_pairs(output)
                .map_err(FrontierUnknown::Arithmetic)?;
            deadline()?;
            Ok(MachineWaitingEnvelope {
                function,
                predecessor,
                waiting_bits: bits,
            })
        };
        match compute() {
            Ok(value) => FrontierOutcome::Complete(value),
            Err(reason) => FrontierOutcome::Unknown(reason),
        }
    }

    /// Reduce consecutive near slopes in the fixed machine array. The first
    /// slope is retained on a merge; the original snapshots remain available
    /// for reconstruction. Zero-width first segments follow the same scan.
    pub fn machine_reduce_for_continuation(&self, budget: FrontierBudget) -> FrontierOutcome<Self> {
        let mut work = Work::new(budget);
        let mut compute = || -> FrontResult<Self> {
            if self.components.len() != 1 {
                return Err(FrontierUnknown::Unsupported(
                    "machine reduction requires one component",
                ));
            }
            if budget.max_components == 0 {
                return Err(FrontierUnknown::ComponentLimit);
            }
            let pairs = self.components[0].pairs();
            if !(2..=12).contains(&pairs.len()) {
                return Err(FrontierUnknown::Unsupported("machine reduction array size"));
            }
            if pairs.len() > budget.max_pairs {
                return Err(FrontierUnknown::PairLimit);
            }
            work.charge(pairs.len().saturating_mul(8).saturating_add(4))?;
            let mut reduced = vec![pairs[0], pairs[1]];
            let mut merged = false;
            for &(slope, end) in &pairs[2..] {
                work.charge(0)?;
                let last = reduced.len() - 1;
                if reduced[last].0.is_finite()
                    && slope.is_finite()
                    && near(reduced[last].0, slope, FrontOperationPolicy::FrontAlgebra)?
                {
                    reduced[last].1 = end;
                    merged = true;
                } else {
                    reduced.push((slope, end));
                }
            }
            // The reducer checks consecutive emitted slopes after merging.
            // The anchor is a value, so comparisons start at segment two.
            for index in 2..reduced.len() {
                work.charge(0)?;
                if reduced[index - 1].0.is_finite()
                    && reduced[index].0.is_finite()
                    && near(
                        reduced[index - 1].0,
                        reduced[index].0,
                        FrontOperationPolicy::FrontAlgebra,
                    )?
                {
                    return Err(FrontierUnknown::Unsupported(
                        "machine reduction consecutive slopes",
                    ));
                }
            }
            let components = if merged {
                vec![PreparedSlopeFunction::from_machine_output_pairs(reduced)
                    .map_err(FrontierUnknown::Arithmetic)?]
                .into()
            } else {
                self.components.clone()
            };
            work.charge(0)?;
            Ok(Self {
                components,
                provenance: Arc::new(FrontierProvenance::MachineReduction {
                    input: self.provenance.clone(),
                    input_front: self.components.clone(),
                    merged,
                }),
                tolerant_alignment: self.tolerant_alignment || merged,
            })
        };
        match compute() {
            Ok(value) => FrontierOutcome::Complete(value),
            Err(reason) => FrontierOutcome::Unknown(reason),
        }
    }

    /// Recovered machine waiting steps for a single zero-length segment.
    /// The lower-slope branch accepts a lower bound at or above the point,
    /// or a lower bound below it within the machine tolerance, and requires
    /// a distinctly later upper bound. `lower` and `upper` are the
    /// reference helper's second and third floating-point arguments. The
    /// returned array and predecessor bit have not passed subsequent native
    /// maximum or upper-bound operations and are not a feasibility result.
    pub fn machine_zero_length_waiting(
        &self,
        lower: f64,
        upper: f64,
        waiting: i32,
        budget: FrontierBudget,
    ) -> FrontierOutcome<MachineWaitingEnvelope> {
        let started = Instant::now();
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine waiting needs finite ordered bounds",
            ));
        }
        if budget.max_components == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        if budget.max_pairs < 2 {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        if budget.max_operations < 4 {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        }
        if budget
            .max_elapsed
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        if self.components.len() != 1
            || !self.components[0].has_machine_only_geometry()
            || self.components[0].has_internal_zero_length()
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine waiting requires one zero-length segment",
            ));
        }
        let pairs = self.components[0].pairs();
        let (value, predecessor) = pairs[0];
        let (slope, end) = pairs[1];
        let wait_slope = f64::from(waiting);
        if value == 1e12 {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "unrecovered machine waiting branch",
            ));
        }
        if slope <= wait_slope {
            if lower >= upper || upper <= predecessor {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "unrecovered lower-slope machine waiting bounds",
                ));
            }
            if lower < predecessor {
                match near(lower, predecessor, FrontOperationPolicy::FrontAlgebra) {
                    Ok(true) => {}
                    Ok(false) => {
                        let upper_near =
                            match near(upper, predecessor, FrontOperationPolicy::FrontAlgebra) {
                                Ok(value) => value,
                                Err(error) => return FrontierOutcome::Unknown(error),
                            };
                        let anchor = match checked(end - predecessor, "machine waiting span")
                            .and_then(|span| {
                                checked(span * slope, "machine waiting multiplication")
                            })
                            .and_then(|product| {
                                checked(value + product, "machine waiting accumulation")
                            }) {
                            Ok(value) => value,
                            Err(error) => {
                                return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error))
                            }
                        };
                        if !upper_near && budget.max_pairs < 3 {
                            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
                        }
                        if budget
                            .max_elapsed
                            .is_some_and(|limit| started.elapsed() >= limit)
                        {
                            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
                        }
                        let function = if upper_near {
                            PreparedSlopeFunction::from_prepared_pairs(vec![
                                (anchor, predecessor),
                                (slope, upper),
                            ])
                        } else {
                            PreparedSlopeFunction::machine_internal_zero_length(
                                anchor,
                                slope,
                                predecessor,
                                wait_slope,
                                upper,
                            )
                        };
                        return match function {
                            Ok(function) => FrontierOutcome::Complete(MachineWaitingEnvelope {
                                function,
                                predecessor: None,
                                waiting_bits: if upper_near { 0 } else { 1 << 2 },
                            }),
                            Err(error) => {
                                FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error))
                            }
                        };
                    }
                    Err(error) => return FrontierOutcome::Unknown(error),
                }
            }
            match near(upper, predecessor, FrontOperationPolicy::FrontAlgebra) {
                Ok(false) => {}
                Ok(true) => {
                    return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                        "unrecovered near-end machine waiting branch",
                    ))
                }
                Err(error) => return FrontierOutcome::Unknown(error),
            }
            let anchor = checked(end - predecessor, "machine waiting span")
                .and_then(|span| checked(span * slope, "machine waiting multiplication"))
                .and_then(|product| checked(value + product, "machine waiting accumulation"))
                .and_then(|v| {
                    checked(lower - end, "machine waiting lower span")
                        .and_then(|span| {
                            checked(span * wait_slope, "machine waiting lower multiplication")
                        })
                        .and_then(|product| {
                            checked(v + product, "machine waiting lower accumulation")
                        })
                });
            let anchor = match anchor {
                Ok(value) => value,
                Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
            };
            if budget.max_operations < 6 {
                return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
            }
            if budget
                .max_elapsed
                .is_some_and(|limit| started.elapsed() >= limit)
            {
                return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
            }
            return match PreparedSlopeFunction::from_prepared_pairs(vec![
                (anchor, lower),
                (wait_slope, upper),
            ]) {
                Ok(function) => FrontierOutcome::Complete(MachineWaitingEnvelope {
                    function,
                    predecessor: Some(predecessor),
                    waiting_bits: 1 << 1,
                }),
                Err(error) => FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
            };
        }
        // MAXSD selects its second operand on equality, including signed
        // zero. Preserve that order for both observed maximum operations.
        let start = if predecessor > lower {
            predecessor
        } else {
            lower
        };
        let finish = if end > upper { end } else { upper };
        let anchor = match checked(start - predecessor, "machine waiting span")
            .and_then(|span| checked(slope * span, "machine waiting multiplication"))
            .and_then(|product| checked(value + product, "machine waiting accumulation"))
        {
            Ok(value) => value,
            Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
        };
        let function = if start == finish {
            PreparedSlopeFunction::machine_zero_length(anchor, wait_slope, start, finish)
        } else {
            PreparedSlopeFunction::from_prepared_pairs(vec![(anchor, start), (wait_slope, finish)])
        };
        if budget
            .max_elapsed
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        match function {
            Ok(function) => FrontierOutcome::Complete(MachineWaitingEnvelope {
                function,
                predecessor: Some(predecessor),
                waiting_bits: 1 << 1,
            }),
            Err(error) => FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
        }
    }

    pub(in crate::graph::coupled::descriptor) fn with_provenance(
        mut self,
        provenance: FrontierProvenance,
        tolerant_alignment: bool,
    ) -> Self {
        self.provenance = Arc::new(provenance);
        self.tolerant_alignment |= tolerant_alignment;
        self
    }
    pub fn zero(lower: f64, upper: f64) -> Result<Self, RawFunctionError> {
        let function = super::RawAffine::new(0., 0.)?.prepare(lower, upper)?;
        Ok(Self::from_prepared(function))
    }

    /// Translate every retained coordinate with a literal addition. Slopes
    /// and anchor values remain unchanged. A rounded coordinate collapse is
    /// unsupported, rather than a reason to drop one of the source pieces.
    pub fn translate(&self, delta: f64, budget: FrontierBudget) -> FrontierOutcome<Self> {
        if self.has_machine_zero_length() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine zero-length segment translation",
            ));
        }
        if !delta.is_finite() {
            return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(
                RawFunctionError::NonFinite("translation"),
            ));
        }
        let mut work = Work::new(budget);
        if self.components.len() > budget.max_components {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        let mut components = Vec::with_capacity(self.components.len());
        for component in self.components.iter() {
            let mut pairs = Vec::with_capacity(component.pairs().len());
            let mut previous = None;
            for &(value_or_slope, coordinate) in component.pairs() {
                if let Err(reason) = work.pair() {
                    return FrontierOutcome::Unknown(reason);
                }
                let translated =
                    match arithmetic(coordinate + delta, "front coordinate translation") {
                        Ok(value) => value,
                        Err(reason) => return FrontierOutcome::Unknown(reason),
                    };
                if previous.is_some_and(|previous| translated <= previous) {
                    return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                        "translation collapsed consecutive coordinates",
                    ));
                }
                pairs.push((value_or_slope, translated));
                previous = Some(translated);
            }
            match PreparedSlopeFunction::from_prepared_pairs(pairs) {
                Ok(component) => {
                    if components.len() >= budget.max_components {
                        return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
                    }
                    components.push(component);
                }
                Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
            }
        }
        if let Err(reason) = validate(&components, &mut work) {
            return FrontierOutcome::Unknown(reason);
        }
        FrontierOutcome::Complete(Self {
            components: components.into(),
            provenance: Arc::new(FrontierProvenance::Translation {
                input: self.provenance.clone(),
                delta,
            }),
            tolerant_alignment: self.tolerant_alignment,
        })
    }

    /// Intersect the component union with the exact requested closed window.
    /// Re-anchor at its lower coordinate and retain native slopes; no secant
    /// fitting or bridging a gap occurs. Unrecovered near-boundary native
    /// truncation cases remain Unknown instead of guessing their topology.
    pub fn restrict(
        &self,
        lower: f64,
        upper: f64,
        policy: FrontOperationPolicy,
        budget: FrontierBudget,
    ) -> FrontierOutcome<Self> {
        if self.has_machine_zero_length() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine zero-length segment restriction",
            ));
        }
        if !lower.is_finite() || !upper.is_finite() {
            return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(
                RawFunctionError::NonFinite("restriction window"),
            ));
        }
        if lower > upper {
            return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(
                RawFunctionError::InvalidDomain,
            ));
        }
        let mut work = Work::new(budget);
        let mut components = Vec::new();
        for function in self.components.iter() {
            if let Err(reason) = work.charge(function.pairs().len()) {
                return FrontierOutcome::Unknown(reason);
            }
            let (start, end) = function.domain();
            let lo = start.max(lower);
            let hi = end.min(upper);
            if lo > hi {
                continue;
            }
            if policy == FrontOperationPolicy::FrontAlgebra {
                for &(_, boundary) in function.pairs() {
                    for coordinate in [lo, hi] {
                        if coordinate != boundary {
                            match near(coordinate, boundary, policy) {
                                Ok(true) => {
                                    return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                                        "nonexact near-boundary truncation needs a topology proof",
                                    ))
                                }
                                Err(reason) => return FrontierOutcome::Unknown(reason),
                                Ok(false) => (),
                            }
                        }
                    }
                }
            }
            let anchor = match value_at(function, lo, policy) {
                Ok(value) => value,
                Err(reason) => return FrontierOutcome::Unknown(reason),
            };
            if let Err(reason) = work.pair() {
                return FrontierOutcome::Unknown(reason);
            }
            let mut pairs = vec![(anchor, lo)];
            if lo != hi {
                for &(slope, segment_end) in &function.pairs()[1..] {
                    if segment_end <= lo {
                        continue;
                    }
                    if let Err(reason) = work.pair() {
                        return FrontierOutcome::Unknown(reason);
                    }
                    pairs.push((slope, segment_end.min(hi)));
                    if segment_end >= hi {
                        break;
                    }
                }
            }
            match PreparedSlopeFunction::from_prepared_pairs(pairs) {
                Ok(component) => {
                    if components.len() >= budget.max_components {
                        return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
                    }
                    components.push(component);
                }
                Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
            }
        }
        if components.is_empty() {
            return FrontierOutcome::Empty;
        }
        if let Err(reason) = validate(&components, &mut work) {
            return FrontierOutcome::Unknown(reason);
        }
        FrontierOutcome::Complete(Self {
            components: components.into(),
            provenance: Arc::new(FrontierProvenance::Restriction {
                input: self.provenance.clone(),
                lower,
                upper,
                policy,
            }),
            tolerant_alignment: self.tolerant_alignment,
        })
    }
    pub fn from_prepared(function: PreparedSlopeFunction) -> Self {
        let tolerant_alignment = function.has_machine_only_geometry();
        let components: Arc<[PreparedSlopeFunction]> = vec![function].into();
        Self {
            provenance: Arc::new(FrontierProvenance::Source(components.clone())),
            components,
            tolerant_alignment,
        }
    }
    /// Components must be strictly disjoint and sorted. Overlaps/touching
    /// components need an envelope, which this addition primitive does not fake.
    pub fn from_components(
        components: Vec<PreparedSlopeFunction>,
        budget: FrontierBudget,
    ) -> FrontierOutcome<Self> {
        if components.is_empty() {
            return FrontierOutcome::Empty;
        }
        let mut work = Work::new(budget);
        if let Err(reason) = validate(&components, &mut work) {
            return FrontierOutcome::Unknown(reason);
        }
        let tolerant_alignment = components
            .iter()
            .any(PreparedSlopeFunction::is_machine_zero_length);
        let components: Arc<[PreparedSlopeFunction]> = components.into();
        FrontierOutcome::Complete(Self {
            provenance: Arc::new(FrontierProvenance::Source(components.clone())),
            components,
            tolerant_alignment,
        })
    }
    pub fn components(&self) -> &[PreparedSlopeFunction] {
        &self.components
    }
    pub(in crate::graph::coupled::descriptor) fn has_machine_zero_length(&self) -> bool {
        self.components
            .iter()
            .any(PreparedSlopeFunction::is_machine_zero_length)
    }
    pub fn provenance(&self) -> &Arc<FrontierProvenance> {
        &self.provenance
    }
    pub fn used_tolerant_alignment(&self) -> bool {
        self.tolerant_alignment
    }
    /// Add active slopes directly, without fitting rounded endpoint values.
    pub fn add(
        &self,
        other: &Self,
        policy: FrontOperationPolicy,
        budget: FrontierBudget,
    ) -> FrontierOutcome<Self> {
        if self
            .components
            .iter()
            .any(PreparedSlopeFunction::has_internal_zero_length)
            || other
                .components
                .iter()
                .any(PreparedSlopeFunction::has_internal_zero_length)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine internal zero-length segment addition",
            ));
        }
        if (self.has_machine_zero_length() || other.has_machine_zero_length())
            && (policy != FrontOperationPolicy::FrontAlgebra
                || self.components.len() != 1
                || other.components.len() != 1
                || !self.components[0].has_machine_only_geometry())
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine zero-length segment addition",
            ));
        }
        let mut work = Work::new(budget);
        let mut output = Vec::new();
        let mut tolerant = self.tolerant_alignment || other.tolerant_alignment;
        let (mut i, mut j) = (0, 0);
        while i < self.components.len() && j < other.components.len() {
            if let Err(reason) = work.charge(1) {
                return FrontierOutcome::Unknown(reason);
            }
            let a = &self.components[i];
            let b = &other.components[j];
            match add_component(a, b, policy, &mut work) {
                Ok(Some((component, aligned))) => {
                    if output.len() >= budget.max_components {
                        return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
                    }
                    output.push(component);
                    tolerant |= aligned;
                }
                Ok(None) => (),
                Err(reason) => return FrontierOutcome::Unknown(reason),
            }
            let ae = a.domain().1;
            let be = b.domain().1;
            if ae <= be {
                i += 1;
            }
            if be <= ae {
                j += 1;
            }
        }
        if output.is_empty() {
            return FrontierOutcome::Empty;
        }
        if let Err(reason) = validate(&output, &mut work) {
            return FrontierOutcome::Unknown(reason);
        }
        FrontierOutcome::Complete(Self {
            components: output.into(),
            provenance: Arc::new(FrontierProvenance::Addition {
                left: self.provenance.clone(),
                right: other.provenance.clone(),
                policy,
            }),
            tolerant_alignment: tolerant,
        })
    }
    /// The largest amount by which this front exceeds `other` on `other`'s
    /// domain, or None when it does not cover that domain. Checked knot-wise
    /// exactly as `covers_below`; a value no higher than (or, under the
    /// native policy, near) `other` contributes zero.
    pub(crate) fn excess(&self, other: &Self, policy: FrontOperationPolicy) -> Option<f64> {
        if self.has_machine_zero_length() || other.has_machine_zero_length() {
            return None;
        }
        let evaluation = match policy {
            FrontOperationPolicy::StrictDomain => EvaluationPolicy::StrictDomain,
            FrontOperationPolicy::FrontAlgebra => EvaluationPolicy::EndpointTolerance,
        };
        let value = |f: &PreparedSlopeFunction, x: f64| f.evaluate(x, evaluation).ok().flatten();
        let gap = |a: f64, b: f64| {
            if a <= b || near(a, b, policy).unwrap_or(false) {
                Some(0.)
            } else {
                Some(a - b).filter(|gap| gap.is_finite())
            }
        };
        let mut knots = Vec::new();
        let mut worst = 0f64;
        for b in other.components.iter() {
            let (lo, hi) = b.domain();
            knots.clear();
            knots.extend(b.pairs().iter().map(|p| p.1));
            knots.extend(
                self.components
                    .iter()
                    .flat_map(|a| a.pairs().iter().map(|p| p.1))
                    .filter(|&x| lo < x && x < hi),
            );
            knots.sort_by(f64::total_cmp);
            knots.dedup();
            // The least excess of any single component covering [l, h];
            // components are continuous, so endpoint gaps bound it.
            let covers = |l: f64, h: f64| {
                let (bl, bh) = (value(b, l)?, value(b, h)?);
                self.components
                    .iter()
                    .filter_map(|a| {
                        let (al, ah) = (value(a, l)?, value(a, h)?);
                        Some(gap(al, bl)?.max(gap(ah, bh)?))
                    })
                    .min_by(f64::total_cmp)
            };
            if knots.len() == 1 {
                worst = worst.max(covers(knots[0], knots[0])?);
                continue;
            }
            for w in knots.windows(2) {
                worst = worst.max(covers(w[0], w[1])?);
            }
        }
        Some(worst)
    }

    /// Compute the backward wrapper's key after a successful machine
    /// transition. This key is a machine quantity, not a source-domain
    /// objective bound. Mode 4 leaves the wrapper's sentinel in place.
    pub fn machine_backward_key(
        &self,
        mode: i32,
        terminal_ceiling: f64,
        budget: FrontierBudget,
    ) -> FrontierOutcome<f64> {
        let mut work = Work::new(budget);
        if self.components.len() != 1 {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine backward key requires one front",
            ));
        }
        if budget.max_components == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        let pairs = self.components[0].pairs();
        if pairs.len() > 12 {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine backward key pair count",
            ));
        }
        if pairs.len() > budget.max_pairs {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        if let Err(reason) = work.charge(1) {
            return FrontierOutcome::Unknown(reason);
        }
        if mode == 4 {
            return FrontierOutcome::Complete(1e12);
        }
        if !terminal_ceiling.is_finite() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine backward key ceiling",
            ));
        }
        let (mut value, mut coordinate) = pairs[0];
        let mut maximum = value;
        for &(slope, end) in &pairs[1..] {
            if let Err(reason) = work.charge(1) {
                return FrontierOutcome::Unknown(reason);
            }
            // The native key scan propagates IEEE arithmetic through a
            // retained empty NaN segment. Its ordered maximum comparison
            // keeps the preceding finite maximum when the next value is NaN.
            value += (end - coordinate) * slope;
            // MAXSD takes the preceding maximum on an equal comparison.
            if value > maximum {
                maximum = value;
            }
            coordinate = end;
        }
        if let Err(reason) = work.charge(0) {
            return FrontierOutcome::Unknown(reason);
        }
        match checked(
            terminal_ceiling - maximum,
            "machine backward key subtraction",
        ) {
            Ok(key) => FrontierOutcome::Complete(key),
            Err(error) => FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
        }
    }

    /// Reproduce the stored forward machine key for the recovered single
    /// internal-zero-width front after a successful extension. `mode` is the
    /// reference resource mode integer; mode 4 retains the sentinel. This
    /// value is diagnostic and is not a source-control bound or a witness.
    pub fn machine_internal_forward_key(
        &self,
        mode: i32,
        budget: FrontierBudget,
    ) -> FrontierOutcome<f64> {
        if mode != 4 {
            return match self.machine_internal_forward_minimum(budget) {
                FrontierOutcome::Complete(witness) => FrontierOutcome::Complete(witness.value),
                FrontierOutcome::Empty => FrontierOutcome::Empty,
                FrontierOutcome::Unknown(reason) => FrontierOutcome::Unknown(reason),
            };
        }
        let mut work = Work::new(budget);
        if self.components.len() != 1 || !self.components[0].has_internal_zero_length() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine forward key requires one internal zero-length segment",
            ));
        }
        if budget.max_components < 1 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        if budget.max_pairs < self.components[0].pairs().len() {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        if let Err(reason) = work.charge(1) {
            return FrontierOutcome::Unknown(reason);
        }
        FrontierOutcome::Complete(1e12)
    }

    /// Ordered endpoint minimum for the recovered single internal-zero-width
    /// machine front. The coordinate identifies the first scanned endpoint
    /// attaining the returned value; it is not a source-control witness, and
    /// endpoint-tolerant reevaluation may select a different value. This is
    /// diagnostic and is not a directed lower bound or route certificate.
    pub fn machine_internal_forward_minimum(
        &self,
        budget: FrontierBudget,
    ) -> FrontierOutcome<FrontAlgebraWitness> {
        let mut work = Work::new(budget);
        if self.components.len() != 1 || !self.components[0].has_internal_zero_length() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine forward minimum requires one internal zero-length segment",
            ));
        }
        if budget.max_components < 1 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        if budget.max_pairs < self.components[0].pairs().len() {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        if let Err(reason) = work.charge(1 + 3 * (self.components[0].pairs().len() - 1)) {
            return FrontierOutcome::Unknown(reason);
        }
        let pairs = self.components[0].pairs();
        let (mut value, mut coordinate) = pairs[0];
        let mut minimum = value;
        let mut minimum_coordinate = coordinate;
        for &(slope, end) in &pairs[1..] {
            value = match checked(end - coordinate, "machine key span")
                .and_then(|span| checked(span * slope, "machine key multiplication"))
                .and_then(|product| checked(value + product, "machine key accumulation"))
            {
                Ok(value) => value,
                Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
            };
            // MINSD uses the previous minimum as its second operand, so it
            // retains that value on equality, including signed-zero ties.
            if value < minimum {
                minimum = value;
                minimum_coordinate = end;
            }
            coordinate = end;
        }
        if let Err(reason) = work.charge(0) {
            return FrontierOutcome::Unknown(reason);
        }
        FrontierOutcome::Complete(FrontAlgebraWitness {
            coordinate: minimum_coordinate,
            value: minimum,
            component: 0,
            provenance: self.provenance.clone(),
        })
    }

    /// Minimum of the computed slope front, not of a separate source program.
    /// Ordered endpoint accumulation suffices for each monotone slope segment.
    pub fn minimum(&self, budget: FrontierBudget) -> FrontierOutcome<FrontAlgebraWitness> {
        if self.has_machine_zero_length() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine zero-length segment minimum",
            ));
        }
        let mut work = Work::new(budget);
        let mut best: Option<(f64, f64, usize)> = None;
        for (component, function) in self.components.iter().enumerate() {
            let (mut value, mut x) = function.pairs()[0];
            for (index, &(slope, end)) in function.pairs().iter().enumerate() {
                if let Err(reason) = work.charge(1) {
                    return FrontierOutcome::Unknown(reason);
                }
                if index > 0 {
                    let next = checked(end - x, "minimum span")
                        .and_then(|span| checked(slope * span, "minimum multiplication"))
                        .and_then(|product| checked(value + product, "minimum accumulation"));
                    match next {
                        Ok(next) => value = next,
                        Err(error) => {
                            return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error))
                        }
                    }
                    x = end;
                }
                if best.as_ref().map_or(true, |old| value < old.1) {
                    best = Some((x, value, component));
                }
            }
        }
        match best {
            Some((coordinate, value, component)) => {
                FrontierOutcome::Complete(FrontAlgebraWitness {
                    coordinate,
                    value,
                    component,
                    provenance: self.provenance.clone(),
                })
            }
            None => FrontierOutcome::Empty,
        }
    }
}
