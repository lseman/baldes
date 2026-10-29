//! MachineForwardState operations.
use super::*;

impl MachineForwardState {
    /// Construct a machine state from a successful, already filtered waiting
    /// envelope. `reduced` retains the input front's machine-reduction history
    /// and `arc` identifies the transition. The caller must supply an envelope
    /// produced by that transition after its lower and upper filters succeed.
    /// This operation does not verify the envelope's association with the arc.
    /// A missing predecessor means the recurrence did not write that field;
    /// the modern wrapper initializes it to `1e12` before the recurrence.
    pub fn from_filtered_waiting(
        reduced: &SlopeFront,
        envelope: &MachineWaitingEnvelope,
        arc: Arc<crate::graph::coupled::descriptor::SlopeRelationArc>,
        mode: i32,
        budget: FrontierBudget,
    ) -> FrontierOutcome<Self> {
        let mut work = Work::new(budget);
        if !matches!(
            reduced.provenance().as_ref(),
            FrontierProvenance::MachineReduction { .. }
        ) {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine forward state requires reduction provenance",
            ));
        }
        let function = &envelope.function;
        let pairs = function.pairs();
        // The wrapper scans at most eleven segments in its twelve-pair
        // fixed array. Ordinary prepared functions have strictly increasing
        // coordinates; machine-only duplicates retain their narrower proven
        // representations. This does not validate the preceding filters.
        let ordinary = !function.is_machine_zero_length() && (2..=12).contains(&pairs.len());
        let duplicate = pairs.len() == 2 && function.is_machine_zero_length()
            || pairs.len() == 3 && function.has_internal_zero_length();
        if !ordinary && !duplicate {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "unrecovered machine forward state geometry",
            ));
        }
        if budget.max_components == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        if pairs.len() > budget.max_pairs {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        let valid_bits = ((1u64 << pairs.len()) - 1) & !1;
        if envelope.waiting_bits & !valid_bits != 0
            || envelope.predecessor.is_some_and(|value| !value.is_finite())
            || pairs.iter().any(|&(a, b)| !a.is_finite() || !b.is_finite())
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "invalid machine forward state fields",
            ));
        }
        let operations = if mode == 4 {
            1
        } else {
            1 + 3 * (pairs.len() - 1)
        };
        if let Err(reason) = work.charge(operations) {
            return FrontierOutcome::Unknown(reason);
        }
        let key = if mode == 4 {
            1e12
        } else {
            let (mut value, mut coordinate) = pairs[0];
            let mut minimum = value;
            for &(slope, endpoint) in &pairs[1..] {
                value = match checked(endpoint - coordinate, "machine key span")
                    .and_then(|span| checked(span * slope, "machine key multiplication"))
                    .and_then(|product| checked(value + product, "machine key accumulation"))
                {
                    Ok(value) => value,
                    Err(error) => {
                        return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error))
                    }
                };
                // MINSD selects the old minimum on equality, including zeros.
                if value < minimum {
                    minimum = value;
                }
                coordinate = endpoint;
            }
            minimum
        };
        if let Err(reason) = work.charge(0) {
            return FrontierOutcome::Unknown(reason);
        }
        let front = SlopeFront {
            components: vec![function.clone()].into(),
            provenance: Arc::new(FrontierProvenance::PairedMachineWaiting {
                input: reduced.provenance().clone(),
                arc,
                predecessor: envelope.predecessor,
                waiting_bits: envelope.waiting_bits,
            }),
            tolerant_alignment: true,
        };
        FrontierOutcome::Complete(Self {
            front,
            waiting_bits: envelope.waiting_bits,
            predecessor: envelope.predecessor,
            key,
        })
    }

    /// Resulting machine front, retaining the reduction history.
    pub fn front(&self) -> &SlopeFront {
        &self.front
    }
    /// Reference waiting provenance bitset.
    pub fn waiting_bits(&self) -> u64 {
        self.waiting_bits
    }
    /// The predecessor written by the recurrence, if any.
    pub fn predecessor(&self) -> Option<f64> {
        self.predecessor
    }
    /// Raw machine state key, without objective normalization.
    pub fn key(&self) -> f64 {
        self.key
    }

    /// Recover `(waiting, carried_coordinate)` for a strictly interior point
    /// of a finite ordinary multi-segment machine front, away from every knot's
    /// machine tolerance band. The carried coordinate is the value passed to
    /// the later arc's inverse; it is not reconstructed by subtracting the
    /// rounded waiting amount. This is not a source or route certificate.
    pub fn machine_multisegment_waiting_at(
        &self,
        coordinate: f64,
        budget: FrontierBudget,
    ) -> FrontierOutcome<(f64, f64)> {
        let mut work = Work::new(budget);
        if budget.max_components == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        if self.front.components.len() != 1 || !coordinate.is_finite() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine multisegment waiting inputs",
            ));
        }
        let function = &self.front.components[0];
        let pairs = function.pairs();
        if function.is_machine_zero_length() || !(3..=12).contains(&pairs.len()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine multisegment waiting geometry",
            ));
        }
        if pairs.len() > budget.max_pairs {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        let Some(operations) = pairs.len().checked_mul(8).and_then(|n| n.checked_add(8)) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        if let Err(reason) = work.charge(operations) {
            return FrontierOutcome::Unknown(reason);
        }
        if coordinate <= pairs[0].1 || coordinate >= pairs[pairs.len() - 1].1 {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine multisegment waiting coordinate",
            ));
        }
        for &(_, knot) in pairs {
            match near(coordinate, knot, FrontOperationPolicy::FrontAlgebra) {
                Ok(false) => {}
                Ok(true) => {
                    return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                        "machine multisegment waiting knot alignment",
                    ));
                }
                Err(reason) => return FrontierOutcome::Unknown(reason),
            }
        }
        let valid_bits = ((1u64 << pairs.len()) - 1) & !1;
        if self.waiting_bits & !valid_bits != 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine multisegment waiting bits",
            ));
        }
        let segment = pairs.partition_point(|pair| pair.1 < coordinate);
        // The strict interior and not-near guards imply 1 <= segment < len.
        match function.evaluate(coordinate, EvaluationPolicy::EndpointTolerance) {
            Ok(Some(value)) if value != 1e12 => {}
            Ok(_) => {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine multisegment waiting lookup sentinel",
                ));
            }
            Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
        }
        let (waiting, carried) = if self.waiting_bits & (1u64 << segment) == 0 {
            (0.0, coordinate)
        } else {
            let preceding = (1..segment)
                .rev()
                .find(|&index| self.waiting_bits & (1u64 << index) == 0);
            let predecessor = match preceding {
                Some(index) => pairs[index].1,
                None => match self.predecessor {
                    Some(value) => value,
                    None => {
                        return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                            "machine multisegment predecessor unavailable",
                        ));
                    }
                },
            };
            if !predecessor.is_finite() || predecessor == 1e12 || predecessor < 0.0 {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine multisegment predecessor rejected",
                ));
            }
            let waiting = match checked(coordinate - predecessor, "machine waiting subtraction") {
                Ok(value) => value,
                Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
            };
            (waiting, predecessor)
        };
        if let Err(reason) = work.charge(0) {
            return FrontierOutcome::Unknown(reason);
        }
        FrontierOutcome::Complete((waiting, carried))
    }

    /// Recover waiting for the bounded internal-duplicate front with bit 1
    /// clear and bit 2 set. This is a machine result, not a source witness.
    /// Near the first knot takes precedence even when it is also near the end.
    pub fn machine_internal_duplicate_waiting_at(
        &self,
        coordinate: f64,
        budget: FrontierBudget,
    ) -> FrontierOutcome<f64> {
        let mut work = Work::new(budget);
        if budget.max_components == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        if budget.max_pairs < 3 {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        if self.front.components.len() != 1 || !coordinate.is_finite() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine duplicate waiting reconstruction inputs",
            ));
        }
        let function = &self.front.components[0];
        let pairs = function.pairs();
        if !function.has_internal_zero_length() || pairs.len() != 3 || self.waiting_bits != 1 << 2 {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine duplicate waiting reconstruction geometry",
            ));
        }
        if let Err(reason) = work.charge(10) {
            return FrontierOutcome::Unknown(reason);
        }
        let (anchor, first) = pairs[0];
        let (slope, duplicate) = pairs[1];
        let (waiting_slope, end) = pairs[2];
        let first_near = match near(coordinate, first, FrontOperationPolicy::FrontAlgebra) {
            Ok(value) => value,
            Err(reason) => return FrontierOutcome::Unknown(reason),
        };
        let (value, waiting) = if first_near {
            (anchor, 0.0)
        } else {
            if coordinate <= first {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine duplicate waiting coordinate below start",
                ));
            }
            if duplicate == 1e12 || duplicate < 0.0 {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine duplicate waiting predecessor rejected",
                ));
            }
            let endpoint = if coordinate > end {
                match near(coordinate, end, FrontOperationPolicy::FrontAlgebra) {
                    Ok(true) => end,
                    Ok(false) => {
                        return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                            "machine duplicate waiting coordinate above end",
                        ));
                    }
                    Err(reason) => return FrontierOutcome::Unknown(reason),
                }
            } else {
                coordinate
            };
            let evaluated = checked(duplicate - first, "machine duplicate waiting span")
                .and_then(|span| checked(span * slope, "machine duplicate waiting multiplication"))
                .and_then(|product| {
                    checked(anchor + product, "machine duplicate waiting accumulation")
                })
                .and_then(|value| {
                    checked(endpoint - duplicate, "machine duplicate waiting span")
                        .and_then(|span| {
                            checked(
                                span * waiting_slope,
                                "machine duplicate waiting multiplication",
                            )
                        })
                        .and_then(|product| {
                            checked(value + product, "machine duplicate waiting accumulation")
                        })
                });
            let value = match evaluated {
                Ok(value) => value,
                Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
            };
            let waiting = match checked(
                coordinate - duplicate,
                "machine duplicate waiting subtraction",
            ) {
                Ok(value) => value,
                Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
            };
            (value, waiting)
        };
        if value == 1e12 {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine duplicate waiting lookup sentinel",
            ));
        }
        if let Err(reason) = work.charge(0) {
            return FrontierOutcome::Unknown(reason);
        }
        FrontierOutcome::Complete(waiting)
    }

    /// Return the directly selected predecessor for the bounded duplicate
    /// waiting case, without recovering it by subtracting rounded waiting.
    pub fn machine_internal_duplicate_predecessor_at(
        &self,
        coordinate: f64,
        budget: FrontierBudget,
    ) -> FrontierOutcome<f64> {
        let Some(remaining) = budget.max_operations.checked_sub(1) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        let started = Instant::now();
        match self.machine_internal_duplicate_waiting_at(
            coordinate,
            FrontierBudget {
                max_operations: remaining,
                ..budget
            },
        ) {
            FrontierOutcome::Complete(_) => {}
            FrontierOutcome::Empty => return FrontierOutcome::Empty,
            FrontierOutcome::Unknown(reason) => return FrontierOutcome::Unknown(reason),
        }
        if budget
            .max_elapsed
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        let pairs = self.front.components[0].pairs();
        match near(coordinate, pairs[0].1, FrontOperationPolicy::FrontAlgebra) {
            Ok(true) => FrontierOutcome::Complete(coordinate),
            Ok(false) => FrontierOutcome::Complete(pairs[1].1),
            Err(reason) => FrontierOutcome::Unknown(reason),
        }
    }

    /// Recover waiting for a coordinate selected within a finite nonzero-width
    /// one-segment machine front. This is not a source-control witness.
    pub fn machine_single_segment_waiting_at(
        &self,
        coordinate: f64,
        budget: FrontierBudget,
    ) -> FrontierOutcome<f64> {
        let mut work = Work::new(budget);
        if budget.max_components == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        if budget.max_pairs < 2 {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        if self.front.components.len() != 1 || !coordinate.is_finite() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine waiting reconstruction inputs",
            ));
        }
        let function = &self.front.components[0];
        let pairs = function.pairs();
        if function.is_machine_zero_length()
            || pairs.len() != 2
            || coordinate < pairs[0].1
            || coordinate > pairs[1].1
            || self.waiting_bits & !(1 << 1) != 0
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine waiting reconstruction geometry",
            ));
        }
        if let Err(reason) = work.charge(5) {
            return FrontierOutcome::Unknown(reason);
        }
        // The caller's lookup rejects its 1e12 sentinel before using the bit.
        match function.evaluate(coordinate, EvaluationPolicy::EndpointTolerance) {
            Ok(Some(value)) if value != 1e12 => {}
            Ok(_) => {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine waiting reconstruction lookup sentinel",
                ));
            }
            Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
        }
        let waiting = if self.waiting_bits & (1 << 1) == 0 {
            0.0
        } else {
            let Some(predecessor) = self.predecessor else {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine waiting reconstruction predecessor unavailable",
                ));
            };
            if predecessor == 1e12 || predecessor < 0.0 {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine waiting reconstruction predecessor rejected",
                ));
            }
            match checked(
                coordinate - predecessor,
                "machine waiting reconstruction subtraction",
            ) {
                Ok(value) => value,
                Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
            }
        };
        if let Err(reason) = work.charge(0) {
            return FrontierOutcome::Unknown(reason);
        }
        FrontierOutcome::Complete(waiting)
    }

    /// Return the coordinate passed to the current arc's inverse while
    /// reconstructing a finite one-segment machine state. When waiting bit 1
    /// is clear this is the selected coordinate; when it is set the reference
    /// carries the stored predecessor directly. In particular, do not recover
    /// it as `coordinate - waiting`: that would introduce another rounded
    /// subtraction. The caller must still evaluate the correct arc inverse.
    /// This operation is not a source-control witness or route certificate.
    pub fn machine_single_segment_predecessor_at(
        &self,
        coordinate: f64,
        budget: FrontierBudget,
    ) -> FrontierOutcome<f64> {
        match self.machine_single_segment_waiting_at(coordinate, budget) {
            FrontierOutcome::Complete(_) => {
                if self.waiting_bits & (1 << 1) == 0 {
                    FrontierOutcome::Complete(coordinate)
                } else {
                    match self.predecessor {
                        Some(predecessor) => FrontierOutcome::Complete(predecessor),
                        None => FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                            "machine waiting reconstruction predecessor unavailable",
                        )),
                    }
                }
            }
            FrontierOutcome::Empty => FrontierOutcome::Empty,
            FrontierOutcome::Unknown(reason) => FrontierOutcome::Unknown(reason),
        }
    }
}
