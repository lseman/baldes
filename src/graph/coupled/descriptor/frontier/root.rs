//! MachineRootState operations.
use super::*;

impl MachineRootState {
    /// Prepare the backward machine root from already prepared upper vectors
    /// of selected sink arcs and explicitly supplied terminal bounds. This
    /// does not verify arc selection, source feasibility, or the reference
    /// initializer's append destination.
    pub fn from_backward_inputs(
        upper_vectors: &[&PreparedSlopeFunction],
        lower: f64,
        upper: f64,
        budget: FrontierBudget,
    ) -> FrontierOutcome<Self> {
        let started = Instant::now();
        if budget.max_components == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        if budget.max_pairs < 2 {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        // The initializer writes the terminal endpoints literally; source
        // window validation remains the caller's separate responsibility.
        if !lower.is_finite() || !upper.is_finite() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine backward root bounds",
            ));
        }
        let mut work = Work::new(budget);
        let mut ceiling = -1e12;
        for function in upper_vectors {
            let pairs = function.pairs();
            if pairs.is_empty() || pairs.len() > 12 {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine backward root upper vector",
                ));
            }
            if pairs.len() > budget.max_pairs {
                return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
            }
            if let Err(reason) = work.charge(pairs.len().saturating_mul(4)) {
                return FrontierOutcome::Unknown(reason);
            }
            let (mut value, mut coordinate) = pairs[0];
            let mut maximum = value;
            for &(slope, end) in &pairs[1..] {
                value = match checked(end - coordinate, "backward root span")
                    .and_then(|span| checked(span * slope, "backward root multiplication"))
                    .and_then(|product| checked(value + product, "backward root accumulation"))
                {
                    Ok(value) => value,
                    Err(error) => {
                        return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error))
                    }
                };
                if value > maximum {
                    maximum = value;
                }
                coordinate = end;
            }
            if maximum > ceiling {
                ceiling = maximum;
            }
        }
        let vector =
            match PreparedSlopeFunction::from_machine_pairs(vec![(ceiling, lower), (0., upper)]) {
                Ok(vector) => vector,
                Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
            };
        let remaining_work = match budget.max_operations.checked_sub(work.operations) {
            Some(remaining) => remaining,
            None => return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit),
        };
        let remaining_time = budget
            .max_elapsed
            .map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining_time.is_some_and(|limit| limit.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        match Self::from_prepared(
            &vector,
            FrontierBudget {
                max_operations: remaining_work,
                max_elapsed: remaining_time,
                ..budget
            },
        ) {
            FrontierOutcome::Complete(mut state) => {
                state.waiting_bits = 0;
                state.predecessor = 1e12;
                state.key = ceiling;
                FrontierOutcome::Complete(state)
            }
            FrontierOutcome::Empty => FrontierOutcome::Empty,
            FrontierOutcome::Unknown(reason) => FrontierOutcome::Unknown(reason),
        }
    }

    /// Prepare an explicit machine root input in an initially empty output.
    ///
    /// The supplied bounds and input correspond to the reference initializer's
    /// arguments; their graph origin, the subsequent domain check, and source
    /// feasibility are not established by this operation. It returns unknown
    /// for unsupported truncation and arithmetic, not source infeasibility.
    /// The initializer's failure-side effects for a nonempty output are not
    /// modeled. Work and elapsed limits are shared by preparation and copying.
    pub fn from_input(
        input: MachineRootInput<'_>,
        lower: f64,
        upper: f64,
        budget: FrontierBudget,
    ) -> FrontierOutcome<Self> {
        let started = Instant::now();
        if budget.max_components == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        let (count, work) = match input {
            MachineRootInput::Affine(_) => (2, Some(8usize)),
            MachineRootInput::Vector(vector) => {
                let count = vector.pairs().len();
                if !(2..=11).contains(&count) {
                    return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                        "machine root source vector pair count",
                    ));
                }
                (
                    count,
                    count.checked_mul(16).and_then(|work| work.checked_add(32)),
                )
            }
        };
        if count > budget.max_pairs {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        let Some(work) = work else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        let Some(remaining_work) = budget.max_operations.checked_sub(work) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        if budget
            .max_elapsed
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        let prepared = match input {
            MachineRootInput::Affine(affine) => affine.prepare_machine(lower, upper),
            MachineRootInput::Vector(vector) => vector.machine_truncate(lower, upper),
        };
        let prepared = match prepared {
            Ok(prepared) => prepared,
            Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
        };
        let remaining_time = budget
            .max_elapsed
            .map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining_time.is_some_and(|time| time.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        Self::from_prepared(
            &prepared,
            FrontierBudget {
                max_operations: remaining_work,
                max_elapsed: remaining_time,
                ..budget
            },
        )
    }

    /// Copy one finite prepared vector into the bounded machine root state.
    /// This does not construct the reference's upstream initial vector.
    pub fn from_prepared(
        input: &PreparedSlopeFunction,
        budget: FrontierBudget,
    ) -> FrontierOutcome<Self> {
        let mut work = Work::new(budget);
        let count = input.pairs().len();
        if count == 0 || count > 12 {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine root pair count",
            ));
        }
        if budget.max_components == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        if count > budget.max_pairs {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        if let Err(reason) = work.charge(count) {
            return FrontierOutcome::Unknown(reason);
        }
        if input
            .pairs()
            .iter()
            .any(|&(a, b)| !a.is_finite() || !b.is_finite())
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine root nonfinite pair",
            ));
        }
        let input = Arc::new(input.clone());
        let segments = count - 1;
        let waiting_bits = ((1u64 << (segments + 1)) - 1) & !1;
        let front = SlopeFront {
            components: vec![input.as_ref().clone()].into(),
            provenance: Arc::new(FrontierProvenance::MachineRoot { input }),
            tolerant_alignment: false,
        };
        if let Err(reason) = work.charge(0) {
            return FrontierOutcome::Unknown(reason);
        }
        FrontierOutcome::Complete(Self {
            front,
            waiting_bits,
            predecessor: 0.0,
            key: 0.0,
        })
    }

    /// Copied machine front and its machine-only provenance.
    pub fn front(&self) -> &SlopeFront {
        &self.front
    }
    /// Forward roots set bits 1 through the segment count; backward roots
    /// start with all waiting bits clear.
    pub fn waiting_bits(&self) -> u64 {
        self.waiting_bits
    }
    /// Initial predecessor: positive zero forward, the sentinel backward.
    pub fn predecessor(&self) -> f64 {
        self.predecessor
    }
    /// Initial key: positive zero forward, the terminal ceiling backward.
    pub fn key(&self) -> f64 {
        self.key
    }
}
