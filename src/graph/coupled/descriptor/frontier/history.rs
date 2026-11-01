//! MachinePathHistory operations.
use super::*;

impl MachinePathHistory {
    /// Split only the latest pricing snapshot, preserving its physical arc,
    /// predecessor history and root identity. No physical transition is added.
    pub(crate) fn split_last(
        &self,
        strategy: i32,
        halfway: f64,
        mode: i32,
        bucket_ends: &[f64],
        bucket_grid: [f64; 3],
        budget: FrontierBudget,
    ) -> FrontierOutcome<Vec<Self>> {
        if !matches!(strategy, 1 | 2) {
            return FrontierOutcome::Complete(vec![self.clone()]);
        }
        let (forward, bits, predecessor, key, ceiling) = match self.last() {
            MachinePathSnapshot::Root(_) => return FrontierOutcome::Complete(vec![self.clone()]),
            MachinePathSnapshot::Forward { state, .. } => (
                true,
                state.waiting_bits,
                state.predecessor.unwrap_or(1e12),
                state.key,
                0.,
            ),
            MachinePathSnapshot::Backward {
                front,
                waiting_bits,
                key,
                ceiling,
                ..
            } => {
                let predecessor = match front.provenance.as_ref() {
                    FrontierProvenance::MachineSplit { predecessor, .. } => *predecessor,
                    _ => 1e12,
                };
                (false, *waiting_bits, predecessor, *key, *ceiling)
            }
        };
        let front = self.last().front();
        let [function] = front.components.as_ref() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "paired split requires one fixed-array front",
            ));
        };
        let input = super::split::SplitState {
            pairs: function.pairs().to_vec(),
            waiting_bits: bits,
            predecessor,
            key,
        };
        let pieces = match super::split::split_state(
            &input,
            forward,
            strategy,
            halfway,
            mode,
            ceiling,
            bucket_ends,
            bucket_grid,
            budget,
        ) {
            Ok(pieces) => pieces,
            Err(reason) => return FrontierOutcome::Unknown(reason),
        };
        if pieces.len() <= 1 {
            return FrontierOutcome::Complete(vec![self.clone()]);
        }
        let mut output = Vec::with_capacity(pieces.len());
        for piece in pieces {
            let function = match PreparedSlopeFunction::from_machine_pairs(piece.pairs) {
                Ok(function) => function,
                Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
            };
            let split_front = SlopeFront {
                components: vec![function].into(),
                provenance: Arc::new(FrontierProvenance::MachineSplit {
                    input: front.provenance.clone(),
                    forward,
                    strategy,
                    halfway,
                    waiting_bits: piece.waiting_bits,
                    predecessor: piece.predecessor,
                }),
                tolerant_alignment: true,
            };
            let snapshot = match self.last() {
                MachinePathSnapshot::Forward { arc, .. } => MachinePathSnapshot::Forward {
                    arc: *arc,
                    state: MachineForwardState {
                        front: split_front,
                        waiting_bits: piece.waiting_bits,
                        predecessor: Some(piece.predecessor),
                        key: piece.key,
                    },
                },
                MachinePathSnapshot::Backward { arc, ceiling, .. } => {
                    MachinePathSnapshot::Backward {
                        arc: *arc,
                        front: split_front,
                        waiting_bits: piece.waiting_bits,
                        key: piece.key,
                        ceiling: *ceiling,
                    }
                }
                MachinePathSnapshot::Root(_) => unreachable!(),
            };
            output.push(Self {
                head: Arc::new(MachinePathNode {
                    previous: self.head.previous.clone(),
                    snapshot,
                }),
                root: self.root.clone(),
                len: self.len,
                max_states: self.max_states,
            });
        }
        FrontierOutcome::Complete(output)
    }
    /// Apply the full backward paired-resource machine dominance wrapper.
    pub fn backward_dominance(
        &self,
        other: &Self,
        mode: i32,
        accumulated: f64,
        budget: FrontierBudget,
    ) -> FrontierOutcome<f64> {
        if self.backward_ceiling().is_none() || other.backward_ceiling().is_none() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "backward dominance requires backward histories",
            ));
        }
        let left = self.last().front().components();
        let right = other.last().front().components();
        if left.len() != 1 || right.len() != 1 {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "backward dominance requires single machine fronts",
            ));
        }
        PairedFrontOperations::machine_backward_dominance(
            &left[0],
            &right[0],
            mode,
            accumulated,
            budget,
        )
    }

    /// Apply the full forward paired-resource machine dominance wrapper.
    pub fn forward_dominance(
        &self,
        other: &Self,
        mode: i32,
        accumulated: f64,
        budget: FrontierBudget,
    ) -> FrontierOutcome<f64> {
        if self.backward_ceiling().is_some() || other.backward_ceiling().is_some() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "forward dominance requires forward histories",
            ));
        }
        let left = self.last().front().components();
        let right = other.last().front().components();
        if left.len() != 1 || right.len() != 1 {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "forward dominance requires single machine fronts",
            ));
        }
        PairedFrontOperations::machine_forward_dominance(
            &left[0],
            &right[0],
            mode,
            accumulated,
            budget,
        )
    }

    /// Compare retained forward feasibility fronts with the ordered machine
    /// scan. This result is not a source-domain dominance certificate.
    pub fn forward_feasibility_comparison(
        &self,
        other: &Self,
        budget: FrontierBudget,
    ) -> FrontierOutcome<()> {
        if self.backward_ceiling().is_some() || other.backward_ceiling().is_some() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "forward comparison requires forward histories",
            ));
        }
        let left = self.last().front().components();
        let right = other.last().front().components();
        if left.len() != 1 || right.len() != 1 {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "forward comparison requires single machine fronts",
            ));
        }
        PairedFrontOperations::machine_forward_feasibility_dominance(&left[0], &right[0], budget)
    }

    /// Compare forward cost fronts and update the raw machine accumulator.
    pub fn forward_cost_comparison(
        &self,
        other: &Self,
        mode: i32,
        accumulated: f64,
        budget: FrontierBudget,
    ) -> FrontierOutcome<f64> {
        if self.backward_ceiling().is_some() || other.backward_ceiling().is_some() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "forward comparison requires forward histories",
            ));
        }
        let left = self.last().front().components();
        let right = other.last().front().components();
        if left.len() != 1 || right.len() != 1 {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "forward comparison requires single machine fronts",
            ));
        }
        PairedFrontOperations::machine_forward_cost_dominance(
            &left[0],
            &right[0],
            mode,
            accumulated,
            budget,
        )
    }

    /// Compare retained backward feasibility fronts under machine tolerance.
    /// Acceptance is not strict source-domain containment.
    pub fn backward_feasibility_comparison(
        &self,
        other: &Self,
        budget: FrontierBudget,
    ) -> FrontierOutcome<()> {
        if self.backward_ceiling().is_none() || other.backward_ceiling().is_none() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "backward comparison requires backward histories",
            ));
        }
        let left = self.last().front().components();
        let right = other.last().front().components();
        if left.len() != 1 || right.len() != 1 {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "backward comparison requires single machine fronts",
            ));
        }
        PairedFrontOperations::machine_backward_feasibility_dominance(&left[0], &right[0], budget)
    }

    /// Compare the retained backward fronts using the ordered machine scan.
    /// Returns the signed updated accumulator, rejection, or unknown. This
    /// does not certify source feasibility or normalized route-cost order.
    pub fn backward_cost_comparison(
        &self,
        other: &Self,
        mode: i32,
        accumulated: f64,
        budget: FrontierBudget,
    ) -> FrontierOutcome<f64> {
        if self.backward_ceiling().is_none() || other.backward_ceiling().is_none() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "backward comparison requires backward histories",
            ));
        }
        let left = self.last().front().components();
        let right = other.last().front().components();
        if left.len() != 1 || right.len() != 1 {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "backward comparison requires single machine fronts",
            ));
        }
        PairedFrontOperations::machine_backward_cost_dominance(
            &left[0],
            &right[0],
            mode,
            accumulated,
            budget,
        )
    }

    /// Immutable root identity, used to reject histories from replaced arc
    /// inputs or a different installed terminal root.
    pub(crate) fn shares_root(&self, root: &Self) -> bool {
        if root.len != 1 {
            return false;
        }
        Arc::ptr_eq(&self.root, &root.root)
    }

    /// Terminal ceiling retained by a backward machine history.
    pub(crate) fn backward_ceiling(&self) -> Option<f64> {
        match &self.head.snapshot {
            MachinePathSnapshot::Root(root)
                if root.predecessor().to_bits() == 1e12f64.to_bits() =>
            {
                Some(root.key())
            }
            MachinePathSnapshot::Root(_) => None,
            MachinePathSnapshot::Backward { ceiling, .. } => Some(*ceiling),
            MachinePathSnapshot::Forward { .. } => None,
        }
    }

    /// Append a machine-only backward transition. This internal operation
    /// carries no source-domain feasibility or objective certificate.
    pub(crate) fn append_backward(
        &self,
        arc: usize,
        front: PreparedSlopeFunction,
        waiting_bits: u64,
        key: f64,
        budget: FrontierBudget,
    ) -> FrontierOutcome<Self> {
        let Some(ceiling) = self.backward_ceiling() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "forward history cannot append backward state",
            ));
        };
        let Some(next_len) = self.len.checked_add(1) else {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        };
        if next_len > self.max_states || budget.max_components == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        if front.pairs().len() > budget.max_pairs {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        let mut work = Work::new(budget);
        if let Err(reason) = work.charge(front.pairs().len().saturating_add(1)) {
            return FrontierOutcome::Unknown(reason);
        }
        if !key.is_finite() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "nonfinite backward machine key",
            ));
        }
        FrontierOutcome::Complete(Self {
            head: Arc::new(MachinePathNode {
                previous: Some(self.head.clone()),
                snapshot: MachinePathSnapshot::Backward {
                    arc,
                    front: SlopeFront::from_prepared(front),
                    waiting_bits,
                    key,
                    ceiling,
                },
            }),
            root: self.root.clone(),
            len: next_len,
            max_states: self.max_states,
        })
    }
    /// Start with an unreduced root. The explicit limit includes the root.
    pub fn from_root(
        root: &MachineRootState,
        max_states: usize,
        budget: FrontierBudget,
    ) -> FrontierOutcome<Self> {
        if max_states == 0 || budget.max_components == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        let pairs = root
            .front
            .components
            .iter()
            .map(|f| f.pairs().len())
            .sum::<usize>();
        if pairs > budget.max_pairs {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        let mut work = Work::new(budget);
        if let Err(reason) = work.charge(pairs.saturating_add(1)) {
            return FrontierOutcome::Unknown(reason);
        }
        let root = Arc::new(MachinePathNode {
            previous: None,
            snapshot: MachinePathSnapshot::Root(root.clone()),
        });
        FrontierOutcome::Complete(Self {
            head: root.clone(),
            root,
            len: 1,
            max_states,
        })
    }

    /// Append a successful transition. The reduction must come from the
    /// latest snapshot and the new state must reference that reduction. Arc
    /// identity is caller supplied and is not independently certified.
    pub fn append(
        &self,
        arc: usize,
        reduced: &SlopeFront,
        state: &MachineForwardState,
        budget: FrontierBudget,
    ) -> FrontierOutcome<Self> {
        let Some(next_len) = self.len.checked_add(1) else {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        };
        if next_len > self.max_states || budget.max_components == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        let previous = self.head.snapshot.front();
        match reduced.provenance.as_ref() {
            FrontierProvenance::MachineReduction {
                input, input_front, ..
            } if Arc::ptr_eq(input, &previous.provenance)
                && Arc::ptr_eq(input_front, &previous.components) => {}
            _ => {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine history reduction does not match prior state",
                ))
            }
        }
        match state.front.provenance.as_ref() {
            FrontierProvenance::PairedMachineWaiting { input, .. }
                if Arc::ptr_eq(input, &reduced.provenance) => {}
            _ => {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine history state does not match reduction",
                ))
            }
        }
        let pairs = state
            .front
            .components
            .iter()
            .map(|f| f.pairs().len())
            .sum::<usize>();
        if pairs > budget.max_pairs {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        let mut work = Work::new(budget);
        if let Err(reason) = work.charge(pairs.saturating_add(1)) {
            return FrontierOutcome::Unknown(reason);
        }
        FrontierOutcome::Complete(Self {
            head: Arc::new(MachinePathNode {
                previous: Some(self.head.clone()),
                snapshot: MachinePathSnapshot::Forward {
                    arc,
                    state: state.clone(),
                },
            }),
            root: self.root.clone(),
            len: next_len,
            max_states: self.max_states,
        })
    }

    /// Number of stored snapshots, including the root.
    pub fn len(&self) -> usize {
        self.len
    }
    /// Histories always contain their root.
    pub fn is_empty(&self) -> bool {
        false
    }
    /// Latest snapshot, before any subsequent reduction.
    pub fn last(&self) -> &MachinePathSnapshot {
        &self.head.snapshot
    }

    /// Materialize borrowed snapshots in root-to-last order. The output
    /// component and work limits apply to the whole traversal.
    pub fn snapshots(&self, budget: FrontierBudget) -> FrontierOutcome<Vec<&MachinePathSnapshot>> {
        if self.len > budget.max_components {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        let mut work = Work::new(budget);
        let mut result = Vec::with_capacity(self.len);
        let mut node = Some(self.head.as_ref());
        while let Some(current) = node {
            if let Err(reason) = work.charge(1) {
                return FrontierOutcome::Unknown(reason);
            }
            result.push(&current.snapshot);
            node = current.previous.as_deref();
        }
        result.reverse();
        if let Err(reason) = work.charge(0) {
            return FrontierOutcome::Unknown(reason);
        }
        FrontierOutcome::Complete(result)
    }
}
