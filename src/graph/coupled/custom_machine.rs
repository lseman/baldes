//! Ordered custom-resource callback arithmetic.

/// Raw state of the custom resource callbacks. Its values do not certify a
/// source-model objective or dominance relation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CustomMachineState {
    pub cost: f64,
    pub value: f64,
}

/// Callback state and whether every arc callback on the prefix or suffix
/// avoided the machine wrapper's rejection sentinel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CustomMachineTrace {
    pub state: CustomMachineState,
    pub accepted: bool,
    /// The label wrapper's accumulator, separate from the callback state cost.
    pub wrapper_cost: f64,
}

impl Default for CustomMachineTrace {
    fn default() -> Self {
        Self {
            state: CustomMachineState::default(),
            accepted: true,
            wrapper_cost: 0.,
        }
    }
}

impl CustomMachineState {
    /// Apply the forward arc callback and return its literal contribution.
    pub fn extend_forward(&mut self, increment: f64, weight: f64) -> f64 {
        self.cost = (increment + self.value) * weight + self.cost;
        self.value += increment;
        self.value * weight
    }

    /// Apply the backward arc callback and return its literal contribution.
    pub fn extend_backward(&mut self, increment: f64, weight: f64) -> f64 {
        self.cost = (weight + self.value) * increment + self.cost;
        self.value += weight;
        self.value * increment
    }

    /// The vertex callback contributes zero and does not change state.
    pub fn extend_vertex(&mut self) -> f64 {
        0.
    }

    /// Cross term for the ordinary forward/backward callback.
    pub fn concatenate(&self, backward: &Self) -> f64 {
        self.value * backward.value
    }

    /// Combine intrinsic callback costs in the wrapper's addition order.
    /// Label wrapper accumulators can differ and are retained separately.
    pub fn concatenate_cost(&self, backward: &Self, previous: f64) -> Option<f64> {
        let cross = self.concatenate(backward);
        Self::contribution_accepted(cross).then(|| ((cross + self.cost) + backward.cost) + previous)
    }

    /// The forward/forward overload contributes zero.
    pub fn concatenate_forward(&self, _other: &Self) -> f64 {
        0.
    }

    /// Apply the forward/forward resource wrapper's ordered accumulation.
    pub fn concatenate_forward_cost(&self, other: &Self, previous: f64) -> f64 {
        ((self.concatenate_forward(other) + self.cost) + other.cost) + previous
    }

    /// The resource's symmetry callback is always false.
    pub fn symmetric() -> bool {
        false
    }

    /// Cost returned by either directional dominance callback, using that
    /// direction's supplied constant parameter.
    pub fn dominance_cost(&self, other: &Self, bound: f64) -> f64 {
        let difference = self.value - other.value;
        let positive = if difference <= 0. { 0. } else { difference };
        (self.cost - other.cost) + positive * bound
    }

    /// The callback's sentinel marks a rejected arc or concatenation in the
    /// native wrapper. The callback has already updated its state on return.
    pub fn contribution_accepted(contribution: f64) -> bool {
        contribution != 1e12
    }
}
