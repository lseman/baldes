//! Heuristic branching work budgets. None of these estimates are proof bounds.

#[derive(Clone, Copy, Debug)]
pub(in crate::master::tree) struct SubtreeEstimate {
    pub size: f64,
    pub priority: f64,
}
impl Default for SubtreeEstimate {
    fn default() -> Self {
        Self {
            size: 1e12,
            priority: 1e12,
        }
    }
}
pub(in crate::master::tree) fn exceeds(a: f64, b: f64) -> bool {
    a - (1e-6 + a.abs().max(b.abs()) * 1e-10) > b
}

pub(in crate::master::tree) struct CandidateBudget {
    maximum: usize,
    pub quota: usize,
    pub estimate: SubtreeEstimate,
    from_history: bool,
}
impl CandidateBudget {
    pub fn new(maximum: usize, ratio: f64, estimate: SubtreeEstimate) -> Self {
        let quota = if ratio > 0. {
            // The native caller caps its floating-point ceiling at 10^6
            // before converting it to an integer.
            maximum.min((ratio * estimate.size).ceil().clamp(0., 1e6) as usize)
        } else {
            maximum.min(1_000_000)
        };
        Self {
            maximum,
            quota,
            estimate,
            from_history: false,
        }
    }
    pub fn use_history(&mut self) {
        self.from_history = true;
    }
    pub fn visit(&mut self, priority: f64) {
        if !self.from_history && exceeds(self.estimate.priority, priority) {
            self.quota = self.maximum;
            self.estimate.size = 1e12;
        }
    }
    pub fn permits(&self, evaluated: usize, ratio: f64, best_size: f64) -> bool {
        if ratio <= 0. {
            return true;
        }
        let size = if exceeds(self.estimate.size, best_size) {
            best_size
        } else {
            self.estimate.size
        };
        !exceeds(evaluated as f64, ratio * size)
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/tree/strong/budget/tests.rs"]
mod tests;
