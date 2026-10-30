//! Phase-local progress, distinct from globally published bounds.
pub(super) struct PhaseWalk {
    pub stage: usize,
    pub bound: f64,
    pub penalty_updates: usize,
    last: usize,
}

impl PhaseWalk {
    pub fn new(phases: usize) -> Self {
        let last = phases - 1;
        Self {
            stage: last,
            bound: -1e12,
            penalty_updates: 0,
            last,
        }
    }

    pub fn restart(&mut self) {
        self.stage = self.last;
        self.bound = -1e12;
        self.penalty_updates = 0;
    }

    pub fn advance(&mut self) {
        self.stage -= 1;
        self.bound = -1e12;
        self.penalty_updates = 0;
    }
}
