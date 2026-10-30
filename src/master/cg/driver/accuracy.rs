/// Munari--Gondzio, Algorithm 1: bounds for the continuous master, distinct
/// from integer incumbents and certificates for stronger pricing domains.
pub(super) struct IpmAccuracy {
    pub(super) lower: f64,
    pub(super) upper: f64,
}

impl IpmAccuracy {
    pub(super) fn new() -> Self {
        Self {
            lower: f64::NEG_INFINITY,
            upper: f64::INFINITY,
        }
    }

    pub(super) fn tolerance(&self) -> f64 {
        if !self.lower.is_finite() || !self.upper.is_finite() {
            return 0.5;
        }
        // epsilon_max = 1, degree of optimality D = 10. A numerical floor
        // avoids requesting a gap below the accurate master's precision.
        (((self.upper - self.lower).max(0.) / (1. + self.upper.abs())) / 10.).clamp(1e-9, 1.)
    }
}
