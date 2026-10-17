//! ColGenStabilization, ported from master/Stabilization.h.
use super::RowSense;
#[derive(Clone, Debug)]
pub struct Parameters {
    pub alpha_factor: f64,
    pub initial_alpha: f64,
    pub beta_factor: f64,
    pub max_updates: i32,
    pub min_gap: f64,
    pub max_gap: f64,
    pub minimum_stage: i32,
    pub tolerance: f64,
}
impl Default for Parameters {
    fn default() -> Self {
        Self {
            alpha_factor: 1.,
            initial_alpha: 0.5,
            beta_factor: 0.,
            max_updates: 10,
            min_gap: 0.,
            max_gap: 0.,
            minimum_stage: 0,
            tolerance: 1e-6,
        }
    }
}
#[derive(Clone, Debug)]
pub struct Feedback {
    pub negative_columns: usize,
    pub subgradient: Vec<f64>,
    pub lagrangian: f64,
    pub phase: i32,
}
#[derive(Clone, Debug)]
pub struct DualPriceSmoothing {
    pub parameters: Parameters,
    pub alpha: f64,
    pub effective_alpha: f64,
    pub angle: f64,
    pub center_bound: f64,
    pub center_phase: i32,
    pub misprices: i32,
    pub active: bool,
    pub center_moved: bool,
    pub center: Vec<f64>,
    pub pricing: Vec<f64>,
    raw: Vec<f64>,
    subgradient: Vec<f64>,
    senses: Vec<RowSense>,
    next_center: Option<(Vec<f64>, Vec<f64>)>,
}
impl Default for DualPriceSmoothing {
    fn default() -> Self {
        Self::new(Parameters::default())
    }
}
impl DualPriceSmoothing {
    pub fn new(parameters: Parameters) -> Self {
        let alpha = if parameters.alpha_factor < 1. {
            parameters.alpha_factor
        } else {
            parameters.initial_alpha
        };
        Self {
            parameters,
            alpha,
            effective_alpha: alpha,
            angle: 0.,
            center_bound: f64::NEG_INFINITY,
            center_phase: i32::MAX,
            misprices: 0,
            active: false,
            center_moved: false,
            center: vec![],
            pricing: vec![],
            raw: vec![],
            subgradient: vec![],
            senses: vec![],
            next_center: None,
        }
    }
    pub fn restart_schedule(&mut self, phase: i32) {
        self.alpha = if self.parameters.alpha_factor < 1. {
            self.parameters.alpha_factor
        } else {
            self.parameters.initial_alpha
        };
        self.effective_alpha = self.alpha;
        self.misprices = 0;
        self.active = false;
        self.center_phase = phase;
    }
    pub fn remap_rows(&mut self, map: &[Option<usize>]) {
        fn compact<T: Clone>(v: &mut Vec<T>, map: &[Option<usize>]) {
            *v = v
                .iter()
                .enumerate()
                .filter(|(i, _)| map.get(*i).is_some_and(Option::is_some))
                .map(|(_, v)| v.clone())
                .collect();
        }
        compact(&mut self.center, map);
        compact(&mut self.raw, map);
        compact(&mut self.pricing, map);
        compact(&mut self.subgradient, map);
        compact(&mut self.senses, map);
        if let Some((c, g)) = &mut self.next_center {
            compact(c, map);
            compact(g, map);
        }
    }
    pub fn after_master(
        &mut self,
        raw: &[f64],
        senses: &[RowSense],
        gap: f64,
        stage: i32,
        initial: bool,
    ) {
        if let Some((c, mut g)) = self.next_center.take() {
            self.center = c;
            if self.parameters.beta_factor > 0. {
                normalize(&mut g);
                self.subgradient = g;
            }
        }
        self.raw = raw.to_vec();
        self.senses = senses.to_vec();
        self.center.resize(raw.len(), 0.);
        self.active = false;
        self.misprices = 0;
        self.effective_alpha = self.alpha;
        self.pricing = raw.to_vec();
        if initial {
            self.effective_alpha = 0.;
            return;
        }
        if stage < self.parameters.minimum_stage
            || !self.gap_accepts(gap)
            || self.alpha <= self.parameters.tolerance
        {
            return;
        }
        self.publish(self.effective_alpha);
        self.active = true;
    }
    pub fn after_pricing(&mut self, f: &Feedback) -> bool {
        self.consider_center(f);
        self.angle = cosine(&self.center, &self.pricing, &f.subgradient);
        if self.parameters.alpha_factor >= 1. && self.misprices == 0 {
            let tolerance = self.parameters.tolerance;
            // The angle test uses its own native threshold, independently of
            // the tolerance for bounds and dual values (ELF 0x2d8fd0).
            if f.negative_columns == 0 || self.angle > 1e-12 {
                self.alpha -= 0.1;
            } else if self.angle < -1e-12
                && self.alpha < 0.999 - (self.alpha.abs().min(0.999) * 1e-10 + tolerance)
            {
                self.alpha += 0.1 * (1. - self.alpha);
            }
        }
        if f.negative_columns > 0 || !self.active {
            return false;
        }
        self.misprices += 1;
        self.effective_alpha = if self.parameters.alpha_factor >= 1. {
            2. * self.effective_alpha - 1.
        } else {
            1. - self.misprices as f64 * (1. - self.alpha)
        };
        if self.misprices <= self.parameters.max_updates
            && self.effective_alpha > self.parameters.tolerance
        {
            self.publish(self.effective_alpha);
            return true;
        }
        self.active = false;
        self.pricing = self.raw.clone();
        true
    }
    pub fn after_iteration(&mut self) {
        self.center_moved = false;
        if self.active {
            self.active = false;
            self.pricing = self.raw.clone();
        }
    }
    fn gap_accepts(&self, gap: f64) -> bool {
        (self.parameters.max_gap <= 0.
            || !greater(gap, self.parameters.max_gap, self.parameters.tolerance))
            && !greater(self.parameters.min_gap, gap, self.parameters.tolerance)
    }
    fn publish(&mut self, alpha: f64) {
        let outward = if self.parameters.beta_factor > 0. {
            self.directional_point()
        } else {
            self.raw.clone()
        };
        self.pricing = outward
            .iter()
            .enumerate()
            .map(|(i, out)| (1. - alpha) * out + alpha * self.center.get(i).copied().unwrap_or(0.))
            .collect();
        for (pi, sense) in self.pricing.iter_mut().zip(&self.senses) {
            if (*sense == RowSense::LessEqual && *pi > self.parameters.tolerance)
                || (*sense == RowSense::GreaterEqual && *pi < -self.parameters.tolerance)
            {
                *pi = 0.;
            }
        }
    }
    fn directional_point(&self) -> Vec<f64> {
        let mut out = self.raw.clone();
        let diff: Vec<_> = self
            .raw
            .iter()
            .enumerate()
            .map(|(i, v)| v - self.center.get(i).copied().unwrap_or(0.))
            .collect();
        let length = norm(&diff);
        if length <= self.parameters.tolerance {
            return out;
        }
        let sub_norm = norm(&self.subgradient);
        let mut beta = self.parameters.beta_factor;
        if beta == 1. {
            if sub_norm <= 1e-12 {
                return out;
            }
            beta = dot(&diff, &self.subgradient) / (length * sub_norm);
        }
        beta = beta.clamp(0., 1.);
        for (i, v) in out.iter_mut().enumerate() {
            let component = if sub_norm > 1e-12 {
                self.subgradient.get(i).copied().unwrap_or(0.) / sub_norm
            } else {
                0.
            };
            *v = self.center.get(i).copied().unwrap_or(0.) + (1. - beta) * diff[i]
                - beta * length * component;
        }
        let step: Vec<_> = out
            .iter()
            .enumerate()
            .map(|(i, v)| v - self.center.get(i).copied().unwrap_or(0.))
            .collect();
        let step_norm = norm(&step);
        if step_norm > 1e-12 {
            for (i, v) in out.iter_mut().enumerate() {
                *v = self.center.get(i).copied().unwrap_or(0.) + step[i] * (length / step_norm);
            }
        }
        out
    }
    fn consider_center(&mut self, f: &Feedback) {
        if !f.lagrangian.is_finite() {
            return;
        }
        if self.center_phase <= f.phase {
            if self.center_bound.is_finite()
                && !greater(f.lagrangian, self.center_bound, self.parameters.tolerance)
            {
                return;
            }
        } else {
            self.center_phase = f.phase;
        }
        self.center_bound = f.lagrangian;
        let mut c = self.pricing.clone();
        c.resize(self.raw.len().max(c.len()), 0.);
        let mut g = f.subgradient.clone();
        g.resize(c.len(), 0.);
        self.next_center = Some((c, g));
        self.center_moved = true;
    }
}
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.;
    for (a, b) in a.iter().zip(b) {
        sum += a * b;
    }
    sum
}
pub fn norm(v: &[f64]) -> f64 {
    dot(v, v).max(0.).sqrt()
}
fn normalize(v: &mut [f64]) {
    let n = norm(v);
    if n > 1e-12 {
        for x in v {
            *x /= n;
        }
    }
}
fn greater(value: f64, bound: f64, tolerance: f64) -> bool {
    value > bound + (value.abs().max(bound.abs()) * 1e-10 + tolerance)
}
fn cosine(center: &[f64], priced: &[f64], g: &[f64]) -> f64 {
    let toward: Vec<_> = center
        .iter()
        .enumerate()
        .map(|(i, v)| priced.get(i).copied().unwrap_or(0.) - v)
        .collect();
    // calculateAngleAtPricingPoint (ELF 0x2d87f0) divides unguarded: a
    // round-off step toward the pricing point still yields an angle, and a
    // zero step gives NaN, which then fails both alpha-update comparisons.
    dot(&toward, g) / (norm(g) * norm(&toward))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn angle_threshold_is_independent_of_bound_tolerance() {
        for (angle, expected_alpha) in [(2e-12, 0.4), (-2e-12, 0.55), (5e-13, 0.5), (-5e-13, 0.5)] {
            let mut smoothing = DualPriceSmoothing::default();
            smoothing.center = vec![0., 0.];
            smoothing.pricing = vec![1., 0.];
            assert!(!smoothing.after_pricing(&Feedback {
                negative_columns: 1,
                subgradient: vec![angle, 1.],
                lagrangian: f64::NEG_INFINITY,
                phase: 1,
            }));
            assert_eq!(smoothing.alpha, expected_alpha, "angle {angle}");
        }
    }
}
