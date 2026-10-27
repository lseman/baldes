//! Correlated affine enclosures for rounded column expressions.
//! Symbols retain shared selected values through cost and row accumulation.
use crate::core::rounding::{add_down, add_up, multiply_down};

#[derive(Clone, Copy, Debug)]
pub(crate) struct Interval {
    pub lo: f64,
    pub hi: f64,
}
impl Interval {
    pub fn point(value: f64) -> Self {
        Self {
            lo: value,
            hi: value,
        }
    }
    pub fn add(self, other: Self) -> Option<Self> {
        Some(Self {
            lo: add_down(self.lo, other.lo)?,
            hi: add_up(self.hi, other.hi)?,
        })
    }
    pub fn scale(self, coefficient: f64) -> Option<Self> {
        self.multiply(Self::point(coefficient))
    }
    pub fn multiply(self, other: Self) -> Option<Self> {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for a in [self.lo, self.hi] {
            for b in [other.lo, other.hi] {
                lo = lo.min(multiply_down(a, b)?);
                hi = hi.max(-multiply_down(-a, b)?);
            }
        }
        Some(Self { lo, hi })
    }
    fn magnitude(self) -> f64 {
        self.lo.abs().max(self.hi.abs())
    }
    fn zero(self) -> bool {
        self.lo == 0. && self.hi == 0.
    }
}
#[derive(Clone, Debug)]
pub(crate) struct Expression {
    constant: Interval,
    coefficients: Vec<Interval>,
}
impl Expression {
    pub fn constant(value: Interval, variables: usize) -> Self {
        Self {
            constant: value,
            coefficients: vec![Interval::point(0.); variables],
        }
    }
    pub fn variable(index: usize, variables: usize) -> Self {
        let mut result = Self::constant(Interval::point(0.), variables);
        result.coefficients[index] = Interval::point(1.);
        result
    }
    pub fn coefficient(&self, index: usize) -> Interval {
        self.coefficients[index]
    }
    pub fn coefficient_count(&self) -> usize {
        self.coefficients.len()
    }
    pub fn constant_interval(&self) -> Interval {
        self.constant
    }
    pub fn range(&self, domains: &[Interval]) -> Option<Interval> {
        if domains.len() != self.coefficients.len() {
            return None;
        }
        self.coefficients
            .iter()
            .zip(domains)
            .try_fold(self.constant, |value, (&coefficient, &domain)| {
                value.add(coefficient.multiply(domain)?)
            })
    }
    fn zero(&self) -> bool {
        self.constant.zero()
            && self
                .coefficients
                .iter()
                .all(|coefficient| coefficient.zero())
    }
    fn rounding(mut self, domains: &[Interval]) -> Option<Self> {
        let magnitude = self.range(domains)?.magnitude();
        // One relative ulp plus one subnormal lattice unit bounds nearest
        // rounding, including cancellation, underflow and signed values.
        if magnitude != 0. {
            let error = add_up(-multiply_down(-magnitude, f64::EPSILON)?, f64::from_bits(1))?;
            self.constant = self.constant.add(Interval {
                lo: -error,
                hi: error,
            })?;
        }
        Some(self)
    }
    pub fn add(&self, other: &Self, domains: &[Interval]) -> Option<Self> {
        if self.coefficients.len() != other.coefficients.len() {
            return None;
        }
        let result = Self {
            constant: self.constant.add(other.constant)?,
            coefficients: self
                .coefficients
                .iter()
                .zip(&other.coefficients)
                .map(|(&a, &b)| a.add(b))
                .collect::<Option<Vec<_>>>()?,
        };
        if self.zero() || other.zero() {
            Some(result)
        } else {
            result.rounding(domains)
        }
    }
    pub fn scale(&self, coefficient: f64, domains: &[Interval]) -> Option<Self> {
        if coefficient == 0. {
            return Some(Self::constant(Interval::point(0.), self.coefficients.len()));
        }
        let result = Self {
            constant: self.constant.scale(coefficient)?,
            coefficients: self
                .coefficients
                .iter()
                .map(|&value| value.scale(coefficient))
                .collect::<Option<Vec<_>>>()?,
        };
        if coefficient.abs() == 1. {
            Some(result)
        } else {
            result.rounding(domains)
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/master/affine_controls_tests.rs"]
mod tests;
