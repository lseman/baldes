//! Directed interval arithmetic for column bounds.
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
}
