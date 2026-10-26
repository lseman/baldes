//! Exact floating-point expansions used by model validation and pricing.
#[derive(Clone, Default)]
pub(crate) struct ExactSum(Vec<f64>);
impl ExactSum {
    pub(crate) fn add(&mut self, mut value: f64) -> Option<()> {
        if !value.is_finite() {
            return None;
        }
        let mut terms = Vec::new();
        for &term in &self.0 {
            let sum = value + term;
            let z = sum - value;
            let error = (value - (sum - z)) + (term - z);
            if !sum.is_finite() || !error.is_finite() {
                return None;
            }
            if error != 0. {
                terms.push(error);
            }
            value = sum;
        }
        if value != 0. {
            terms.push(value);
        }
        self.0 = terms;
        Some(())
    }
    pub(crate) fn product(&mut self, value: f64, integer: f64) -> Option<()> {
        let product = value * integer;
        // Integer multipliers preserve the subnormal lattice, so an FMA
        // residual cannot hide a fractional subnormal product error.
        let error = value.mul_add(integer, -product);
        self.add(error)?;
        self.add(product)
    }
    /// Add an upper enclosure of a product with an arbitrary finite factor.
    /// Products on the binary64 subnormal lattice retain the exact expansion.
    pub(crate) fn product_upper(&mut self, value: f64, factor: f64) -> Option<()> {
        self.product(value, factor)?;
        if value == 0. || factor == 0. {
            return Some(());
        }
        let lattice_exponent = |value: f64| {
            let bits = value.to_bits();
            let exponent = ((bits >> 52) & 0x7ff) as i32;
            let significand =
                (bits & ((1u64 << 52) - 1)) | if exponent == 0 { 0 } else { 1u64 << 52 };
            let scale = if exponent == 0 {
                -1074
            } else {
                exponent - 1075
            };
            scale + significand.trailing_zeros() as i32
        };
        if lattice_exponent(value) + lattice_exponent(factor) < -1074 {
            // The exact residual can lie between adjacent subnormals. FMA
            // rounds that residual, so compensate upward by one lattice unit.
            // With at most 106 product bits this case has a subnormal residual;
            // normal residuals and integer-factor products remain exact.
            self.add(f64::from_bits(1))?;
        }
        Some(())
    }
    pub(crate) fn sign(&self) -> f64 {
        self.0.last().copied().unwrap_or(0.)
    }
    pub(crate) fn compare(&self, value: f64) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;
        if !value.is_finite() {
            return None;
        }
        let sign = self.sign();
        if sign == 0. {
            return 0f64.partial_cmp(&value);
        }
        // Opposite signs can be compared without overflowing the difference.
        if sign > 0. && value <= 0. {
            return Some(Ordering::Greater);
        }
        if sign < 0. && value >= 0. {
            return Some(Ordering::Less);
        }
        let mut difference = self.clone();
        difference.add(-value)?;
        difference.sign().partial_cmp(&0.)
    }
    pub(crate) fn upper(&self) -> Option<f64> {
        let mut value = 0.;
        for &term in &self.0 {
            let sum = value + term;
            let z = sum - value;
            let error = (value - (sum - z)) + (term - z);
            if !sum.is_finite() || !error.is_finite() {
                return None;
            }
            value = if error > 0. {
                if sum == 0. {
                    f64::from_bits(1)
                } else {
                    f64::from_bits(if sum > 0. {
                        sum.to_bits() + 1
                    } else {
                        sum.to_bits() - 1
                    })
                }
            } else {
                sum
            };
        }
        value.is_finite().then_some(value)
    }
}
