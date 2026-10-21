//! Downward binary64 operations for conservative solver bounds.

fn next_down(value: f64) -> f64 {
    if value == 0. {
        return -f64::from_bits(1);
    }
    f64::from_bits(if value > 0. {
        value.to_bits() - 1
    } else {
        value.to_bits() + 1
    })
}
pub(crate) fn add_down(a: f64, b: f64) -> Option<f64> {
    let sum = a + b;
    if !sum.is_finite() {
        return None;
    }
    let z = sum - a;
    let error = (a - (sum - z)) + (b - z);
    if !error.is_finite() {
        return None;
    }
    Some(if error < 0. { next_down(sum) } else { sum })
}
pub(crate) fn multiply_down(a: f64, b: f64) -> Option<f64> {
    let product = a * b;
    if !product.is_finite() {
        return None;
    }
    if a == 0. || b == 0. {
        return Some(product);
    }
    let error = a.mul_add(b, -product);
    // Tiny exact FMA residuals can themselves underflow. One downward ulp
    // safely encloses that range without relying on the residual's sign.
    Some(
        if error < 0. || product.abs() < 4. * f64::MIN_POSITIVE / f64::EPSILON {
            next_down(product)
        } else {
            product
        },
    )
}

pub(crate) fn add_up(a: f64, b: f64) -> Option<f64> {
    Some(-add_down(-a, -b)?)
}
