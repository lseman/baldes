use super::*;
use std::cmp::Ordering;

// Compare binary64 endpoints with an independently accumulated i128
// rational whose denominator is 2^scale. No directed arithmetic is used here.
fn compare(value: f64, numerator: i128, scale: i32) -> Ordering {
    if value == 0. {
        return 0.cmp(&numerator);
    }
    let bits = value.to_bits();
    let mantissa = ((bits & ((1_u64 << 52) - 1)) | (1_u64 << 52)) as i128;
    let signed = if bits >> 63 == 0 { mantissa } else { -mantissa };
    let shift = ((bits >> 52) & 0x7ff) as i32 - 1023 - 52 + scale;
    if shift >= 0 {
        (signed << shift).cmp(&numerator)
    } else {
        signed.cmp(&(numerator << -shift))
    }
}
fn draw(state: &mut u64, n: i32) -> i32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 32) % n as u64) as i32
}
#[test]
fn enclosures_match_independent_integer_rationals() {
    let mut seed = 0x5035_cafe;
    for _ in 0..10_000 {
        let c = draw(&mut seed, 2049) - 1024;
        let ce = draw(&mut seed, 21);
        let cost = c as f64 / 2_f64.powi(ce);
        let mut exact = (c as i128) << (40 - ce);
        let mut terms = vec![];
        for _ in 0..1 + draw(&mut seed, 8) {
            let a = draw(&mut seed, 2049) - 1024;
            let b = draw(&mut seed, 2049) - 1024;
            let ae = draw(&mut seed, 21);
            let be = draw(&mut seed, 21);
            exact -= ((a as i128) * (b as i128)) << (40 - ae - be);
            terms.push((a as f64 / 2_f64.powi(ae), b as f64 / 2_f64.powi(be)));
        }
        let bounds = enclose(cost, terms).unwrap();
        assert!(compare(bounds.lower, exact, 40) != Ordering::Greater);
        assert!(compare(bounds.upper, exact, 40) != Ordering::Less);
    }
}
#[test]
fn full_mantissa_products_enclose_exact_104_bit_rationals() {
    let mut seed = 0x5035_104_u64;
    for _ in 0..5_000 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let a = f64::from_bits((1023_u64 << 52) | (seed & ((1_u64 << 52) - 1)));
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let b = f64::from_bits((1023_u64 << 52) | (seed & ((1_u64 << 52) - 1)));
        let ma = ((a.to_bits() & ((1_u64 << 52) - 1)) | (1_u64 << 52)) as i128;
        let mb = ((b.to_bits() & ((1_u64 << 52) - 1)) | (1_u64 << 52)) as i128;
        let numerator = (3_i128 << 104) - ma * mb;
        let bounds = enclose(3., [(a, b)]).unwrap();
        assert!(compare(bounds.lower, numerator, 104) != Ordering::Greater);
        assert!(compare(bounds.upper, numerator, 104) != Ordering::Less);
    }
}

#[test]
fn cancellation_and_fractional_underflow_do_not_turn_into_certified_zero() {
    let large = 2_f64.powi(54);
    for cost in [-1., 1.] {
        let terms = [(1_f64, large), (-1., large)];
        let numerical = terms
            .into_iter()
            .fold(cost, |sum, (a, b)| (-a).mul_add(b, sum));
        assert_eq!(numerical, 0.);
        let bounds = enclose(cost, terms).unwrap();
        assert_eq!(bounds.lower, cost);
        assert_eq!(bounds.upper, cost);
    }
    let tiny = f64::from_bits(1);
    let integer_product = enclose(0., [(1., tiny)]).unwrap();
    assert_eq!(integer_product.lower, -tiny);
    assert_eq!(integer_product.upper, -tiny);
    let bounds = enclose(0., [(0.5, tiny)]).unwrap();
    assert!(bounds.lower < 0. && bounds.upper >= 0.);
    assert_eq!((-0.5_f64).mul_add(tiny, 0.), 0.);
    assert!(enclose(f64::MAX, [(-f64::MAX, 2.)]).is_none());
    assert!(enclose(0., [(f64::NAN, 1.)]).is_none());
}
