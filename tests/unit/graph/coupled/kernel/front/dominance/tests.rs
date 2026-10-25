use super::*;
fn front(points: &[[f64; 2]]) -> CostFront {
    CostFront {
        points: points.to_vec().into(),
    }
}
#[test]
fn normalized_constant_and_rising_fronts_use_directional_compensation() {
    let constant = front(&[[0., 8.], [4., 8.]]);
    let rising = front(&[[0., -2.], [4., 2.]]);
    assert_eq!(constant.dominance_penalty(&rising), Some(0.));
    assert_eq!(rising.dominance_penalty(&constant), Some(4.));
    assert_eq!(rising.dominance_penalty(&rising), Some(0.));
    assert_eq!(
        constant.dominance_penalty(&front(&[[-1., 0.], [4., 0.]])),
        None
    );
}
#[test]
fn correlated_snapshot_orders_shared_literal_interpolation_operands() {
    let a = front(&[[0., 0.], [4., 4.]]);
    let b = front(&[[0., 0.], [4., 8.]]);
    assert_eq!(a.dominance_penalty(&b), Some(0.));
    // Future rounded additions remain outside this snapshot guarantee.
    let current = front(&[[0., 0.], [1., 1.]]);
    let zero = front(&[[0., 0.], [1., 0.]]);
    assert_eq!(current.dominance_penalty(&zero), Some(1.));
}
#[test]
fn complete_front_bound_encloses_every_sample_and_retains_interior_knots() {
    let fronts = [
        front(&[[-2., 6.], [0., 0.], [1., 7.], [4., 2.], [6., -3.]]),
        front(&[[0., 4.], [0.5, -2.], [2., 5.], [4., 0.]]),
        front(&[[0., 0.], [4., 0.]]),
        front(&[[0., -1e12], [1., 1e12], [4., -1e12]]),
        front(&[[2., 3.]]),
    ];
    for a in &fronts {
        for b in &fronts {
            let domain = b.domain();
            let Some(penalty) = a.dominance_penalty(b) else {
                continue;
            };
            assert!(penalty >= 0. && penalty.is_finite());
            for i in 0..=512 {
                let x = domain.lower + (domain.upper - domain.lower) * (i as f64 / 512.);
                // Outward comparisons also protect the test's subtraction
                // from rounding in the opposite direction to the proof.
                let a_value = Range::point(a.value(x).unwrap()).subtract(Range::point(a.minimum()));
                let b_value = Range::point(b.value(x).unwrap()).subtract(Range::point(b.minimum()));
                assert!(a_value.lo - b_value.hi <= penalty);
            }
        }
    }
    let peak = front(&[[0., 0.], [1., 9.], [2., 0.]]);
    let flat = front(&[[0., 0.], [2., 0.]]);
    assert_eq!(peak.dominance_penalty(&flat), Some(9.));
}
#[test]
fn compensation_declines_overflow_but_supports_fixed_domains() {
    let huge = front(&[[0., -1e308], [1., 1e308]]);
    let zero = front(&[[0., 0.], [1., 0.]]);
    assert_eq!(huge.dominance_penalty(&zero), None);
    let fixed = front(&[[0.5, 2.]]);
    assert_eq!(zero.dominance_penalty(&fixed), Some(0.));
    assert_eq!(fixed.dominance_penalty(&zero), None);
}
