use super::*;
#[test]
fn preimages_retain_exact_arguments_through_flat_and_nonmonotone_segments() {
    let function =
        PiecewiseLinearFunction::new(vec![[-2., 3.], [0., 1.], [1., 1.], [3., -1.], [4., 2.]])
            .unwrap();
    let domain = Range { lo: -2., hi: 4. };
    for index in 0..=128 {
        let x = -2. + 6. * (index as f64 / 128.);
        let y = function.value(x).unwrap().unwrap();
        let inverse = function.preimage(domain, Range::point(y)).unwrap();
        assert!(
            inverse.lo <= x && x <= inverse.hi,
            "x={x} y={y} inverse={inverse:?}"
        );
    }
    assert!(function
        .preimage(domain, Range { lo: 10., hi: 20. })
        .is_none());
}
