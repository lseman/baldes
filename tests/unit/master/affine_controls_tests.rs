use super::*;

#[test]
fn correlated_objective_and_row_bounds_keep_the_shared_selected_value() {
    let domains = [Interval { lo: 3., hi: 8. }];
    let gamma = Expression::variable(0, 1);
    let reduced = gamma
        .add(&gamma.scale(-2., &domains).unwrap(), &domains)
        .unwrap();
    let interval = reduced.range(&domains).unwrap();
    assert!(interval.lo <= -8. && interval.lo > -8. - 1e-12);
    assert!(interval.hi >= -3. && interval.hi < -3. + 1e-12);
}

#[test]
fn affine_enclosures_cover_ordered_rounded_products_and_cancellation() {
    for scale in [0., 0.1, 1., 2., 1e-300, 1e100] {
        for constant in [0., -1., 1e10, -1e100] {
            let domains = [Interval { lo: -8., hi: 8. }, Interval { lo: -3., hi: 5. }];
            let first = Expression::variable(0, 2).scale(scale, &domains).unwrap();
            let second = Expression::variable(1, 2).scale(0.3, &domains).unwrap();
            let value = Expression::constant(Interval::point(constant), 2)
                .add(&first, &domains)
                .unwrap()
                .add(&second, &domains)
                .unwrap()
                .add(&first.scale(-1., &domains).unwrap(), &domains)
                .unwrap();
            let bounds = value.range(&domains).unwrap();
            for a in [-8., -0.1, 0., 1., 8.] {
                for b in [-3., -0.5, 0., 3., 5.] {
                    let actual = ((constant + a * scale) + b * 0.3) - a * scale;
                    assert!(
                        bounds.lo <= actual && actual <= bounds.hi,
                        "scale={scale} constant={constant} actual={actual} {bounds:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn affine_bounds_refuse_nonfinite_arithmetic_and_mismatched_symbols() {
    let domain = [Interval { lo: 1., hi: 2. }];
    let value = Expression::variable(0, 1);
    assert!(value.scale(f64::INFINITY, &domain).is_none());
    assert!(value.range(&[]).is_none());
    assert!(value
        .add(&Expression::constant(Interval::point(0.), 0), &domain)
        .is_none());
}
