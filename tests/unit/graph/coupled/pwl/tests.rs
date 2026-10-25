use super::*;
#[test]
fn interval_images_enclose_literal_evaluation_at_knots_and_segment_interiors() {
    for points in [
        vec![[-2., -3.], [0., 2.], [1., -1.], [3., 4.]],
        vec![[0., -1e16], [1., 1.], [2., 1e16]],
    ] {
        let function = PiecewiseLinearFunction::new(points).unwrap();
        let domain = function.prepared_domain();
        for i in 0..=32 {
            for j in i..=32 {
                let lo = domain.lower + (domain.upper - domain.lower) * (i as f64 / 32.);
                let hi = domain.lower + (domain.upper - domain.lower) * (j as f64 / 32.);
                let image = function.range(Range { lo, hi }).unwrap();
                for point in &function.points {
                    if lo <= point[0] && point[0] <= hi {
                        assert!(image.lo <= point[1] && point[1] <= image.hi);
                    }
                }
                for k in 0..=8 {
                    let x = lo + (hi - lo) * (k as f64 / 8.);
                    let y = function.value(x).unwrap().unwrap();
                    assert!(
                        image.lo <= y && y <= image.hi,
                        "x={x} y={y} image={image:?}"
                    );
                }
            }
        }
    }
}
