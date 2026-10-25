use super::*;

#[test]
fn correction_step_bound_matches_materialized_directions() {
    let q = Point {
        x: vec![0.5, 2., 3.],
        y: vec![0., -1.],
        s: vec![2., 1., 5.],
        v: vec![1., 2.],
        w: vec![5., 1.],
        tau: 1.,
        kappa: 1.5,
    };
    let mut rng = crate::rng::Xoroshiro128Plus::from_seed(0x01f0_cafe);
    for _ in 0..1000 {
        let mut values = |n| {
            (0..n)
                .map(|_| (rng.uniform_below(257) as f64 - 128.) / 16.)
                .collect()
        };
        let direction = Point {
            x: values(3),
            y: values(2),
            s: values(3),
            v: values(2),
            w: values(2),
            tau: -0.25,
            kappa: 0.5,
        };
        let correction = Point {
            x: values(3),
            y: values(2),
            s: values(3),
            v: values(2),
            w: values(2),
            tau: 0.125,
            kappa: -2.,
        };
        let mut combined = direction.clone();
        combined.add_direction(&correction);
        assert_eq!(
            q.alpha_with_correction(&direction, &correction).to_bits(),
            q.alpha(&combined).to_bits()
        );
    }
}
