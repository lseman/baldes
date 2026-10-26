use super::*;

fn graph(value: f64, upper: f64) -> Graph<2> {
    let mut g = Graph::<2>::new(3, 1).unwrap();
    g.add_arc(0, 1, 0.).unwrap();
    g.arcs[0].consumption[0] = value;
    for v in &mut g.vertices {
        v.bounds[0] = Window::new(0., upper).unwrap();
    }
    g
}

#[test]
fn binary_resource_grid_accepts_exact_fractions_and_rejects_ambiguous_scales() {
    for value in [0., 1., 1.5, 0.125, MIN_QUANTUM] {
        assert!(validate(&graph(value, 1_000_000.), 0).is_ok(), "{value}");
    }
    for value in [0.1, MIN_QUANTUM / 2., f64::from_bits(1)] {
        assert!(validate(&graph(value, 1.), 0).is_err(), "{value}");
    }
    assert!(validate(&graph(0.5, MAX_UNITS / 2.), 0).is_ok());
    assert!(validate(&graph(0.5, MAX_UNITS), 0).is_err());
    assert!(validate(&graph(-0.5, 1.), 0).is_err());
    let mut g = graph(1., 10.);
    g.initial[0] = Some(f64::NAN);
    assert!(validate(&g, 0).is_err());
    let mut g = graph(1., 10.);
    g.arcs[0].bounds[0] = Some(Window::new(0.1, 1.).unwrap());
    assert!(validate(&g, 0).is_err());
}

#[cfg(feature = "multi-time-windows")]
#[test]
fn resource_grid_checks_interior_window_endpoints() {
    let mut g = graph(1., 10.);
    g.set_time_windows(
        1,
        0,
        &[Window::new(0., 0.1).unwrap(), Window::new(2., 10.).unwrap()],
    )
    .unwrap();
    assert!(validate(&g, 0).is_err());
    g.set_time_windows(
        1,
        0,
        &[Window::new(0., 0.5).unwrap(), Window::new(2., 10.).unwrap()],
    )
    .unwrap();
    assert!(validate(&g, 0).is_ok());
}

#[test]
fn accepted_grid_additions_and_dominance_keep_distinct_resource_points() {
    let mut rng = crate::rng::Xoroshiro128Plus::from_seed(73_u64);
    for quantum in [MIN_QUANTUM, 0.125, 0.5, 1.] {
        for _ in 0..10_000 {
            let seed = rng.next();
            let left = seed & ((1_u64 << 52) - 1);
            let seed = rng.next();
            let right = seed & ((1_u64 << 52) - 1);
            let a = left as f64 * quantum;
            let b = right as f64 * quantum;
            // Integer sum below 2^53 is an independent exact oracle.
            assert_eq!(a + b, (left + right) as f64 * quantum);
            let next = (left + 1) as f64 * quantum;
            assert!(next > a + crate::labeling::EPSILON);
            assert!(a < next - crate::labeling::EPSILON);
        }
    }
}
