use super::floor_isize;

#[test]
fn automatic_grid_supports_large_resource_windows() {
    use super::{BucketGraph, Graph};
    let mut graph = Graph::new(3, 1).unwrap();
    for vertex in 0..3 {
        graph.set_bounds(vertex, 0, 0., 100_000_000.).unwrap();
    }
    for (tail, head) in [(0, 1), (1, 2)] {
        let arc = graph.add_arc(tail, head, 0.).unwrap();
        graph.arc_mut(arc).unwrap().consumption[0] = 1_000_000.;
    }
    let automatic = BucketGraph::prepare(&graph, [0.; 2], 100, true).unwrap();
    assert_eq!(automatic.step[0], 1_000_000.);
    let explicit = BucketGraph::prepare(&graph, [1_000_000.; 2], 100, true).unwrap();
    for forward in [true, false] {
        assert_eq!(
            automatic.direction(forward).corners,
            explicit.direction(forward).corners
        );
        assert_eq!(
            automatic.direction(forward).current_arcs,
            explicit.direction(forward).current_arcs
        );
    }
    graph.set_bounds(1, 0, 0., 1e13).unwrap();
    assert!(BucketGraph::prepare(&graph, [0.; 2], 100, true).is_err());
}

#[test]
fn bucket_floor_matches_standard_floor_at_numeric_boundaries() {
    for value in [
        0.,
        -0.,
        1.,
        -1.,
        1.5,
        -1.5,
        1e-6,
        -1e-6,
        isize::MIN as f64,
        isize::MAX as f64,
        f64::MIN,
        f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ] {
        for bits in [
            value.to_bits().wrapping_sub(1),
            value.to_bits(),
            value.to_bits().wrapping_add(1),
        ] {
            let x = f64::from_bits(bits);
            assert_eq!(floor_isize(x), x.floor() as isize, "x={x:?}");
        }
    }
    for n in -1000..1000 {
        let x = n as f64 / 7.;
        assert_eq!(floor_isize(x), x.floor() as isize, "x={x:?}");
    }
    // Sample the entire IEEE-754 bit range, including subnormal values
    // and overflow, against the standard-library operation.
    let mut bits = 0x8e71_23a4_917b_cdef_u64;
    for _ in 0..10000 {
        bits = bits.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = f64::from_bits(bits);
        assert_eq!(floor_isize(x), x.floor() as isize, "bits={bits:x} x={x:?}");
    }
}
