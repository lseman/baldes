use super::floor_isize;

#[test]
fn automatic_grid_supports_large_resource_windows() {
    use super::{BucketGraph, Graph};
    let mut graph = Graph::<2>::new(3, 1).unwrap();
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
    let mut rng = crate::rng::Xoroshiro128Plus::from_seed(0x8e71_23a4_917b_cdef);
    for _ in 0..10000 {
        let bits = rng.next();
        let x = f64::from_bits(bits);
        assert_eq!(floor_isize(x), x.floor() as isize, "bits={bits:x} x={x:?}");
    }
}

fn dense_graph(resources: usize) -> super::Graph<2> {
    let mut graph = super::Graph::<2>::new(5, resources).unwrap();
    for v in 0..5 {
        for r in 0..resources {
            graph.set_bounds(v, r, 0., 20.).unwrap();
        }
    }
    for tail in 0..4 {
        for head in 1..5 {
            if tail == head {
                continue;
            }
            for consumption in [1., 2., 3.] {
                let id = graph.add_arc(tail, head, 0.).unwrap();
                for r in 0..resources {
                    graph.arc_mut(id).unwrap().consumption[r] = consumption / (r + 1) as f64;
                }
            }
        }
    }
    graph
}

#[test]
fn halfway_flag_updates_preserve_sparse_arc_ids_and_pruned_storage_order() {
    use super::BucketGraph;
    for (resources, bidirectional) in [(1, false), (1, true), (2, false), (2, true)] {
        let graph = dense_graph(resources);
        let mut grid = BucketGraph::prepare(&graph, [4.; 2], 20, bidirectional).unwrap();
        let mut serial = 0;
        for direction in [&mut grid.forward, &mut grid.backward] {
            for bucket in direction.buckets.iter_mut().flatten() {
                bucket.arcs.retain(|arc| arc.physical % 5 != 0);
                bucket.arcs.reverse();
                for arc in &mut bucket.arcs {
                    arc.id = serial * 11 + 7;
                    serial += 1;
                }
            }
        }
        for movement in [true, false, true, true, false] {
            let mut expected = grid.clone();
            let middle = if movement {
                expected.middle
                    + (expected.upper[expected.order.len() - 1][0] - expected.middle) * 0.05
            } else {
                expected.middle - (expected.middle - expected.lower[0][0]) * 0.05
            };
            expected.middle = (middle * 10. + 0.5).floor() / 10.;
            grid.move_middle(&graph, movement);
            assert_eq!(grid.middle.to_bits(), expected.middle.to_bits());
            for direction in [true, false] {
                let original = &expected.direction(direction).buckets;
                let actual = &grid.direction(direction).buckets;
                assert_eq!(original.len(), actual.len());
                for (old, new) in original.iter().flatten().zip(actual.iter().flatten()) {
                    assert_eq!(old.arcs.len(), new.arcs.len());
                    for (a, b) in old.arcs.iter().zip(&new.arcs) {
                        assert_eq!(
                            (a.id, a.from, a.physical, a.jump),
                            (b.id, b.from, b.physical, b.jump)
                        );
                        assert_eq!(b.flags, expected.flags(&graph, direction, a));
                    }
                }
            }
        }
    }
}

#[test]
fn bucket_refinement_transfers_fixings_and_scales_existing_depths() {
    use super::BucketGraph;
    for (resources, bidirectional) in [(1, false), (1, true), (2, false), (2, true)] {
        let graph = dense_graph(resources);
        let mut grid = BucketGraph::prepare(&graph, [4.; 2], 20, bidirectional).unwrap();
        for direction in [&mut grid.forward, &mut grid.backward] {
            for bucket in direction.buckets.iter_mut().flatten() {
                bucket.arcs.retain(|arc| arc.physical != 0);
            }
        }
        let old_depths = [grid.forward.depth.clone(), grid.backward.depth.clone()];
        let versions = (grid.version, grid.backward_version);
        grid.decrease_steps(&graph).unwrap();
        assert_eq!(grid.step[0], if resources == 1 { 2. } else { 2.9 });
        assert_eq!(grid.version, versions.0 + 1);
        assert_eq!(grid.backward_version, versions.1 + i32::from(bidirectional));
        for (i, direction) in [true, false].into_iter().enumerate() {
            if !direction && !bidirectional {
                continue;
            }
            let d = grid.direction(direction);
            assert!(d
                .buckets
                .iter()
                .flatten()
                .flat_map(|b| &b.arcs)
                .all(|a| a.physical != 0));
            let factor = if resources == 1 { 2. } else { 1.4 };
            let expected: Vec<_> = old_depths[i]
                .iter()
                .map(|&depth| {
                    if depth < 1 {
                        1
                    } else {
                        (depth as f64 * factor) as i32
                    }
                })
                .collect();
            assert_eq!(d.depth, expected);
        }
        grid.step[0] = 0.05;
        let before = format!("{grid:?}");
        assert!(grid.decrease_steps(&graph).is_err());
        assert_eq!(format!("{grid:?}"), before);
    }
}
