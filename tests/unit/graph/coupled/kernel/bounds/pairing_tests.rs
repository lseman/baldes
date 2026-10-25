use super::*;
fn kernel() -> CoupledKernel {
    let mut graph = Graph::<2>::new(4, 0).unwrap();
    for (a, b) in [(0, 1), (0, 2), (1, 3), (2, 3)] {
        graph.add_arc(a, b, 0.).unwrap();
    }
    CoupledKernel::prepare(
        &graph,
        &[
            CoupledResource::Selection {
                initial: 3,
                allowed: vec![1, 2, 1, 2],
                vertices: vec![],
                relaxed: false,
            },
            CoupledResource::SimultaneousPickupDelivery {
                capacity: 20.,
                deliveries: vec![1.; 4],
                pickups: vec![2.; 4],
                tolerance: 0.,
            },
            CoupledResource::CumulativeCost {
                initial: 0.,
                increments: vec![1., 2., 3., 4.],
                weights: vec![-2., 3., -1., 2.],
                sink_upper: None,
                source_upper: None,
            },
        ],
    )
    .unwrap()
}
#[test]
fn hull_pair_bounds_enclose_every_signed_join_and_guard_membership() {
    let kernel = kernel();
    let forward: Vec<_> = [0, 1]
        .iter()
        .map(|&arc| {
            kernel
                .extend(&kernel.root(true), arc, true)
                .unwrap()
                .unwrap()
                .0
        })
        .collect();
    let backward: Vec<_> = [2, 3]
        .iter()
        .map(|&arc| {
            kernel
                .extend(&kernel.root(false), arc, false)
                .unwrap()
                .unwrap()
                .0
        })
        .collect();
    let (mut f, mut b) = (CoupledStateBounds::default(), CoupledStateBounds::default());
    for state in &forward {
        f.include(Some(state));
    }
    for state in &backward {
        b.include(Some(state));
    }
    b.prepare(&kernel);
    let pair = kernel.pairing_summary(&f, &b);
    assert!(pair.possible && pair.correction_lower.is_finite());
    for x in &forward {
        for y in &backward {
            if let Some(cost) = kernel.join(x, y).unwrap() {
                assert!(pair.correction_lower <= cost);
            }
        }
    }
    let mut singleton = CoupledStateBounds::default();
    singleton.include(Some(&forward[0]));
    assert!(singleton.contains(Some(&forward[0])));
    assert!(!singleton.contains(Some(&forward[1])));
    let mut disjoint = CoupledStateBounds::default();
    disjoint.include(Some(&backward[1]));
    assert!(!kernel.pairing_summary(&singleton, &disjoint).possible);
    let cached = kernel.completion_correction_lower(true, Some(&forward[0]), &b);
    b.initial_dot = None;
    assert_eq!(
        cached,
        kernel.completion_correction_lower(true, Some(&forward[0]), &b)
    );
    b.prepare(&kernel);
    b.include(Some(&backward[0]));
    assert!(b.initial_dot.is_none());
    let mut invalid = forward[0].clone();
    invalid.numeric[0] = f64::NAN;
    assert!(!singleton.contains(Some(&invalid)));
    let mut unknown = CoupledStateBounds::default();
    unknown.include(None);
    let summary = kernel.pairing_summary(&singleton, &unknown);
    assert!(summary.possible);
    assert_eq!(summary.correction_lower, f64::NEG_INFINITY);
}
#[cfg(feature = "clp")]
#[test]
fn phase_i_replaces_the_completion_cache_identity() {
    let original = kernel();
    let mut zero = original.clone();
    assert!(std::sync::Arc::ptr_eq(
        original.cache_identity(),
        zero.cache_identity()
    ));
    zero.suppress_costs();
    assert!(!std::sync::Arc::ptr_eq(
        original.cache_identity(),
        zero.cache_identity()
    ));
    assert!(!zero.has_costs());
    assert!(!std::sync::Arc::ptr_eq(
        &original.transitions.arcs,
        &zero.transitions.arcs
    ));
    assert!(original.has_costs());
    let clone = original.clone();
    assert!(std::sync::Arc::ptr_eq(
        &original.transitions.arcs,
        &clone.transitions.arcs
    ));
}
