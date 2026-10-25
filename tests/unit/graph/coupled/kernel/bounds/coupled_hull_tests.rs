use super::*;

fn kernel() -> CoupledKernel {
    let mut graph: Graph<2> = Graph::new(2, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    CoupledKernel::prepare(
        &graph,
        &[
            CoupledResource::Selection {
                initial: 3,
                allowed: vec![3],
                vertices: vec![],
                relaxed: false,
            },
            CoupledResource::SimultaneousPickupDelivery {
                capacity: 10.,
                deliveries: vec![0.],
                pickups: vec![0.],
                tolerance: 0.,
            },
            CoupledResource::CumulativeCost {
                initial: 0.,
                increments: vec![0.],
                weights: vec![1.],
                sink_upper: None,
                source_upper: None,
            },
        ],
    )
    .unwrap()
}
#[test]
fn bucket_hulls_tighten_cost_and_preserve_union_compatibility() {
    let kernel = kernel();
    let mut query = kernel.root(true);
    query.numeric[0] = 2.;
    query.selection[0] = 1;
    let mut local = kernel.root(false);
    local.selection[0] = 2;
    local.suffix.as_mut().unwrap().coefficients[0] = 3.;
    let mut other = kernel.root(false);
    other.suffix.as_mut().unwrap().coefficients[0] = -10.;
    let mut bucket = CoupledStateBounds::default();
    bucket.include(Some(&local));
    let mut global = bucket.clone();
    global.include(Some(&other));
    assert_eq!(
        kernel.completion_correction_lower(true, Some(&query), &bucket),
        6.
    );
    assert_eq!(
        kernel.completion_correction_lower(true, Some(&query), &global),
        -20.
    );
    assert!(!kernel.completion_feasible(true, Some(&query), &bucket));
    assert!(kernel.completion_feasible(true, Some(&query), &global));
}
#[test]
fn load_envelopes_reject_only_when_all_opposite_states_exceed_capacity() {
    let kernel = kernel();
    let mut query = kernel.root(true);
    query.loads[0] = [6., 0., 0.];
    let mut full = kernel.root(false);
    full.loads[0] = [5., 0., 0.];
    let mut bucket = CoupledStateBounds::default();
    bucket.include(Some(&full));
    assert!(!kernel.completion_feasible(true, Some(&query), &bucket));
    let mut smaller = kernel.root(false);
    smaller.loads[0] = [2., 0., 0.];
    bucket.include(Some(&smaller));
    assert!(kernel.completion_feasible(true, Some(&query), &bucket));
    bucket.include(None);
    assert!(kernel.completion_feasible(true, Some(&query), &bucket));
}
