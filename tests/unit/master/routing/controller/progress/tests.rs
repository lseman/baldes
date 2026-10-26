use super::*;
use crate::cuts::knapsack::{FenchelParameters, RouteLoadKnapsackCut};

fn empty_cut_state() -> CutState {
    let mut graph = Graph::<2>::new(3, 1).unwrap();
    for vertex in 0..3 {
        graph.set_bounds(vertex, 0, 0., 10.).unwrap();
    }
    graph.add_arc(0, 1, 1.).unwrap();
    graph.add_arc(1, 2, 1.).unwrap();
    let problem = RoutingContext::set_partitioning(graph, 1.).unwrap();
    let mut options = RootOptions::default();
    options.capture_state = true;
    options.enumerate = false;
    super::super::super::solve(problem, options)
        .unwrap()
        .snapshot
        .unwrap()
        .cut
        .clone()
}

#[test]
fn enabled_families_count_live_rows_without_double_counting() {
    let mut state = empty_cut_state();
    assert!(state.progress_counts(&[]).is_empty());
    state.options.enabled = true;
    state.rows = vec![Some(0), None, Some(1), None];
    state.kpath_indices = vec![2, 3];
    state.kpath_separator = Some(RootKPathSeparator::Standalone(
        KPathSeparator::new(vec![0., 1.], 2.).unwrap(),
    ));
    let mut capacity = CapacitySeparator::new(vec![0., 1.], 2.).unwrap();
    capacity.parameters.use_two_path = true;
    state.capacity_separator = Some(capacity);
    state.capacity_cuts = [
        CapacityKind::Capacity,
        CapacityKind::Complement,
        CapacityKind::TwoPath,
        CapacityKind::Capacity,
    ]
    .iter()
    .enumerate()
    .map(|(id, &kind)| CapacityCut {
        id,
        kind,
        customers: vec![],
        other: vec![],
        rhs: 0,
        k: 0,
        demand: 0.,
        violation: 0.,
        edges: Default::default(),
    })
    .collect();
    state.capacity_rows = vec![Some(2), Some(3), Some(4), None];
    let mut separator = ResourceKnapsackSeparator::default();
    separator.parameters.fenchel = Some(FenchelParameters::default());
    state.knapsack_separator = Some(separator);
    let knapsack: Vec<_> = [
        KnapsackKind::Steps,
        KnapsackKind::Fenchel,
        KnapsackKind::Fenchel,
    ]
    .iter()
    .map(|&kind| crate::cuts::knapsack::ResourceKnapsackCut {
        id: 0,
        row: 0,
        cut_class: 0,
        graph_resources: Default::default(),
        dual: 0.,
        cut: RouteLoadKnapsackCut {
            kind,
            numerator: 1,
            denominator: 1,
            rhs: 0,
            lhs: 0.,
            violation: 0.,
            thresholds: vec![],
            coefficients: Default::default(),
        },
    })
    .collect();
    state.knapsack_rows = vec![Some(5), Some(6), None];
    assert_eq!(
        state.progress_counts(&knapsack),
        vec![
            ("Rank1", 1),
            ("Capacity", 2),
            ("2Path", 1),
            ("KPath", 1),
            ("Knapsack", 1),
            ("Fenchel", 1),
        ]
    );
    state.options.enabled = false;
    let capacity = state.capacity_separator.as_mut().unwrap();
    capacity.parameters.enabled = false;
    state.kpath_separator = None;
    state
        .knapsack_separator
        .as_mut()
        .unwrap()
        .parameters
        .max_one_k = 0;
    assert_eq!(state.progress_counts(&knapsack), vec![("Fenchel", 1)]);
    state
        .knapsack_separator
        .as_mut()
        .unwrap()
        .parameters
        .fenchel = None;
    assert!(state.progress_counts(&knapsack).is_empty());
}
