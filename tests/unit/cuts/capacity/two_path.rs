use super::*;

fn prepared() -> Separation {
    let separator = CapacitySeparator::new(vec![0., 1., 2., 1., 2., 1., 2.], 100.).unwrap();
    separator.build(false).unwrap()
}

fn run_growth(state: &mut Separation, memoized: bool) -> (Vec<Bits1024>, TwoPath) {
    let n = state.n;
    let mut growth = TwoPath {
        size: 0,
        cut_weight: 0.,
        demand: 0.,
        vector_18: vec![0.; n + 1],
        vector_30: vec![0.; n + 1],
        customers: Bits1024::default(),
    };
    for i in 0..n {
        for &nb in &state.graph.neighbours[i] {
            growth.vector_30[i] += state.graph.flow[i][nb];
        }
    }
    let mut visited = HashSet::new();
    if memoized {
        state.kohl_greedy::<true>(&mut growth, &mut visited);
    } else {
        state.kohl_greedy::<false>(&mut growth, &mut visited);
    }
    (state.two_path_subsets.iter(), growth)
}

#[test]
fn exact_dyadic_growth_preserves_leaves_order_and_restored_state() {
    let mut baseline = prepared();
    let n = baseline.n;
    for i in 0..n {
        for j in i + 1..n {
            let value = if (i + j) % 3 == 0 { 0.25 } else { 0.5 };
            baseline.graph.flow[i][j] = value;
            baseline.graph.flow[j][i] = value;
            baseline.graph.neighbours[i].push(j);
            baseline.graph.neighbours[j].push(i);
        }
    }
    assert!(baseline.exact_two_path_inputs());
    let mut optimized = baseline.clone();
    let (expected, original) = run_growth(&mut baseline, false);
    let (actual, result) = run_growth(&mut optimized, true);
    assert_eq!(actual, expected);
    assert_eq!(result.customers, original.customers);
    assert_eq!(result.size, original.size);
    assert_eq!(result.demand.to_bits(), original.demand.to_bits());
    assert_eq!(result.cut_weight.to_bits(), original.cut_weight.to_bits());
    assert_eq!(result.vector_18, original.vector_18);
    assert_eq!(result.vector_30, original.vector_30);
}

#[test]
fn non_grid_or_large_inputs_disable_subset_memoization() {
    let mut state = prepared();
    assert!(state.exact_two_path_inputs());
    state.demands[0] = 0.1;
    assert!(!state.exact_two_path_inputs());
    state.demands[0] = 1.;
    state.graph.flow[0][1] = 1e100;
    assert!(!state.exact_two_path_inputs());
}
