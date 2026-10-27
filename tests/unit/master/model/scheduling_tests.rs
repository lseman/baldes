use super::*;

fn cold_node(policy: NetworkPricingSchedule) -> NodeSnapshot<2> {
    let networks = [-1., -4., -2.]
        .into_iter()
        .enumerate()
        .map(|(id, cost)| {
            let mut graph = Graph::<2>::new(3, 0).unwrap();
            graph.add_arc(0, 1, cost).unwrap();
            graph.add_arc(1, 2, 0.).unwrap();
            Network {
                id: id as i32,
                graph,
                multiplicity: Window::new(0., 1.).unwrap(),
                arc_entries: vec![vec![]; 2],
                route_entries: vec![],
                activity_entries: vec![],
            }
        })
        .collect();
    let mut options = Options::default();
    options.network_pricing_schedule = policy;
    options.preprocessing.enabled = false;
    options.cg.time_limit = std::time::Duration::ZERO;
    solve(
        Problem {
            rows: vec![],
            networks,
        },
        options,
    )
    .unwrap()
    .snapshot
}

#[test]
fn partial_exact_oracles_never_supply_a_complete_bound_or_stale_gradient() {
    for policy in [
        NetworkPricingSchedule::Cyclic,
        NetworkPricingSchedule::Sorted,
        NetworkPricingSchedule::PromisingFirst,
        NetworkPricingSchedule::UnvisitedFirst,
    ] {
        let mut node = cold_node(policy);
        let phase = node.state.oracles[0]
            .solver
            .phases
            .iter()
            .position(|p| p.exact())
            .unwrap();
        node.state.deadline = Instant::now() + std::time::Duration::from_secs(10);
        let duals = vec![0.; node.state.rhs.len()];
        for expected_network in 0..3 {
            let answer = node.state.price(phase, &duals).unwrap();
            assert_eq!(answer.columns.len(), 1);
            assert_eq!(
                node.state.routes[answer.columns[0].tag as usize].network,
                expected_network
            );
            assert!(!answer.exact);
            assert!(!node.state.scheduler.complete);
            assert_eq!(node.state.correction, None);
            assert_eq!(node.state.gradient, None);
        }
        // Previous visits do not count as coverage at a new dual point. Even
        // identical prices require a complete traversal for a fresh bound.
        let answer = node.state.price(phase, &duals).unwrap();
        assert!(answer.exact);
        assert!(node.state.scheduler.complete);
        assert_eq!(node.state.correction, Some(-7.));
        assert!(node.state.gradient.is_some());
        let mut changed = duals.clone();
        let (row, coefficient) = node.state.oracles[2].route_entries[0];
        changed[row] = 5. / coefficient;
        assert!(node.state.price(phase, &changed).unwrap().exact);
        assert_eq!(node.state.correction, Some(-12.));
        // Formulation forks do not inherit score/cursor/completion state.
        let fork = node.fork().unwrap();
        assert!(!fork.state.scheduler.complete);
        assert_eq!(
            fork.state.scheduler.fresh().order(phase, vec![0, 1, 2]),
            vec![0, 1, 2]
        );
        node.state.deadline = Instant::now();
        assert!(node.state.price(phase, &duals).is_err());
        assert!(!node.state.scheduler.complete);
        assert_eq!(node.state.correction, None);
        assert_eq!(node.state.gradient, None);
    }
}

#[test]
fn disabled_and_empty_network_families_need_no_oracle_calls() {
    let mut node = cold_node(NetworkPricingSchedule::Cyclic);
    node.state.deadline = Instant::now() + std::time::Duration::from_secs(10);
    for oracle in &mut node.state.oracles {
        oracle.network.multiplicity.upper = 0.;
    }
    let answer = node
        .state
        .price(0, &vec![0.; node.state.rhs.len()])
        .unwrap();
    assert!(answer.exact);
    assert!(node.state.scheduler.complete);
    assert!(answer.columns.is_empty());
    assert!(node.state.oracles.iter().all(|o| o.calls == 0));
    assert_eq!(node.state.correction, Some(0.));
}

#[test]
fn insignificant_new_routes_do_not_stop_a_scan_before_real_improvement() {
    for policy in [
        NetworkPricingSchedule::Cyclic,
        NetworkPricingSchedule::Sorted,
    ] {
        let mut node = cold_node(policy);
        node.state.scheduler.tolerance = 2.;
        node.state.deadline = Instant::now() + std::time::Duration::from_secs(10);
        let answer = node
            .state
            .price(0, &vec![0.; node.state.rhs.len()])
            .unwrap();
        assert_eq!(answer.columns.len(), 2);
        assert_eq!(
            node.state
                .oracles
                .iter()
                .map(|o| o.calls)
                .collect::<Vec<_>>(),
            vec![1, 1, 0]
        );
        assert!(!answer.exact);
    }
}

#[test]
fn native_queue_duplicate_columns_require_a_fresh_complete_traversal() {
    for policy in [
        NetworkPricingSchedule::PromisingFirst,
        NetworkPricingSchedule::UnvisitedFirst,
    ] {
        let mut node = cold_node(policy);
        let phase = node.state.oracles[0]
            .solver
            .phases
            .iter()
            .position(|p| p.exact())
            .unwrap();
        node.state.deadline = Instant::now() + std::time::Duration::from_secs(10);
        node.state.scheduler.max_unpromising = 2;
        let duals = vec![0.; node.state.rhs.len()];
        let first = node.state.price(phase, &duals).unwrap();
        assert_eq!(node.state.oracles.iter().map(|o| o.calls).sum::<usize>(), 2);
        assert!(!first.exact);
        assert_eq!(node.state.correction, None);
        let second = node.state.price(phase, &duals).unwrap();
        assert!(!second.exact);
        assert_eq!(node.state.correction, None);
        // All returned columns already exist now, so no budget can interrupt
        // the fresh full pass needed for a correction at these duals.
        let third = node.state.price(phase, &duals).unwrap();
        assert!(third.exact);
        assert!(node.state.scheduler.complete);
        assert_eq!(node.state.correction, Some(-7.));
    }
}
