use super::*;

#[test]
fn cyclic_wraps_skips_disabled_slots_and_resets_at_phase_changes() {
    let mut scheduler = Scheduler::new(NetworkPricingSchedule::Cyclic, 4, 1e-6);
    assert_eq!(scheduler.order(2, vec![0, 2, 3]), vec![0, 2, 3]);
    scheduler.record(0, -1.);
    assert_eq!(scheduler.order(2, vec![0, 2, 3]), vec![2, 3, 0]);
    scheduler.record(3, -1.);
    assert_eq!(scheduler.order(2, vec![0, 2, 3]), vec![0, 2, 3]);
    scheduler.record(2, -1.);
    assert_eq!(scheduler.order(0, vec![0, 2, 3]), vec![0, 2, 3]);
    assert!(scheduler.order(0, vec![]).is_empty());
}

#[test]
fn sorted_prioritizes_unseen_then_recent_minima_with_stable_ties() {
    let mut scheduler = Scheduler::new(NetworkPricingSchedule::Sorted, 4, 1e-6);
    assert_eq!(scheduler.order(2, vec![0, 1, 2, 3]), vec![0, 1, 2, 3]);
    scheduler.record(0, -10.);
    scheduler.record(2, -20.);
    assert_eq!(scheduler.order(2, vec![0, 1, 2, 3]), vec![1, 3, 2, 0]);
    scheduler.record(1, -10.);
    scheduler.record(3, 2.);
    assert_eq!(scheduler.order(2, vec![0, 1, 2, 3]), vec![2, 0, 1, 3]);
    scheduler.record(2, 3.);
    assert_eq!(scheduler.order(2, vec![0, 1, 2, 3]), vec![0, 1, 3, 2]);
    assert_eq!(scheduler.order(0, vec![0, 1, 2, 3]), vec![0, 1, 2, 3]);
}

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
        NetworkPricingSchedule::NativePromisingFirst,
        NetworkPricingSchedule::NativeUnvisitedFirst,
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
        assert!(fork.state.scheduler.scores.iter().all(Option::is_none));
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
fn native_queues_replay_observed_modern_batches_and_preserve_phase_order() {
    for policy in [
        NetworkPricingSchedule::NativePromisingFirst,
        NetworkPricingSchedule::NativeUnvisitedFirst,
    ] {
        let mut scheduler = Scheduler::new(policy, 2, 1e-6);
        assert_eq!(scheduler.order(2, vec![0, 1]), vec![0, 1]);
        // Native root observer: network 7 returns zero and adds five columns.
        assert!(scheduler.finish_network(0, 5));
        assert_eq!(scheduler.order(2, vec![0, 1]), vec![1, 0]);
        // Network 42 returns zero and adds one accepted column.
        assert!(scheduler.finish_network(1, 1));
        assert_eq!(scheduler.order(2, vec![0, 1]), vec![0, 1]);
        assert!(!scheduler.finish_network(0, 0));
        assert!(!scheduler.finish_network(1, 0));
        assert_eq!(scheduler.order(1, vec![0, 1]), vec![0, 1]);
        assert!(scheduler.finish_network(0, 1));
        assert_eq!(scheduler.order(0, vec![0, 1]), vec![1, 0]);
    }
}

#[test]
fn native_queue_limit_counts_only_after_new_columns_and_skips_disabled_slots() {
    let mut scheduler = Scheduler::new(NetworkPricingSchedule::NativePromisingFirst, 5, 1e-6);
    scheduler.max_unpromising = 2;
    assert_eq!(scheduler.order(2, vec![0, 2, 3, 4]), vec![0, 2, 3, 4]);
    assert!(!scheduler.finish_network(0, 0));
    assert!(!scheduler.finish_network(2, 1));
    assert!(scheduler.finish_network(3, 0));
    assert_eq!(scheduler.order(2, vec![0, 2, 4]), vec![4, 0, 2]);
    // Even a limit of one never stops a no-new-column pass.
    scheduler.max_unpromising = 1;
    for index in [4, 0, 2] {
        assert!(!scheduler.finish_network(index, 0));
    }
    assert_eq!(scheduler.order(2, vec![0, 2, 3, 4]), vec![3, 4, 0, 2]);
}

#[test]
fn recovered_positive_status_queue_equations_differ_from_modern_column_success() {
    for (policy, expected) in [
        (NetworkPricingSchedule::NativePromisingFirst, vec![0, 1, 2]),
        (NetworkPricingSchedule::NativeUnvisitedFirst, vec![1, 2, 0]),
    ] {
        let mut scheduler = Scheduler::new(policy, 3, 1e-6);
        scheduler.order(2, vec![0, 1, 2]);
        // Static positive-status branch, not observed from the normal modern
        // wrapper. A column-producing status zero goes to the other queue.
        assert!(!scheduler.finish_classified(0, 1, true));
        assert_eq!(scheduler.order(2, vec![0, 1, 2]), expected);
    }
}

#[test]
fn native_queue_duplicate_columns_require_a_fresh_complete_traversal() {
    for policy in [
        NetworkPricingSchedule::NativePromisingFirst,
        NetworkPricingSchedule::NativeUnvisitedFirst,
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

#[test]
fn fresh_preserves_queue_budget_and_resets_all_scan_state() {
    let mut scheduler = Scheduler::new(NetworkPricingSchedule::NativeUnvisitedFirst, 3, 2e-6);
    scheduler.max_unpromising = 2;
    scheduler.order(2, vec![0, 1, 2]);
    scheduler.record(0, -4.);
    assert!(!scheduler.finish_network(0, 1));
    assert!(scheduler.finish_network(1, 0));
    scheduler.complete = true;
    let mut fresh = scheduler.fresh();
    assert_eq!(fresh.policy, scheduler.policy);
    assert_eq!(fresh.tolerance, 2e-6);
    assert_eq!(fresh.max_unpromising, 2);
    assert_eq!(fresh.cursor, 0);
    assert_eq!(fresh.phase, None);
    assert!(fresh.scores.iter().all(Option::is_none));
    assert!(!fresh.complete);
    assert!(fresh.secondary.is_empty());
    assert!(fresh.promising.is_empty());
    assert!(fresh.unpromising.is_empty());
    assert!(!fresh.found_columns);
    assert_eq!(fresh.unpromising_seen, 0);
    assert_eq!(fresh.order(0, vec![0, 1, 2]), vec![0, 1, 2]);
    assert!(!fresh.finish_network(0, 1));
    assert!(fresh.finish_network(1, 0));
}

#[test]
fn native_status_budgets_keep_skips_and_failures_distinct_from_solved_networks() {
    let mut scheduler = Scheduler::new(NetworkPricingSchedule::NativePromisingFirst, 4, 1e-6);
    scheduler.max_promising = 1;
    scheduler.max_unpromising = 1;
    assert_eq!(scheduler.order(2, vec![0, 1, 2, 3]), vec![0, 1, 2, 3]);
    // A skipped visit belongs to U but never consumes the solved-network budget.
    assert!(!scheduler.finish_native_status(0, 1, -2).unwrap());
    assert_eq!(scheduler.unpromising_seen, 0);
    assert!(scheduler.finish_native_status(1, 0, 0).unwrap());
    assert_eq!(scheduler.order(1, vec![0, 1, 2, 3]), vec![2, 3, 0, 1]);
    // Positive status, rather than column count, consumes its independent budget.
    assert!(scheduler.finish_native_status(2, 0, 1).unwrap());
    assert_eq!(scheduler.order(0, vec![0, 1, 2, 3]), vec![2, 3, 0, 1]);
    scheduler.complete = true;
    assert!(scheduler.finish_native_status(2, 0, -1).is_err());
    assert!(!scheduler.complete);
    let fresh = scheduler.fresh();
    assert_eq!(fresh.max_promising, 1);
    assert_eq!(fresh.max_unpromising, 1);
}
