use super::*;

impl Scheduler {
    fn finish_classified(&mut self, index: usize, new_columns: usize, promising: bool) -> bool {
        self.finish_status(index, new_columns, i32::from(promising))
            .unwrap()
    }
}

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

#[test]
fn native_queues_replay_observed_modern_batches_and_preserve_phase_order() {
    for policy in [
        NetworkPricingSchedule::PromisingFirst,
        NetworkPricingSchedule::UnvisitedFirst,
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
    let mut scheduler = Scheduler::new(NetworkPricingSchedule::PromisingFirst, 5, 1e-6);
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
        (NetworkPricingSchedule::PromisingFirst, vec![0, 1, 2]),
        (NetworkPricingSchedule::UnvisitedFirst, vec![1, 2, 0]),
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
fn fresh_preserves_queue_budget_and_resets_all_scan_state() {
    let mut scheduler = Scheduler::new(NetworkPricingSchedule::UnvisitedFirst, 3, 2e-6);
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
    let mut scheduler = Scheduler::new(NetworkPricingSchedule::PromisingFirst, 4, 1e-6);
    scheduler.max_promising = 1;
    scheduler.max_unpromising = 1;
    assert_eq!(scheduler.order(2, vec![0, 1, 2, 3]), vec![0, 1, 2, 3]);
    // A skipped visit belongs to U but never consumes the solved-network budget.
    assert!(!scheduler.finish_status(0, 1, -2).unwrap());
    assert_eq!(scheduler.unpromising_seen, 0);
    assert!(scheduler.finish_status(1, 0, 0).unwrap());
    assert_eq!(scheduler.order(1, vec![0, 1, 2, 3]), vec![2, 3, 0, 1]);
    // Positive status, rather than column count, consumes its independent budget.
    assert!(scheduler.finish_status(2, 0, 1).unwrap());
    assert_eq!(scheduler.order(0, vec![0, 1, 2, 3]), vec![2, 3, 0, 1]);
    scheduler.complete = true;
    assert!(scheduler.finish_status(2, 0, -1).is_err());
    assert!(!scheduler.complete);
    let fresh = scheduler.fresh();
    assert_eq!(fresh.max_promising, 1);
    assert_eq!(fresh.max_unpromising, 1);
}
