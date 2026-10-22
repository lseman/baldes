use super::*;
#[test]
fn registered_original_names_resolve_ties_without_merging_histories() {
    let mut history = BranchEvaluationHistory::default();
    let a = arc();
    let mut b = arc();
    if let BranchConstraint::ArcFlow { arc, .. } = &mut b.down {
        *arc = 2;
    }
    if let BranchConstraint::ArcFlow { arc, .. } = &mut b.up {
        *arc = 2;
    }
    for candidate in [&a, &b] {
        for side in 0..2 {
            history.record(candidate, side, 0, EvaluationKind::ExactPricing, 0., 1., 0.);
        }
    }
    let mut options = BranchingOptions::arc_flow_only();
    options.generator_names.insert(&a.up, "x_1_20_1_OspV");
    options.generator_names.insert(&b.down, "x_1_10_1_OspV");
    let mut candidates = vec![a.clone(), b.clone()];
    history.select_admitted(
        &mut candidates,
        2,
        4,
        &options,
        Some(&Default::default()),
        10.,
    );
    assert_eq!(candidates[0].down, b.down);
    assert_eq!(history.phase_costs.len(), 2);
    options.generator_names.insert(&a.down, "same");
    options.generator_names.insert(&b.down, "same");
    assert_eq!(history.phase_costs.len(), 2);
    assert_eq!(options.generator_names.get(&a.up), Some("same"));
}
fn arc() -> BranchCandidate {
    BranchCandidate {
        down: BranchConstraint::ArcFlow {
            arc: 1,
            sense: RowSense::LessEqual,
            rhs: 0.,
        },
        up: BranchConstraint::ArcFlow {
            arc: 1,
            sense: RowSense::GreaterEqual,
            rhs: 1.,
        },
        value: 0.5,
    }
}
fn resource() -> BranchCandidate {
    let c = crate::master::branching::AccumulatedResourceGroupBranch::new(1, vec![7], 0, 2., false)
        .unwrap();
    BranchCandidate {
        up: BranchConstraint::AccumulatedResourceGroup(c.with_upper(true)),
        down: BranchConstraint::AccumulatedResourceGroup(c),
        value: 0.5,
    }
}
#[test]
fn an_empty_registered_family_excludes_lower_priority_evaluated_history() {
    let mut h = BranchEvaluationHistory::default();
    let r = resource();
    let a = arc();
    for (c, gain) in [(&a, 1000.), (&r, 1.)] {
        for side in 0..2 {
            h.record(c, side, 0, EvaluationKind::ExactPricing, 0., gain, 0.);
        }
    }
    let mut options = BranchingOptions {
        accumulated_resource: true,
        packing_set_assignment: false,
        ryan_foster: false,
        ..Default::default()
    };
    options.priorities.number_of_vehicles = 2.;
    let mut supplied = vec![r.clone(), a.clone()];
    h.select_admitted(
        &mut supplied,
        2,
        4,
        &options,
        Some(&BranchHistoryOptions::default()),
        f64::INFINITY,
    );
    assert_eq!(supplied.len(), 1);
    assert_eq!(supplied[0].down, r.down);
    // Disabling the intervening registration advances cutoff to arc priority
    // one, so the same evaluated arc is admitted and wins on its score.
    options.number_of_vehicles = false;
    let mut supplied = vec![r.clone(), a.clone()];
    h.select_admitted(
        &mut supplied,
        2,
        4,
        &options,
        Some(&BranchHistoryOptions::default()),
        f64::INFINITY,
    );
    assert_eq!(supplied[0].down, a.down);
    assert_eq!(supplied[1].down, r.down);
}
#[test]
fn retained_resource_history_can_compete_with_a_fresh_vehicle_candidate() {
    let mut h = BranchEvaluationHistory::default();
    let r = resource();
    for side in 0..2 {
        h.record(&r, side, 0, EvaluationKind::ExactPricing, 0., 2., 0.);
    }
    let v = BranchCandidate {
        down: BranchConstraint::NumberOfVehicles {
            graph_id: 0,
            sense: RowSense::LessEqual,
            rhs: 1.,
        },
        up: BranchConstraint::NumberOfVehicles {
            graph_id: 0,
            sense: RowSense::GreaterEqual,
            rhs: 2.,
        },
        value: 1.5,
    };
    let mut supplied = vec![v.clone(), r.clone()];
    let options = BranchingOptions {
        accumulated_resource: true,
        ..Default::default()
    };
    h.select_admitted(
        &mut supplied,
        1,
        4,
        &options,
        Some(&BranchHistoryOptions::default()),
        f64::INFINITY,
    );
    assert_eq!(supplied[0].down, r.down);
    assert_eq!(supplied[1].down, v.down);
    // Root selection/no-history keeps only fresh output and never invents
    // an observation or treats a retained template as freshly separated.
    let mut root = vec![v.clone(), r];
    h.select_admitted(&mut root, 1, 4, &options, None, f64::INFINITY);
    assert_eq!(root.len(), 1);
    assert_eq!(root[0].down, v.down);
}
#[test]
fn cutoff_advances_only_when_the_fresh_quota_remains_open() {
    let buckets = [11., 10., 2., 1., 0.1];
    assert_eq!(separate(&[10., 10., 1.], 100, &buckets), (vec![0, 1], 2.));
    assert_eq!(separate(&[2., 2., 1.], 100, &buckets), (vec![0, 1], 1.));
    assert_eq!(separate(&[1., 1.], 100, &buckets), (vec![0, 1], 0.1));
    assert_eq!(separate(&[1., 1.], 1, &buckets), (vec![0, 1], 1.));
    assert_eq!(separate(&[], 100, &buckets), (vec![], 0.1));
    assert!(admitted(2., 2. + 5e-7));
    assert!(!admitted(1., 2.));
}
#[test]
fn priority_groups_keep_same_integer_bucket_and_support_custom_model_priorities() {
    assert_eq!(
        separate(&[10.9, 10.1, 9.9], 100, &[10.9, 10.1, 9.9, 1.]),
        (vec![0, 1], 9.9)
    );
    assert_eq!(
        separate(&[-0.1, -0.9, -1.1], 100, &[-0.1, -0.9, -1.1]),
        (vec![0, 1], -1.1)
    );
}
