use super::super::PairedFrontOperations;
use super::*;

fn function(pairs: &[(f64, f64)]) -> PreparedSlopeFunction {
    PreparedSlopeFunction::from_prepared_pairs(pairs.to_vec()).unwrap()
}
fn complete<T: std::fmt::Debug>(outcome: FrontierOutcome<T>) -> T {
    match outcome {
        FrontierOutcome::Complete(value) => value,
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn machine_reduction_merges_near_slopes_across_ordinary_front() {
    let front =
        SlopeFront::from_prepared(function(&[(1., 0.), (2., 1.), (2.00005, 2.), (-1., 3.)]));
    let reduced = complete(front.machine_reduce_for_continuation(FrontierBudget::default()));
    assert_eq!(
        reduced.components()[0].pairs(),
        &[(1., 0.), (2., 2.), (-1., 3.)]
    );
    assert!(matches!(
        reduced.provenance.as_ref(),
        FrontierProvenance::MachineReduction { merged: true, .. }
    ));
}

#[test]
fn backward_machine_root_uses_sink_upper_maximum_and_literal_fields() {
    let upper = [
        function(&[(2., 0.), (3., 1.), (-1., 2.)]),
        function(&[(4., 0.), (0., 2.)]),
    ];
    let root = complete(MachineRootState::from_backward_inputs(
        &[&upper[0], &upper[1]],
        1.,
        4.,
        FrontierBudget::default(),
    ));
    assert_eq!(root.front().components()[0].pairs(), &[(5., 1.), (0., 4.)]);
    assert_eq!(root.waiting_bits(), 0);
    assert_eq!(root.predecessor(), 1e12);
    assert_eq!(root.key(), 5.);
    let empty = complete(MachineRootState::from_backward_inputs(
        &[],
        1.,
        4.,
        FrontierBudget::default(),
    ));
    assert_eq!(empty.key(), -1e12);
    let signed = function(&[(-0.0, 0.), (0., 1.)]);
    let signed_root = complete(MachineRootState::from_backward_inputs(
        &[&signed],
        0.,
        1.,
        FrontierBudget::default(),
    ));
    assert_eq!(signed_root.key().to_bits(), (-0.0f64).to_bits());
}

#[test]
fn backward_machine_key_scans_interior_endpoints_and_preserves_mode_sentinel() {
    let front = SlopeFront::from_prepared(function(&[(2., 0.), (3., 1.), (-1., 2.)]));
    let budget = FrontierBudget::default();
    assert_eq!(complete(front.machine_backward_key(5, 10., budget)), 5.);
    assert_eq!(complete(front.machine_backward_key(4, 10., budget)), 1e12);
    assert!(matches!(
        front.machine_backward_key(
            5,
            10.,
            FrontierBudget {
                max_pairs: 2,
                ..budget
            },
        ),
        FrontierOutcome::Unknown(FrontierUnknown::PairLimit)
    ));
}

#[test]
fn duplicate_waiting_carries_the_direct_predecessor_and_respects_budget() {
    let function =
        PreparedSlopeFunction::from_machine_pairs(vec![(2., 1.), (3., 1.), (4., 3.)]).unwrap();
    let state = MachineForwardState {
        front: SlopeFront::from_prepared(function),
        waiting_bits: 1 << 2,
        predecessor: None,
        key: 0.,
    };
    let budget = FrontierBudget::default();
    assert_eq!(
        complete(state.machine_internal_duplicate_predecessor_at(1., budget)),
        1.
    );
    assert_eq!(
        complete(state.machine_internal_duplicate_predecessor_at(2., budget)),
        1.
    );
    assert!(matches!(
        state.machine_internal_duplicate_predecessor_at(
            2.,
            FrontierBudget {
                max_operations: 10,
                ..budget
            },
        ),
        FrontierOutcome::Unknown(FrontierUnknown::WorkLimit)
    ));
}

#[test]
fn machine_history_backtracks_through_a_multisegment_inverse() {
    let budget = FrontierBudget::default();
    let root = complete(MachineRootState::from_prepared(
        &function(&[(0., 0.), (0., 3.)]),
        budget,
    ));
    let history = complete(MachinePathHistory::from_root(&root, 2, budget));
    let state = MachineForwardState {
        front: SlopeFront::from_prepared(function(&[(0., 0.), (0., 3.)])),
        waiting_bits: 0,
        predecessor: None,
        key: 0.,
    };
    let history = MachinePathHistory {
        head: Arc::new(MachinePathNode {
            previous: Some(history.head.clone()),
            snapshot: MachinePathSnapshot::Forward { arc: 7, state },
        }),
        root: history.root.clone(),
        len: 2,
        max_states: 2,
    };
    let inverse = function(&[(0., 0.), (0.5, 1.), (0.5, 3.)]);
    let coordinates = complete(PairedFrontOperations::machine_backtrack_coordinates(
        &history,
        2.,
        &[(7, &inverse)],
        budget,
    ));
    assert_eq!(coordinates, vec![1., 2.]);
    assert!(matches!(
        PairedFrontOperations::machine_backtrack_coordinates(
            &history,
            2.,
            &[(8, &inverse)],
            budget,
        ),
        FrontierOutcome::Unknown(_)
    ));
}

#[test]
fn native_slope_addition_and_literal_replay_have_distinct_witnesses() {
    let a = function(&[(1073741824., 0.), (1., 1.)]);
    let b = function(&[(-1073741824., 0.), (0., 1.)]);
    let c = function(&[(0., 0.), (-1., 1.)]);
    let budget = FrontierBudget::default();
    let ab = complete(SlopeFront::from_prepared(a.clone()).add(
        &SlopeFront::from_prepared(b.clone()),
        FrontOperationPolicy::FrontAlgebra,
        budget,
    ));
    assert_eq!(ab.components()[0].pairs(), &[(0., 0.), (1., 1.)]);
    let abc = complete(ab.add(
        &SlopeFront::from_prepared(c.clone()),
        FrontOperationPolicy::FrontAlgebra,
        budget,
    ));
    assert_eq!(abc.components()[0].pairs(), &[(0., 0.), (0., 1.)]);
    assert_eq!(complete(abc.minimum(budget)).value, 0.);
    // This argument is outside the native start-snap band. The difference
    // is operation scope, not a difference in leaf boundary policy.
    let x = 2_f64.powi(-12) + 2_f64.powi(-24);
    let replay = SourceReplayWitness::evaluate_shared_argument(
        &[a, b, c],
        x,
        EvaluationPolicy::EndpointTolerance,
    )
    .unwrap()
    .unwrap();
    assert_eq!(replay.value(), -2_f64.powi(-24));
    assert!(matches!(
        &**abc.provenance(),
        FrontierProvenance::Addition { .. }
    ));
}

#[test]
fn native_alignment_preserves_left_coordinates_and_active_slopes() {
    let budget = FrontierBudget::default();
    let a = SlopeFront::from_prepared(function(&[(0., 0.), (1., 5.), (2., 10.)]));
    let b = SlopeFront::from_prepared(function(&[(0., 0.), (3., 5.00001), (4., 10.)]));
    let native = complete(a.add(&b, FrontOperationPolicy::FrontAlgebra, budget));
    assert_eq!(
        native.components()[0].pairs(),
        &[(0., 0.), (4., 5.), (6., 10.)]
    );
    assert!(native.used_tolerant_alignment());
    let strict = complete(a.add(&b, FrontOperationPolicy::StrictDomain, budget));
    assert_eq!(
        strict.components()[0].pairs(),
        &[(0., 0.), (4., 5.), (5., 5.00001), (6., 10.)]
    );
    assert!(!strict.used_tolerant_alignment());
}

#[test]
fn strict_policy_does_not_expand_a_nearby_source_start() {
    let a = SlopeFront::from_prepared(function(&[(1., 0.), (2., 10.)]));
    let b = SlopeFront::from_prepared(function(&[(3., 0.00001), (4., 10.)]));
    let native = complete(a.add(
        &b,
        FrontOperationPolicy::FrontAlgebra,
        FrontierBudget::default(),
    ));
    assert_eq!(native.components()[0].pairs(), &[(4., 0.), (6., 10.)]);
    let strict = complete(a.add(
        &b,
        FrontOperationPolicy::StrictDomain,
        FrontierBudget::default(),
    ));
    assert_eq!(strict.components()[0].domain().0, 0.00001);
    assert_eq!(strict.components()[0].pairs()[0].0, 4.00002);
    let a = SlopeFront::from_prepared(function(&[(1., 0.), (2., 10.00001)]));
    let b = SlopeFront::from_prepared(function(&[(3., 0.), (4., 10.)]));
    let native = complete(a.add(
        &b,
        FrontOperationPolicy::FrontAlgebra,
        FrontierBudget::default(),
    ));
    assert_eq!(native.components()[0].pairs(), &[(4., 0.), (6., 10.00001)]);
    let strict = complete(a.add(
        &b,
        FrontOperationPolicy::StrictDomain,
        FrontierBudget::default(),
    ));
    assert_eq!(strict.components()[0].domain().1, 10.);
}

#[test]
fn disconnected_domains_and_singleton_minima_are_not_bridged() {
    let budget = FrontierBudget::default();
    let components = complete(SlopeFront::from_components(
        vec![function(&[(3., 0.), (1., 1.)]), function(&[(-2., 3.)])],
        budget,
    ));
    let other = SlopeFront::from_prepared(function(&[(0., 0.), (1., 4.)]));
    let result = complete(components.add(&other, FrontOperationPolicy::StrictDomain, budget));
    assert_eq!(result.components().len(), 2);
    assert_eq!(result.components()[0].domain(), (0., 1.));
    assert_eq!(result.components()[1].pairs(), &[(1., 3.)]);
    let witness = complete(result.minimum(budget));
    assert_eq!(
        (witness.coordinate, witness.value, witness.component),
        (3., 1., 1)
    );
}

#[test]
fn empty_unknown_limits_and_arithmetic_are_distinct() {
    let a = SlopeFront::from_prepared(function(&[(1., 0.), (0., 1.)]));
    let far = SlopeFront::from_prepared(function(&[(2., 2.), (0., 3.)]));
    assert!(matches!(
        a.add(
            &far,
            FrontOperationPolicy::StrictDomain,
            FrontierBudget::default()
        ),
        FrontierOutcome::Empty
    ));
    let zero = FrontierBudget {
        max_operations: 0,
        ..FrontierBudget::default()
    };
    assert!(matches!(
        a.add(&a, FrontOperationPolicy::StrictDomain, zero),
        FrontierOutcome::Unknown(FrontierUnknown::WorkLimit)
    ));
    assert!(matches!(
        a.minimum(zero),
        FrontierOutcome::Unknown(FrontierUnknown::WorkLimit)
    ));
    let huge = SlopeFront::from_prepared(function(&[(f64::MAX, 0.)]));
    assert!(matches!(
        huge.add(
            &huge,
            FrontOperationPolicy::StrictDomain,
            FrontierBudget::default()
        ),
        FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(_))
    ));
    let near_point = SlopeFront::from_prepared(function(&[(0., 1.00001)]));
    assert!(matches!(
        a.add(
            &near_point,
            FrontOperationPolicy::FrontAlgebra,
            FrontierBudget::default()
        ),
        FrontierOutcome::Unknown(FrontierUnknown::Unsupported(_))
    ));
}

#[test]
fn zero_translation_and_restriction_preserve_slope_arithmetic() {
    let budget = FrontierBudget::default();
    assert_eq!(
        SlopeFront::zero(2., 2.).unwrap().components()[0].pairs(),
        &[(0., 2.)]
    );
    assert_eq!(
        SlopeFront::zero(0., 2.).unwrap().components()[0].pairs(),
        &[(0., 0.), (0., 2.)]
    );
    let source = SlopeFront::from_prepared(function(&[(1., 0.), (2., 2.)]));
    let interior = complete(source.restrict(0.5, 1.5, FrontOperationPolicy::FrontAlgebra, budget));
    assert_eq!(interior.components()[0].pairs(), &[(2., 0.5), (2., 1.5)]);
    let singleton = complete(source.restrict(1., 1., FrontOperationPolicy::FrontAlgebra, budget));
    assert_eq!(singleton.components()[0].pairs(), &[(3., 1.)]);
    let shifted = complete(interior.translate(4., budget));
    assert_eq!(shifted.components()[0].pairs(), &[(2., 4.5), (2., 5.5)]);
    assert!(matches!(
        &**shifted.provenance(),
        FrontierProvenance::Translation { .. }
    ));
}

#[test]
fn restriction_retains_gaps_and_translation_collapse_is_unknown() {
    let budget = FrontierBudget::default();
    let source = complete(SlopeFront::from_components(
        vec![
            function(&[(0., 0.), (1., 1.)]),
            function(&[(2., 3.), (-1., 4.)]),
        ],
        budget,
    ));
    let restricted =
        complete(source.restrict(0.5, 3.5, FrontOperationPolicy::StrictDomain, budget));
    assert_eq!(restricted.components().len(), 2);
    assert_eq!(restricted.components()[0].pairs(), &[(0.5, 0.5), (1., 1.)]);
    assert_eq!(restricted.components()[1].pairs(), &[(2., 3.), (-1., 3.5)]);
    assert!(matches!(
        source.restrict(1.5, 2., FrontOperationPolicy::StrictDomain, budget),
        FrontierOutcome::Empty
    ));
    assert!(matches!(
        source.translate(1e20, budget),
        FrontierOutcome::Unknown(FrontierUnknown::Unsupported(_))
    ));
    assert!(matches!(
        source.restrict(0.00001, 0.5, FrontOperationPolicy::FrontAlgebra, budget),
        FrontierOutcome::Unknown(FrontierUnknown::Unsupported(_))
    ));
    assert!(matches!(
        source.translate(
            1.,
            FrontierBudget {
                max_operations: 0,
                ..budget
            }
        ),
        FrontierOutcome::Unknown(FrontierUnknown::WorkLimit)
    ));
}

#[test]
fn machine_dependent_bounds_match_reference_crossings_and_waiting_bits() {
    // Observed by internal/reference/metasolver_paired_operators.py, ELF 71f320.
    for (pairs, bound, bits, expected, expected_bits) in [
        (
            vec![(0., 0.), (2., 1.), (0., 3.)],
            1.,
            2,
            vec![(1., 0.), (0., 0.5), (2., 1.), (0., 3.)],
            6,
        ),
        (
            vec![(2., 0.), (0., 1.), (2., 3.)],
            3.,
            4,
            vec![(3., 0.), (0., 1.5), (2., 3.)],
            6,
        ),
        (
            vec![(2., 0.), (0., 1.), (0., 3.)],
            1.,
            6,
            vec![(2., 0.), (0., 3.)],
            2,
        ),
    ] {
        let envelope = MachineWaitingEnvelope {
            function: function(&pairs),
            waiting_bits: bits,
            predecessor: Some(0.),
        };
        let actual = complete(envelope.apply_lower_bound(
            &function(&[(bound, 0.), (0., 3.)]),
            FrontierBudget::default(),
        ));
        assert_eq!(actual.function().pairs(), expected);
        assert_eq!(actual.waiting_bits(), expected_bits);
        assert_eq!(actual.predecessor(), Some(0.));
    }
    // The upper filter clips its connected retained region; its caller keeps
    // the existing bits and predecessor rather than remapping them.
    for (pairs, expected) in [
        (vec![(0., 0.), (2., 2.)], vec![(0., 0.), (2., 0.5)]),
        (vec![(2., 0.), (-2., 2.)], vec![(1., 0.5), (-2., 2.)]),
    ] {
        let envelope = MachineWaitingEnvelope {
            function: function(&pairs),
            waiting_bits: 2,
            predecessor: Some(0.),
        };
        let actual = complete(
            envelope.apply_upper_bound(&function(&[(1., 0.), (0., 2.)]), FrontierBudget::default()),
        );
        assert_eq!(actual.function().pairs(), expected);
        assert_eq!(actual.waiting_bits(), 2);
    }
    // A front entirely above the bound is an empty (infeasible) state, as
    // the native filter's count -1, not an unknown status.
    let above = MachineWaitingEnvelope {
        function: function(&[(6., 0.), (0., 2.)]),
        waiting_bits: 0,
        predecessor: None,
    };
    assert!(matches!(
        above.apply_upper_bound(&function(&[(5.5, 0.), (0., 2.)]), FrontierBudget::default()),
        FrontierOutcome::Empty
    ));
}

#[test]
fn forward_machine_mixed_scan_matches_reference_without_source_equivalence() {
    let input = function(&[(2., 0.), (2., 1.), (-2., 3.), (2., 4.)]);
    let front = SlopeFront::from_prepared(input.clone());
    let actual = complete(front.machine_waiting(0., 4., 1, FrontierBudget::default()));
    // Reference full forward extension ELF 720890 with identity mapping,
    // zero increment and nonbinding dependent bounds produces this same scan.
    assert_eq!(actual.function().pairs(), &[(2., 0.), (-2., 3.), (1., 4.)]);
    assert_eq!(actual.waiting_bits(), 4);
    assert_eq!(actual.predecessor(), None);
    // Independent piecewise-linear waiting oracle: a linear objective on
    // each source segment attains its minimum at an endpoint.
    let source_minimum = [0., 1., 3.]
        .into_iter()
        .map(|x| {
            input
                .evaluate(x, EvaluationPolicy::StrictDomain)
                .unwrap()
                .unwrap()
                + (3. - x)
        })
        .fold(f64::INFINITY, f64::min);
    assert_eq!(source_minimum, 0.);
    assert_eq!(
        actual
            .function()
            .evaluate(3., EvaluationPolicy::StrictDomain)
            .unwrap(),
        Some(-4.)
    );
}
