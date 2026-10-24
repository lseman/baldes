use super::*;

#[test]
fn all_small_incidence_splits_match_independent_pair_presence() {
    let pairs = vec![(0, 1), (1, 2), (2, 3), (0, 1), (3, 3)];
    for mode in [Mode::PairCount, Mode::AnyPair] {
        let row = PresenceRow::new(4, pairs.clone(), mode).unwrap();
        for a in 0_u32..16 {
            for b in 0_u32..16 {
                let mut left = row.empty_state();
                let mut right = row.empty_state();
                let mut left_increment = 0;
                let mut right_increment = 0;
                for event in 0..4 {
                    if a & (1 << event) != 0 {
                        left_increment += row.visit(&mut left, event).unwrap();
                        assert_eq!(row.visit(&mut left, event).unwrap(), 0);
                    }
                    // Reverse traversal must represent the same incidence.
                    let event = 3 - event;
                    if b & (1 << event) != 0 {
                        right_increment += row.visit(&mut right, event).unwrap();
                        assert_eq!(row.visit(&mut right, event).unwrap(), 0);
                    }
                }
                let mask = a | b;
                let count = pairs
                    .iter()
                    .filter(|&&(x, y)| mask & (1 << x) != 0 && mask & (1 << y) != 0)
                    .count();
                let expected = match mode {
                    Mode::PairCount => count,
                    Mode::AnyPair => usize::from(count > 0),
                };
                let joined = row.join(&left, &right).unwrap();
                assert_eq!(row.coefficient(&joined).unwrap(), expected);
                assert_eq!(left_increment, row.coefficient(&left).unwrap());
                assert_eq!(right_increment, row.coefficient(&right).unwrap());
                assert_eq!(
                    left_increment as i64
                        + right_increment as i64
                        + row.join_correction(&left, &right).unwrap(),
                    expected as i64
                );
                assert_eq!(
                    row.join(&left, &right).unwrap(),
                    row.join(&right, &left).unwrap()
                );
                assert_eq!(row.join(&joined, &joined).unwrap(), joined);
            }
        }
    }
}

#[test]
fn arbitrary_event_counts_and_failed_visits_preserve_state() {
    let empty = PresenceRow::new(0, vec![], Mode::AnyPair).unwrap();
    assert_eq!(empty.coefficient(&empty.empty_state()).unwrap(), 0);
    assert_eq!(
        empty
            .join_correction(&empty.empty_state(), &empty.empty_state())
            .unwrap(),
        0
    );
    let row = PresenceRow::new(130, vec![(0, 64), (64, 129)], Mode::PairCount).unwrap();
    let mut state = row.empty_state();
    assert_eq!(row.visit(&mut state, 64).unwrap(), 0);
    let saved = state.clone();
    assert!(row.visit(&mut state, 130).is_err());
    assert_eq!(state, saved);
    assert_eq!(row.visit(&mut state, 129).unwrap(), 1);
    assert_eq!(row.visit(&mut state, 0).unwrap(), 1);
    assert_eq!(row.coefficient(&state).unwrap(), 2);
    let other = PresenceRow::new(129, vec![], Mode::AnyPair)
        .unwrap()
        .empty_state();
    assert!(row.join(&state, &other).is_err());
    assert!(row.coefficient(&other).is_err());
    assert!(PresenceRow::new(1, vec![(0, 1)], Mode::PairCount).is_err());
}

#[test]
fn equal_coefficients_do_not_imply_equal_future_behavior() {
    let row = PresenceRow::new(3, vec![(0, 2)], Mode::AnyPair).unwrap();
    let mut a = row.empty_state();
    let mut b = row.empty_state();
    row.visit(&mut a, 0).unwrap();
    row.visit(&mut b, 1).unwrap();
    assert_eq!(row.coefficient(&a).unwrap(), row.coefficient(&b).unwrap());
    assert_ne!(a, b);
    assert_eq!(row.visit(&mut a, 2).unwrap(), 1);
    assert_eq!(row.visit(&mut b, 2).unwrap(), 0);
}

#[test]
fn physical_arc_pricing_matches_replay_at_every_bidirectional_split() {
    // Parallel arcs 0/1 represent the same original variable. Arc 2 emits two
    // variables; duplicate events on arc 3 must not change presence or cost.
    let mapping = vec![vec![0], vec![0], vec![1, 2], vec![2, 2], vec![3]];
    for mode in [Mode::PairCount, Mode::AnyPair] {
        for dual in [-0.375, 0., 1e-12, 0.625] {
            let pricing = PresencePricing::new(
                5,
                vec![PricingInput {
                    row: PresenceRow::new(4, vec![(0, 1), (2, 3), (0, 1)], mode).unwrap(),
                    arc_events: mapping.clone(),
                    dual,
                }],
            )
            .unwrap();
            for route in [vec![0, 2, 4], vec![1, 3, 2, 0, 4, 4], vec![3, 3], vec![]] {
                let events: std::collections::HashSet<_> = route
                    .iter()
                    .flat_map(|&a| mapping[a].iter().copied())
                    .collect();
                let count = [(0, 1), (2, 3), (0, 1)]
                    .iter()
                    .filter(|&&(a, b)| events.contains(&a) && events.contains(&b))
                    .count();
                let coefficient = match mode {
                    Mode::PairCount => count,
                    Mode::AnyPair => usize::from(count > 0),
                };
                assert_eq!(
                    pricing.route_coefficients(&route).unwrap(),
                    vec![coefficient]
                );
                for split in 0..=route.len() {
                    let mut a = pricing.empty_state();
                    let mut b = pricing.empty_state();
                    let mut cost = 0.;
                    for &arc in &route[..split] {
                        let (next, increment) = pricing.extend(&a, arc).unwrap();
                        a = next;
                        cost += increment;
                    }
                    for &arc in route[split..].iter().rev() {
                        let (next, increment) = pricing.extend(&b, arc).unwrap();
                        b = next;
                        cost += increment;
                    }
                    cost += pricing.join_cost(&a, &b).unwrap();
                    assert!((cost + dual * coefficient as f64).abs() <= 1e-15);
                }
            }
        }
    }
}

#[test]
fn pricing_mapping_validation_and_failures_are_atomic() {
    let row = PresenceRow::new(2, vec![(0, 1)], Mode::AnyPair).unwrap();
    assert!(PresencePricing::new(
        1,
        vec![PricingInput {
            row: row.clone(),
            arc_events: vec![],
            dual: 0.
        }]
    )
    .is_err());
    assert!(PresencePricing::new(
        1,
        vec![PricingInput {
            row: row.clone(),
            arc_events: vec![vec![2]],
            dual: 0.
        }]
    )
    .is_err());
    assert!(PresencePricing::new(
        1,
        vec![PricingInput {
            row: row.clone(),
            arc_events: vec![vec![]],
            dual: f64::NAN
        }]
    )
    .is_err());
    let pricing = PresencePricing::new(
        1,
        vec![PricingInput {
            row,
            arc_events: vec![vec![0, 1]],
            dual: 0.5,
        }],
    )
    .unwrap();
    let before = pricing.empty_state();
    let saved = before.clone();
    assert!(pricing.extend(&before, 1).is_err());
    assert_eq!(before, saved);
    assert!(pricing.extend(&None, 0).is_err());
    let empty = PresencePricing::new(1, vec![]).unwrap();
    assert_eq!(empty.extend(&None, 0).unwrap(), (None, 0.));
    assert_eq!(empty.join_cost(&None, &None).unwrap(), 0.);
}

#[test]
fn multiple_presence_rows_keep_independent_coefficients_and_reject_cost_overflow() {
    let pricing = PresencePricing::new(
        2,
        vec![
            PricingInput {
                row: PresenceRow::new(2, vec![(0, 1), (0, 1)], Mode::PairCount).unwrap(),
                arc_events: vec![vec![0], vec![1]],
                dual: 0.25,
            },
            PricingInput {
                row: PresenceRow::new(1, vec![(0, 0)], Mode::AnyPair).unwrap(),
                arc_events: vec![vec![0], vec![0]],
                dual: -0.5,
            },
        ],
    )
    .unwrap();
    assert_eq!(pricing.route_coefficients(&[0, 1, 0]).unwrap(), vec![2, 1]);
    let (a, ac) = pricing.extend(&pricing.empty_state(), 0).unwrap();
    let (b, bc) = pricing.extend(&pricing.empty_state(), 1).unwrap();
    assert_eq!(ac + bc + pricing.join_cost(&a, &b).unwrap(), 0.);
    let overflowing = PresencePricing::new(
        1,
        vec![PricingInput {
            row: PresenceRow::new(1, vec![(0, 0), (0, 0)], Mode::PairCount).unwrap(),
            arc_events: vec![vec![0]],
            dual: f64::MAX,
        }],
    )
    .unwrap();
    let before = overflowing.empty_state();
    let saved = before.clone();
    assert!(overflowing.extend(&before, 0).is_err());
    assert_eq!(before, saved);
    // Master coefficient replay does not evaluate dual products.
    assert_eq!(overflowing.route_coefficients(&[0]).unwrap(), vec![2]);
}

#[test]
fn directed_remaining_and_join_envelopes_cover_every_event_incidence() {
    let mapping: Vec<_> = (0..4).map(|e| vec![e]).collect();
    for mode in [Mode::PairCount, Mode::AnyPair] {
        for dual in [-0.375, -0.1, -1e-12, 0., 0.1, 0.625, f64::from_bits(1)] {
            let pricing = PresencePricing::new(
                4,
                vec![PricingInput {
                    row: PresenceRow::new(4, vec![(0, 1), (2, 3), (0, 1)], mode).unwrap(),
                    arc_events: mapping.clone(),
                    dual,
                }],
            )
            .unwrap();
            for mask in 0_u32..16 {
                let route: Vec<_> = (0..4).filter(|&e| mask & (1 << e) != 0).collect();
                let mut a = pricing.empty_state();
                for &arc in &route {
                    a = pricing.extend(&a, arc).unwrap().0;
                }
                let ca = pricing.route_coefficients(&route).unwrap()[0];
                let remaining = pricing.remaining_cost_lower_bound(&a).unwrap().unwrap();
                let join = pricing.join_correction_lower_bound(&a).unwrap().unwrap();
                for other in 0_u32..16 {
                    let suffix: Vec<_> = (0..4).filter(|&e| other & (1 << e) != 0).collect();
                    let mut b = pricing.empty_state();
                    for &arc in &suffix {
                        b = pricing.extend(&b, arc).unwrap().0;
                    }
                    let combined: Vec<_> = route.iter().chain(&suffix).copied().collect();
                    let union = pricing.route_coefficients(&combined).unwrap()[0];
                    assert!(remaining <= -dual * (union - ca) as f64);
                    assert!(join <= pricing.join_cost(&a, &b).unwrap());
                }
            }
        }
    }
}

#[test]
fn presence_envelope_overflow_is_unknown_and_negative_dual_overlap_is_allowed() {
    let pricing = PresencePricing::new(
        1,
        vec![PricingInput {
            row: PresenceRow::new(1, vec![(0, 0), (0, 0)], Mode::PairCount).unwrap(),
            arc_events: vec![vec![0]],
            dual: f64::MAX,
        }],
    )
    .unwrap();
    assert_eq!(
        pricing
            .remaining_cost_lower_bound(&pricing.empty_state())
            .unwrap(),
        None
    );
    assert_eq!(
        pricing
            .join_correction_lower_bound(&pricing.empty_state())
            .unwrap(),
        None
    );
    let negative = PresencePricing::new(
        1,
        vec![PricingInput {
            row: PresenceRow::new(1, vec![(0, 0)], Mode::AnyPair).unwrap(),
            arc_events: vec![vec![0]],
            dual: -0.5,
        }],
    )
    .unwrap();
    let state = negative.extend(&negative.empty_state(), 0).unwrap().0;
    assert_eq!(
        negative.remaining_cost_lower_bound(&state).unwrap(),
        Some(0.)
    );
    assert_eq!(
        negative.join_correction_lower_bound(&state).unwrap(),
        Some(-0.5)
    );
    assert_eq!(negative.join_cost(&state, &state).unwrap(), -0.5);
}

#[test]
fn unchanged_presence_transitions_share_state_and_siblings_remain_independent() {
    let pricing = PresencePricing::new(
        4,
        vec![PricingInput {
            row: PresenceRow::new(2, vec![(0, 1)], Mode::PairCount).unwrap(),
            arc_events: vec![vec![], vec![0], vec![0, 0], vec![1]],
            dual: 0.75,
        }],
    )
    .unwrap();
    let root = pricing.empty_state();
    let (idle, cost) = pricing.extend(&root, 0).unwrap();
    assert_eq!(cost, 0.);
    assert!(Arc::ptr_eq(root.as_ref().unwrap(), idle.as_ref().unwrap()));
    let (left, cost) = pricing.extend(&root, 1).unwrap();
    assert_eq!(cost, 0.); // A new bit matters even before it completes a pair.
    assert!(!Arc::ptr_eq(root.as_ref().unwrap(), left.as_ref().unwrap()));
    assert_ne!(left, root);
    let (repeated, cost) = pricing.extend(&left, 2).unwrap();
    assert_eq!(cost, 0.);
    assert!(Arc::ptr_eq(
        left.as_ref().unwrap(),
        repeated.as_ref().unwrap()
    ));
    let (complete, cost) = pricing.extend(&left, 3).unwrap();
    assert_eq!(cost, -0.75);
    let (right, cost) = pricing.extend(&root, 3).unwrap();
    assert_eq!(cost, 0.);
    assert_eq!(pricing.join_cost(&left, &right).unwrap(), -0.75);
    assert_ne!(left, complete);
    assert!(Arc::ptr_eq(root.as_ref().unwrap(), idle.as_ref().unwrap()));
    // The fast path must still reject invalid states and physical arcs.
    assert!(pricing.extend(&None, 0).is_err());
    assert!(pricing.extend(&root, 4).is_err());
}
