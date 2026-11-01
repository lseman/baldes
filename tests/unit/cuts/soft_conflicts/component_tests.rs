use super::*;

fn sequence(sense: ComponentSense, threshold: f64) -> ComponentSequence {
    ComponentSequence {
        bounds: vec![ComponentBound {
            coordinate: 0,
            threshold,
            sense,
        }],
    }
}

#[test]
fn component_membership_matches_reference_capture() {
    let capture: serde_json::Value =
        serde_json::from_str(include_str!("component_capture.json")).unwrap();
    for case in capture["membership_cases"].as_array().unwrap() {
        let sense = if case["sense"] == "G" {
            ComponentSense::GreaterEqual
        } else {
            ComponentSense::LessEqual
        };
        let sequence = sequence(sense, case["bound"].as_f64().unwrap());
        assert_eq!(
            sequence.satisfied_by(|_| case["value"].as_f64()),
            case["accepted"].as_bool().unwrap(),
            "{case}"
        );
    }
}

#[test]
fn component_sparse_values_and_prefix_sibling_match_recovered_comparisons() {
    let greater = sequence(ComponentSense::GreaterEqual, 2.);
    assert!(!greater.satisfied_by(|_| None));
    assert!(greater.satisfied_by(|_| Some(2. - 5e-7)));
    assert!(!greater.satisfied_by(|_| Some(2. - 2e-6)));
    let mut conjunction = greater;
    conjunction.bounds.push(ComponentBound {
        coordinate: 1,
        threshold: 3.,
        sense: ComponentSense::LessEqual,
    });
    let sibling = conjunction.sibling().unwrap();
    assert_eq!(sibling.bounds[0], conjunction.bounds[0]);
    assert_eq!(sibling.bounds[1].threshold, 4.);
    assert_eq!(sibling.bounds[1].sense, ComponentSense::GreaterEqual);
    assert!(sequence(ComponentSense::LessEqual, -100.).satisfied_by(|_| None));
}

#[test]
fn component_join_counts_repeated_arcs_and_signed_nonmonotone_transitions() {
    let row =
        PresenceRow::component_class(1, sequence(ComponentSense::LessEqual, 1.), vec![(0, 1)])
            .unwrap();
    let pricing = PresencePricing::new(
        2,
        vec![PricingInput {
            row,
            arc_events: vec![vec![0], vec![1]],
            dual: -3.,
        }],
    )
    .unwrap();
    let empty = pricing.empty_state();
    let (left, a) = pricing.extend(&empty, 0).unwrap();
    let (right, b) = pricing.extend(&empty, 1).unwrap();
    assert_eq!(a + b + pricing.join_cost(&left, &right).unwrap(), 3.);
    let (twice, _) = pricing.extend(&left, 0).unwrap();
    assert_ne!(left, twice);
    assert_eq!(pricing.join_cost(&twice, &right).unwrap(), -3.);
    let (terminal, _) = pricing.extend(&left, 1).unwrap();
    let (_, decrement) = pricing.extend(&terminal, 0).unwrap();
    assert_eq!(decrement, -3.);
    assert!(
        pricing
            .remaining_cost_lower_bound(&terminal)
            .unwrap()
            .unwrap()
            <= decrement
    );
    assert_eq!(pricing.route_coefficients(&[0, 0, 1]).unwrap(), vec![0]);
    assert_eq!(pricing.route_coefficients(&[1]).unwrap(), vec![1]);
}

#[test]
fn component_siblings_reject_fractional_and_unrepresentable_unit_steps() {
    assert!(sequence(ComponentSense::LessEqual, 0.5).sibling().is_err());
    assert!(sequence(ComponentSense::GreaterEqual, 0.5)
        .sibling()
        .is_err());
    let limit = (1_u64 << 53) as f64;
    assert!(sequence(ComponentSense::LessEqual, limit)
        .sibling()
        .is_err());
    assert!(sequence(ComponentSense::GreaterEqual, -limit)
        .sibling()
        .is_err());
    assert_eq!(
        sequence(ComponentSense::GreaterEqual, limit)
            .sibling()
            .unwrap()
            .bounds[0]
            .threshold,
        limit - 1.
    );
}
