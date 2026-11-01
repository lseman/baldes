use super::*;

fn number(value: &serde_json::Value) -> f64 {
    value.as_f64().unwrap()
}

fn state(value: &serde_json::Value) -> SplitState {
    SplitState {
        pairs: value["pairs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| (number(&p[0]), number(&p[1])))
            .collect(),
        waiting_bits: value["waiting_bits"].as_u64().unwrap(),
        predecessor: number(&value["predecessor"]),
        key: number(&value["key"]),
    }
}

#[test]
fn paired_split_states_match_captured_reference_arrays_and_failure_boundaries() {
    let cases: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../../fixtures/paired_split_states.json"
    ))
    .unwrap();
    let cases = cases.as_array().unwrap();
    assert_eq!(cases.len(), 547);
    for (index, case) in cases.iter().enumerate() {
        let input = state(&case["input"]);
        let ends: Vec<f64> = case["bucket_ends"]
            .as_array()
            .map(|xs| xs.iter().map(number).collect())
            .unwrap_or_else(|| vec![1., 2., 3., 4.]);
        let result = split_state(
            &input,
            case["forward"].as_bool().unwrap(),
            case["strategy"].as_i64().unwrap() as i32,
            number(&case["halfway"]),
            case["mode"].as_i64().unwrap() as i32,
            100.,
            &ends,
            case["bucket_grid"]
                .as_array()
                .map(|xs| [number(&xs[0]), number(&xs[1]), number(&xs[2])])
                .unwrap_or([0., 4., 1.]),
            FrontierBudget::default(),
        );
        if case["expected_error"].as_bool().unwrap() {
            assert!(result.is_err(), "case {index}: {case}");
            continue;
        }
        let result = result.unwrap_or_else(|e| panic!("case {index}: {case}: {e:?}"));
        let expected = case["expected"].as_array().unwrap();
        assert_eq!(result.len(), expected.len(), "case {index}: {case}");
        for (actual, expected) in result.iter().zip(expected) {
            let expected = state(expected);
            assert_eq!(actual.pairs, expected.pairs, "case {index}: {case}");
            assert_eq!(
                actual.waiting_bits, expected.waiting_bits,
                "case {index}: {case}"
            );
            assert_eq!(
                actual.predecessor.to_bits(),
                expected.predecessor.to_bits(),
                "case {index}: {case}"
            );
            assert_eq!(
                actual.key.to_bits(),
                expected.key.to_bits(),
                "case {index}: {case}"
            );
        }
    }
}

#[test]
fn paired_split_work_and_output_limits_are_failures_not_empty_domains() {
    let input = SplitState {
        pairs: vec![(0., 0.), (1., 2.), (2., 4.)],
        waiting_bits: 6,
        predecessor: -1.,
        key: 0.,
    };
    for budget in [
        FrontierBudget {
            max_operations: 0,
            ..Default::default()
        },
        FrontierBudget {
            max_components: 1,
            ..Default::default()
        },
        FrontierBudget {
            max_pairs: 1,
            ..Default::default()
        },
        FrontierBudget {
            max_elapsed: Some(Duration::ZERO),
            ..Default::default()
        },
    ] {
        assert!(split_state(
            &input,
            true,
            2,
            3.,
            0,
            100.,
            &[1., 2., 3., 4.],
            [0., 4., 1.],
            budget
        )
        .is_err());
    }
}
