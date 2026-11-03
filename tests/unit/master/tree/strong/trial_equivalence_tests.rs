use super::retain_distinct_trials;

#[test]
fn ordered_equal_trials_match_native_across_and_within_families() {
    let capture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../fixtures/cuts/strong_ordered_equal_trials_native.json"
    ))
    .unwrap();
    for case in capture["cases"].as_array().unwrap() {
        let mut candidates: Vec<_> = case["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|candidate| {
                (
                    candidate["name"].as_str().unwrap().to_owned(),
                    candidate["score"].as_f64().unwrap(),
                    candidate["bounds"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|value| value.as_f64().unwrap())
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        retain_distinct_trials(&mut candidates, |c| (c.1, c.2.clone()));
        let actual: Vec<_> = candidates.iter().take(3).map(|c| c.0.as_str()).collect();
        let expected: Vec<_> = case["advanced"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap())
            .collect();
        assert_eq!(actual, expected, "{}", case["source"]);
    }
}

#[test]
fn multiway_equivalence_compares_every_child_and_requires_equal_arity() {
    let mut candidates = vec![
        (0, 12., vec![2., 6.]),
        (1, 12., vec![2., 6., 8.]),
        (2, 12., vec![8., 6., 2.]),
        (3, 12., vec![2., 6., 9.]),
    ];
    retain_distinct_trials(&mut candidates, |c| (c.1, c.2.clone()));
    assert_eq!(
        candidates.iter().map(|c| c.0).collect::<Vec<_>>(),
        [0, 1, 2, 3]
    );
}

#[test]
fn product_phase_keeps_distinct_children_beyond_the_next_candidate_quota() {
    // Four-network reference RMP scores, including a reversed child order.
    let mut candidates = vec![
        (7, 300., [6.333333333333333, 605.8333333333333]),
        (103, 300., [605.8333333333333, 6.333333333333333]),
        (19, 300., [6.333333333333333, 605.8333333333333]),
        (42, 266.6666666666664, [6.5, 405.8333333333333]),
    ];
    retain_distinct_trials(&mut candidates, |c| (c.1, c.2));
    assert_eq!(
        candidates.iter().map(|c| c.0).collect::<Vec<_>>(),
        [7, 103, 42]
    );
}

#[test]
fn equal_product_alone_does_not_make_candidates_equivalent() {
    let mut candidates = vec![
        (0, 12., [2., 6.]),
        (1, 12., [3., 4.]),
        (2, 12. + 5e-7, [6. + 5e-7, 2.]),
        (3, 12., [2. + 1e-4, 6.]),
    ];
    retain_distinct_trials(&mut candidates, |c| (c.1, c.2));
    assert_eq!(
        candidates.iter().map(|c| c.0).collect::<Vec<_>>(),
        [0, 1, 2, 3]
    );
}

#[test]
fn equivalence_respects_relative_slack_score_groups_and_nonfinite_estimates() {
    let mut candidates = vec![
        (0, 100., [1e12, 4.]),
        (1, 100., [1e12 + 50., 4.]),
        (2, 100., [1e12 + 200., 4.]),
        (3, 99., [1e12, 4.]),
        (4, 99., [f64::INFINITY, 4.]),
        (5, 99., [f64::INFINITY, 4.]),
    ];
    retain_distinct_trials(&mut candidates, |c| (c.1, c.2));
    assert_eq!(
        candidates.iter().map(|c| c.0).collect::<Vec<_>>(),
        [0, 2, 3, 4, 5]
    );
}

#[test]
fn tree_depth_near_scores_use_first_child_creation_order() {
    use super::tree_depth_score_less;
    // Observed native score/first-child-ID pairs, including both input orders.
    for values in [
        vec![(-2., 0), (-1.9999999, 1)],
        vec![(-1.9999999, 1), (-2., 0)],
    ] {
        let mut ranked = values;
        crate::cuts::shared::stdsort::stable_sort(&mut ranked, |&a, &b| {
            tree_depth_score_less(a, b)
        });
        assert_eq!(ranked, [(-2., 0), (-1.9999999, 1)]);
    }
    let mut ranked = vec![(-2., 9), (-2., 3), (-1.9, 15)];
    crate::cuts::shared::stdsort::stable_sort(&mut ranked, |&a, &b| tree_depth_score_less(a, b));
    assert_eq!(ranked, [(-1.9, 15), (-2., 3), (-2., 9)]);
    // The wider tree-depth band scales with score magnitude.
    assert!(tree_depth_score_less((-20., 0), (-19.999999, 1)));
}
