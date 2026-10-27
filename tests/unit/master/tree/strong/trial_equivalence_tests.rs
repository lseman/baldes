use super::retain_distinct_trials;

#[test]
fn product_phase_keeps_distinct_children_beyond_the_next_candidate_quota() {
    // Actual four-network reference RMP scores. Three triangle networks
    // have identical child estimates; the fourth has a different tradeoff.
    let mut candidates = vec![
        (7, 300., [6.333333333333333, 605.8333333333333]),
        (103, 300., [605.8333333333333, 6.333333333333333]),
        (19, 300., [6.333333333333333, 605.8333333333333]),
        (42, 266.6666666666664, [6.5, 405.8333333333333]),
    ];
    retain_distinct_trials(&mut candidates, |c| (c.1, c.2));
    assert_eq!(candidates.iter().map(|c| c.0).collect::<Vec<_>>(), [7, 42]);
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
        [0, 1, 3]
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
