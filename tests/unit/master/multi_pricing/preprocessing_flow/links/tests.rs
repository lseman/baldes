use super::*;
use std::time::Duration;

#[test]
fn exact_shared_link_elimination_is_bounded_and_rejects_rounded_coefficients() {
    let deadline = Instant::now() + Duration::from_secs(10);
    let target = Constraint {
        entries: vec![(0, 1.), (1, 1.)],
        sense: RowSense::GreaterEqual,
        rhs: 1.5,
    };
    let equality = Constraint {
        entries: vec![(0, -1.), (1, -1.), (2, 1.)],
        sense: RowSense::Equal,
        rhs: 0.,
    };
    let (derived, spent, complete) =
        derive(&[target.clone()], &[equality.clone()], 10_000, deadline);
    assert!(complete && spent < 10_000);
    assert!(derived
        .iter()
        .any(|r| r.entries == [(2, 1.)] && r.rhs == 1.5 && r.sense == RowSense::GreaterEqual));
    assert_eq!(
        derive(&[target.clone()], &[equality.clone()], 0, deadline).1,
        0
    );
    assert!(!derive(&[target.clone()], &[equality.clone()], 0, deadline).2);
    assert!(
        !derive(
            &[target.clone()],
            &[equality.clone()],
            10_000,
            Instant::now()
        )
        .2
    );
    let mut rounded = target.clone();
    rounded.rhs = 1e16;
    let mut shifted = equality.clone();
    shifted.rhs = 1.;
    assert!(derive(&[rounded], &[shifted], 10_000, deadline)
        .0
        .is_empty());
    let mut rounded = target;
    rounded.entries[1].1 = 1e16;
    let mut shifted = equality;
    shifted.entries[1].1 = 1.;
    assert!(derive(&[rounded], &[shifted], 10_000, deadline)
        .0
        .is_empty());
}
