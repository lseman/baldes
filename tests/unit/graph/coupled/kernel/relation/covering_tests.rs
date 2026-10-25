use super::*;
fn front(pieces: &[([f64; 2], [f64; 2])]) -> RelationFront {
    RelationFront::from_pieces(
        pieces
            .iter()
            .map(|&(left, right)| RelationPiece {
                metadata: 0,
                left,
                right,
                waiting: false,
            })
            .collect(),
    )
}

#[test]
fn relation_dominance_applies_absolute_and_relative_tolerance() {
    let flat = |value| front(&[([0., value], [2., value])]);
    assert!(flat(1. + 0.5e-4).covers_below(&flat(1.)));
    assert!(!flat(1. + 2e-4).covers_below(&flat(1.)));
    assert!(flat(1e8 + 0.01).covers_below(&flat(1e8)));
    assert!(!flat(1e8 + 0.1).covers_below(&flat(1e8)));
    assert!(flat(1. - 0.5e-4).covers_above(&flat(1.)));
    assert!(!flat(1. - 2e-4).covers_above(&flat(1.)));
}

#[test]
fn lower_wider_front_covers_and_crossing_or_narrower_fronts_do_not() {
    let b = front(&[([1., 5.], [3., 7.])]);
    // Lower on a wider domain, with an interior breakpoint inside b.
    let a = front(&[([0., 4.], [2., 4.]), ([2., 4.], [4., 6.])]);
    assert!(a.covers_below(&b));
    assert!(!b.covers_below(&a));
    // Equal endpoints but above b in the interior.
    let bump = front(&[([1., 5.], [2., 8.]), ([2., 8.], [3., 7.])]);
    assert!(!bump.covers_below(&b));
    // Below b but missing part of its domain.
    assert!(!front(&[([1., 0.], [2.5, 0.])]).covers_below(&b));
    // A gap inside b's domain is not covered.
    let gap = front(&[([0., 0.], [1.5, 0.]), ([2., 0.], [4., 0.])]);
    assert!(!gap.covers_below(&b));
    // Singleton pieces are compared at their point.
    let point = front(&[([2., 6.], [2., 6.])]);
    assert!(a.covers_below(&point));
    assert!(!front(&[([2., 6.5], [2., 6.5])]).covers_below(&point));
    assert!(a.covers_below(&a));
}
