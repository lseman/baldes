use super::*;
fn front(value: f64) -> RelationFront {
    RelationFront::from_pieces(vec![RelationPiece {
        metadata: 0,
        left: [0., value],
        right: [1., value],
        waiting: false,
    }])
}
fn sized_front(count: usize, value: f64) -> RelationFront {
    RelationFront::from_pieces(
        (0..count)
            .map(|i| RelationPiece {
                metadata: 0,
                left: [i as f64, value],
                right: [(i + 1) as f64, value],
                waiting: false,
            })
            .collect(),
    )
}
#[test]
fn explicit_piece_metadata_keeps_original_storage_sizes() {
    assert_eq!(std::mem::size_of::<RelationPiece>(), 40);
    assert_eq!(
        std::mem::size_of::<RelationFront>(),
        std::mem::size_of::<Vec<RelationPiece>>()
    );
}
#[test]
fn invalid_cached_endpoint_index_falls_back_to_literal_arithmetic() {
    let mut f = front(0.5);
    f.pieces[0].metadata =
        RelationFront::META_VALID | RelationFront::META_EXACT | RelationFront::META_INDEX;
    assert_eq!(f.minimum(), 0.5);
    assert!(!f.exact_cache_eligible());
}
#[test]
fn minimum_endpoint_index_preserves_signed_zero_semantics() {
    for count in [1, 8] {
        for ys in [[0_f64, -0_f64], [-0., 0.], [-0., -0.], [0., 0.]] {
            let pieces = (0..count)
                .map(|i| RelationPiece {
                    metadata: 0,
                    left: [i as f64, ys[0]],
                    right: [(i + 1) as f64, ys[1]],
                    waiting: false,
                })
                .collect();
            let mut front = RelationFront::from_pieces(pieces);
            assert!(front.metadata().is_some());
            for cached in [true, false] {
                if !cached {
                    front.pieces[0].metadata = 0;
                }
                let minimum = front.minimum();
                assert_eq!(minimum, 0., "count={count}, cached={cached}");
                // f64::min may choose either input for mixed signed-zero
                // ties, independently in the cache builder and fallback.
                // Equal-sign endpoints must retain their original bits.
                if ys[0].to_bits() == ys[1].to_bits() {
                    assert_eq!(minimum.to_bits(), ys[0].to_bits());
                }
            }
        }
    }
}
#[test]
fn publishing_clipped_or_changed_pieces_rebuilds_all_derived_words() {
    let original = sized_front(8, 2.);
    let mut changed = original.pieces.clone();
    changed[7].right[1] = -1000.;
    let rebuilt = RelationFront::from_pieces(changed);
    assert_eq!(rebuilt.minimum(), -1000.);
    assert_ne!(
        original.rejection_fingerprint(),
        rebuilt.rejection_fingerprint()
    );
    let piece = original.pieces[0].clipped(0.25, 0.75).unwrap().unwrap();
    assert_eq!(piece.metadata, 0);
    let clipped = RelationFront::from_pieces(vec![piece]);
    assert_eq!(clipped.minimum(), 2.);
    assert!(!clipped.exact_cache_eligible());
    let mut marked = original.pieces[0].clone();
    marked.metadata ^= u32::MAX;
    assert_eq!(marked, original.pieces[0]);
}
#[test]
fn only_wide_fronts_pay_for_a_rejection_digest() {
    for value in [0., 1., 2.] {
        let short = sized_front(7, value);
        assert_eq!(short.rejection_fingerprint(), None);
        let wide = sized_front(8, value);
        assert!(wide.rejection_fingerprint().is_some());
        assert_eq!(short, sized_front(7, value));
        assert_eq!(wide, sized_front(8, value));
    }
}
#[test]
fn matching_fingerprint_does_not_replace_literal_piece_equality() {
    let a = sized_front(8, 1.);
    let mut b = sized_front(8, 2.);
    b.pieces[1].metadata = a.pieces[1].metadata;
    b.pieces[2].metadata = a.pieces[2].metadata;
    assert_eq!(a.rejection_fingerprint(), b.rejection_fingerprint());
    assert_ne!(a, b);
    assert_eq!(a, a.clone());
}
#[test]
fn signed_zero_equalities_and_waiting_provenance_survive_fingerprints() {
    let a = sized_front(8, 0.);
    let b = sized_front(8, -0.);
    assert_eq!(a.rejection_fingerprint(), b.rejection_fingerprint());
    assert_eq!(a, b);
    let mut pieces = a.pieces.clone();
    pieces[0].waiting = true;
    assert_ne!(a, RelationFront::from_pieces(pieces));
}
#[test]
fn cached_minimum_and_exactness_match_retained_original_pieces() {
    let f = RelationFront::from_pieces(vec![
        RelationPiece {
            metadata: 0,
            left: [-2., 3.],
            right: [0., -1.],
            waiting: false,
        },
        RelationPiece {
            metadata: 0,
            left: [1., 2.],
            right: [3., 4.],
            waiting: true,
        },
    ]);
    let literal = f.pieces.iter().fold(f64::INFINITY, |minimum, p| {
        minimum.min(p.left[1]).min(p.right[1])
    });
    assert_eq!(f.minimum().to_bits(), literal.to_bits());
    assert!(f.exact_cache_eligible());
    let fractional = RelationFront::from_pieces(vec![RelationPiece {
        metadata: 0,
        left: [0., 0.],
        right: [3., 1.],
        waiting: false,
    }]);
    assert!(!fractional.exact_cache_eligible());
}
#[test]
fn nonfinite_piece_values_cannot_gain_an_exact_certificate_or_nan_equality() {
    let nan = front(f64::NAN);
    assert!(!nan.exact_cache_eligible());
    assert_ne!(nan, nan.clone());
    assert!(!front(f64::INFINITY).exact_cache_eligible());
}
