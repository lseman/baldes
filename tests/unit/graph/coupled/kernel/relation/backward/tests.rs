use super::*;
#[test]
fn compatible_join_keeps_cross_combinations_of_endpoint_singletons() {
    let forward = RelationFront::from_pieces(vec![
        RelationPiece {
            metadata: 0,
            left: [0., 0.],
            right: [0., 0.],
            waiting: false,
        },
        RelationPiece {
            metadata: 0,
            left: [0., 10.],
            right: [1., 10.],
            waiting: false,
        },
    ]);
    let b = BoundedCostFront {
        exact: true,
        unit: 1.,
        ceiling: 10.,
        upper: RelationFront::from_pieces(vec![
            RelationPiece {
                metadata: 0,
                left: [0., 0.],
                right: [0., 0.],
                waiting: false,
            },
            RelationPiece {
                metadata: 0,
                left: [0., 10.],
                right: [1., 10.],
                waiting: false,
            },
        ]),
    };
    assert_eq!(b.checked_compatible_join(&forward), Some(Some(0.)));
}
#[test]
fn fractional_fronts_are_unknown_and_exact_incompatible_fronts_are_distinct() {
    let b = BoundedCostFront::terminal(
        Window {
            lower: 0.,
            upper: 4.,
        },
        10.,
    )
    .unwrap();
    let exact = RelationFront::from_pieces(vec![RelationPiece {
        metadata: 0,
        left: [0., 11.],
        right: [4., 11.],
        waiting: false,
    }]);
    assert_eq!(b.checked_compatible_join(&exact), Some(None));
    let fractional = RelationFront::from_pieces(vec![RelationPiece {
        metadata: 0,
        left: [0., 0.5],
        right: [4., 0.5],
        waiting: false,
    }]);
    assert_eq!(b.checked_compatible_join(&fractional), None);
}
