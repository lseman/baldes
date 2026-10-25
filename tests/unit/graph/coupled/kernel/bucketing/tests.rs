use super::*;
use crate::graph::coupled::PiecewiseLinearFunction;
use crate::graph::Window;

fn example() -> RelationFront {
    RelationFront::initial(
        &PiecewiseLinearFunction {
            points: vec![[0., 3.], [1., 0.], [2., 4.], [3., 2.]],
        },
        Window {
            lower: 0.,
            upper: 3.,
        },
    )
    .unwrap()
    .unwrap()
}

#[test]
fn views_borrow_every_original_piece_and_keep_global_normalization() {
    let source = example();
    let policy = FrontBucketPolicy::PieceGroups {
        pieces_per_view: NonZeroUsize::new(1).unwrap(),
        max_views: NonZeroUsize::new(3).unwrap(),
    };
    let mut views = FrontBucketViews::new(&source, policy);
    assert_eq!(views.len(), 3);
    for original in source.pieces() {
        let view = views.next().unwrap();
        assert!(std::ptr::eq(view.source, &source));
        assert!(std::ptr::eq(&view.pieces[0], original));
        assert_eq!(
            view.normalization_minimum.to_bits(),
            source.minimum().to_bits()
        );
        assert_eq!(
            view.base_hull(),
            Some([original.left[0], original.right[0]])
        );
    }
    assert!(views.next().is_none());
    assert!(views.next().is_none());
}

#[test]
fn disabled_or_over_budget_keeps_one_whole_front_view() {
    let source = example();
    for policy in [
        FrontBucketPolicy::Disabled,
        FrontBucketPolicy::PieceGroups {
            pieces_per_view: NonZeroUsize::new(1).unwrap(),
            max_views: NonZeroUsize::new(2).unwrap(),
        },
    ] {
        let mut views = FrontBucketViews::new(&source, policy);
        assert_eq!(views.len(), 1);
        let view = views.next().unwrap();
        assert_eq!(view.pieces, source.pieces());
        assert!(views.next().is_none());
    }
}
