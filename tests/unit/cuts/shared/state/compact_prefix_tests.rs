use super::*;
#[test]
fn omitted_words_match_zero_extended_states_in_all_readers() {
    let prefix = [1, 0];
    let mut padded = [0; LIMITED_WORDS];
    padded[..2].copy_from_slice(&prefix);
    let mut other = [0; LIMITED_WORDS];
    other[0] = 2;
    other[18] = 2;
    let descriptors = vec![
        DominanceDescriptor {
            word: 0,
            mask: 3,
            denominator: 3,
            cost: 0.25,
            ..Default::default()
        },
        DominanceDescriptor {
            word: 18,
            mask: 3,
            denominator: 3,
            cost: 1.25,
            ..Default::default()
        },
    ];
    for reverse in [false, true] {
        for initial in [-2., -0.75, -0.1] {
            let mut full_gap = initial;
            let mut short_gap = initial;
            let (a, b, sa, sb): (&[u32], &[u32], &[u32], &[u32]) = if reverse {
                (&other, &padded, &other, &prefix)
            } else {
                (&padded, &other, &prefix, &other)
            };
            assert_eq!(
                packed_dominate(&descriptors, a, b, &mut full_gap),
                packed_dominate(&descriptors, sa, sb, &mut short_gap)
            );
            assert_eq!(full_gap.to_bits(), short_gap.to_bits());
        }
    }
    let mut info = CutResourceInfo::default();
    info.vertices.resize_with(1, VertexInfo::default);
    info.vertices[0].dominance = descriptors;
    assert_eq!(
        concatenation_cost(&info, 0, (&padded, 0.1), (&other, 0.2), -4.).to_bits(),
        concatenation_cost(&info, 0, (&prefix, 0.1), (&other, 0.2), -4.).to_bits()
    );
    let transitions = [
        ArcTransition {
            source_word: 18,
            target_word: 19,
            mask: 1,
            denominator: 2,
            increment: 1,
            overflow_cost: 0.5,
            ..Default::default()
        },
        ArcTransition {
            source_word: 0,
            target_word: 0,
            mask: 1,
            denominator: 2,
            increment: 1,
            overflow_cost: 0.5,
            ..Default::default()
        },
    ];
    let mut full_cost = -0.;
    let mut short_cost = -0.;
    let expected = changed_along_arc(&padded, &mut full_cost, &transitions);
    let actual = changed_along_arc(&prefix, &mut short_cost, &transitions);
    assert_eq!(actual, expected);
    assert_eq!(actual[19], 1);
    assert_eq!(full_cost.to_bits(), short_cost.to_bits());
}
