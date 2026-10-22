use super::*;

#[test]
fn sparse_state_support_tracks_changed_rows_across_zero_multipliers() {
    let columns: Vec<_> = (0..129)
        .map(|i| {
            let mut packing = vec![99; 128];
            packing[0] = i % 2;
            packing[63] = (i / 2) % 2;
            packing[127] = 1;
            let mut covering = packing.clone();
            covering[0] = 1 - packing[0];
            covering[63] = 1 - packing[63];
            ColumnInfo {
                packing_set_count: packing,
                covering_set_count: covering,
                ..Default::default()
            }
        })
        .collect();
    let indices: Vec<i32> = (0..129).rev().collect();
    let mut bitmaps = vec![];
    for (row, weight, constant) in [(0, 2, 1), (63, 3, 1), (0, 0, 0)] {
        let mut multipliers = vec![0; 128];
        multipliers[row] = weight;
        multipliers[127] = constant;
        for covering in [false, true] {
            if covering {
                compute_state_bitmap::<true>(&mut bitmaps, 4, &multipliers, &columns, &indices);
            } else {
                compute_state_bitmap::<false>(&mut bitmaps, 4, &multipliers, &columns, &indices);
            }
            for (state, bitmap) in bitmaps.iter().enumerate() {
                for position in 0..192 {
                    let expected = if position < indices.len() {
                        let column = indices[position];
                        let count = if row == 0 {
                            column % 2
                        } else {
                            (column / 2) % 2
                        };
                        let count = if covering { 1 - count } else { count };
                        state as i32 == weight * count + constant
                    } else {
                        false
                    };
                    assert_eq!(
                        bitmap[position / 64] & (1 << (position % 64)) != 0,
                        expected
                    );
                }
            }
        }
    }
}

#[test]
fn reused_state_bitmaps_clear_old_assignments_and_truncated_words() {
    let columns: Vec<_> = (0..129)
        .map(|i| ColumnInfo {
            packing_set_count: vec![(i % 2) as i32, ((i + 1) % 2) as i32],
            covering_set_count: vec![((i + 1) % 2) as i32, (i % 2) as i32],
            ..Default::default()
        })
        .collect();
    let mut bitmaps = vec![vec![u64::MAX; 4]; 5];
    for n in [129, 65, 1, 0, 129] {
        let indices: Vec<i32> = (0..n).collect();
        for multipliers in [[1, 2], [2, 1], [0, 0]] {
            for covering in [false, true] {
                if covering {
                    compute_state_bitmap::<true>(&mut bitmaps, 2, &multipliers, &columns, &indices);
                } else {
                    compute_state_bitmap::<false>(
                        &mut bitmaps,
                        2,
                        &multipliers,
                        &columns,
                        &indices,
                    );
                }
                assert_eq!(bitmaps.len(), 3);
                for (state, bitmap) in bitmaps.iter().enumerate() {
                    assert_eq!(bitmap.len(), (n as usize + 63) / 64);
                    for position in 0..bitmap.len() * 64 {
                        let expected = if position < n as usize {
                            let counts = if covering {
                                &columns[position].covering_set_count
                            } else {
                                &columns[position].packing_set_count
                            };
                            state as i32 == counts[0] * multipliers[0] + counts[1] * multipliers[1]
                        } else {
                            false
                        };
                        assert_eq!(
                            bitmap[position / 64] & (1 << (position % 64)) != 0,
                            expected
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn randomized_row_sets_are_reproducible_with_independent_searches() {
    fn check<const COVERING: bool>() {
        let parent = Separation {
            num_packing_sets: 8,
            num_covering_sets: 8,
            ..Default::default()
        };
        let columns: Vec<_> = (0..8)
            .map(|row| {
                let mut counts = vec![0; 8];
                counts[row] = 1;
                ColumnInfo {
                    value: 1.,
                    packing_set_count: counts.clone(),
                    covering_set_count: counts,
                    packing_set_sequence: vec![row as i32],
                    ..Default::default()
                }
            })
            .collect();
        let elementary: Vec<_> = (0..8).collect();
        let mut left = LocalSearch::<COVERING>::new(&parent, &columns, &elementary);
        let mut right = LocalSearch::<COVERING>::new(&parent, &columns, &elementary);
        let mut unrelated = LocalSearch::<COVERING>::new(&parent, &columns, &elementary);
        let draw = |search: &mut LocalSearch<COVERING>| {
            let mut rows = vec![];
            if COVERING {
                search.build_covering_row_set(&columns, &mut rows, &[1, 1, 1], 0.5);
            } else {
                search.build_packing_row_set(&parent, &columns, &mut rows, &[1, 1, 1], 2, 0.5);
            }
            rows
        };
        for _ in 0..20 {
            let expected = draw(&mut left);
            for _ in 0..4 {
                draw(&mut unrelated);
            }
            assert_eq!(expected, draw(&mut right));
            assert_eq!(expected.len(), 3);
            let mut unique = expected;
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(unique.len(), 3);
        }
    }
    check::<false>();
    check::<true>();
}
