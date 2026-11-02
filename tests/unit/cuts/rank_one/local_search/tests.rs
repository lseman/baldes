use super::*;

#[test]
fn bounded_search_draws_match_reference_distribution_including_rejection() {
    let mut rng = MinStdRandom::new();
    // Captured by calling the binary's bounded distribution with state 1.
    for (range, expected, state) in [
        (1, 0, 16_807),
        (2, 0, 282_475_249),
        (3, 2, 1_622_650_073),
        (5, 2, 984_943_658),
        (1_024, 545, 1_144_108_930),
        (1_073_741_824, 470_211_271, 470_211_272),
        (1_073_741_824, 101_027_543, 101_027_544),
        (1_073_741_824, 823_564_439, 823_564_440),
        (7, 3, 1_115_438_165),
        (31, 25, 1_784_484_492),
    ] {
        assert_eq!(rng.uniform_below(range), expected);
        assert_eq!(rng.state, state);
    }
}

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
    let mut active = vec![];
    for (row, weight, constant) in [(0, 2, 1), (63, 3, 1), (0, 0, 0)] {
        let mut multipliers = vec![0; 128];
        multipliers[row] = weight;
        multipliers[127] = constant;
        for covering in [false, true] {
            if covering {
                compute_state_bitmap::<true>(
                    &mut bitmaps,
                    4,
                    &multipliers,
                    &columns,
                    &indices,
                    &mut active,
                );
            } else {
                compute_state_bitmap::<false>(
                    &mut bitmaps,
                    4,
                    &multipliers,
                    &columns,
                    &indices,
                    &mut active,
                );
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
    let mut active = vec![];
    for n in [129, 65, 1, 0, 129] {
        let indices: Vec<i32> = (0..n).collect();
        for multipliers in [[1, 2], [2, 1], [0, 0]] {
            for covering in [false, true] {
                if covering {
                    compute_state_bitmap::<true>(
                        &mut bitmaps,
                        2,
                        &multipliers,
                        &columns,
                        &indices,
                        &mut active,
                    );
                } else {
                    compute_state_bitmap::<false>(
                        &mut bitmaps,
                        2,
                        &multipliers,
                        &columns,
                        &indices,
                        &mut active,
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

#[test]
fn packing_search_starts_match_captured_reference_binary() {
    let counts = [
        [1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 1],
    ];
    let values = [1.0, 0.5, 0.5, 0.5];
    let columns: Vec<_> = counts
        .into_iter()
        .zip(values)
        .map(|(counts, value)| ColumnInfo {
            value,
            elementary: true,
            packing_set_count: counts.into(),
            ..Default::default()
        })
        .collect();
    let parent = Separation {
        num_packing_sets: 5,
        ..Default::default()
    };
    let mut search = LocalSearch::<false>::new(&parent, &columns, &[0, 1, 2, 3]);
    let expected = [
        ([0, 1, 4, 2], 984943658),
        ([2, 0, 1, 4], 1457850878),
        ([3, 4, 1, 2], 1115438165),
        ([4, 0, 1, 3], 1137522503),
        ([3, 0, 2, 1], 143542612),
        ([2, 3, 1, 4], 1998097157),
        ([4, 2, 0, 3], 1404280278),
        ([2, 3, 4, 1], 1636807826),
        ([1, 0, 4, 2], 704877633),
        ([3, 4, 2, 0], 784558821),
        ([1, 4, 3, 2], 1617819336),
        ([3, 0, 2, 4], 1899894091),
        ([1, 2, 4, 0], 1025921153),
        ([1, 2, 3, 0], 357571490),
        ([2, 4, 3, 0], 130060903),
        ([4, 2, 1, 0], 685118024),
        ([4, 1, 0, 2], 194847408),
        ([4, 0, 2, 1], 824938981),
        ([1, 4, 2, 0], 997389814),
        ([4, 0, 3, 2], 1654001669),
    ];
    for (rows, state) in expected {
        let mut actual = Vec::new();
        search.build_packing_row_set(&parent, &columns, &mut actual, &[2, 1, 1, 1], 3, 0.5);
        assert_eq!(actual, rows);
        assert_eq!(search.rng.state, state);
    }
}

#[test]
fn covering_search_starts_match_captured_reference_binary() {
    let counts = [
        [1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 1],
    ];
    let values = [1.0, 0.5, 0.5, 0.5];
    let columns: Vec<_> = counts
        .into_iter()
        .zip(values)
        .map(|(counts, value)| ColumnInfo {
            value,
            elementary: true,
            covering_set_count: counts.into(),
            ..Default::default()
        })
        .collect();
    let parent = Separation {
        num_covering_sets: 5,
        ..Default::default()
    };
    let mut search = LocalSearch::<true>::new(&parent, &columns, &[0, 1, 2, 3]);
    let expected = [
        [3, 2, 4, 1],
        [3, 2, 1, 0],
        [4, 3, 1, 2],
        [0, 2, 4, 1],
        [0, 1, 2, 3],
        [1, 0, 4, 3],
        [2, 0, 1, 4],
        [2, 3, 4, 0],
        [2, 0, 4, 3],
        [3, 2, 4, 0],
        [4, 3, 2, 1],
        [4, 3, 1, 0],
        [0, 1, 4, 2],
        [1, 0, 2, 4],
        [2, 0, 1, 4],
        [0, 1, 2, 4],
        [1, 3, 2, 4],
        [3, 4, 1, 2],
        [2, 3, 1, 4],
        [4, 0, 2, 1],
    ];
    for rows in expected {
        let mut actual = Vec::new();
        search.build_covering_row_set(&columns, &mut actual, &[1, 1, 1, 1], 0.5);
        assert_eq!(actual, rows);
    }
}

#[test]
fn covering_random_stream_continues_across_search_rounds() {
    let mut parent = Separation {
        num_covering_sets: 5,
        ..Default::default()
    };
    parent.params.ls_num_iterations = 1;
    let columns = vec![ColumnInfo {
        value: 0.5,
        covering_set_count: vec![1; 5],
        ..Default::default()
    }];
    let mut first = LocalSearch::<true>::new(&parent, &columns, &[0]);
    let mut cuts = CandidateSet::new();
    first.run(&mut parent, &columns, &mut cuts, 4, 2.0, &[1; 4], -3, 0);
    let mut continued = first.covering_rng.clone();
    let mut next = LocalSearch::<true>::new(&parent, &columns, &[0]);
    assert_eq!(next.covering_rng.next(), continued.next());
    assert_ne!(next.covering_rng.next(), GnuRand::new(1).next());
}
