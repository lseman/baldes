use super::*;
use std::time::Instant;

impl Separation {
    /// Three-row packing cuts.
    pub(super) fn reference_pack(&mut self, columns: &[ColumnInfo], cuts: &mut CandidateSet) {
        let n = self.num_packing_sets.max(0) as usize;
        let mut pv = vec![vec![0.0f64; n]; n];
        for col in columns {
            if n == 0 {
                continue;
            }
            let present: Vec<usize> = (0..n).filter(|&p| col.packing_set_count[p] != 0).collect();
            for a in 1..present.len() {
                for b in 0..a {
                    let (pa, pb) = (present[a], present[b]);
                    pv[pb][pa] += col.value;
                    pv[pa][pb] += col.value;
                }
            }
        }
        for i in 0..n {
            for j in i + 1..n {
                let vij = pv[i][j];
                if vij == 0.0 {
                    continue;
                }
                for k in j + 1..n {
                    let approx = vij + pv[i][k] + pv[j][k];
                    if !(approx - 1.0 > self.params.violation_tolerance) {
                        continue;
                    }
                    let mut lhs = 0.0;
                    for col in columns {
                        let pc = &col.packing_set_count;
                        let (a, b, c) = (pc[i] != 0, pc[j] != 0, pc[k] != 0);
                        if if a { b || c } else { b && c } {
                            lhs += col.value;
                        }
                    }
                    if !(lhs - 1.0 > self.params.violation_tolerance) {
                        continue;
                    }
                    let cut = self.candidate(
                        1.0,
                        0,
                        3,
                        -1,
                        2,
                        vec![1, 1, 1],
                        vec![i as i32, j as i32, k as i32],
                    );
                    self.insert_candidate(cuts, rounded_score((1.0 - lhs) * 1e8 + 0.5), cut);
                }
            }
        }
    }

    /// Three-row covering cuts.
    pub(super) fn reference_cov(&mut self, columns: &[ColumnInfo], cuts: &mut CandidateSet) {
        let n = self.num_covering_sets.max(0) as usize;
        for i in 0..n {
            for j in i + 1..n {
                for k in j + 1..n {
                    let mut lhs = 0.0;
                    for col in columns {
                        let cc = &col.covering_set_count;
                        let sum = cc[j] + cc[i] + cc[k] + 1;
                        lhs = ((sum / 2) as f64) * col.value + lhs;
                    }
                    if !(2.0 - lhs > self.params.violation_tolerance) {
                        continue;
                    }
                    let cut = self.candidate(
                        2.0,
                        1,
                        3,
                        -1,
                        -2,
                        vec![-1, -1, -1],
                        vec![i as i32, j as i32, k as i32],
                    );
                    self.insert_candidate(cuts, rounded_score((lhs - 2.0) * 1e8 + 0.5), cut);
                }
            }
        }
    }
}

fn fixture(n: usize, count: usize) -> (Separation, Vec<ColumnInfo>) {
    let mut seed = 12345u64;
    let mut columns = Vec::new();
    for _ in 0..count {
        let mut counts = vec![0; n];
        for _ in 0..12 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            counts[(seed >> 32) as usize % n] += 1;
        }
        columns.push(ColumnInfo {
            value: n as f64 / (12.0 * count as f64),
            packing_set_count: counts.clone(),
            covering_set_count: counts,
            ..Default::default()
        });
    }
    // Fractional triangle packings provide known violated three-row cuts.
    for base in (0..n.saturating_sub(2)).step_by(3) {
        for pair in [[base, base + 1], [base, base + 2], [base + 1, base + 2]] {
            let mut counts = vec![0; n];
            for row in pair {
                counts[row] = 1;
            }
            columns.push(ColumnInfo {
                value: 0.5,
                packing_set_count: counts.clone(),
                covering_set_count: counts,
                ..Default::default()
            });
        }
    }
    (
        Separation {
            num_packing_sets: n as i32,
            num_covering_sets: n as i32,
            ..Default::default()
        },
        columns,
    )
}

fn run(
    mut state: Separation,
    columns: &[ColumnInfo],
    covering: bool,
    reference: bool,
) -> (Vec<SeparatedCut>, Vec<(f64, usize)>) {
    let mut cuts = CandidateSet::new();
    match (covering, reference) {
        (false, false) => state.three_rows_pack(columns, &mut cuts),
        (true, false) => state.three_rows_cov(columns, &mut cuts),
        (false, true) => state.reference_pack(columns, &mut cuts),
        (true, true) => state.reference_cov(columns, &mut cuts),
    }
    (state.arena, cuts.ordered())
}

#[test]
fn three_rows_parallel_matches_original_order_and_scores() {
    for n in [0, 3, 20, 64, 100] {
        let (state, columns) = fixture(n, if n == 0 { 0 } else { 90 });
        for covering in [false, true] {
            let expected = run(state.clone(), &columns, covering, true);
            for threads in [1, 2, 4] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let actual = pool.install(|| run(state.clone(), &columns, covering, false));
                assert_eq!(
                    actual, expected,
                    "n={n}, covering={covering}, threads={threads}"
                );
            }
        }
    }
}

#[test]
fn three_rows_parallel_keeps_known_fractional_triangle_cuts() {
    let (mut state, columns) = fixture(99, 0);
    state.round_cut_counter = 17;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    for covering in [false, true] {
        let expected = run(state.clone(), &columns, covering, true);
        assert_eq!(expected.0.len(), 33);
        assert_eq!(expected.0[0].id, 17);
        assert_eq!(expected.0[0].set_ids, vec![0, 1, 2]);
        let actual = pool.install(|| run(state.clone(), &columns, covering, false));
        assert_eq!(actual, expected);
    }
}

#[test]
fn three_row_packing_violation_boundary_matches_captured_binary() {
    // Native 0x90e780 with tolerance 0.02 and the three pair columns.
    for (last, expected) in [
        (0.01999998, None),
        (0.01999999, None),
        (0.02, Some(-0.02_f64)),
        (0.02000001, Some(-0.02000001_f64)),
        (0.02000002, Some(-0.02000002_f64)),
    ] {
        let columns = [(0.5, [1, 0, 1]), (0.5, [1, 1, 0]), (last, [0, 1, 1])]
            .into_iter()
            .map(|(value, counts)| ColumnInfo {
                value,
                packing_set_count: counts.into(),
                ..Default::default()
            })
            .collect::<Vec<_>>();
        let state = Separation {
            num_packing_sets: 3,
            ..Default::default()
        };
        let (arena, ordered) = run(state, &columns, false, false);
        if let Some(score) = expected {
            assert_eq!(ordered.len(), 1, "last={last}");
            assert_eq!(ordered[0].0.to_bits(), score.to_bits(), "last={last}");
            let cut = &arena[ordered[0].1];
            assert_eq!(cut.set_ids, [0, 1, 2]);
            assert_eq!(cut.numerators, [1, 1, 1]);
            assert_eq!((cut.rhs, cut.denominator), (1., 2));
        } else {
            assert!(ordered.is_empty(), "last={last}");
        }
    }
}

#[test]
fn three_row_packing_candidates_follow_captured_binary_order_with_ties() {
    // Captured from the native three-row method on a controlled five-set input.
    let weights = [
        0.25, 0.625, 0.375, 0.625, 0.125, 0.5, 0.625, 0.625, 0.625, 0.625,
    ];
    let mut columns = Vec::new();
    let mut weights = weights.into_iter();
    for i in 0..5 {
        for j in i + 1..5 {
            let mut counts = vec![0; 5];
            counts[i] = 1;
            counts[j] = 1;
            columns.push(ColumnInfo {
                value: weights.next().unwrap(),
                packing_set_count: counts,
                ..Default::default()
            });
        }
    }
    let state = Separation {
        num_packing_sets: 5,
        ..Default::default()
    };
    let (arena, ordered) = run(state, &columns, false, false);
    let actual: Vec<_> = ordered
        .into_iter()
        .map(|(score, handle)| (score, arena[handle].set_ids.clone()))
        .collect();
    assert_eq!(
        actual,
        vec![
            (-0.875, vec![0, 2, 4]),
            (-0.875, vec![2, 3, 4]),
            (-0.75, vec![1, 3, 4]),
            (-0.625, vec![0, 2, 3]),
            (-0.625, vec![0, 3, 4]),
            (-0.5, vec![0, 1, 4]),
            (-0.375, vec![1, 2, 4]),
            (-0.25, vec![1, 2, 3]),
            (-0.125, vec![0, 1, 3]),
        ]
    );
}

#[test]
fn three_row_covering_candidates_follow_captured_binary_order_with_ties() {
    // Captured from the native three-row covering method on a controlled input.
    let columns = [
        (0.5, [0, 0, 2, 2, 1]),
        (0.125, [0, 1, 1, 2, 2]),
        (0.125, [2, 0, 0, 2, 2]),
        (0.0, [2, 0, 1, 1, 1]),
        (0.25, [2, 2, 2, 0, 1]),
    ]
    .into_iter()
    .map(|(value, counts)| ColumnInfo {
        value,
        covering_set_count: counts.into(),
        ..Default::default()
    })
    .collect::<Vec<_>>();
    let state = Separation {
        num_covering_sets: 5,
        ..Default::default()
    };
    let (arena, ordered) = run(state, &columns, true, false);
    let actual: Vec<_> = ordered
        .into_iter()
        .map(|(score, handle)| (score, arena[handle].set_ids.clone()))
        .collect();
    assert_eq!(
        actual,
        vec![
            (-0.5, vec![0, 1, 2]),
            (-0.5, vec![0, 1, 3]),
            (-0.25, vec![0, 1, 4]),
            (-0.125, vec![1, 2, 3]),
        ]
    );
}

#[test]
#[ignore = "release benchmark; run with --ignored --nocapture"]
fn benchmark_three_rows_parallel() {
    for (n, count) in [(50, 200), (100, 500), (200, 1000)] {
        let (state, columns) = fixture(n, count);
        for covering in [false, true] {
            let expected = run(state.clone(), &columns, covering, true);
            for threads in [1, 2, 4, 8] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                pool.install(|| {
                    let mut serial = Vec::new();
                    let mut parallel = Vec::new();
                    for iteration in 0..7 {
                        for reference in if iteration % 2 == 0 { [true, false] } else { [false, true] } {
                            let start = Instant::now();
                            let actual = run(state.clone(), &columns, covering, reference);
                            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                            assert_eq!(actual, expected);
                            std::hint::black_box(actual);
                            if iteration > 0 {
                                if reference { serial.push(elapsed); } else { parallel.push(elapsed); }
                            }
                        }
                    }
                    serial.sort_by(f64::total_cmp);
                    parallel.sort_by(f64::total_cmp);
                    println!("n={n} columns={count} covering={covering} threads={threads} candidates={} serial_ms={:.3} parallel_ms={:.3} speedup={:.2}", expected.0.len(), serial[3], parallel[3], serial[3]/parallel[3]);
                });
            }
        }
    }
}

#[test]
fn one_row_packing_from_covering_sets_matches_reference_candidates() {
    let capture =
        include_str!("../../../fixtures/enumeration/one-row-packing-covering-capture.txt");
    for (case_id, line) in capture.lines().enumerate() {
        let mut values = line.split_whitespace();
        let mut next = || values.next().unwrap().parse::<u64>().unwrap();
        let columns: Vec<_> = (0..5)
            .map(|_| ColumnInfo {
                value: f64::from_bits(next()),
                covering_set_count: (0..5).map(|_| next() as i32).collect(),
                ..Default::default()
            })
            .collect();
        let count = next() as usize;
        let expected: Vec<_> = (0..count).map(|_| (next(), next() as i32)).collect();
        assert!(values.next().is_none());
        let mut state = Separation {
            num_packing_sets: 5,
            num_covering_sets: 5,
            ..Default::default()
        };
        state.params.use_cov_sets_for_one_row_packing = true;
        let mut candidates = CandidateSet::new();
        state.one_row_pack(&columns, &mut candidates);
        let actual = candidates.ordered();
        assert_eq!(actual.len(), expected.len(), "case {case_id}");
        for ((score, handle), (expected_bits, expected_set)) in actual.into_iter().zip(expected) {
            let cut = &state.arena[handle];
            assert_eq!(score.to_bits(), expected_bits, "case {case_id}");
            assert_eq!(
                (cut.rhs, cut.kind, cut.num_rows, cut.class, cut.denominator),
                (0., 2, 1, -1, 2)
            );
            assert_eq!(cut.numerators, [1]);
            assert_eq!(cut.set_ids, [expected_set], "case {case_id}");
        }
    }
}

#[test]
fn one_row_separators_match_native_candidate_captures() {
    for (covering, capture) in [
        (
            false,
            include_str!("../../../fixtures/enumeration/one-row-packing-native-capture.txt"),
        ),
        (
            true,
            include_str!("../../../fixtures/enumeration/one-row-covering-native-capture.txt"),
        ),
    ] {
        for (case_id, line) in capture.lines().enumerate() {
            let mut values = line.split_whitespace();
            let mut next = || values.next().unwrap().parse::<u64>().unwrap();
            let columns: Vec<_> = (0..5)
                .map(|_| {
                    let value = f64::from_bits(next());
                    let counts: Vec<_> = (0..5).map(|_| next() as i32).collect();
                    ColumnInfo {
                        value,
                        packing_set_count: counts.clone(),
                        covering_set_count: counts,
                        ..Default::default()
                    }
                })
                .collect();
            let count = next() as usize;
            let expected: Vec<_> = (0..count).map(|_| (next(), next() as i32)).collect();
            assert!(values.next().is_none());
            let mut state = Separation {
                num_packing_sets: 5,
                num_covering_sets: 5,
                ..Default::default()
            };
            let mut candidates = CandidateSet::new();
            if covering {
                state.one_row_cov(&columns, &mut candidates);
            } else {
                state.one_row_pack(&columns, &mut candidates);
            }
            let actual = candidates.ordered();
            assert_eq!(
                actual.len(),
                expected.len(),
                "covering={covering} case={case_id}"
            );
            for ((score, handle), (bits, set)) in actual.into_iter().zip(expected) {
                let cut = &state.arena[handle];
                assert_eq!(score.to_bits(), bits, "covering={covering} case={case_id}");
                assert_eq!(
                    (cut.rhs, cut.kind, cut.num_rows, cut.class, cut.denominator),
                    if covering {
                        (1., 1, 1, -1, -2)
                    } else {
                        (0., 0, 1, -1, 2)
                    }
                );
                assert_eq!(cut.numerators, if covering { [-1] } else { [1] });
                assert_eq!(cut.set_ids, [set], "covering={covering} case={case_id}");
            }
        }
    }
}
