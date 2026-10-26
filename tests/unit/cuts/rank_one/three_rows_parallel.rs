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
