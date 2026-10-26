use super::*;
use crate::cuts::rank_one::separation::{
    five_or_more_row_denominator, five_or_more_row_numerators, five_or_more_row_rhs, GraphData,
    Incidence, Parameters,
};
use std::time::Instant;

// Serial oracle from 90a0d692: preserve RNG consumption and candidate insertion.
impl<const COVERING: bool> LocalSearch<COVERING> {
    /// `num_iterations` randomized starts, each improved by local search.
    #[allow(clippy::too_many_arguments)]
    fn reference_run(
        &mut self,
        parent: &mut Separation,
        columns: &[ColumnInfo],
        cuts: &mut CandidateSet,
        num_rows_cut: i32,
        rhs: f64,
        numerators: &[i32],
        denominator: i32,
        class: i32,
    ) {
        if self.num_rows < num_rows_cut {
            return;
        }
        let mut seen: HashSet<Vec<i32>> = HashSet::new();
        // The bound depends on the column counts and multiplier multiset,
        // which stay fixed across restarts of this class. Initialize lazily
        // so a search with no starts still performs no bound calculation.
        let mut max_state = None;
        for _ in 0..self.num_iterations {
            let mut rows = vec![];
            if COVERING {
                self.build_covering_row_set(columns, &mut rows, numerators, 0.5);
            } else {
                self.build_packing_row_set(
                    parent,
                    columns,
                    &mut rows,
                    numerators,
                    denominator,
                    0.5,
                );
            }
            let mut multipliers: Vec<i32> = numerators[..num_rows_cut as usize].to_vec();
            let max_state = *max_state
                .get_or_insert_with(|| self.compute_max_state(columns, rows.len(), &multipliers));
            let violation = self.run_local_search(
                parent,
                columns,
                &mut rows,
                &mut multipliers,
                denominator,
                rhs,
                max_state,
            );
            let mut key = rows.clone();
            key.sort();
            if seen.contains(&key) {
                continue;
            }
            let ratio = violation / rhs;
            if !(ratio > self.violation_tolerance) {
                continue;
            }
            if num_rows_cut == 4 {
                if !self.skip_subcut_checks {
                    let violated = if COVERING {
                        subcuts::three_rows_cov_violated(
                            parent,
                            columns,
                            &rows,
                            &parent.permutations_4rows,
                        )
                    } else {
                        subcuts::three_rows_pack_violated(
                            parent,
                            columns,
                            &rows,
                            &parent.permutations_4rows,
                        )
                    };
                    if violated {
                        continue;
                    }
                }
            } else if num_rows_cut == 5 && !self.skip_subcut_checks {
                let violated = if COVERING {
                    subcuts::three_rows_cov_violated(
                        parent,
                        columns,
                        &rows,
                        &parent.permutations_5rows,
                    ) || subcuts::four_rows_cov_violated(
                        parent,
                        columns,
                        &rows,
                        &self.column_indices,
                    )
                } else {
                    subcuts::three_rows_pack_violated(
                        parent,
                        columns,
                        &rows,
                        &parent.permutations_5rows,
                    ) || subcuts::four_rows_pack_violated(
                        parent,
                        columns,
                        &rows,
                        &self.column_indices,
                    )
                };
                if violated {
                    continue;
                }
            }
            if COVERING {
                for m in multipliers.iter_mut().take(num_rows_cut as usize) {
                    *m = -*m;
                }
            }
            let id = parent.round_cut_counter;
            parent.round_cut_counter += 1;
            let cut = parent.push_candidate(SeparatedCut {
                id,
                rhs,
                kind: COVERING as i32,
                num_rows: num_rows_cut,
                class,
                denominator: if COVERING { -denominator } else { denominator },
                numerators: multipliers,
                set_ids: rows,
                memory_type: self.cut_memory_type,
                memory: BTreeMap::new(),
            });
            parent.insert_candidate(cuts, (0.5 - 1e8 * ratio).floor() / 1e8, cut);
            seen.insert(key);
        }
    }
}

fn fixture(
    n: usize,
    count: usize,
    iterations: i32,
    checks: bool,
    neighbours: bool,
) -> (Separation, Vec<ColumnInfo>) {
    let mut parent = Separation::new(Parameters {
        max_num_rows: 8,
        ls_num_iterations: iterations,
        skip_subcut_checks: !checks,
        ..Default::default()
    });
    let graph = GraphData {
        elem_set_count: n as i32,
        packing_set_count: n as i32,
        covering_set_count: n as i32,
        packing_neighbourhood: if neighbours {
            (0..n)
                .map(|i| (0..n.min(8)).map(|d| ((i + d) % n) as i32).collect())
                .collect()
        } else {
            Vec::new()
        },
        ..Default::default()
    };
    assert!(parent.prepare(&[&graph], vec![(0, Incidence::default())]));
    parent.round_cut_counter = 17;
    parent.arena.push(SeparatedCut {
        id: 16,
        ..Default::default()
    });
    let mut rng = Xoroshiro128Plus::new(24);
    let columns = (0..count)
        .map(|_| {
            let mut counts = vec![0; n];
            let mut sequence = Vec::new();
            while sequence.len() < n.min(6) {
                let row = rng.uniform_below(n as u64) as usize;
                if counts[row] == 0 {
                    counts[row] = 1;
                    sequence.push(row as i32);
                }
            }
            ColumnInfo {
                value: n as f64 / (n.min(6) * count) as f64,
                elementary: true,
                packing_set_count: counts.clone(),
                covering_set_count: counts,
                packing_set_sequence: sequence,
                ..Default::default()
            }
        })
        .collect();
    (parent, columns)
}

fn call<const COVERING: bool>(
    search: &mut LocalSearch<COVERING>,
    parent: &mut Separation,
    columns: &[ColumnInfo],
    cuts: &mut CandidateSet,
    rows: i32,
    kind: usize,
    reference: bool,
) {
    let (rhs, numerators, denominator) = if rows == 4 {
        if COVERING {
            (2.0, &[1, 1, 1, 1][..], 3)
        } else {
            (1.0, &[2, 1, 1, 1][..], 3)
        }
    } else {
        (
            five_or_more_row_rhs(COVERING as usize, kind),
            five_or_more_row_numerators(COVERING as usize, kind),
            five_or_more_row_denominator(COVERING as usize, kind),
        )
    };
    if reference {
        search.reference_run(
            parent,
            columns,
            cuts,
            rows,
            rhs,
            numerators,
            denominator,
            kind as i32,
        );
    } else {
        search.run(
            parent,
            columns,
            cuts,
            rows,
            rhs,
            numerators,
            denominator,
            kind as i32,
        );
    }
}

fn check<const COVERING: bool>(iterations: i32, checks: bool, neighbours: bool) {
    let (parent, columns) = fixture(20, 35, iterations, checks, neighbours);
    let elementary: Vec<_> = (0..columns.len() as i32).collect();
    let mut expected_parent = parent.clone();
    let mut expected_search = LocalSearch::<COVERING>::new(&parent, &columns, &elementary);
    let mut expected_cuts = CandidateSet::new();
    expected_cuts.insert(0.123, 16, 0);
    let mut snapshots = Vec::new();
    let classes: Vec<_> = std::iter::once((4, 0))
        .chain((0..if COVERING { 10 } else { 7 }).map(|kind| (5, kind)))
        .chain([(6, 7), (4, 0)])
        .collect();
    // Six rows are packing-only here; covering has no six-row class table.
    let classes: Vec<_> = classes
        .into_iter()
        .filter(|&(rows, _)| !COVERING || rows != 6)
        .collect();
    for &(rows, kind) in &classes {
        call(
            &mut expected_search,
            &mut expected_parent,
            &columns,
            &mut expected_cuts,
            rows,
            kind,
            true,
        );
        snapshots.push((
            expected_parent.arena.clone(),
            expected_cuts.ordered(),
            expected_search.rng.clone().next(),
            expected_parent.round_cut_counter,
        ));
    }
    for threads in [1, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let mut actual_parent = parent.clone();
            let mut actual_search = LocalSearch::<COVERING>::new(&parent, &columns, &elementary);
            let mut actual_cuts = CandidateSet::new();
            actual_cuts.insert(0.123, 16, 0);
            for (&(rows, kind), expected) in classes.iter().zip(&snapshots) {
                call(&mut actual_search, &mut actual_parent, &columns, &mut actual_cuts, rows, kind, false);
                assert_eq!(&(actual_parent.arena.clone(), actual_cuts.ordered(),
                    actual_search.rng.clone().next(), actual_parent.round_cut_counter), expected,
                    "covering={COVERING}, rows={rows}, kind={kind}, starts={iterations}, checks={checks}, neighbours={neighbours}, threads={threads}");
            }
        });
    }
}

#[test]
fn parallel_starts_preserve_rng_candidates_scores_and_class_sequence() {
    for iterations in [0, 1, 31, 32, 65] {
        check::<false>(iterations, false, false);
        check::<true>(iterations, false, false);
    }
    check::<false>(65, true, true);
    check::<true>(65, true, false);
}

#[test]
fn too_few_rows_does_not_advance_rng_or_insert_cuts() {
    let (mut parent, columns) = fixture(3, 12, 65, false, false);
    let elementary: Vec<_> = (0..columns.len() as i32).collect();
    let mut search = LocalSearch::<false>::new(&parent, &columns, &elementary);
    let next = search.rng.clone().next();
    let mut cuts = CandidateSet::new();
    call(&mut search, &mut parent, &columns, &mut cuts, 4, 0, false);
    assert_eq!(search.rng.clone().next(), next);
    assert!(cuts.is_empty());
    assert_eq!(parent.arena.len(), 1);
}

#[test]
fn parallel_starts_keep_first_duplicate_and_covering_signs_across_batches() {
    fn check<const COVERING: bool>(rows: i32) {
        let (parent, _) = fixture(rows as usize, rows as usize, 65, false, false);
        let columns: Vec<_> = (0..rows)
            .map(|r| {
                let counts: Vec<_> = (0..rows)
                    .map(|i| {
                        if COVERING {
                            (i == r) as i32
                        } else {
                            (i != r) as i32
                        }
                    })
                    .collect();
                ColumnInfo {
                    value: if COVERING { 0.25 } else { 0.5 },
                    elementary: true,
                    packing_set_sequence: counts
                        .iter()
                        .enumerate()
                        .filter(|(_, c)| **c != 0)
                        .map(|(r, _)| r as i32)
                        .collect(),
                    packing_set_count: counts.clone(),
                    covering_set_count: counts,
                    ..Default::default()
                }
            })
            .collect();
        let elementary: Vec<_> = (0..rows).collect();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let mut outputs = Vec::new();
        for reference in [true, false] {
            let mut state = parent.clone();
            let mut search = LocalSearch::<COVERING>::new(&state, &columns, &elementary);
            let mut cuts = CandidateSet::new();
            pool.install(|| {
                call(
                    &mut search,
                    &mut state,
                    &columns,
                    &mut cuts,
                    rows,
                    0,
                    reference,
                )
            });
            assert_eq!(cuts.len(), 1);
            assert_eq!(state.arena[1].id, 17);
            assert_eq!(state.arena[1].set_ids, (0..rows).collect::<Vec<_>>());
            assert_eq!(state.arena[1].denominator < 0, COVERING);
            assert!(state.arena[1]
                .numerators
                .iter()
                .all(|&m| (m < 0) == COVERING));
            outputs.push((state.arena, cuts.ordered(), search.rng.clone().next()));
        }
        assert_eq!(outputs[0], outputs[1]);
    }
    for rows in [4, 5] {
        check::<false>(rows);
        check::<true>(rows);
    }
}

fn measure<const COVERING: bool>(rows: i32, iterations: i32) {
    let (parent, columns) = fixture(100, 256, iterations, false, true);
    let elementary: Vec<_> = (0..columns.len() as i32).collect();
    for threads in [1, 4, 8, 32] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let mut samples = [Vec::new(), Vec::new()];
            let mut expected = None;
            for iteration in 0..4 {
                for reference in if iteration % 2 == 0 { [true, false] } else { [false, true] } {
                    let mut state = parent.clone();
                    let mut search = LocalSearch::<COVERING>::new(&state, &columns, &elementary);
                    let mut cuts = CandidateSet::new();
                    let start = Instant::now();
                    call(&mut search, &mut state, &columns, &mut cuts, rows, 0, reference);
                    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    let actual = (state.arena, cuts.ordered(), search.rng.clone().next());
                    if let Some(ref expected) = expected { assert_eq!(&actual, expected); }
                    else { expected = Some(actual.clone()); }
                    std::hint::black_box(actual);
                    if iteration > 0 { samples[reference as usize].push(elapsed); }
                }
            }
            samples.iter_mut().for_each(|sample| sample.sort_by(f64::total_cmp));
            println!("covering={COVERING} rows={rows} starts={iterations} threads={threads} candidates={} serial_ms={:.3} parallel_ms={:.3} speedup={:.2}",
                expected.unwrap().0.len() - 1, samples[1][1], samples[0][1], samples[1][1] / samples[0][1]);
        });
    }
}

#[test]
#[ignore = "release benchmark; run with --ignored --nocapture"]
fn benchmark_parallel_local_search_starts() {
    let iterations = std::env::var("BALDES_LS_BENCH_STARTS")
        .ok()
        .map(|s| s.parse().unwrap())
        .unwrap_or(256);
    for rows in [4, 5] {
        measure::<false>(rows, iterations);
        measure::<true>(rows, iterations);
    }
}
