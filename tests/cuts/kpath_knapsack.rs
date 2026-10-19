use baldes::core::options::Options;
use baldes::cuts::capacity::{CapacitySeparator, TimeWindows};
use baldes::cuts::knapsack::{
    knapsack_parameters, KnapsackKind, KnapsackParameters, KnapsackSeparator,
};
use baldes::cuts::kpath::{kpath_parameters, KPathParameters, KPathSeparator};
use baldes::pricing::Solver;
use baldes::Graph;
use std::collections::{BTreeMap, BTreeSet};

// ============================================================================
// Route load knapsack cut tests (ported from test_knapsack.py)
// ============================================================================

#[test]
fn default_mode_has_rounding_off_and_one_k_equal_one() {
    let sep = KnapsackSeparator::new();
    assert_eq!(sep.get_parameter("max_cuts_per_round").unwrap(), 100.0);
    assert_eq!(sep.get_parameter("use_rounding").unwrap(), 0.0);
    assert_eq!(sep.get_parameter("max_one_k").unwrap(), 1.0);
    assert!((sep.get_parameter("violation_tolerance").unwrap() - 0.02).abs() < 1e-12);
    assert!(sep.separate(&[3, 4], &[1.0, 1.0], 7).unwrap().is_empty());
}

#[test]
fn rounding_cut_recovered_by_hand() {
    let mut sep = KnapsackSeparator::new();
    sep.set_parameter("use_rounding", 1.0).unwrap();
    sep.set_parameter("max_one_k", 0.0).unwrap();
    let cuts = sep.separate(&[3], &[4.0 / 3.0], 4).unwrap();
    let by_multiplier: BTreeMap<(i32, i32), _> = cuts
        .iter()
        .map(|c| ((c.numerator, c.denominator), c))
        .collect();
    let keys: BTreeSet<(i32, i32)> = by_multiplier.keys().copied().collect();
    assert_eq!(keys, BTreeSet::from([(1, 3), (2, 3)]));

    let two_thirds = by_multiplier[&(2, 3)];
    assert_eq!(two_thirds.kind, KnapsackKind::Rounding);
    assert_eq!(two_thirds.rhs, 2);
    assert_eq!(two_thirds.coefficient(3), 2);
    assert!((two_thirds.lhs - 8.0 / 3.0).abs() < 1e-7);
    assert!((two_thirds.violation - 2.0 / 3.0).abs() < 1e-7);

    let one_third = by_multiplier[&(1, 3)];
    assert_eq!(one_third.rhs, 1);
    assert!((one_third.violation - 1.0 / 3.0).abs() < 1e-7);
}

fn brute_force_best_knapsack(loads: &[i32], values: &[f64], rhs: i32) -> Option<f64> {
    let mut by: BTreeMap<i32, f64> = BTreeMap::new();
    for (&l, &v) in loads.iter().zip(values) {
        *by.entry(l).or_insert(0.0) += v;
    }
    let mut best: Option<f64> = None;
    for &q in by.keys() {
        for j in 1..=q {
            let w = (j as f64) / (q as f64);
            let t = (w * (rhs as f64) + 1e-6).floor() as i32;
            let s: f64 = by
                .iter()
                .map(|(&l, &v)| (w * (l as f64) + 1e-6).floor() * v)
                .sum();
            if (t as f64) * (s - (t as f64)) > 0.02 {
                let rel = (s - (t as f64)) / (t.max(1) as f64);
                best = Some(best.map_or(rel, |b| b.max(rel)));
            }
        }
    }
    best
}

struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as u32
    }
    fn randint(&mut self, lo: i32, hi: i32) -> i32 {
        let span = (hi - lo + 1) as u32;
        lo + (self.next_u32() % span) as i32
    }
    fn random(&mut self) -> f64 {
        (self.next_u32() as f64) / (u32::MAX as f64)
    }
}

#[test]
fn matches_brute_force_over_all_multipliers() {
    let mut rng = Lcg(4);
    let mut cases = 0;
    let choices = [0.25, 0.5, 0.75, 1.0];
    for _ in 0..80 {
        let n = rng.randint(2, 6) as usize;
        let loads: Vec<i32> = (0..n).map(|_| rng.randint(1, 9)).collect();
        let values: Vec<f64> = (0..n)
            .map(|_| choices[rng.randint(0, 3) as usize])
            .collect();
        let max_load = *loads.iter().max().unwrap();
        let total: f64 = loads
            .iter()
            .zip(&values)
            .map(|(&l, &v)| (l as f64) * v)
            .sum();
        let rhs = max_load.max(total.ceil() as i32);

        let mut sep = KnapsackSeparator::new();
        sep.set_parameter("use_rounding", 1.0).unwrap();
        sep.set_parameter("max_one_k", 0.0).unwrap();
        let cuts = sep.separate(&loads, &values, rhs).unwrap();
        let expected = brute_force_best_knapsack(&loads, &values, rhs);
        match expected {
            None => assert!(cuts.is_empty()),
            Some(exp) => {
                cases += 1;
                assert!(!cuts.is_empty());
                let rel = cuts[0].violation / (cuts[0].rhs.max(1) as f64);
                assert!((rel - exp).abs() < 1e-6);
            }
        }
    }
    assert!(cases >= 15);
}

#[test]
fn rounding_coefficients_above_capacity_match_reported_lhs() {
    let mut sep = KnapsackSeparator::new();
    sep.set_parameter("use_rounding", 1.0).unwrap();
    sep.set_parameter("max_one_k", 0.0).unwrap();
    let loads = [1, 7];
    let values = [0.75, 0.25];
    let cuts = sep.separate(&loads, &values, 2).unwrap();
    let mults: BTreeSet<(i32, i32)> = cuts.iter().map(|c| (c.numerator, c.denominator)).collect();
    assert_eq!(mults, BTreeSet::from([(6, 7), (5, 7), (1, 1)]));
    for cut in &cuts {
        assert_eq!(cut.coefficient(-1), 0);
        assert_eq!(cut.coefficient(0), 0);
        assert_eq!(cut.coefficient(7), cut.rhs);
        assert_eq!(cut.coefficient(i32::MAX), cut.rhs);
        let lhs: f64 = loads
            .iter()
            .zip(&values)
            .map(|(&l, &v)| (cut.coefficient(l) as f64) * v)
            .sum();
        assert!((lhs - cut.lhs).abs() < 1e-9);
        assert!((cut.violation - (cut.lhs - cut.rhs as f64)).abs() < 1e-9);
        assert!((cut.violation - (-0.75)).abs() < 1e-9);
    }
}

#[test]
fn knapsack_cuts_hold_for_every_integer_solution() {
    let mut rng = Lcg(9);
    let mut checked = 0;
    let choices = [0.5, 1.0, 1.5];
    for _ in 0..25 {
        let rhs = rng.randint(6, 14);
        let len = rng.randint(2, 5) as usize;
        let loads: Vec<i32> = (0..len).map(|_| rng.randint(1, 7)).collect();
        let values: Vec<f64> = (0..len)
            .map(|_| choices[rng.randint(0, 2) as usize])
            .collect();
        let mut sep = KnapsackSeparator::new();
        sep.set_parameter("use_rounding", 1.0).unwrap();
        sep.set_parameter("max_one_k", 0.0).unwrap();
        let cuts = sep.separate(&loads, &values, rhs).unwrap();
        for a in 0..=7 {
            for b in a..=7 {
                for c in b..=7 {
                    let combo: Vec<i32> = [a, b, c].into_iter().filter(|&x| x > 0).collect();
                    if combo.iter().sum::<i32>() > rhs {
                        continue;
                    }
                    for cut in &cuts {
                        let lhs: i32 = combo.iter().map(|&l| cut.coefficient(l)).sum();
                        assert!(lhs <= cut.rhs);
                        checked += 1;
                    }
                }
            }
        }
    }
    assert!(checked > 100);
}

#[test]
fn max_cuts_per_round_and_validation() {
    let mut sep = KnapsackSeparator::new();
    sep.set_parameter("use_rounding", 1.0).unwrap();
    sep.set_parameter("max_one_k", 0.0).unwrap();
    sep.set_parameter("max_cuts_per_round", 2.0).unwrap();
    let loads = [5, 7, 9, 11];
    let values = [0.9, 0.9, 0.9, 0.9];
    assert!(sep.separate(&loads, &values, 20).unwrap().len() <= 2);
    assert!(sep.separate(&[1, 2], &[1.0], 5).is_err());
    assert!(sep.separate(&[1], &[-1.0], 5).is_err());
    assert!(sep.set_parameter("nope", 1.0).is_err());
}

fn one_k_max_sum(rhs: i32, thresholds: &[i32]) -> i32 {
    let mut best = vec![0i32; (rhs + 1) as usize];
    for cap in 1..=rhs {
        let mut m = best[(cap - 1) as usize];
        for w in 1..=cap {
            let step = thresholds.iter().filter(|&&t| w >= t).count() as i32;
            m = m.max(best[(cap - w) as usize] + step);
        }
        best[cap as usize] = m;
    }
    best[rhs as usize]
}

#[test]
fn one_k_cut_by_hand() {
    let sep = KnapsackSeparator::new();
    let cuts = sep.separate(&[4, 5], &[1.5, 1.5], 10).unwrap();
    assert!(!cuts.is_empty());
    assert!(cuts
        .iter()
        .all(|c| c.kind == KnapsackKind::Steps && c.rhs == 6 && c.thresholds.len() == 6));
    let best = &cuts[0];
    assert_eq!(best.thresholds, vec![4, 4, 4, 7, 7, 7]);
    assert!((best.violation - 3.0).abs() < 1e-9);
    assert_eq!(best.coefficient(3), 0);
    assert_eq!(best.coefficient(4), 3);
    assert_eq!(best.coefficient(7), 6);
    assert!((best.lhs - (best.violation + 6.0)).abs() < 1e-9);
}

#[test]
fn one_k_families_are_valid_and_gated_by_k() {
    let mut rng = Lcg(21);
    let mut counts = BTreeMap::from([(6, 0usize), (8, 0usize), (10, 0usize)]);
    let choices = [0.5, 1.0, 1.5, 2.0];
    for _ in 0..60 {
        let rhs = rng.randint(12, 26);
        let len = rng.randint(3, 7) as usize;
        let loads: Vec<i32> = (0..len).map(|_| rng.randint(1, rhs)).collect();
        let values: Vec<f64> = (0..len)
            .map(|_| choices[rng.randint(0, 3) as usize])
            .collect();
        for (k, allowed) in [(1, vec![6]), (8, vec![6, 8]), (10, vec![6, 8, 10])] {
            let mut sep = KnapsackSeparator::new();
            sep.set_parameter("max_one_k", k as f64).unwrap();
            for cut in sep.separate(&loads, &values, rhs).unwrap() {
                assert_eq!(cut.kind, KnapsackKind::Steps);
                assert!(allowed.contains(&cut.rhs));
                let mut sorted = cut.thresholds.clone();
                sorted.sort_unstable();
                assert_eq!(cut.thresholds, sorted);
                assert_eq!(cut.thresholds.len() as i32, cut.rhs);
                assert!(one_k_max_sum(rhs, &cut.thresholds) <= cut.rhs);
                let lhs: f64 = loads
                    .iter()
                    .zip(&values)
                    .map(|(&l, &v)| (cut.coefficient(l) as f64) * v)
                    .sum();
                assert!((lhs - cut.lhs).abs() < 1e-9);
                assert!(lhs > (cut.rhs as f64) + 0.02);
                *counts.get_mut(&cut.rhs).unwrap() += 1;
            }
        }
    }
    assert!(counts[&6] > 0);
}

#[test]
fn one_k_off_when_k_is_zero() {
    let mut sep = KnapsackSeparator::new();
    sep.set_parameter("max_one_k", 0.0).unwrap();
    assert!(sep.separate(&[4, 5], &[1.5, 1.5], 10).unwrap().is_empty());
}

// ============================================================================
// Strong k-path cut tests (ported from test_kpath.py)
// ============================================================================

#[test]
fn basic_violation_and_coefficient() {
    let mut sep = KPathSeparator::new(vec![0.0, 3.0, 3.0, 1.0], 4.0).unwrap();
    let cuts = sep.separate(&[(vec![1, 2], 1.0), (vec![3], 1.0)]).unwrap();
    let cut = cuts.iter().find(|c| c.customers == vec![1, 2]).unwrap();
    assert_eq!(cut.rhs, 2);
    assert!((cut.violation - 1.0).abs() < 1e-9);
    assert!((cut.weight - 1.0).abs() < 1e-9);
    assert_eq!(cut.coefficient(&[1]).unwrap(), 1);
    assert_eq!(cut.coefficient(&[3]).unwrap(), 0);
    assert_eq!(cut.coefficient(&[3, 2]).unwrap(), 1);
    assert_eq!(cut.coefficient(&[]).unwrap(), 0);
}

#[test]
fn a_column_is_counted_once_per_set() {
    let mut sep = KPathSeparator::new(vec![0.0, 3.0, 3.0], 4.0).unwrap();
    let cut = &sep.separate(&[(vec![1, 2], 0.5)]).unwrap()[0];
    assert!((cut.weight - 0.5).abs() < 1e-9);
}


fn random_fractional_point(
    rng: &mut Lcg,
    n: usize,
    demand: &[f64],
    capacity: f64,
    routes: usize,
) -> Vec<(Vec<usize>, f64)> {
    let choices = [0.25, 0.5, 0.75, 1.0];
    let mut columns = Vec::new();
    for _ in 0..routes {
        let mut pool: Vec<usize> = (1..=n).collect();
        for i in (1..pool.len()).rev() {
            let j = (rng.next_u32() as usize) % (i + 1);
            pool.swap(i, j);
        }
        let mut route = Vec::new();
        let mut load = 0.0;
        for v in pool {
            if load + demand[v] <= capacity && rng.random() < 0.6 {
                route.push(v);
                load += demand[v];
            }
        }
        if !route.is_empty() {
            columns.push((route, choices[rng.randint(0, 3) as usize]));
        }
    }
    columns
}

fn brute_force_best_kpath_violation(
    columns: &[(Vec<usize>, f64)],
    demand: &[f64],
    capacity: f64,
    n: usize,
) -> f64 {
    let mut best = 0.0f64;
    for mask in 1..(1usize << n) {
        let mut d_sum = 0.0;
        for i in 0..n {
            if (mask & (1 << i)) != 0 {
                d_sum += demand[i + 1];
            }
        }
        let k = (d_sum / capacity).ceil();
        let mut weight = 0.0;
        for (route, value) in columns {
            if route.iter().any(|&v| (mask & (1 << (v - 1))) != 0) {
                weight += *value;
            }
        }
        best = best.max(k - weight);
    }
    best
}

fn set_partitions(items: &[usize]) -> Vec<Vec<Vec<usize>>> {
    if items.is_empty() {
        return vec![vec![]];
    }
    let first = items[0];
    let rest = &items[1..];
    let mut out = Vec::new();
    for mask in 0..(1usize << rest.len()) {
        let mut block = vec![first];
        let mut remaining = Vec::new();
        for (i, &v) in rest.iter().enumerate() {
            if (mask & (1 << i)) != 0 {
                block.push(v);
            } else {
                remaining.push(v);
            }
        }
        for mut tail in set_partitions(&remaining) {
            let mut full = vec![block.clone()];
            full.append(&mut tail);
            out.push(full);
        }
    }
    out
}

#[test]
fn kpath_cuts_hold_for_every_integer_solution() {
    let mut rng = Lcg(11);
    let mut checked = 0;
    for _ in 0..12 {
        let n = rng.randint(4, 5) as usize;
        let capacity = rng.randint(5, 9) as f64;
        let mut demand = vec![0.0];
        for _ in 0..n {
            demand.push(rng.randint(1, 4) as f64);
        }
        let mut sep = KPathSeparator::new(demand.clone(), capacity).unwrap();
        let cols = random_fractional_point(&mut rng, n, &demand, capacity, 6);
        let cuts = sep.separate(&cols).unwrap();
        let items: Vec<usize> = (1..=n).collect();
        for part in set_partitions(&items) {
            if part
                .iter()
                .any(|b| b.iter().map(|&v| demand[v]).sum::<f64>() > capacity)
            {
                continue;
            }
            for cut in &cuts {
                let lhs: i32 = part.iter().map(|r| cut.coefficient(r).unwrap()).sum();
                assert!(lhs >= cut.rhs);
                checked += 1;
            }
        }
    }
    assert!(checked > 100);
}

#[test]
fn detects_a_violation_whenever_one_exists() {
    let mut rng = Lcg(5);
    let mut cases = 0;
    let mut found = 0;
    for _ in 0..60 {
        let n = rng.randint(4, 8) as usize;
        let capacity = rng.randint(5, 10) as f64;
        let mut demand = vec![0.0];
        for _ in 0..n {
            demand.push(rng.randint(1, 4) as f64);
        }
        let columns = random_fractional_point(&mut rng, n, &demand, capacity, 6);
        let best = brute_force_best_kpath_violation(&columns, &demand, capacity, n);
        if best <= 1e-6 {
            continue;
        }
        cases += 1;
        let mut sep = KPathSeparator::new(demand, capacity).unwrap();
        let cuts = sep.separate(&columns).unwrap();
        if !cuts.is_empty() {
            found += 1;
        }
        assert!(cuts
            .iter()
            .all(|c| c.violation > 0.0 && c.violation <= best + 1e-6));
    }
    assert!(cases >= 20);
    assert_eq!(found, cases);
}

#[test]
fn exact_separation_finds_the_most_violated_unrounded_set() {
    let mut rng = Lcg(23);
    let mut hits = 0;
    let mut trials = 0;
    for _ in 0..40 {
        let n = rng.randint(4, 7) as usize;
        let capacity = rng.randint(6, 10) as f64;
        let mut demand = vec![0.0];
        for _ in 0..n {
            demand.push(rng.randint(1, 4) as f64);
        }
        let columns = random_fractional_point(&mut rng, n, &demand, capacity, 6);
        let best = brute_force_best_kpath_violation(&columns, &demand, capacity, n);
        if best <= 1e-6 {
            continue;
        }
        trials += 1;
        let mut sep = KPathSeparator::new(demand, capacity).unwrap();
        let cuts = sep.separate(&columns).unwrap();
        if !cuts.is_empty() && (cuts[0].violation - best).abs() < 1e-6 {
            hits += 1;
        }
    }
    assert!(trials >= 15);
    assert!((hits as f64) >= 0.8 * (trials as f64));
}

#[test]
fn two_path_cut_needs_time_windows_and_uses_k_equal_two() {
    let travel = vec![
        vec![0.0, 5.0, 5.0],
        vec![5.0, 0.0, 50.0],
        vec![5.0, 50.0, 0.0],
    ];
    let columns = vec![(vec![1, 2], 0.5), (vec![1], 0.5), (vec![2], 0.5)];
    let mut sep = KPathSeparator::new(vec![0.0, 1.0, 1.0], 10.0).unwrap();
    assert!(sep
        .separate(&columns)
        .unwrap()
        .into_iter()
        .all(|c| !c.two_path));

    let mut sep = KPathSeparator::new(vec![0.0, 1.0, 1.0], 10.0).unwrap();
    sep.set_time_windows(TimeWindows {
        ready: vec![0.0, 8.0, 8.0],
        due: vec![100.0, 12.0, 12.0],
        service: vec![0.0, 0.0, 0.0],
        travel,
    })
    .unwrap();
    let cuts = sep.separate(&columns).unwrap();
    let two_path: Vec<_> = cuts.into_iter().filter(|c| c.two_path).collect();
    assert_eq!(two_path.len(), 1);
    assert_eq!(two_path[0].customers, vec![1, 2]);
    assert_eq!(two_path[0].rhs, 2);
    assert!((two_path[0].violation - 0.5).abs() < 1e-9);
}

#[test]
fn strong_k_path_implies_the_capacity_cut() {
    let mut rng = Lcg(2);
    for _ in 0..30 {
        let n = rng.randint(4, 7) as usize;
        let capacity = rng.randint(5, 9) as f64;
        let mut demand = vec![0.0];
        for _ in 0..n {
            demand.push(rng.randint(1, 4) as f64);
        }
        let columns = random_fractional_point(&mut rng, n, &demand, capacity, 6);
        let _cap_sep = CapacitySeparator::new(demand.clone(), capacity).unwrap();
        for a in 1..=n {
            for b in (a + 1)..=n {
                let members = BTreeSet::from([a, b]);
                let k = ((demand[a] + demand[b]) / capacity).ceil();
                let kpath_lhs: f64 = columns
                    .iter()
                    .filter(|(r, _)| r.iter().any(|v| members.contains(v)))
                    .map(|(_, val)| *val)
                    .sum();
                let mut crossing = 0.0;
                for (route, value) in &columns {
                    let mut previous = 0usize;
                    for &v in route.iter().chain(std::iter::once(&0usize)) {
                        if members.contains(&previous) != members.contains(&v) {
                            crossing += *value;
                        }
                        previous = v;
                    }
                }
                assert!(crossing >= 2.0 * kpath_lhs - 1e-9);
                if kpath_lhs >= k - 1e-9 {
                    assert!(crossing >= 2.0 * k - 1e-9);
                }
            }
        }
    }
}

#[test]
fn kpath_validation() {
    assert!(KPathSeparator::new(vec![0.0, 1.0], 0.0).is_err());
    let mut sep = KPathSeparator::new(vec![0.0, 1.0, 1.0], 4.0).unwrap();
    assert!(sep.separate(&[(vec![1, 9], 1.0)]).is_err());
    assert!(sep.set_parameter("nope", 1.0).is_err());
    assert_eq!(sep.get_parameter("max_cuts_per_round").unwrap(), 100.0);
    assert_eq!(sep.get_parameter("max_two_path_set_size").unwrap(), 15.0);
    assert_eq!(sep.get_parameter("pulse_buckets").unwrap(), 10.0);
    sep.set_parameter("max_cuts_per_round", 1.0).unwrap();
    let mut sep2 = KPathSeparator::new(vec![0.0, 3.0, 3.0, 3.0], 4.0).unwrap();
    let cuts = sep2
        .separate(&[
            (vec![1], 1.0),
            (vec![2], 1.0),
            (vec![3], 0.5),
            (vec![1, 2, 3], 0.25),
        ])
        .unwrap();
    assert!(cuts.len() <= 100);
}

// ============================================================================
// Cut options tests (ported from test_cut_options.py)
// ============================================================================

#[test]
fn knapsack_and_kpath_options_configure_separators() {
    let mut options = Options::default();
    let mut knap = KnapsackSeparator::new();
    knap.configure(&options).unwrap();
    assert_eq!(knap.get_parameter("use_rounding").unwrap(), 0.0);
    assert_eq!(knap.get_parameter("max_one_k").unwrap(), 1.0);

    options.set("BALDESresConsKnapsackCutsMode", 9.0).unwrap();
    options
        .set("BALDESresConsKnapsackCutsMaxNumPerRound", 5.0)
        .unwrap();
    knap.configure(&options).unwrap();
    assert_eq!(knap.get_parameter("use_rounding").unwrap(), 1.0);
    assert_eq!(knap.get_parameter("max_one_k").unwrap(), 9.0);
    assert_eq!(knap.get_parameter("max_cuts_per_round").unwrap(), 5.0);

    options.set("BALDESresConsKnapsackCutsMode", 0.0).unwrap();
    knap.configure(&options).unwrap();
    assert_eq!(knap.get_parameter("use_rounding").unwrap(), 0.0);
    assert_eq!(knap.get_parameter("max_one_k").unwrap(), 0.0);
    assert!(knap.separate(&[4, 5], &[1.5, 1.5], 10).unwrap().is_empty());

    options
        .set("BALDESresConsKnapsackCutsMaxNumPerRound", 0.0)
        .unwrap();
    assert!(knapsack_parameters(&options, KnapsackParameters::default()).is_err());

    let mut kpath = KPathSeparator::new(vec![0.0, 3.0, 3.0], 4.0).unwrap();
    options.set("BALDESrankOneCutsMaxNumPerRound", 3.0).unwrap();
    kpath.configure(&options).unwrap();
    assert_eq!(kpath.get_parameter("max_cuts_per_round").unwrap(), 3.0);

    options.set("BALDESrankOneCutsMaxNumPerRound", 0.0).unwrap();
    assert!(kpath_parameters(&options, KPathParameters::default()).is_err());
}

// ============================================================================
// Labeling, enumeration, and pricing integration with strong k-path cuts
// ============================================================================

#[test]
fn exhaustive_mixed_cut_lifecycle_preserves_enumerated_routes() {
    use baldes::cuts::CutClass;

    // Deliberately independent of cut preparation, route reconstruction and
    // coefficient helpers: enumerate every elementary path by DFS.
    fn paths(g: &Graph, v: usize, seen: u32, path: &mut Vec<i32>, out: &mut Vec<Vec<i32>>) {
        if v == g.sink {
            out.push(path.clone());
            return;
        }
        for a in g.arcs.iter().filter(|a| a.tail == v) {
            if seen & (1 << a.head) != 0 {
                continue;
            }
            path.push(a.id);
            paths(g, a.head, seen | (1 << a.head), path, out);
            path.pop();
        }
    }

    for covering in [false, true] {
        for scan in [0., 1.] {
            for cyclic in [false, true] {
                let mut solver = Solver::new(6, 1).unwrap();
                solver.graph.elementary = true;
                solver.enumeration_preserve_arc_paths = true;
                for v in 0..6 {
                    solver.graph.set_bounds(v, 0, 0., 5.).unwrap();
                }
                for tail in 0..5 {
                    for head in 1..6 {
                        if head == tail || (!cyclic && head < tail) {
                            continue;
                        }
                        let id = (tail * 19 + head * 3) as i32;
                        solver
                            .graph
                            .add_arc_with_id(id, tail, head, head.abs_diff(tail) as f64 * 0.25)
                            .unwrap();
                        solver.graph.arc_mut(id).unwrap().consumption[0] = 1.;
                    }
                }
                solver
                    .options
                    .set("BALDESuseExactComplBoundsInEnumeration", scan)
                    .unwrap();
                // Build two rank-one rows with identical support and one k-path.
                for dual in [-0.25, -0.5] {
                    let i = solver
                        .cuts
                        .add_strong_k_path_cut(6, &[1, 2, 3], 1, dual)
                        .unwrap();
                    let row = &mut solver.cuts.cuts[i];
                    row.visit_once = false;
                    row.denominator = 3;
                    row.class = if covering {
                        CutClass::Covering
                    } else {
                        CutClass::Packing
                    };
                }
                solver
                    .cuts
                    .add_strong_k_path_cut(6, &[1, 3], 1, -0.75)
                    .unwrap();
                let original = solver.cuts.cuts.clone();
                let mut all_paths = vec![];
                paths(&solver.graph, 0, 1, &mut vec![], &mut all_paths);
                assert_eq!(all_paths.len(), if cyclic { 65 } else { 16 });

                for stage in 0..7 {
                    match stage {
                        1 => {
                            for row in &mut solver.cuts.cuts {
                                row.dual *= if row.class == CutClass::Covering {
                                    -2.
                                } else {
                                    2.
                                };
                            }
                        }
                        2 => {
                            solver.cuts.cuts.reverse();
                            for arc in &mut solver.graph.arcs {
                                arc.reduced_cost += 0.25;
                            }
                        }
                        3 => {
                            let row = solver.cuts.cuts.iter_mut().find(|r| !r.visit_once).unwrap();
                            row.weights[3] = 0;
                            row.ordered_support = Some(vec![(1, 1), (2, 1)]);
                        }
                        4 => solver.cuts.retire(1),
                        5 => {
                            solver.cuts.retired.clear();
                            solver.cuts.cuts = original.clone();
                        }
                        6 => solver.cuts.cuts.clear(),
                        _ => {}
                    }
                    let mut expected = BTreeMap::new();
                    for ids in &all_paths {
                        let arcs: Vec<_> = ids
                            .iter()
                            .map(|id| solver.graph.arcs.iter().find(|a| a.id == *id).unwrap())
                            .collect();
                        let mut cost: f64 = arcs.iter().map(|a| a.reduced_cost).sum();
                        let physical: Vec<_> =
                            arcs.iter().map(|a| (a.tail, a.head, a.id)).collect();
                        for (i, row) in solver.cuts.cuts.iter().enumerate() {
                            if solver.cuts.retired.contains(&i) {
                                continue;
                            }
                            let sum: i32 = arcs.iter().map(|a| row.weights[a.head]).sum();
                            let coefficient = if row.visit_once {
                                i32::from(sum > 0)
                            } else if row.class == CutClass::Covering {
                                (sum + row.denominator - 1) / row.denominator
                            } else {
                                sum / row.denominator
                            };
                            assert_eq!(
                                solver
                                    .cuts
                                    .route_coefficient(&solver.graph, i, &physical, true)
                                    .unwrap(),
                                coefficient
                            );
                            cost -= row.dual * coefficient as f64;
                        }
                        expected.insert(ids.clone(), cost);
                    }
                    // Stage zero runs actual pricing. Later stages first exercise
                    // repricing of the retained complete universe.
                    let priced = solver.price_phase(0, 100, false).unwrap();
                    assert!(priced.exact);
                    assert_eq!(
                        priced.minimum_reduced_cost,
                        expected.values().copied().min_by(f64::total_cmp).unwrap()
                    );
                    if stage > 0 {
                        let actual: BTreeMap<_, _> = priced
                            .columns
                            .iter()
                            .map(|r| (r.arcs.clone(), r.cost))
                            .collect();
                        assert_eq!(
                            actual, expected,
                            "retained stage {stage}, covering={covering}, scan={scan}"
                        );
                    }
                    let routes = solver.enumerate(100.).unwrap();
                    let actual: BTreeMap<_, _> = routes
                        .iter()
                        .map(|r| {
                            let mut nodes = vec![0];
                            for id in &r.arcs {
                                let a = solver.graph.arcs.iter().find(|a| a.id == *id).unwrap();
                                assert_eq!(*nodes.last().unwrap(), a.tail);
                                nodes.push(a.head);
                            }
                            assert_eq!(
                                r.nodes,
                                nodes.iter().map(|&v| v as i32).collect::<Vec<_>>()
                            );
                            (r.arcs.clone(), r.cost)
                        })
                        .collect();
                    assert_eq!(routes.len(), actual.len());
                    assert_eq!(
                        actual, expected,
                        "enumeration stage {stage}, covering={covering}, scan={scan}"
                    );
                    // Strict cutoff at an exactly representable cost, including
                    // repricing a retained universe and building a new one.
                    for cutoff in [2., 3., 4.] {
                        // Inspection removes paths, so restore the full universe
                        // before testing a different cutoff.
                        solver.enumerate(100.).unwrap();
                        let below: BTreeMap<_, _> = expected
                            .iter()
                            .filter(|(_, cost)| **cost < cutoff)
                            .map(|(path, cost)| (path.clone(), *cost))
                            .collect();
                        let inspected = solver.inspect_enumeration(cutoff).unwrap().unwrap();
                        assert_eq!(
                            inspected
                                .iter()
                                .map(|r| (r.arcs.clone(), r.cost))
                                .collect::<BTreeMap<_, _>>(),
                            below
                        );
                        let filtered = solver.enumerate(cutoff).unwrap();
                        assert_eq!(
                            filtered
                                .iter()
                                .map(|r| (r.arcs.clone(), r.cost))
                                .collect::<BTreeMap<_, _>>(),
                            below
                        );
                    }
                    // Discard all cached state and compare fresh pricing and
                    // enumeration with the same oracle, then retain the universe
                    // for the next lifecycle mutation.
                    solver.restore_arcs();
                    let fresh = solver.price_phase(0, 100, false).unwrap();
                    assert!(fresh.exact);
                    assert_eq!(
                        fresh.minimum_reduced_cost,
                        expected.values().copied().min_by(f64::total_cmp).unwrap(),
                        "fresh stage {stage}, covering={covering}, scan={scan}, cyclic={cyclic}"
                    );
                    let duals: Vec<_> = solver.cuts.cuts.iter().map(|row| row.dual).collect();
                    for row in &mut solver.cuts.cuts {
                        row.dual *= 2.;
                    }
                    solver.refresh_cuts().unwrap();
                    for (row, dual) in solver.cuts.cuts.iter_mut().zip(duals) {
                        row.dual = dual;
                    }
                    let updated = solver.price_phase(0, 100, false).unwrap();
                    assert!(updated.exact);
                    assert_eq!(updated.minimum_reduced_cost, fresh.minimum_reduced_cost);
                    let rebuilt = solver.enumerate(100.).unwrap();
                    assert_eq!(
                        rebuilt
                            .iter()
                            .map(|r| (r.arcs.clone(), r.cost))
                            .collect::<BTreeMap<_, _>>(),
                        expected
                    );
                }
            }
        }
    }
}

#[test]
fn enumeration_reprices_current_arc_and_strong_k_path_duals() {
    for covering in [false, true] {
        let mut solver = Solver::new(4, 0).unwrap();
        for (tail, head, cost) in [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (0, 3, 4.0)] {
            solver.graph.add_arc(tail, head, cost).unwrap();
        }
        use baldes::cuts::{CutClass, MemoryKind, RankOneCut};
        let class = if covering {
            CutClass::Covering
        } else {
            CutClass::Packing
        };
        solver.cuts.cuts.push(RankOneCut {
            graph_support: BTreeMap::new(),
            graph_pricing_support: BTreeMap::new(),
            native_kpath_support: None,
            weights: vec![0, 1, 1, 0],
            memory: vec![false, true, true, false],
            graph_dependent_memory: false,
            graph_memories: BTreeMap::new(),
            memory_kind: MemoryKind::Vertex,
            arc_memory: BTreeSet::new(),
            graph_arc_memories: BTreeMap::new(),
            arc_weights: BTreeMap::new(),
            visit_once: false,
            native_type2: false,
            denominator: 3,
            dual: 0.0,
            class,
            rhs: 0,
            kind: -1,
            id: 0,
            ordered_support: Some(vec![(1, 1), (2, 1)]),
        });
        solver
            .cuts
            .add_strong_k_path_cut(4, &[1, 2], 1, 0.0)
            .unwrap();

        assert_eq!(
            solver.run_fixing_enumeration(0, 0.0, 10.0).unwrap().state,
            2
        );
        solver.cuts.cuts[0].dual = -2.0;
        solver.cuts.cuts[1].dual = -3.0;

        let result = solver.price_phase(0, 10, false).unwrap();
        assert!(result.exact);
        let expected_path_cost = if covering { 8.0 } else { 6.0 };
        let got: Vec<(Vec<i32>, f64)> = result
            .columns
            .iter()
            .map(|r| (r.arcs.clone(), r.cost))
            .collect();
        assert_eq!(
            got,
            vec![(vec![3], 4.0), (vec![0, 1, 2], expected_path_cost)]
        );

        // Now test active full-memory enumeration and bidirectional labeling directly
        solver.restore_arcs();
        solver.cuts.cuts[0].dual = 0.0;
        let priced = solver.price_phase(0, 10, false).unwrap();
        assert!(priced.exact);
        assert_eq!(priced.minimum_reduced_cost, 4.0);
        let enum_routes = solver.enumerate(10.0).unwrap();
        let mut enum_costs: Vec<(Vec<i32>, f64)> = enum_routes
            .iter()
            .map(|r| (r.arcs.clone(), r.cost))
            .collect();
        enum_costs.sort_by(|a, b| a.1.total_cmp(&b.1));
        assert_eq!(enum_costs, vec![(vec![3], 4.0), (vec![0, 1, 2], 6.0)]);
    }
}

#[test]
fn bidirectional_pricing_and_non_elementary_memory_with_strong_k_path_cut() {
    let mut g = Graph::new(5, 1).unwrap();
    for v in 0..5 {
        g.set_bounds(v, 0, 0.0, 10.0).unwrap();
    }
    // Path A: 0 -> 1 -> 3 -> 2 -> 4 (visits {1, 2} separated by 3, cost 4)
    // Path B: 0 -> 3 -> 4 (does not visit {1, 2}, cost 6)
    for (u, v, cost, d) in [
        (0, 1, 1.0, 1.0),
        (1, 3, 1.0, 1.0),
        (3, 2, 1.0, 1.0),
        (2, 4, 1.0, 1.0),
        (0, 3, 3.0, 3.0),
        (3, 4, 3.0, 2.0),
    ] {
        let id = g.add_arc(u, v, cost).unwrap();
        g.arc_mut(id).unwrap().consumption[0] = d;
    }

    let mut solver = Solver::new(5, 1).unwrap();
    solver.graph = g;
    // Strong k-path cut on {1, 2} with dual -3.0 (so visiting {1, 2} adds +3.0 to reduced cost).
    // With vertex 3 in memory, visiting 1 -> 3 -> 2 is charged once (+3.0 -> total 7.0).
    let idx = solver
        .cuts
        .add_strong_k_path_cut(5, &[1, 2], 1, -3.0)
        .unwrap();
    let res = solver.price_phase(0, 10, false).unwrap();
    assert!(res.exact);
    assert!((res.minimum_reduced_cost - 6.0).abs() < 1e-9);
    let path_a = res
        .columns
        .iter()
        .find(|r| r.nodes == vec![0, 1, 3, 2, 4])
        .unwrap();
    assert!((path_a.cost - 7.0).abs() < 1e-9);
    assert_eq!(
        solver
            .cuts
            .coefficient(&solver.graph, idx, &[1, 3, 2], false)
            .unwrap(),
        1
    );

    // Without vertex 3 in memory, leaving {1, 2} at 3 forgets the visit and charges twice (+6.0 -> total 10.0).
    solver.cuts.cuts[idx].memory[3] = false;
    solver.restore_arcs();
    let res_forgotten = solver.price_phase(0, 10, false).unwrap();
    let path_a_forgotten = res_forgotten
        .columns
        .iter()
        .find(|r| r.nodes == vec![0, 1, 3, 2, 4])
        .unwrap();
    assert!((path_a_forgotten.cost - 10.0).abs() < 1e-9);
    assert_eq!(
        solver
            .cuts
            .coefficient(&solver.graph, idx, &[1, 3, 2], false)
            .unwrap(),
        2
    );
}

#[cfg(feature = "clp")]
#[test]
fn root_master_with_kpath_cuts() {
    use baldes::master::root::{solve, KPathCutConfig, RootOptions, RootProblem};

    let mut g = Graph::new(5, 1).unwrap();
    for v in 0..5 {
        g.set_bounds(v, 0, 0.0, 4.0).unwrap();
    }
    let demands = vec![0.0, 3.0, 3.0, 1.0];
    for u in 0..4 {
        for v in 1..=4 {
            if u == v || (u == 0 && v == 4) {
                continue;
            }
            let d = if v < 4 { demands[v] } else { 0.0 };
            let cost = match (u, v) {
                (0, 1) | (1, 4) | (0, 2) | (2, 4) => 5.0,
                (1, 2) | (2, 1) => 1.0,
                _ => 2.0,
            };
            let id = g.add_arc(u, v, cost).unwrap();
            g.arc_mut(id).unwrap().consumption[0] = d;
        }
    }
    let problem = RootProblem::set_partitioning(g, 3.0).unwrap();
    let options = RootOptions::default().with_kpath_cuts(KPathCutConfig::new(demands, 4.0));
    let out = solve(problem, options).unwrap();
    assert!(out.master.lp.is_finite());
}

#[test]
fn separated_kpath_memory_survives_registration_and_pricing() {
    let mut separator = KPathSeparator::new(vec![0., 2., 2., 1.], 5.).unwrap();
    let cut = separator
        .separate(&[(vec![1, 2], 0.5), (vec![3], 1.)])
        .unwrap()
        .into_iter()
        .find(|c| c.customers == [1, 2])
        .unwrap();
    assert!(cut.memory.is_empty());
    assert!(cut.coefficient(&[1, 999]).is_err());
    assert!(cut.coefficient_with_memory(&[1, 999], true).is_err());
    let mut solver = Solver::new(5, 1).unwrap();
    for v in 0..5 {
        solver.graph.set_bounds(v, 0, 0., 5.).unwrap();
    }
    for (u, v, load) in [(0, 1, 2.), (1, 3, 1.), (3, 2, 2.), (2, 4, 0.)] {
        let id = solver.graph.add_arc(u, v, -2.).unwrap();
        solver.graph.arc_mut(id).unwrap().consumption[0] = load;
    }
    let index = solver
        .cuts
        .register_strong_k_path_cut(5, 0, &cut, -3.)
        .unwrap();
    assert_eq!(
        solver
            .cuts
            .coefficient(&solver.graph, index, &[1, 3, 2], false)
            .unwrap(),
        2
    );
    assert_eq!(
        cut.coefficient_with_memory(&[0, 1, 3, 2, 4], false)
            .unwrap(),
        2
    );
    assert_eq!(
        solver
            .cuts
            .coefficient(&solver.graph, index, &[1, 3, 2], true)
            .unwrap(),
        1
    );
    let priced = solver.price_phase(0, 10, false).unwrap();
    assert_eq!(priced.columns[0].cost, -2.);
    solver.graph.graph_id = 1;
    assert_eq!(
        solver
            .cuts
            .coefficient(&solver.graph, index, &[1, 3, 2], false)
            .unwrap(),
        0
    );
    assert_eq!(
        solver
            .cuts
            .coefficient(&solver.graph, index, &[1, 3, 2], true)
            .unwrap(),
        1
    );
    assert_eq!(
        solver.price_phase(0, 10, false).unwrap().columns[0].cost,
        -8.
    );
    assert_eq!(solver.enumerate(10.).unwrap()[0].cost, -5.);
    assert_eq!(
        solver.price_phase(0, 10, false).unwrap().columns[0].cost,
        -5.
    );
}

#[test]
fn kpath_memory_ignores_enumerated_columns() {
    use baldes::cuts::FractionalColumn;
    let mut sep = KPathSeparator::new(vec![0., 2., 2., 1.], 5.).unwrap();
    for enumerated in [false, true] {
        let cuts = sep
            .separate_fractional(&[
                FractionalColumn {
                    vertices: vec![1, 3, 2],
                    value: 0.5,
                    graph_id: 0,
                    enumerated,
                },
                FractionalColumn {
                    vertices: vec![3],
                    value: 1.,
                    graph_id: 0,
                    enumerated: false,
                },
            ])
            .unwrap();
        let cut = cuts.iter().find(|c| c.customers == [1, 2]).unwrap();
        assert_eq!(cut.memory, if enumerated { vec![] } else { vec![3] });
    }
}


fn knapsack_pricing_solver(bidirectional: bool) -> Solver {
    use baldes::cuts::knapsack::{KnapsackColumn, KnapsackRow, ResourceKnapsackSeparator};
    let mut separator = ResourceKnapsackSeparator {
        parameters: KnapsackParameters {
            use_rounding: true,
            max_one_k: 0,
            ..Default::default()
        },
        rows: vec![KnapsackRow::new(0, 0, 1., 4)],
        ..Default::default()
    };
    let mut cuts = separator
        .separate(&[KnapsackColumn {
            graph_id: 0,
            resources: vec![3.],
            value: 1.5,
        }])
        .unwrap();
    cuts.retain(|c| c.cut.numerator == 2 && c.cut.denominator == 3);
    assert_eq!(cuts.len(), 1);
    cuts[0].dual = -2.;
    let mut solver = Solver::new(4, 1).unwrap();
    solver
        .set_option(
            "BALDESuseBidirectionalSearch",
            if bidirectional { 1. } else { 0. },
        )
        .unwrap();
    for v in 0..4 {
        solver.graph.set_bounds(v, 0, 0., 4.).unwrap();
    }
    // Competing routes have the same raw cost but different knapsack loads.
    for (u, v, load) in [(0, 1, 1.), (1, 3, 0.), (0, 2, 3.), (2, 3, 0.)] {
        let id = solver.graph.add_arc(u, v, -3.).unwrap();
        solver.graph.arc_mut(id).unwrap().consumption[0] = load;
    }
    solver.cuts.knapsack_cuts = cuts;
    solver
}

#[test]
fn knapsack_pricing_costs_match_master_coefficients_in_both_directions() {
    for bidirectional in [false, true] {
        let mut solver = knapsack_pricing_solver(bidirectional);
        let priced = solver.price_phase(0, 10, false).unwrap();
        assert!(priced.exact);
        assert_eq!(priced.minimum_reduced_cost, -6.);
        for route in &priced.columns {
            let coefficient = solver.cuts.knapsack_cuts[0].coefficient(0, &route.resources);
            assert_eq!(route.cost, route.original_cost + 2. * coefficient as f64);
        }
        assert!(priced
            .columns
            .iter()
            .any(|r| r.nodes == [0, 2, 3] && r.cost == -2.));
        solver.cuts.knapsack_cuts[0].dual = -4.;
        let repriced = solver.price_phase(0, 10, false).unwrap();
        assert!(repriced
            .columns
            .iter()
            .any(|r| r.nodes == [0, 2, 3] && r.cost == 2.));
    }
}

#[test]
fn knapsack_enumeration_and_repricing_keep_load_dependent_costs() {
    let mut solver = knapsack_pricing_solver(true);
    let routes = solver.enumerate(10.).unwrap();
    assert_eq!(routes.len(), 2);
    assert!(routes.iter().any(|r| r.nodes == [0, 2, 3] && r.cost == -2.));
    solver.cuts.knapsack_cuts[0].dual = -4.;
    let repriced = solver.price_phase(0, 10, false).unwrap();
    assert!(repriced
        .columns
        .iter()
        .any(|r| r.nodes == [0, 2, 3] && r.cost == 2.));
    assert!(solver.enumeration.is_some());
    solver.cuts.knapsack_cuts[0].dual = 0.;
    let repriced = solver.price_phase(0, 10, false).unwrap();
    assert!(repriced.columns.iter().all(|r| r.cost == -6.));
}

#[cfg(feature = "clp")]
#[test]
fn knapsack_root_cuts_close_fractional_capacity_gap_and_survive_snapshot() {
    use baldes::cuts::knapsack::KnapsackRow;
    use baldes::master::{
        column_generation::Status,
        root::{KnapsackCutConfig, RootOptions},
    };
    use baldes::vrptw::{Instance, Rounding};
    let instance = Instance::parse("knapsack-root\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n3 1 1 1 0 100 0\n").unwrap();
    let scaled = RootOptions::default()
        .with_knapsack_cuts(KnapsackCutConfig::new(vec![KnapsackRow::new(0, 1, 2., 6)]));
    let scaled = instance.solve_root(8, Rounding::None, scaled).unwrap();
    assert_eq!(scaled.master.status, Status::Optimal);
    assert!(scaled.master.lp.is_finite());
    for enumerate in [false, true] {
        for original_resource in [1, 7] {
            // Sum of the three unit customer-cover rows gives total load <= 3.
            let mut config =
                KnapsackCutConfig::new(vec![KnapsackRow::new(0, original_resource, 1., 3)]);
            if original_resource == 7 {
                config.resource_indices = BTreeMap::from([(12, 0), (7, 1)]);
            }
            config.parameters.use_rounding = true;
            config.parameters.max_one_k = 0;
            let mut options = RootOptions::default().with_knapsack_cuts(config);
            options.enumerate = enumerate;
            options.capture_state = true;
            let out = instance.solve_root(8, Rounding::None, options).unwrap();
            assert_eq!(out.master.status, Status::Optimal);
            assert!(out.cuts > 0);
            assert!(out.master.cut_rounds > 0);
            assert!((out.master.lp - (4. + 2f64.sqrt())).abs() < 1e-8);
            assert!(out.snapshot.as_ref().unwrap().active_cuts() > 0);
            let mut child_options = RootOptions::default();
            child_options.enumerate = enumerate;
            child_options.node_depth = 1;
            let child =
                baldes::master::root::inherit(out.snapshot.as_ref().unwrap(), &[], child_options)
                    .unwrap();
            assert_eq!(child.master.status, Status::Optimal);
            assert!(child.inherited_cuts > 0);
            assert!((child.master.lp - out.master.lp).abs() < 1e-8);
        }
    }
}

#[test]
fn knapsack_pricing_matches_exhaustive_routes_with_two_resources() {
    use baldes::cuts::knapsack::{KnapsackColumn, KnapsackRow, ResourceKnapsackSeparator};
    fn visit(
        g: &Graph,
        cuts: &[baldes::cuts::knapsack::ResourceKnapsackCut],
        vertex: usize,
        seen: u64,
        q: [f64; 2],
        cost: f64,
        best: &mut f64,
    ) {
        if vertex == g.sink {
            let penalty: f64 = cuts
                .iter()
                .map(|c| -c.dual * c.coefficient(0, &q) as f64)
                .sum();
            *best = best.min(cost + penalty);
            return;
        }
        for arc in g.arcs.iter().filter(|a| a.tail == vertex) {
            if seen & (1 << arc.head) != 0 {
                continue;
            }
            if let Some(next) = baldes::labeling::extend_resources(g, arc, q, true) {
                visit(
                    g,
                    cuts,
                    arc.head,
                    seen | (1 << arc.head),
                    next,
                    cost + arc.cost,
                    best,
                );
            }
        }
    }
    let mut rng = Lcg(0x5ba9);
    for trial in 0..40 {
        let mut sep = ResourceKnapsackSeparator {
            parameters: KnapsackParameters {
                use_rounding: true,
                max_one_k: 0,
                ..Default::default()
            },
            rows: vec![KnapsackRow::new(0, 0, 1., 4), KnapsackRow::new(0, 1, 1., 7)],
            ..Default::default()
        };
        let mut cuts = sep
            .separate(&[KnapsackColumn {
                graph_id: 0,
                resources: vec![3., 5.],
                value: 1.5,
            }])
            .unwrap();
        let mut rows = BTreeSet::new();
        cuts.retain(|c| rows.insert(c.row));
        assert_eq!(cuts.len(), 2);
        for c in &mut cuts {
            c.dual = -rng.randint(1, 4) as f64;
        }
        let mut solver = Solver::new(6, 2).unwrap();
        solver
            .set_option(
                "BALDESuseBidirectionalSearch",
                if trial % 2 == 0 { 1. } else { 0. },
            )
            .unwrap();
        for v in 0..6 {
            solver.graph.set_bounds(v, 0, 0., 6.).unwrap();
            solver.graph.set_bounds(v, 1, 0., 7.).unwrap();
            solver.graph.set_ng(v, &[0, 1, 2, 3, 4, 5]).unwrap();
        }
        let loads: Vec<_> = (0..6)
            .map(|v| {
                if v == 0 || v == 5 {
                    [0., 0.]
                } else {
                    [rng.randint(1, 3) as f64, rng.randint(1, 3) as f64]
                }
            })
            .collect();
        for u in 0..5 {
            for v in 1..=5 {
                if u == v {
                    continue;
                }
                let cost = rng.randint(-5, 5) as f64;
                let id = solver.graph.add_arc(u, v, cost).unwrap();
                solver.graph.arc_mut(id).unwrap().consumption = loads[v];
            }
        }
        solver.cuts.knapsack_cuts = cuts;
        let mut best = f64::INFINITY;
        visit(
            &solver.graph,
            &solver.cuts.knapsack_cuts,
            0,
            1,
            [0., 0.],
            0.,
            &mut best,
        );
        let result = solver.price_phase(0, 150, false).unwrap();
        assert!(
            (result.minimum_reduced_cost - best).abs() < 1e-9,
            "trial {trial}: {} != {best}",
            result.minimum_reduced_cost
        );
        for route in result.columns {
            let penalty: f64 = solver
                .cuts
                .knapsack_cuts
                .iter()
                .map(|c| -c.dual * c.coefficient(0, &route.resources) as f64)
                .sum();
            assert!((route.cost - route.original_cost - penalty).abs() < 1e-9);
        }
    }
}


#[test]
fn knapsack_retrieval_remaps_resources_and_retains_inactive_slots() {
    use baldes::cuts::knapsack::{KnapsackPricing, ResourceKnapsackCut, RouteLoadKnapsackCut};
    let mapping = BTreeMap::from([(7, 0), (3, 1)]);
    let mut cache = KnapsackPricing::default();
    let mut cut = ResourceKnapsackCut {
        id: 0,
        row: 0,
        cut_class: 0,
        graph_resources: BTreeMap::from([(4, (7, 2.))]),
        dual: -1.234567895,
        cut: RouteLoadKnapsackCut {
            kind: KnapsackKind::Rounding,
            numerator: 1,
            denominator: 1,
            rhs: 1,
            lhs: 0.,
            violation: 0.,
            thresholds: vec![],
            coefficients: BTreeMap::from([(0, 0), (3, 1)]),
        },
    };
    cache.retrieve(&[cut.clone()], 4, &mapping).unwrap();
    assert!(cache.active);
    assert_eq!(cache.cost(&[3., 0.]), 1.2345679);
    assert_eq!(cache.resources.len(), 2);
    assert!(cache.resources[&1].is_empty());
    cache.retrieve(&[], 4, &mapping).unwrap();
    assert!(!cache.active);
    assert_eq!(cache.resources.len(), 2);
    assert!(cache.resources.values().all(BTreeMap::is_empty));
    cut.cut.coefficients.clear();
    cache.retrieve(&[cut.clone()], 4, &mapping).unwrap();
    assert!(cache.active); // Native active flag tracks association, not keys.
    cut.dual = 1.;
    assert!(cache.retrieve(&[cut], 4, &mapping).is_err());
    assert!(!cache.active);
    assert!(cache.resources.values().all(BTreeMap::is_empty));
}










#[test]
fn mapped_knapsack_separation_retains_original_ids() {
    use baldes::cuts::knapsack::{KnapsackColumn, KnapsackRow, ResourceKnapsackSeparator};
    let mut direct = ResourceKnapsackSeparator::default();
    direct.parameters.use_rounding = true;
    direct.parameters.max_one_k = 0;
    direct.rows = vec![KnapsackRow::new(4, 7, 1., 4), KnapsackRow::new(4, 3, 1., 3)];
    let mut mapped = direct.clone();
    let mut resources = vec![0.; 8];
    resources[7] = 3.;
    resources[3] = 2.;
    let expected = direct
        .separate(&[KnapsackColumn {
            graph_id: 4,
            resources,
            value: 1.5,
        }])
        .unwrap();
    let indices = BTreeMap::from([(4, BTreeMap::from([(7, 1), (3, 0)]))]);
    let actual = mapped
        .separate_mapped(
            &[KnapsackColumn {
                graph_id: 4,
                resources: vec![2., 3.],
                value: 1.5,
            }],
            &indices,
        )
        .unwrap();
    assert!(!actual.is_empty());
    assert_eq!(actual, expected);
    assert_eq!(mapped.num_separate_calls, direct.num_separate_calls);
    assert_eq!(mapped.next_cut_id, direct.next_cut_id);
    for cut in &actual {
        let mut original = vec![0.; 8];
        original[7] = 3.;
        original[3] = 2.;
        assert_eq!(
            cut.coefficient_mapped(4, &[2., 3.], &indices[&4]),
            cut.coefficient(4, &original)
        );
        assert!(cut.graph_resources[&4].0 == 7 || cut.graph_resources[&4].0 == 3);
    }
}

#[test]
fn original_knapsack_ids_survive_pricing_and_enumeration() {
    for bidirectional in [false, true] {
        let mut solver = knapsack_pricing_solver(bidirectional);
        for cut in &mut solver.cuts.knapsack_cuts {
            cut.graph_resources.get_mut(&0).unwrap().0 = 7;
        }
        solver.knapsack_resource_indices = BTreeMap::from([(7, 0)]);
        assert_eq!(
            solver.knapsack_coefficient(&solver.cuts.knapsack_cuts[0], &[3.]),
            2
        );
        let priced = solver.price_phase(0, 10, false).unwrap();
        let route = priced
            .columns
            .iter()
            .find(|r| r.nodes == [0, 2, 3])
            .unwrap();
        assert_eq!(route.cost, -2.);
        let routes = solver.enumerate(10.).unwrap();
        assert_eq!(
            routes.iter().find(|r| r.nodes == [0, 2, 3]).unwrap().cost,
            -2.
        );
        solver.cuts.knapsack_cuts[0].dual = -4.;
        let routes = solver.price_phase(0, 10, false).unwrap();
        assert_eq!(
            routes
                .columns
                .iter()
                .find(|r| r.nodes == [0, 2, 3])
                .unwrap()
                .cost,
            2.
        );
        assert!(solver.enumeration.is_some());
    }
}
