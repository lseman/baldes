use super::*;
use std::collections::{BTreeSet, HashSet};

fn fixture(n: usize, columns: &[(Vec<usize>, f64)], set_limit: i32, states: usize) -> Separation {
    let mut separation = Separation::prepare(
        &vec![1.; n],
        100.,
        Params {
            max_num_per_round: 100,
            max_kohl_set_size: set_limit,
            max_kohl_states: states,
            num_pulse_buckets: 10,
            violation_tolerance: 1e-6,
            max_shrink_removals: 1000,
        },
        None,
    )
    .unwrap();
    for (column, (customers, value)) in columns.iter().enumerate() {
        separation.columns.push(ColumnInfo {
            column,
            value: *value,
            flag: false,
            customers: customers.clone(),
            vertices: vec![],
        });
        for &customer in customers {
            separation.customer_columns[customer].push(column);
        }
    }
    separation
}

fn search(separation: &mut Separation) -> (usize, usize) {
    let mut tp = TwoPathCandidate {
        size: 0,
        cut_weight: 0.,
        demand: 0.,
        column_touched: vec![false; separation.columns.len()],
        touched_columns: vec![],
        customers: Bits1024::default(),
    };
    let mut visited = HashSet::new();
    let mut remaining = separation.params.max_kohl_states;
    separation.kohl_greedy_procedure(&mut tp, &mut visited, &mut remaining);
    assert_eq!(tp.customers, Bits1024::default());
    assert_eq!(tp.size, 0);
    assert!(tp.cut_weight.abs() < 1e-10);
    assert!(tp.demand.abs() < 1e-10);
    assert!(tp.column_touched.iter().all(|&touched| !touched));
    assert!(tp.touched_columns.is_empty());
    (visited.len(), remaining)
}

#[test]
fn memoized_kohl_leaves_match_exhaustive_small_subset_oracle() {
    let columns = vec![
        (vec![0, 1, 2], 0.7),
        (vec![1, 3], 0.4),
        (vec![2, 4], 0.8),
        (vec![0, 5], 0.5),
    ];
    let n = 6;
    let weight = |mask: usize| -> f64 {
        columns
            .iter()
            .filter(|(rows, _)| rows.iter().any(|&r| mask & (1 << r) != 0))
            .map(|(_, value)| value)
            .sum()
    };
    for limit in [2, 4, 15] {
        let mut separation = fixture(n, &columns, limit, 10_000);
        let (visited, remaining) = search(&mut separation);
        assert!(visited <= 1 << n);
        assert!(remaining > 0);
        let mut expected = BTreeSet::new();
        for mask in 0usize..1 << n {
            let size = mask.count_ones() as i32;
            if size > limit || weight(mask) >= 2. - 1e-6 {
                continue;
            }
            if size < limit
                && (0..n).any(|r| mask & (1 << r) == 0 && weight(mask | (1 << r)) < 2. - 1e-6)
            {
                continue;
            }
            let mut bits = Bits1024::default();
            for r in 0..n {
                if mask & (1 << r) != 0 {
                    bits.set(r);
                }
            }
            expected.insert(bits);
        }
        assert_eq!(
            separation
                .two_path_subsets
                .iter()
                .into_iter()
                .collect::<BTreeSet<_>>(),
            expected
        );
    }
}

#[test]
fn kohl_shared_route_search_is_bounded_and_unwinds_on_budget_exhaustion() {
    // Every permutation of up to 15 out of 20 customers has flow one.
    // The old search repeats equivalent subsets factorially before deduplication.
    let columns = vec![((0..20).collect(), 1.)];
    for budget in [0, 1, 128, 1000] {
        let mut separation = fixture(20, &columns, 15, budget);
        let (visited, remaining) = search(&mut separation);
        assert_eq!(visited, budget);
        assert_eq!(remaining, 0);
        assert!(separation.two_path_subsets.iter().len() <= budget);
    }
}

#[test]
fn kohl_work_budget_parameter_is_validated() {
    let mut separator = KPathSeparator::new(vec![0., 1., 1.], 2.).unwrap();
    assert_eq!(
        separator.get_parameter("max_two_path_states").unwrap(),
        10_000.
    );
    for value in [0., 1., 1234.] {
        separator
            .set_parameter("max_two_path_states", value)
            .unwrap();
        assert_eq!(
            separator.get_parameter("max_two_path_states").unwrap(),
            value
        );
    }
    for value in [-1., 0.5, f64::NAN, f64::INFINITY, i32::MAX as f64 + 1.] {
        assert!(separator
            .set_parameter("max_two_path_states", value)
            .is_err());
    }
}
