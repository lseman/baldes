use super::*;

fn cut() -> ResourceKnapsackCut {
    ResourceKnapsackCut {
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
    }
}

#[test]
fn retrieval_handles_coefficient_differences_outside_i32_range() {
    let mut cut = cut();
    cut.dual = -0.5;
    cut.cut.coefficients = BTreeMap::from([(0, i32::MIN), (2, i32::MAX)]);
    let mut pricing = KnapsackPricing::default();
    pricing
        .retrieve(&[cut], 4, &BTreeMap::from([(7, 0)]))
        .unwrap();
    let steps = &pricing.resources[&0];
    // Captured from the pinned binary's retrieval routine with the same cut.
    assert_eq!(steps[&0].to_bits(), 13_965_662_444_475_908_096);
    assert_eq!(steps[&2].to_bits(), 4_742_290_407_616_937_984);
}

#[test]
fn exact_pricing_and_master_coefficients_agree_above_native_integer_range() {
    let mut cut = cut();
    cut.graph_resources = BTreeMap::from([(7, (0, 1.))]);
    cut.cut.coefficients = BTreeMap::from([(0, 0), (1, 1), (3, 4), (6, 9)]);
    cut.dual = -1.;
    let mut pricing = KnapsackPricing::default();
    pricing
        .retrieve_exact(&[cut.clone()], 7, &BTreeMap::from([(0, 0)]))
        .unwrap();
    for line in include_str!("coefficient_range_native.txt").lines() {
        let mut fields = line.split_whitespace();
        let load: f64 = fields.next().unwrap().parse().unwrap();
        let native: i32 = fields.next().unwrap().parse().unwrap();
        let native_pricing_bits: u64 = fields.next().unwrap().parse().unwrap();
        assert!(fields.next().is_none());
        let coefficient = cut.coefficient(7, &[load]);
        assert_eq!(coefficient, 9);
        assert_eq!(
            cut.coefficient_mapped(7, &[load], &BTreeMap::from([(0, 0)])),
            coefficient
        );
        assert_eq!(pricing.cost(&[load]), f64::from(coefficient));
        assert_eq!(pricing.cost(&[load]).to_bits(), native_pricing_bits);
        assert_eq!(native, if load < 2_147_483_648. { 9 } else { 0 });
    }
}

#[test]
fn nonempty_retrieval_preserves_exact_and_rounded_costs_without_snapshot() {
    let mapping = BTreeMap::from([(7, 0), (3, 1)]);
    for exact in [false, true] {
        let mut cached = KnapsackPricing::default();
        let mut cuts = vec![cut()];
        for dual in [-1.234567895, -1e-6, -1e-12, -0., 0., -2.] {
            cuts[0].dual = dual;
            cached.refresh_cached(&cuts, 4, &mapping, 2, exact).unwrap();
            let mut fresh = KnapsackPricing::default();
            if exact {
                fresh.retrieve_exact(&cuts, 4, &mapping).unwrap();
            } else {
                fresh.retrieve(&cuts, 4, &mapping).unwrap();
            }
            assert_eq!(cached, fresh);
            for load in [0., 1.499, 1.5, 3., 10.] {
                assert_eq!(
                    cached.cost(&[load, 0.]).to_bits(),
                    fresh.cost(&[load, 0.]).to_bits()
                );
            }
            assert!(cached.source.is_none());
            assert!(!cached.refresh_cached(&cuts, 4, &mapping, 2, exact).unwrap());
            assert!(cached.source.is_none());
        }
    }
}

#[test]
fn cached_retrieval_detects_public_output_and_source_mutations_and_retains_old_slots() {
    let mapping = BTreeMap::from([(7, 0), (3, 1)]);
    for exact in [false, true] {
        let mut cache = KnapsackPricing::default();
        let mut cuts = vec![cut()];
        cache.refresh_cached(&cuts, 4, &mapping, 2, exact).unwrap();
        cache.resources.get_mut(&0).unwrap().insert(3, 99.);
        cache.active = false;
        assert!(cache.refresh_cached(&cuts, 4, &mapping, 2, exact).unwrap());
        let expected = cache.clone();
        assert!(cache.source.is_none());
        cuts[0].cut.coefficients.insert(5, 2);
        assert!(cache.refresh_cached(&cuts, 4, &mapping, 2, exact).unwrap());
        assert_ne!(cache, expected);
        assert!(cache.source.is_none());
        // Mappings are part of the source; compatibility retrieval deliberately
        // retains old empty resource slots, unlike exact retrieval.
        let changed = BTreeMap::from([(7, 1)]);
        cache.refresh_cached(&cuts, 4, &changed, 2, exact).unwrap();
        let mut fresh = expected;
        if exact {
            fresh.retrieve_exact(&cuts, 4, &changed).unwrap();
        } else {
            fresh.retrieve(&cuts, 4, &changed).unwrap();
        }
        assert_eq!(cache, fresh);
        cache
            .refresh_cached(&[], 4, &BTreeMap::new(), 2, exact)
            .unwrap();
        assert!(!cache.active);
        assert!(cache.resources.values().all(BTreeMap::is_empty));
    }
}

#[test]
fn cached_retrieval_errors_are_atomic_and_unused_nan_fields_remain_accepted() {
    let mapping = BTreeMap::from([(7, 0)]);
    for exact in [false, true] {
        let mut cache = KnapsackPricing::default();
        let cuts = vec![cut()];
        cache.refresh_cached(&cuts, 4, &mapping, 1, exact).unwrap();
        let before = cache.clone();
        assert!(cache.source.is_none());
        for dual in [f64::NAN, f64::INFINITY, 1.] {
            let mut bad = cuts.clone();
            bad[0].dual = dual;
            assert!(cache.refresh_cached(&bad, 4, &mapping, 1, exact).is_err());
            assert_eq!(cache, before);
            assert!(cache.source.is_none());
        }
        assert!(cache
            .refresh_cached(&cuts, 4, &BTreeMap::from([(7, 1)]), 1, exact)
            .is_err());
        assert_eq!(cache, before);
        let mut unused = cuts.clone();
        unused[0].graph_resources.insert(9, (7, f64::NAN));
        cache
            .refresh_cached(&unused, 4, &mapping, 1, exact)
            .unwrap();
        assert_eq!(cache, before);
        cache
            .refresh_cached(&unused, 4, &mapping, 1, exact)
            .unwrap();
        assert_eq!(cache, before);
    }
}

#[test]
fn empty_registry_snapshot_shares_and_detects_public_state_and_mapping_changes() {
    for exact in [false, true] {
        let mapping = BTreeMap::new();
        let mut cache = KnapsackPricing::default();
        cache.refresh_cached(&[], 4, &mapping, 2, exact).unwrap();
        let source = cache.source.as_ref().unwrap().clone();
        assert!(!cache.refresh_cached(&[], 4, &mapping, 2, exact).unwrap());
        assert!(std::sync::Arc::ptr_eq(
            &source,
            cache.source.as_ref().unwrap()
        ));
        assert!(std::sync::Arc::ptr_eq(
            &source,
            cache.clone().source.as_ref().unwrap()
        ));
        cache.resources.get_mut(&0).unwrap().insert(3, 99.);
        cache.active = true;
        assert!(cache.refresh_cached(&[], 4, &mapping, 2, exact).unwrap());
        assert!(!cache.active);
        assert!(cache.resources.values().all(BTreeMap::is_empty));
        let before = cache.clone();
        let source = cache.source.as_ref().unwrap().clone();
        assert!(cache
            .refresh_cached(&[], 4, &BTreeMap::from([(7, 2)]), 2, exact)
            .is_err());
        assert_eq!(cache, before);
        assert!(std::sync::Arc::ptr_eq(
            &source,
            cache.source.as_ref().unwrap()
        ));
        assert!(!cache.refresh_cached(&[], 9, &mapping, 2, exact).unwrap());
        assert!(!std::sync::Arc::ptr_eq(
            &source,
            cache.source.as_ref().unwrap()
        ));
        cache.refresh_cached(&[], 9, &mapping, 3, exact).unwrap();
        assert_eq!(cache.resources.len(), 3);
    }
}

#[test]
fn backward_partial_cost_matches_native_two_resource_breakpoints() {
    let mut graph = crate::Graph::<2>::new(2, 2).unwrap();
    graph.set_bounds(1, 0, 0., 8.).unwrap();
    graph.set_bounds(1, 1, 0., 6.).unwrap();
    let mut pricing = KnapsackPricing::default();
    pricing
        .resources
        .insert(0, BTreeMap::from([(0, 0.125), (2, 1.5), (4, 3.)]));
    pricing
        .resources
        .insert(1, BTreeMap::from([(-1, 0.25), (1, 0.75), (3, 2.)]));
    for (case, line) in include_str!("backward_cost_native.txt").lines().enumerate() {
        let mut fields = line.split_whitespace();
        let first: f64 = fields.next().unwrap().parse().unwrap();
        let second: f64 = fields.next().unwrap().parse().unwrap();
        let expected: u64 = fields.next().unwrap().parse().unwrap();
        assert_eq!(
            pricing
                .partial_cost(&graph, &[first, second], false)
                .to_bits(),
            expected,
            "case {case}: {line}"
        );
    }
}

#[test]
fn retrieval_matches_native_rounded_overlapping_steps_and_resource_mapping() {
    let prices = [
        0.,
        0.000000999999,
        0.000001,
        0.000001000001,
        0.1,
        0.100000005,
        0.123456789,
        1.234567895,
    ];
    let mut a = cut();
    a.graph_resources = BTreeMap::from([(7, (0, 1.))]);
    a.cut.coefficients = BTreeMap::from([(0, 0), (2, 1), (5, 2)]);
    let mut b = a.clone();
    b.graph_resources = BTreeMap::from([(7, (0, 2.))]);
    b.cut.coefficients = BTreeMap::from([(1, 1), (2, 2), (4, 2), (7, 4)]);
    let mut c = a.clone();
    c.graph_resources = BTreeMap::from([(7, (1, 1.))]);
    c.cut.coefficients = BTreeMap::from([(-1, 1), (3, 2)]);
    c.dual = -0.312345678;
    let mut d = a.clone();
    d.graph_resources = BTreeMap::from([(8, (0, 1.))]);
    d.cut.coefficients = BTreeMap::from([(0, 1)]);
    d.dual = -1.;
    let mut pricing = KnapsackPricing::default();
    let mapping = BTreeMap::from([(0, 1), (1, 0)]);
    for (case, row) in include_str!("retrieve_native.txt").lines().enumerate() {
        let mut fields = row.split_whitespace();
        let i: usize = fields.next().unwrap().parse().unwrap();
        let j: usize = fields.next().unwrap().parse().unwrap();
        assert_eq!(fields.next().unwrap(), "1", "native return in case {case}");
        let active = fields.next().unwrap() == "1";
        a.dual = -prices[i];
        b.dual = -prices[j];
        pricing
            .retrieve(&[a.clone(), b.clone(), c.clone(), d.clone()], 7, &mapping)
            .unwrap();
        assert_eq!(pricing.active, active, "case {case}");
        for _ in 0..2 {
            let slot: usize = fields.next().unwrap().parse().unwrap();
            let n: usize = fields.next().unwrap().parse().unwrap();
            let expected: BTreeMap<i32, u64> = (0..n)
                .map(|_| {
                    (
                        fields.next().unwrap().parse().unwrap(),
                        fields.next().unwrap().parse().unwrap(),
                    )
                })
                .collect();
            let actual: BTreeMap<_, _> = pricing.resources[&slot]
                .iter()
                .map(|(&key, &value)| (key, value.to_bits()))
                .collect();
            assert_eq!(actual, expected, "case {case}, slot {slot}");
        }
        assert!(fields.next().is_none());
    }
}

#[test]
fn rounding_separator_matches_native_candidate_order_and_coefficients() {
    let mut parameters = crate::cuts::knapsack::KnapsackParameters::default();
    parameters.max_cuts_per_round = 64;
    parameters.use_rounding = true;
    parameters.max_one_k = 0;
    parameters.violation_tolerance = 1e-6;
    let separator = crate::cuts::knapsack::KnapsackSeparator::with_parameters(parameters).unwrap();
    for (case, row) in include_str!("rounding_native.txt").lines().enumerate() {
        let mut fields = row.split_whitespace();
        assert_eq!(fields.next().unwrap().parse::<usize>().unwrap(), case);
        let rhs: i32 = fields.next().unwrap().parse().unwrap();
        let count: usize = fields.next().unwrap().parse().unwrap();
        let mut loads = Vec::new();
        let mut values = Vec::new();
        for q in 0..=8 {
            let n = (case * 11 + q * 7 + case * q * 3) % 9;
            if n != 0 {
                loads.push(q as i32);
                values.push(n as f64 / if case % 5 == 0 { 128. } else { 8. });
            }
        }
        let cuts = separator.separate(&loads, &values, rhs).unwrap();
        assert_eq!(cuts.len(), count, "case {case}");
        for (index, cut) in cuts.iter().enumerate() {
            let multiplier: u64 = fields.next().unwrap().parse().unwrap();
            let native_rhs: i32 = fields.next().unwrap().parse().unwrap();
            let n: usize = fields.next().unwrap().parse().unwrap();
            let coefficients: BTreeMap<i32, i32> = (0..n)
                .map(|_| {
                    (
                        fields.next().unwrap().parse().unwrap(),
                        fields.next().unwrap().parse().unwrap(),
                    )
                })
                .collect();
            assert_eq!(
                (cut.numerator as f64 / cut.denominator as f64).to_bits(),
                multiplier,
                "case {case}, candidate {index}"
            );
            assert_eq!(cut.rhs, native_rhs, "case {case}, candidate {index}");
            assert_eq!(
                cut.coefficients, coefficients,
                "case {case}, candidate {index}"
            );
        }
        assert!(fields.next().is_none());
    }
}

#[test]
fn one_k_separator_matches_native_candidates_and_cap() {
    for (case, row) in include_str!("one_k_native.txt").lines().enumerate() {
        let mut fields = row.split_whitespace();
        assert_eq!(fields.next().unwrap().parse::<usize>().unwrap(), case);
        let rhs: i32 = fields.next().unwrap().parse().unwrap();
        let max_one_k: i32 = fields.next().unwrap().parse().unwrap();
        let cap: usize = fields.next().unwrap().parse().unwrap();
        let count: usize = fields.next().unwrap().parse().unwrap();
        let mut parameters = crate::cuts::knapsack::KnapsackParameters::default();
        parameters.use_rounding = false;
        parameters.max_one_k = max_one_k;
        parameters.max_cuts_per_round = cap;
        parameters.violation_tolerance = 1e-6;
        let separator =
            crate::cuts::knapsack::KnapsackSeparator::with_parameters(parameters).unwrap();
        let mut loads = Vec::new();
        let mut values = Vec::new();
        for q in 0..=rhs as usize {
            let n = (case * 11 + q * 7 + case * q * 3) % 9;
            if n != 0 {
                loads.push(q as i32);
                values.push(n as f64 / if case % 7 == 0 { 128. } else { 8. });
            }
        }
        let cuts = separator.separate(&loads, &values, rhs).unwrap();
        assert_eq!(cuts.len(), count, "case {case}");
        for (index, cut) in cuts.iter().enumerate() {
            let native_rhs: i32 = fields.next().unwrap().parse().unwrap();
            let n: usize = fields.next().unwrap().parse().unwrap();
            let thresholds: Vec<i32> = (0..n)
                .map(|_| fields.next().unwrap().parse().unwrap())
                .collect();
            let n: usize = fields.next().unwrap().parse().unwrap();
            let coefficients: BTreeMap<i32, i32> = (0..n)
                .map(|_| {
                    (
                        fields.next().unwrap().parse().unwrap(),
                        fields.next().unwrap().parse().unwrap(),
                    )
                })
                .collect();
            assert_eq!(thresholds.first(), Some(&0));
            assert_eq!(cut.rhs, native_rhs, "case {case}, candidate {index}");
            assert_eq!(
                cut.thresholds,
                thresholds[1..],
                "case {case}, candidate {index}"
            );
            assert_eq!(
                cut.coefficients, coefficients,
                "case {case}, candidate {index}"
            );
        }
        assert!(fields.next().is_none());
    }
}

#[test]
fn rounding_and_one_k_share_native_candidate_order_and_cap() {
    for (case, row) in include_str!("mixed_native.txt").lines().enumerate() {
        let mut fields = row.split_whitespace();
        assert_eq!(fields.next().unwrap().parse::<usize>().unwrap(), case);
        let rhs: i32 = fields.next().unwrap().parse().unwrap();
        let max_one_k: i32 = fields.next().unwrap().parse().unwrap();
        let cap: usize = fields.next().unwrap().parse().unwrap();
        let count: usize = fields.next().unwrap().parse().unwrap();
        let mut parameters = crate::cuts::knapsack::KnapsackParameters::default();
        parameters.use_rounding = true;
        parameters.max_one_k = max_one_k;
        parameters.max_cuts_per_round = cap;
        parameters.violation_tolerance = 1e-6;
        let separator =
            crate::cuts::knapsack::KnapsackSeparator::with_parameters(parameters).unwrap();
        let mut loads = Vec::new();
        let mut values = Vec::new();
        for q in 0..=rhs as usize {
            let n = (case * 11 + q * 7 + case * q * 3) % 9;
            if n != 0 {
                loads.push(q as i32);
                values.push(n as f64 / if case % 7 == 0 { 128. } else { 8. });
            }
        }
        let cuts = separator.separate(&loads, &values, rhs).unwrap();
        assert_eq!(cuts.len(), count, "case {case}");
        for (index, cut) in cuts.iter().enumerate() {
            let rounding = fields.next().unwrap() == "1";
            let multiplier: u64 = fields.next().unwrap().parse().unwrap();
            let native_rhs: i32 = fields.next().unwrap().parse().unwrap();
            let n: usize = fields.next().unwrap().parse().unwrap();
            let thresholds: Vec<i32> = (0..n)
                .map(|_| fields.next().unwrap().parse().unwrap())
                .collect();
            let n: usize = fields.next().unwrap().parse().unwrap();
            let coefficients: BTreeMap<i32, i32> = (0..n)
                .map(|_| {
                    (
                        fields.next().unwrap().parse().unwrap(),
                        fields.next().unwrap().parse().unwrap(),
                    )
                })
                .collect();
            if rounding {
                assert!(matches!(
                    cut.kind,
                    crate::cuts::knapsack::KnapsackKind::Rounding
                ));
                assert_eq!(
                    (cut.numerator as f64 / cut.denominator as f64).to_bits(),
                    multiplier,
                    "case {case}, candidate {index}"
                );
                assert!(thresholds.is_empty());
            } else {
                assert!(matches!(
                    cut.kind,
                    crate::cuts::knapsack::KnapsackKind::Steps
                ));
                assert_eq!(thresholds.first(), Some(&0));
            }
            assert_eq!(cut.rhs, native_rhs, "case {case}, candidate {index}");
            assert_eq!(
                cut.thresholds,
                if rounding {
                    &thresholds[..]
                } else {
                    &thresholds[1..]
                },
                "case {case}, candidate {index}"
            );
            assert_eq!(
                cut.coefficients, coefficients,
                "case {case}, candidate {index}"
            );
        }
        assert!(fields.next().is_none());
    }
}

#[test]
fn native_paired_cut_dual_capture_preserves_master_coefficient_charge() {
    let mut item = cut();
    item.graph_resources = BTreeMap::from([(7, (1, 1.))]);
    item.cut.coefficients = BTreeMap::from([(0, 0), (3, 1), (6, 2), (9, 3), (12, 4)]);
    for line in include_str!("paired_master_dual_native.txt").lines() {
        let mut fields = line.split_whitespace();
        let _: usize = fields.next().unwrap().parse().unwrap();
        let native_price: f64 = fields.next().unwrap().parse().unwrap();
        let native_bound: f64 = fields.next().unwrap().parse().unwrap();
        item.dual = -native_price;
        let mut pricing = KnapsackPricing::default();
        pricing
            .retrieve_exact(&[item.clone()], 7, &BTreeMap::from([(1, 1)]))
            .unwrap();
        let resources = [2., 3.];
        assert_eq!(item.coefficient(7, &resources), 1);
        assert_eq!(pricing.cost(&resources), native_price);
        assert_eq!(native_bound + pricing.cost(&resources), 0.);
    }
}
