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
