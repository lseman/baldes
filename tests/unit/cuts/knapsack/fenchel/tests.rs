use super::*;



#[test]
fn validates_options_and_input() {
    let mut options = FenchelParameters::default();
    options.coefficient_scale = 0;
    assert!(FenchelSeparator::with_parameters(options).is_err());
    assert!(FenchelSeparator::default().separate(&[1], &[], 4).is_err());
    assert!(FenchelSeparator::default()
        .separate(&[-1], &[1.0], 4)
        .is_err());
    assert!(FenchelSeparator::default()
        .separate(&[1], &[f64::NAN], 4)
        .is_err());
}

#[cfg(not(feature = "clp"))]
#[test]
fn reports_missing_backend() {
    assert!(FenchelSeparator::default()
        .separate(&[1], &[1.0], 4)
        .unwrap_err()
        .message
        .contains("clp"));
}

#[cfg(feature = "clp")]
fn assert_valid_on_every_integer_packing(cut: &RouteLoadKnapsackCut, capacity: i32) {
    // Deliberately include loads absent from the separator's input, and
    // combinations containing several copies of the same load.
    fn visit(cut: &RouteLoadKnapsackCut, next: i32, remaining: i32, activity: i32) {
        assert!(
            activity <= cut.rhs,
            "invalid table {:?}, activity={activity}",
            cut.coefficients
        );
        if next > remaining {
            return;
        }
        for count in 0..=remaining / next {
            visit(
                cut,
                next + 1,
                remaining - count * next,
                activity + count * cut.coefficient(next),
            );
        }
    }
    visit(cut, 1, capacity, 0);
    assert_eq!(cut.coefficient(0), 0);
    for q in 1..=capacity + 2 {
        assert!(cut.coefficient(q - 1) <= cut.coefficient(q));
    }
}

#[cfg(feature = "clp")]
#[test]
fn separates_fractional_packings_and_future_loads() {
    // 3*y_3 + 2*y_2 <=4 permits (1.2,0.1), but the integer hull also
    // satisfies 2*y_3 + y_2 <=2.
    let report = FenchelSeparator::default()
        .separate(&[2, 3], &[0.1, 1.2], 4)
        .unwrap();
    assert_eq!(report.status, FenchelStatus::Separated);
    let cut = report.cut.unwrap();
    assert_eq!(cut.kind, KnapsackKind::Fenchel);
    assert!(cut.violation > 0.0);
    assert_valid_on_every_integer_packing(&cut, 4);
    assert!(cut.coefficient(4) >= cut.coefficient(3));
    let mut rng = crate::rng::Xoroshiro128Plus::from_seed(719_u64);
    for capacity in 2..=12 {
        for _ in 0..12 {
            let mut loads = Vec::new();
            let mut values = Vec::new();
            for load in 1..=capacity {
                let seed = rng.next();
                if seed % 3 != 0 {
                    continue;
                }
                loads.push(load);
                values.push(((seed >> 32) % 40) as f64 / 20.0);
            }
            let report = FenchelSeparator::default()
                .separate(&loads, &values, capacity)
                .unwrap();
            if let Some(cut) = report.cut {
                assert_valid_on_every_integer_packing(&cut, capacity);
                let lhs: f64 = loads
                    .iter()
                    .zip(&values)
                    .map(|(&q, &x)| x * f64::from(cut.coefficient(q)))
                    .sum();
                assert!((lhs - cut.lhs).abs() < 1e-8);
                assert!(lhs > f64::from(cut.rhs));
            }
        }
    }
}

#[cfg(feature = "clp")]
#[test]
fn feasible_convex_combinations_and_work_limits() {
    let separator = FenchelSeparator::default();
    // Half a packing with two 2s, half a packing with one 3.
    assert!(separator
        .separate(&[2, 3], &[1.0, 0.5], 4)
        .unwrap()
        .cut
        .is_none());
    let mut limits = FenchelParameters::default();
    limits.max_iterations = 0;
    assert_eq!(
        FenchelSeparator::with_parameters(limits.clone())
            .unwrap()
            .separate(&[2, 3], &[0.1, 1.2], 4)
            .unwrap()
            .status,
        FenchelStatus::IterationLimit
    );
    limits.time_limit = Duration::ZERO;
    assert_eq!(
        FenchelSeparator::with_parameters(limits)
            .unwrap()
            .separate(&[2, 3], &[0.1, 1.2], 4)
            .unwrap()
            .status,
        FenchelStatus::TimeLimit
    );
    assert_eq!(
        separator.separate(&[2], &[1.0], 5000).unwrap().status,
        FenchelStatus::DomainLimit
    );
    let overloaded = separator
        .separate(&[0, 5], &[3.0, 0.25], 4)
        .unwrap()
        .cut
        .unwrap();
    assert_eq!(overloaded.rhs, 0);
    assert_eq!(overloaded.coefficient(4), 0);
    assert_eq!(overloaded.coefficient(5), 1);
    assert_valid_on_every_integer_packing(&overloaded, 4);
}

#[cfg(feature = "clp")]
#[test]
fn integrated_resource_separator_remaps_and_limits_combined_families() {
    use crate::cuts::knapsack::*;
    let mut separator = ResourceKnapsackSeparator {
        rows: vec![KnapsackRow::new(8, 7, 2.0, 4)],
        parameters: KnapsackParameters {
            use_rounding: false,
            max_one_k: 0,
            fenchel: Some(FenchelParameters::default()),
            ..Default::default()
        },
        ..Default::default()
    };
    let cuts = separator
        .separate_mapped(
            &[
                KnapsackColumn {
                    graph_id: 8,
                    resources: vec![1.0],
                    value: 0.1,
                },
                KnapsackColumn {
                    graph_id: 8,
                    resources: vec![1.5],
                    value: 1.2,
                },
            ],
            &BTreeMap::from([(8, BTreeMap::from([(7, 0)]))]),
        )
        .unwrap();
    assert_eq!(cuts.len(), 1);
    assert_eq!(cuts[0].cut.kind, KnapsackKind::Fenchel);
    assert_eq!(
        cuts[0].coefficient_mapped(8, &[2.0], &BTreeMap::from([(7, 0)])),
        cuts[0].cut.coefficient(4)
    );
    assert_valid_on_every_integer_packing(&cuts[0].cut, 4);
    separator.parameters.use_rounding = true;
    separator.parameters.max_one_k = 1;
    separator.parameters.max_cuts_per_round = 1;
    assert_eq!(
        separator
            .separate_mapped(
                &[KnapsackColumn {
                    graph_id: 8,
                    resources: vec![1.5],
                    value: 1.2
                },],
                &BTreeMap::from([(8, BTreeMap::from([(7, 0)]))])
            )
            .unwrap()
            .len(),
        1
    );
}

#[cfg(feature = "clp")]
#[test]
fn root_fenchel_only_cuts_survive_enumeration_and_child_restore() {
    use crate::inputs::{Instance, Rounding};
    use crate::master::{
        column_generation::Status,
        root::{inherit, RootOptions},
    };
    let instance = Instance::parse("fenchel-root\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n3 1 1 1 0 100 0\n").unwrap();
    for enumerate in [false, true] {
        let config = instance.fenchel_cut_config().unwrap();
        assert_eq!(config.parameters.max_one_k, 0);
        let mut options = RootOptions::default().with_knapsack_cuts(config);
        options.enumerate = enumerate;
        options.capture_state = true;
        let out = instance.solve_root(8, Rounding::None, options).unwrap();
        assert_eq!(out.master.status, Status::Optimal);
        assert!(out.cuts > 0);
        assert!((out.master.lp - (4.0 + 2.0_f64.sqrt())).abs() < 1e-7);
        let snapshot = out.snapshot.as_ref().unwrap();
        assert!(snapshot.active_cuts() > 0);
        let mut child_options = RootOptions::default();
        child_options.enumerate = enumerate;
        child_options.node_depth = 1;
        let child = inherit(snapshot, &[], child_options).unwrap();
        assert_eq!(child.master.status, Status::Optimal);
        assert!(child.inherited_cuts > 0);
        assert!((child.master.lp - out.master.lp).abs() < 1e-7);
    }
    let mut scaled = instance.clone();
    for customer in &mut scaled.customers {
        customer.demand *= 50.0;
    }
    scaled.capacity *= 50.0;
    let config = scaled.fenchel_cut_config().unwrap();
    assert_eq!(config.rows[0].rhs, 3);
    assert_eq!(config.rows[0].graph_resources[&0], (1, 0.02));
    scaled.customers[1].demand = 0.5;
    assert!(scaled.fenchel_cut_config().is_err());
}

#[cfg(feature = "clp")]
#[test]
fn generated_cut_prices_and_enumerates_with_scaled_resources_in_both_directions() {
    use crate::cuts::knapsack::ResourceKnapsackCut;
    use crate::pricing::Solver;
    let cut = FenchelSeparator::default()
        .separate(&[4], &[1.5], 6)
        .unwrap()
        .cut
        .unwrap();
    for bidirectional in [false, true] {
        let mut solver = Solver::<2>::new(4, 2).unwrap();
        solver.graph.graph_id = 8;
        solver.graph.elementary = true;
        solver.knapsack_resource_indices = BTreeMap::from([(7, 1)]);
        solver
            .set_option("BALDESuseBidirectionalSearch", f64::from(bidirectional))
            .unwrap();
        for v in 0..4 {
            solver.graph.set_bounds(v, 0, 0.0, 5.0).unwrap();
            solver.graph.set_bounds(v, 1, 0.0, 2.0).unwrap();
        }
        for (from, to, cost) in [
            (0, 1, -4.0),
            (1, 2, -3.0),
            (2, 3, 0.0),
            (0, 2, -2.0),
            (1, 3, 0.0),
        ] {
            let id = solver.graph.add_arc(from, to, cost).unwrap();
            solver.graph.arc_mut(id).unwrap().consumption =
                [1.0, if to == 3 { 0.0 } else { 1.0 }].into();
        }
        solver.cuts.knapsack_cuts.push(ResourceKnapsackCut {
            id: 0,
            row: 0,
            cut_class: 1,
            graph_resources: BTreeMap::from([(8, (7, 2.0))]),
            cut: cut.clone(),
            dual: -0.123456789,
        });
        assert!(!solver.knapsack_exact_coefficients); // kind selects exact cache automatically
        let priced = solver.price_phase(0, 20, false).unwrap();
        let expected = -7.0 + 0.123456789 * f64::from(cut.coefficient(4));
        assert!((priced.minimum_reduced_cost - expected).abs() < 1e-9);
        let mut child = solver.fork_for_node();
        let enumerated = child.enumerate(1000.0).unwrap();
        assert!(!enumerated.is_empty());
        for route in &enumerated {
            let load = (route.resources[1] * 2.0 + 1e-6).floor() as i32;
            let penalty = 0.123456789 * f64::from(cut.coefficient(load));
            assert!((route.cost - route.original_cost - penalty).abs() < 1e-9);
        }
        let best = enumerated
            .iter()
            .map(|r| r.cost)
            .fold(f64::INFINITY, f64::min);
        assert!((best - expected).abs() < 1e-9);
    }
}
