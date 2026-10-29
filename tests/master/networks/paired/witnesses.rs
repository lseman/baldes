use super::*;

#[test]
fn repaired_paired_witnesses_return_the_joint_schedule_across_snapshots() {
    use baldes::master::{
        model::{SynchronizedService, SynchronizedServiceMember},
        tree::{ModelTreeOptions, TreeStatus},
    };
    let base: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/joint-synchronized-model.json"
    ))
    .unwrap();
    for representation in 0..4 {
        let mut source = base.clone();
        let points = control_point_source_model(source["networks"][0].clone());
        for r in 0..2 {
            if representation & (1 << r) != 0 {
                source["networks"][0]["coupled_resources"][r] =
                    points["coupled_resources"][r].clone();
            }
        }
        for quantity in [1., 2.] {
            source["networks"][0]["multiplicity"] = serde_json::json!([quantity, quantity]);
            for mode in [BoundMode::Numerical, BoundMode::Certified] {
                let (problem, registry) = model::format::Model::from_json(&source.to_string())
                    .unwrap()
                    .compile::<2>()
                    .unwrap();
                let mut options = Options::default();
                options.cg.stabilize = false;
                options.cg.bound_mode = mode;
                options.synchronized_services = vec![SynchronizedService {
                    id: 17,
                    resource: 0,
                    vehicles_required: quantity as u32 + 1,
                    reward: 1.,
                    start: [0, 10],
                    members: vec![
                        SynchronizedServiceMember {
                            network: 42,
                            vertex: 1,
                        },
                        SynchronizedServiceMember {
                            network: 7,
                            vertex: 1,
                        },
                    ],
                }];
                registry.configure_pricing(&mut options);
                let root = model::solve(problem, options.clone()).unwrap();
                for start in [None, Some((8, 9))] {
                    let snapshot = match start {
                        Some((lo, hi)) => {
                            root.snapshot.with_synchronized_start(17, lo, hi).unwrap()
                        }
                        None => root.snapshot.with_synchronized_start(17, 0, 10).unwrap(),
                    };
                    let mut tree = ModelTreeOptions::default();
                    tree.node = options.clone();
                    registry.configure_tree(&mut tree).unwrap();
                    let result = snapshot.solve_tree(tree).unwrap();
                    assert_eq!(
                        result.status,
                        TreeStatus::Optimal,
                        "{representation} {quantity} {mode:?}"
                    );
                    assert_eq!(result.incumbent, -10. * quantity + 1.);
                    for fixed in &result.routes {
                        assert_eq!(
                            fixed.route.route.resource_profile[1][0],
                            if start.is_some() { 8. } else { 5. }
                        );
                    }
                    let primary: Vec<_> = result
                        .routes
                        .iter()
                        .filter(|r| r.route.network == 42)
                        .collect();
                    assert_eq!(primary.iter().map(|r| r.quantity).sum::<f64>(), quantity);
                    for resource in 0..2 {
                        let total: f64 = primary
                            .iter()
                            .map(|r| {
                                r.quantity
                                    * r.route
                                        .route
                                        .coupled_values
                                        .iter()
                                        .find(|(index, _)| *index == resource)
                                        .unwrap()
                                        .1
                            })
                            .sum();
                        assert_eq!(total, 4.);
                    }
                    if quantity == 1. {
                        assert_eq!(primary.len(), 1);
                        assert_eq!(
                            primary[0].route.route.coupled_values,
                            vec![(0, 4.), (1, 4.)]
                        );
                    }
                }
                // The narrowed child must not rewrite the parent's cached profiles.
                assert_eq!(
                    root.snapshot.solve(options).unwrap().master.status,
                    Status::Optimal
                );
            }
        }
    }
}

#[test]
fn cross_network_and_cross_path_rows_recover_joint_source_controls() {
    use baldes::master::tree::{ModelTreeOptions, TreeStatus};
    for (cross_network, fixture) in [
        (
            true,
            include_str!("../../../fixtures/coupled/cross-network-interior-model.json"),
        ),
        (
            false,
            include_str!("../../../fixtures/coupled/cross-path-interior-model.json"),
        ),
    ] {
        let base: serde_json::Value = serde_json::from_str(fixture).unwrap();
        for representation in 0..4 {
            let mut source = base.clone();
            if cross_network {
                for i in 0..2 {
                    if representation & (1 << i) != 0 {
                        source["networks"][i] =
                            control_point_source_model(source["networks"][i].clone());
                    }
                }
            } else {
                let points = control_point_source_model(source["networks"][0].clone());
                for i in 0..2 {
                    if representation & (1 << i) != 0 {
                        source["networks"][0]["coupled_resources"][i] =
                            points["coupled_resources"][i].clone();
                    }
                }
            }
            for counts in [[1., 1.], [2., 2.], [2., 1.], [3., 1.], [3., 3.]] {
                if cross_network {
                    for i in 0..2 {
                        source["networks"][i]["multiplicity"] =
                            serde_json::json!([counts[i], counts[i]]);
                    }
                } else {
                    source["networks"][0]["multiplicity"] =
                        serde_json::json!([counts[0] + counts[1], counts[0] + counts[1]]);
                    for i in 0..2 {
                        source["rows"][i]["rhs"] = counts[i].into();
                    }
                }
                for mode in [BoundMode::Numerical, BoundMode::Certified] {
                    for schedule in [
                        model::NetworkPricingSchedule::FullSweep,
                        model::NetworkPricingSchedule::Cyclic,
                        model::NetworkPricingSchedule::Sorted,
                        model::NetworkPricingSchedule::PromisingFirst,
                        model::NetworkPricingSchedule::UnvisitedFirst,
                    ] {
                        let (problem, registry) =
                            model::format::Model::from_json(&source.to_string())
                                .unwrap()
                                .compile::<2>()
                                .unwrap();
                        let mut options = ModelTreeOptions::default();
                        options.node.cg.stabilize = false;
                        options.node.cg.bound_mode = mode;
                        options.node.network_pricing_schedule = schedule;
                        registry.configure_tree(&mut options).unwrap();
                        let tree = baldes::master::tree::solve(problem, options).unwrap();
                        let expected = -10. * counts[0] + 2. * counts[1];
                        assert_eq!(
                            tree.status,
                            TreeStatus::Optimal,
                            "{cross_network} {representation} {counts:?} {mode:?} {schedule:?}"
                        );
                        assert_eq!(tree.incumbent, expected);
                        assert!(tree.bound <= expected && tree.bound > expected - 1e-7);
                        let mut quantities = [0., 0.];
                        let mut gamma = [0., 0.];
                        for fixed in &tree.routes {
                            let second = if cross_network {
                                fixed.route.network == 7
                            } else {
                                fixed.route.route.arcs == vec![1]
                            };
                            let i = usize::from(second);
                            assert!(fixed.quantity > 0. && fixed.quantity.fract() == 0.);
                            quantities[i] += fixed.quantity;
                            let resource = if cross_network { 0 } else { i };
                            gamma[i] += fixed.quantity
                                * fixed
                                    .route
                                    .route
                                    .coupled_values
                                    .iter()
                                    .find(|(r, _)| *r == resource)
                                    .unwrap()
                                    .1;
                            if !cross_network {
                                assert_eq!(
                                    fixed
                                        .route
                                        .route
                                        .coupled_values
                                        .iter()
                                        .find(|(r, _)| *r == 1 - i)
                                        .unwrap()
                                        .1,
                                    0.
                                );
                            }
                        }
                        assert_eq!(quantities, counts);
                        assert_eq!(gamma, [4., 4.]);
                    }
                }
            }
        }
    }
}

#[test]
fn narrow_dyadic_domains_recover_integral_copy_blocks() {
    use baldes::master::{
        model,
        tree::{ModelTreeOptions, TreeStatus},
    };
    let positive: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/dyadic-copy-model.json"
    ))
    .unwrap();
    for sign in [1., -1.] {
        let mut source = positive.clone();
        if sign < 0. {
            source["rows"][0]["rhs"] = (-8.).into();
            for network in source["networks"].as_array_mut().unwrap() {
                let resource = &mut network["coupled_resources"][0];
                resource["initial"]["pairs"][0][0] = (-1.25).into();
                resource["initial"]["pairs"][1][0] = (-1.).into();
                for arc in resource["arcs"].as_array_mut().unwrap() {
                    arc["lower"]["pairs"][0][0] = (-8.).into();
                    arc["upper"]["pairs"][0][0] = 0.into();
                }
            }
        }
        for mode in [BoundMode::Numerical, BoundMode::Certified] {
            let (problem, registry) = model::format::Model::from_json(&source.to_string())
                .unwrap()
                .compile::<2>()
                .unwrap();
            let mut options = ModelTreeOptions::default();
            options.node.cg.stabilize = false;
            options.node.cg.bound_mode = mode;
            registry.configure_tree(&mut options).unwrap();
            let tree = baldes::master::tree::solve(problem, options).unwrap();
            assert_eq!(tree.status, TreeStatus::Optimal);
            assert_eq!(tree.incumbent, -24.);
            let mut quantities = [0.; 2];
            let mut gamma = [0.; 2];
            for fixed in tree.routes {
                let i = usize::from(fixed.route.network == 7);
                assert!(fixed.quantity > 0. && fixed.quantity.fract() == 0.);
                quantities[i] += fixed.quantity;
                gamma[i] += fixed.quantity
                    * fixed
                        .route
                        .route
                        .coupled_values
                        .iter()
                        .find(|(r, _)| *r == 0)
                        .unwrap()
                        .1;
            }
            assert_eq!(quantities, [3., 3.]);
            assert_eq!(gamma, [4. * sign; 2]);
        }
    }
}

#[test]
fn narrow_dyadic_domains_recover_exact_three_copy_gamma_totals() {
    use baldes::master::tree::{ModelTreeOptions, TreeStatus};

    let source = include_str!("../../../fixtures/coupled/dyadic-narrow-interior-model.json");
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let (problem, registry) = model::format::Model::from_json(source)
            .unwrap()
            .compile::<2>()
            .unwrap();
        let mut options = ModelTreeOptions::default();
        options.node.cg.stabilize = false;
        options.node.cg.bound_mode = mode;
        registry.configure_tree(&mut options).unwrap();
        let tree = baldes::master::tree::solve(problem, options).unwrap();
        assert_eq!(tree.status, TreeStatus::Optimal, "{tree:?}");
        assert_eq!(tree.incumbent, -24.);
        assert!(tree.bound <= -24. && tree.bound > -24. - 1e-7);
        let mut quantities = [0., 0.];
        let mut totals = [0., 0.];
        let mut controls = [
            std::collections::BTreeSet::new(),
            std::collections::BTreeSet::new(),
        ];
        for fixed in &tree.routes {
            let index = usize::from(fixed.route.network == 7);
            let gamma = fixed.route.route.coupled_values[0].1;
            assert!((0.125..=0.15625).contains(&gamma));
            assert!(fixed.quantity > 0. && fixed.quantity.fract() == 0.);
            quantities[index] += fixed.quantity;
            totals[index] += fixed.quantity * gamma;
            controls[index].insert(gamma.to_bits());
        }
        assert_eq!(quantities, [3., 3.]);
        assert!((totals[0] - 0.40625).abs() <= f64::EPSILON);
        assert!((totals[1] - 0.40625).abs() <= f64::EPSILON);
        assert!(controls.iter().all(|values| values.len() > 1));
    }
}

#[test]
fn narrow_fractional_domains_recover_exact_three_copy_gamma_totals() {
    use baldes::master::tree::{ModelTreeOptions, TreeStatus};

    let mut source: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/dyadic-narrow-interior-model.json"
    ))
    .unwrap();
    let expected_gamma = 1.0333333333333333_f64 - 1.;
    let expected_total = expected_gamma * 3.;
    source["rows"][0]["rhs"] = (expected_total * 2.).into();
    for network in source["networks"].as_array_mut().unwrap() {
        network["coupled_resources"][0]["initial_base"]["lower"] = 1.03.into();
        network["coupled_resources"][0]["initial_base"]["upper"] = 1.04.into();
    }
    let source = source.to_string();
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let (problem, registry) = model::format::Model::from_json(&source)
            .unwrap()
            .compile::<2>()
            .unwrap();
        let mut options = ModelTreeOptions::default();
        options.node.cg.stabilize = false;
        options.node.cg.bound_mode = mode;
        registry.configure_tree(&mut options).unwrap();
        let tree = baldes::master::tree::solve(problem, options).unwrap();
        assert_eq!(tree.status, TreeStatus::Optimal, "{tree:?}");
        assert_eq!(tree.incumbent, -24.);
        let mut quantities = [0., 0.];
        let mut totals = [0., 0.];
        let mut controls = [
            std::collections::BTreeSet::new(),
            std::collections::BTreeSet::new(),
        ];
        for fixed in &tree.routes {
            let index = usize::from(fixed.route.network == 7);
            let gamma = fixed.route.route.coupled_values[0].1;
            assert!((0.03..=0.04).contains(&gamma));
            quantities[index] += fixed.quantity;
            totals[index] += fixed.quantity * gamma;
            controls[index].insert(gamma.to_bits());
        }
        assert_eq!(quantities, [3., 3.]);
        assert_eq!(totals, [expected_total, expected_total]);
        // Exact weighted totals above are the contract; an optimal mixture
        // need not contain the homogeneous three-copy control itself.
        assert!(controls.iter().all(|values| !values.is_empty()));
    }
}

#[test]
fn paired_cyclic_pricing_keeps_convexity_duals_on_the_network_row() {
    let source = include_str!("../../../fixtures/coupled/paired-cyclic-pricing-model.json");
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let problem = model::format::Model::from_json(source)
            .unwrap()
            .problem::<2>()
            .unwrap();
        let mut options = Options::default();
        options.cg.stabilize = false;
        options.cg.artificial_cost = 100.;
        options.cg.bound_mode = mode;
        let root = model::solve(problem, options).unwrap();
        assert_eq!(root.master.status, Status::Optimal);
        // Independent oracle: the direct two-arc path costs -20+1+1.
        assert_eq!(root.master.lp, -18.);
        assert!((root.master.bound + 18.).abs() < 1e-8);
        assert_eq!(root.duals, vec![-18.]);
        assert_eq!(root.master.pricing_calls, 3);
    }
}
