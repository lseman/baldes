use super::*;
use crate::master::preprocessing::Bounds;
use std::time::Duration;

fn root() -> Output<2> {
    let problem = format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/networks/triangle.json"
    )))
    .unwrap()
    .problem()
    .unwrap();
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.cg.artificial_cost = 100.;
    solve(problem, options).unwrap()
}

fn compact() -> NodeSnapshot<2> {
    let mut node = root().snapshot;
    let removed: Vec<_> = node
        .warm
        .columns
        .iter()
        .enumerate()
        .filter_map(|(c, m)| {
            m.as_ref()
                .filter(|m| m.tag >= 0 && node.master.values[c].abs() < 1e-12)
                .map(|_| c)
        })
        .collect();
    let (_, map) = node.master.remove_cuts(&[], &removed).unwrap();
    node.warm.columns = node
        .warm
        .columns
        .drain(..)
        .zip(&map)
        .filter_map(|(c, m)| m.map(|_| c))
        .collect();
    node.warm.artificials = node
        .warm
        .artificials
        .iter()
        .filter_map(|&c| map[c])
        .collect();
    node.warm.cut_artificials = node
        .warm
        .cut_artificials
        .iter()
        .filter_map(|&(r, c)| Some((r, map[c]?)))
        .collect();
    let retained: HashSet<_> = node
        .warm
        .columns
        .iter()
        .flatten()
        .map(|c| c.tag as usize)
        .collect();
    let mut remap = HashMap::new();
    node.state.routes = node
        .state
        .routes
        .into_iter()
        .enumerate()
        .filter_map(|(old, r)| {
            if retained.contains(&old) {
                remap.insert(old, remap.len());
                Some(r)
            } else {
                None
            }
        })
        .collect();
    for c in node.warm.columns.iter_mut().flatten() {
        c.tag = remap[&(c.tag as usize)] as i32;
    }
    node.state.tags = node
        .state
        .routes
        .iter()
        .enumerate()
        .map(|(tag, r)| ((r.network, r.route.identity()), tag as i32))
        .collect();
    assert!(node.state.route_upper_bounds.is_empty());
    node.warm.generated_pool.clear();
    assert_eq!(node.state.routes.len(), 3);
    node
}

fn donor_with_active_columns() -> NodeSnapshot<2> {
    let mut donor = root().snapshot;
    for route in &mut donor.state.routes {
        let route = Arc::make_mut(route);
        route.route.original_cost = -9999.;
        route.route.cost = -9999.;
        route.route.resources.fill(f64::NAN);
        for point in &mut route.route.resource_profile {
            point.fill(f64::NAN);
        }
    }
    donor
}

fn pool() -> Pool {
    let mut pool = Pool::default();
    pool.absorb(&donor_with_active_columns());
    pool
}

#[test]
fn active_donor_columns_are_inactive_and_available_in_the_recipient() {
    let donor = donor_with_active_columns();
    let active: HashSet<_> = donor
        .warm
        .columns
        .iter()
        .flatten()
        .map(|column| column.tag)
        .collect();
    let expected: HashSet<_> = donor
        .warm
        .columns
        .iter()
        .flatten()
        .chain(&donor.warm.generated_pool)
        .filter_map(|column| {
            usize::try_from(column.tag)
                .ok()
                .and_then(|tag| donor.state.routes.get(tag))
                .map(|route| (route.network, route.route.identity()))
        })
        .collect();
    assert!(
        active.iter().any(|tag| *tag >= 0),
        "fixture must have active donor columns"
    );
    let mut pool = Pool::default();
    pool.absorb(&donor);
    assert_eq!(pool.identities, expected);
    for tag in active.iter().copied().filter(|tag| *tag >= 0) {
        let route = &donor.state.routes[tag as usize];
        assert!(pool
            .identities
            .contains(&(route.network, route.route.identity())));
    }
    let mut recipient = compact();
    assert_eq!(
        pool.seed(&mut recipient, Instant::now() + Duration::from_secs(10))
            .unwrap(),
        6,
        "recipient-active paths are skipped while donor-active, recipient-inactive paths are offered"
    );
}

#[test]
fn registry_only_routes_are_not_treated_as_donor_pool_columns() {
    let mut donor = root().snapshot;
    let mut speculative = donor.state.routes[0].as_ref().clone();
    speculative.route.arcs = vec![i32::MAX];
    let speculative_path = (speculative.network, speculative.route.identity());
    donor.state.routes.push(Arc::new(speculative));

    let expected: HashSet<_> = donor
        .warm
        .columns
        .iter()
        .flatten()
        .chain(&donor.warm.generated_pool)
        .filter_map(|column| {
            usize::try_from(column.tag)
                .ok()
                .and_then(|tag| donor.state.routes.get(tag))
                .map(|route| (route.network, route.route.identity()))
        })
        .collect();
    assert!(!expected.contains(&speculative_path));

    let mut pool = Pool::default();
    pool.absorb(&donor);
    assert_eq!(pool.identities, expected);
    assert!(!pool.identities.contains(&speculative_path));
}

#[test]
fn distinct_donor_schedules_for_one_physical_path_are_both_shared() {
    let mut donor = root().snapshot;
    let (source_tag, source_column) = donor
        .warm
        .columns
        .iter()
        .flatten()
        .find(|column| column.tag >= 0)
        .map(|column| (column.tag as usize, column.clone()))
        .unwrap();
    let mut alias = donor.state.routes[source_tag].as_ref().clone();
    for point in alias.route.resource_profile.iter_mut().skip(1) {
        point[0] += 1.;
    }
    alias.route.resources[0] += 1.;
    let network = alias.network;
    let identity = alias.route.identity();
    let tag = donor.state.routes.len();
    donor.state.routes.push(Arc::new(alias));
    let mut column = source_column;
    column.tag = tag as i32;
    column.identity = format!("{network}:{identity}");
    donor.warm.generated_pool.push(column);

    let mut pool = Pool::default();
    pool.absorb(&donor);
    assert!(pool
        .identities
        .contains(&(network, donor.state.routes[source_tag].route.identity())));
    assert!(pool.identities.contains(&(network, identity.clone())));

    let mut recipient = compact();
    let added = pool
        .seed(&mut recipient, Instant::now() + Duration::from_secs(10))
        .unwrap();
    assert_eq!(added, 7);
    let transferred = recipient
        .state
        .routes
        .iter()
        .find(|route| route.network == network && route.route.identity() == identity)
        .unwrap();
    assert_eq!(transferred.route.resource_profile[1][0], 1.);
}

#[test]
fn shared_paths_replay_recipient_rows_costs_and_tags_without_transferring_proofs() {
    let mut node = compact()
        .with_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 10.,
            },
            &[RowCoefficients {
                network: 7,
                arcs: vec![0.; 10],
                per_route: 2.,
            }],
        )
        .unwrap();
    let physical = node.master.rhs.len() - 1;
    let columns = node.master.costs.len();
    let bound = node.warm.bound;
    let pool = pool();
    assert_eq!(
        pool.seed(&mut node, Instant::now() + Duration::from_secs(10))
            .unwrap(),
        6
    );
    assert_eq!(node.master.costs.len(), columns);
    assert_eq!(node.warm.bound, bound);
    assert_eq!(node.state.routes.len(), 9);
    for c in &node.warm.generated_pool {
        assert!((c.cost - 1.).abs() < 1e-12);
        let route = &node.state.routes[c.tag as usize];
        assert!(route.route.resources.iter().all(|x| x.is_finite()));
        assert!(route
            .route
            .resource_profile
            .iter()
            .flatten()
            .all(|x| x.is_finite()));
        let coefficient = c
            .entries
            .iter()
            .find(|(row, _)| *row == physical)
            .map_or(0., |(_, v)| *v);
        assert_eq!(coefficient, if route.network == 7 { 2. } else { 0. });
        assert_eq!(
            node.state.tags[&(route.network, route.route.identity())],
            c.tag
        );
    }
    assert_eq!(
        pool.seed(&mut node, Instant::now() + Duration::from_secs(10))
            .unwrap(),
        0
    );
    let child = node
        .with_multiplicity(
            42,
            Bounds {
                lower: 0.,
                upper: 0.,
            },
        )
        .unwrap();
    let mut options = Options::default();
    options.cg.stabilize = false;
    let result = child.solve(options).unwrap();
    assert_eq!(result.master.status, cg::Status::Optimal);
    assert!((result.master.bound - 2.).abs() < 1e-8);
}

#[test]
fn shared_paths_do_not_expand_universes_or_disabled_networks() {
    let pool = pool();
    let mut node = compact();
    node.state.universe = Some(vec![0, 1, 2]);
    assert_eq!(
        pool.seed(&mut node, Instant::now() + Duration::from_secs(10))
            .unwrap(),
        0
    );
    assert_eq!(node.state.routes.len(), 3);
    let mut node = compact();
    node.state
        .oracles
        .iter_mut()
        .find(|o| o.network.id == 7)
        .unwrap()
        .network
        .multiplicity
        .upper = 0.;
    assert_eq!(
        pool.seed(&mut node, Instant::now() + Duration::from_secs(10))
            .unwrap(),
        3
    );
    assert!(node
        .warm
        .generated_pool
        .iter()
        .all(|c| node.state.routes[c.tag as usize].network == 42));
}

#[test]
fn shared_paths_cannot_bypass_a_profile_alias_zero_cap() {
    let mut node = root().snapshot;
    let column = node
        .warm
        .columns
        .iter()
        .enumerate()
        .find_map(|(i, c)| {
            c.as_ref()
                .filter(|c| c.tag >= 0 && node.master.values[i].abs() < 1e-12)
                .map(|c| (i, c.tag as usize))
        })
        .unwrap();
    let (_, map) = node.master.remove_cuts(&[], &[column.0]).unwrap();
    node.warm.columns = node
        .warm
        .columns
        .drain(..)
        .zip(&map)
        .filter_map(|(c, m)| m.map(|_| c))
        .collect();
    node.warm.artificials = node
        .warm
        .artificials
        .iter()
        .filter_map(|&c| map[c])
        .collect();
    node.warm.generated_pool.clear();
    node.state.route_upper_bounds.insert(column.1, 0.);
    let mut alias = node.state.routes[column.1].as_ref().clone();
    for point in alias.route.resource_profile.iter_mut().skip(1) {
        point[0] += 1.;
    }
    alias.route.resources[0] += 1.;
    let tag = node.state.routes.len();
    node.state
        .tags
        .insert((alias.network, alias.route.identity()), tag as i32);
    node.state.routes.push(Arc::new(alias));
    assert_eq!(
        pool()
            .seed(&mut node, Instant::now() + Duration::from_secs(10))
            .unwrap(),
        0
    );
    assert_eq!(node.state.route_upper_bounds[&column.1], 0.);
}

#[test]
fn shared_paths_reconstruct_synchronized_recipient_schedules() {
    let mut node = compact();
    node.state
        .oracles
        .iter_mut()
        .find(|o| o.network.id == 7)
        .unwrap()
        .synchronized_windows
        .push((
            3,
            0,
            Window {
                lower: 5.,
                upper: 10.,
            },
        ));
    pool()
        .seed(&mut node, Instant::now() + Duration::from_secs(10))
        .unwrap();
    let route = node
        .state
        .routes
        .iter()
        .find(|r| r.network == 7 && r.route.arcs == [4, 5])
        .unwrap();
    assert_eq!(route.route.resource_profile[1][0], 5.);
    assert_eq!(route.route.resources[0], 5.);
    assert_eq!(route.route.original_cost, 1.);
}

#[test]
fn shared_pool_overflow_does_not_publish_a_partial_registry() {
    let mut pool = pool();
    pool.routes
        .sort_by_key(|r| r.network == 7 && r.route.arcs.contains(&4));
    let mut coefficients = vec![0.; 10];
    coefficients[4] = f64::MAX;
    let mut node = compact()
        .with_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 10.,
            },
            &[RowCoefficients {
                network: 7,
                arcs: coefficients,
                per_route: f64::MAX,
            }],
        )
        .unwrap();
    let tags = node.state.tags.clone();
    let columns = node.master.costs.len();
    let bound = node.warm.bound;
    assert!(pool
        .seed(&mut node, Instant::now() + Duration::from_secs(10))
        .is_err());
    assert_eq!(node.state.tags, tags);
    assert_eq!(node.state.routes.len(), 3);
    assert!(node.warm.generated_pool.is_empty());
    assert_eq!(node.master.costs.len(), columns);
    assert_eq!(node.warm.bound, bound);
}

#[test]
fn optional_tree_pool_keeps_original_numerical_and_certified_optima() {
    for mode in [cg::BoundMode::Numerical, cg::BoundMode::Certified] {
        let node = compact();
        let mut options = ModelTreeOptions::default();
        options.node.cg.stabilize = false;
        options.node.cg.bound_mode = mode;
        options.share_inactive_columns = true;
        options.branching = NetworkBranchingPolicy::Phased;
        let result = node.solve_tree(options).unwrap();
        assert_eq!(result.status, crate::master::tree::TreeStatus::Optimal);
        assert_eq!(result.incumbent, 2.);
        assert!((result.bound - 2.).abs() < 1e-8);
    }
}

#[test]
fn inactive_scan_is_opt_in_and_preserves_restricted_snapshot_proof_bounds() {
    let mut node = compact()
        .with_multiplicity(
            42,
            Bounds {
                lower: 0.,
                upper: 0.,
            },
        )
        .unwrap();
    pool()
        .seed(&mut node, Instant::now() + Duration::from_secs(10))
        .unwrap();
    assert!(!node.warm.generated_pool.is_empty());
    for mode in [cg::BoundMode::Numerical, cg::BoundMode::Certified] {
        let mut options = Options::default();
        options.cg.stabilize = false;
        options.cg.bound_mode = mode;
        let off = node.solve(options.clone()).unwrap();
        options.search_inactive_columns = true;
        let on = node.solve(options).unwrap();
        assert_eq!(off.master.status, cg::Status::Optimal);
        assert_eq!(on.master.status, cg::Status::Optimal);
        assert!((off.master.lp - 2.).abs() < 1e-8);
        assert!((on.master.lp - 2.).abs() < 1e-8);
        assert!((off.master.bound - on.master.bound).abs() < 1e-8);
        assert!(
            on.master.pricing_calls < off.master.pricing_calls,
            "retained improving paths must avoid a fresh oracle call: on={}, off={}",
            on.master.pricing_calls,
            off.master.pricing_calls
        );
    }
}

#[test]
fn paired_column_values_match_registered_native_values_and_reject_stale_pool_metadata() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/coupled/metasolver-paired-preparation.json"
    )))
    .unwrap();
    let mut source = corpus["cases"][0]["model"].clone();
    for arc in source["networks"][0]["paired_front_programs"][0]["arcs"]
        .as_array_mut()
        .unwrap()
    {
        arc["dependent"]["function"]["pairs"][0][0] = serde_json::json!(1.);
        for pair in arc["dependent"]["function"]["pairs"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .skip(1)
        {
            pair[0] = serde_json::json!(0.);
        }
    }
    let problem = format::Model::from_json(&source.to_string())
        .unwrap()
        .problem::<2>()
        .unwrap();
    let mut options = Options::default();
    options.cg.stabilize = false;
    let output = solve(problem, options).unwrap();
    assert_eq!(output.master.lp, -18.);
    let oracle = &output.snapshot.state.oracles[0];
    let route = &output
        .routes
        .iter()
        .find(|r| r.route.arcs.len() == 2)
        .unwrap()
        .route;
    // Actual registered native columns map the same constant-increment pair
    // to dependent value2; the arc variables carry physical cost-20.
    assert_eq!(route.coupled_values, vec![(0, 2.)]);
    assert!(compatible_schedule(oracle, route));
    let domain = crate::master::routing::RoutingContext::set_partitioning(
        oracle.network.graph.as_ref().clone(),
        1.,
    )
    .unwrap();
    assert!(crate::master::routing::exact_route_feasible(&domain, route).is_some());
    let mut stale = route.clone();
    stale.coupled_values[0].1 = 3.;
    assert!(!compatible_schedule(oracle, &stale));
    assert_eq!(
        crate::master::routing::exact_route_feasibility(&domain, &stale),
        Some(true)
    );
    assert!(crate::master::routing::exact_route_feasible(&domain, &stale).is_none());
    stale.coupled_values.clear();
    assert!(!compatible_schedule(oracle, &stale));
    assert!(crate::master::routing::exact_route_feasible(&domain, &stale).is_none());
}
