use super::*;

fn point() -> (Output<2>, Options) {
    let problem = format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/networks/triangle.json"
    )))
    .unwrap()
    .problem::<2>()
    .unwrap();
    let mut options = Options::default();
    options.enumeration = None;
    options.gap_enumeration = false;
    options.reduced_cost_fixing = false;
    (solve(problem, options.clone()).unwrap(), options)
}

fn install(oracle: &mut Oracle<2>, shared: usize, c: &RowCoefficients) {
    if c.per_route != 0. {
        Arc::make_mut(&mut oracle.network.route_entries).push((shared, c.per_route));
    }
    for (entries, &v) in Arc::make_mut(&mut oracle.network.arc_entries)
        .iter_mut()
        .zip(&c.arcs)
    {
        if v != 0. {
            entries.push((shared, v));
        }
    }
}
fn full(oracle: &Oracle<2>, route: &Route, offset: usize, physical: usize) -> Result<f64> {
    Ok(entries(oracle, route, offset)?
        .into_iter()
        .find(|&(r, _)| r == physical)
        .map_or(0., |(_, v)| v))
}

#[test]
fn appended_plain_row_matches_full_replay_bits_and_errors() {
    let (root, _) = point();
    let offset = root.snapshot.state.shared_offset;
    let shared = root.snapshot.state.shared_rows.len();
    let physical = offset + shared;
    let choices = [
        0.,
        -0.,
        1.,
        -1.,
        1e16,
        -1e16,
        1e-300,
        -1e-300,
        f64::MAX,
        -f64::MAX,
    ];
    let mut rng = 19u64;
    for k in 0..256 {
        let mut state = root.snapshot.state.fork_state();
        let oracle = &mut state.oracles[0];
        let c = RowCoefficients {
            network: oracle.network.id,
            per_route: choices[k % choices.len()],
            arcs: oracle
                .network
                .graph
                .arcs
                .iter()
                .map(|_| {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    choices[(rng >> 32) as usize % choices.len()]
                })
                .collect(),
        };
        install(oracle, shared, &c);
        let mut route = root.routes[0].route.clone();
        route.arcs = (0..k % 33)
            .map(|i| oracle.network.graph.arcs[i % oracle.network.graph.arcs.len()].id)
            .collect();
        match (
            full(oracle, &route, offset, physical),
            appended_row_value(oracle, &route, offset, shared, physical, Some(&c)),
        ) {
            (Ok(a), Ok(b)) => assert_eq!(a.to_bits(), b.to_bits(), "case{k}"),
            (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string()),
            other => panic!("different result {other:?}"),
        }
        route.arcs = vec![i32::MAX];
        assert_eq!(
            full(oracle, &route, offset, physical)
                .unwrap_err()
                .to_string(),
            appended_row_value(oracle, &route, offset, shared, physical, Some(&c))
                .unwrap_err()
                .to_string()
        );
    }
    let oracle = &root.snapshot.state.oracles[0];
    assert_eq!(
        appended_row_value(
            oracle,
            &root.routes[0].route,
            offset,
            shared,
            physical,
            None
        )
        .unwrap()
        .to_bits(),
        0f64.to_bits()
    );
}

#[test]
fn rich_oracle_and_physical_index_mismatch_keep_full_replay() {
    let (root, _) = point();
    let mut state = root.snapshot.state.fork_state();
    let offset = state.shared_offset;
    let shared = state.shared_rows.len();
    let oracle = &mut state.oracles[0];
    let route = &root.routes[0].route;
    let c = RowCoefficients {
        network: oracle.network.id,
        arcs: vec![1.; oracle.network.graph.arcs.len()],
        per_route: 1.,
    };
    install(oracle, shared, &c);
    assert_eq!(
        appended_row_value(oracle, route, offset, shared, 0, Some(&c))
            .unwrap()
            .to_bits(),
        full(oracle, route, offset, 0).unwrap().to_bits()
    );
    oracle.activity_rows.push(AccumulatedActivityEntry {
        resource: usize::MAX,
        row: 0,
        coefficient: 1.,
    });
    let expected = full(oracle, route, offset, offset + shared)
        .unwrap_err()
        .to_string();
    assert_eq!(
        appended_row_value(oracle, route, offset, shared, offset + shared, Some(&c))
            .unwrap_err()
            .to_string(),
        expected
    );
}

#[test]
fn public_branch_row_preserves_fixed_rhs_metadata_and_parent() {
    let (root, options) = point();
    let parent = root.snapshot.fix_route(0, 1.).unwrap();
    let parent_rhs = parent.master.rhs.clone();
    let parent_costs = parent.master.costs.clone();
    let oracle = &parent.state.oracles[0];
    let c = RowCoefficients {
        network: oracle.network.id,
        arcs: vec![0.25; oracle.network.graph.arcs.len()],
        per_route: 0.5,
    };
    let physical = parent.master.rhs.len();
    let child = parent
        .with_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 100.,
            },
            &[c],
        )
        .unwrap();
    let expected_rhs = child
        .state
        .fixed_routes
        .iter()
        .try_fold(100., |rhs, f| {
            let o = child
                .state
                .oracles
                .iter()
                .find(|o| o.network.id == f.route.network)
                .unwrap();
            full(o, &f.route.route, child.state.shared_offset, physical)
                .map(|v| rhs - f.quantity * v)
        })
        .unwrap();
    assert_eq!(child.master.rhs[physical].to_bits(), expected_rhs.to_bits());
    for metadata in child
        .warm
        .columns
        .iter()
        .flatten()
        .chain(&child.warm.generated_pool)
    {
        if metadata.tag < 0 {
            continue;
        }
        let route = &child.state.routes[metadata.tag as usize];
        let o = child
            .state
            .oracles
            .iter()
            .find(|o| o.network.id == route.network)
            .unwrap();
        let expected = full(o, &route.route, child.state.shared_offset, physical).unwrap();
        let actual = metadata
            .entries
            .iter()
            .find(|&&(r, _)| r == physical)
            .map_or(0., |&(_, v)| v);
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    assert_eq!(parent.master.rhs, parent_rhs);
    assert_eq!(parent.master.costs, parent_costs);
    let solved = child.solve(options).unwrap();
    assert!(solved.master.lp.is_finite());
}
