use super::*;
use crate::master::cg::LpMaster;
use crate::master::solvers::mip::MipStatus;
#[test]
fn backend_failure_after_private_insertion_preserves_checkpoint_and_retry() {
    let problem = super::super::format::Model::from_json(include_str!(
        "../../../../../examples/networks/triangle.json"
    ))
    .unwrap()
    .problem::<2>()
    .unwrap();
    let root = solve(problem, Options::default()).unwrap();
    let mut node = root.snapshot.fork().unwrap();
    // Retire real columns while retaining their route registry and pool.
    // Transfer must then perform actual backend insertions, not only reuse
    // a tiny root that happened to generate its entire universe already.
    let removed: Vec<_> = node
        .warm
        .columns
        .iter()
        .enumerate()
        .filter_map(|(index, column)| column.as_ref().map(|_| index))
        .collect();
    let (_, map) = node.master.remove(&[], &removed).unwrap();
    node.warm.columns = node
        .warm
        .columns
        .into_iter()
        .zip(&map)
        .filter_map(|(column, retained)| retained.map(|_| column))
        .collect();
    for column in &mut node.warm.artificials {
        *column = map[*column].unwrap();
    }
    for (_, column) in &mut node.warm.cut_artificials {
        *column = map[*column].unwrap();
    }
    let columns = node.master.costs().to_vec();
    let route_count = node.state.routes.len();
    let tags = node.state.tags.clone();
    let caps = node.state.route_upper_bounds.clone();
    let mut configured = cg::Options {
        warm_start: Some(node.warm.clone()),
        ..Default::default()
    };
    let mut inserted = 0;
    let failed = transfer_with(
        &mut node.master,
        &mut node.state,
        &mut configured,
        &EnumerationOptions::default(),
        |destination, column| {
            inserted += 1;
            if inserted == 2 {
                return Err(invalid("injected backend insertion failure"));
            }
            destination.add_column(column.cost, &column.entries)?;
            Ok(())
        },
        Instant::now,
    );
    assert!(failed.is_err());
    assert_eq!(inserted, 2);
    assert_eq!(node.master.costs(), columns);
    assert_eq!(node.state.routes.len(), route_count);
    assert_eq!(node.state.tags, tags);
    assert_eq!(node.state.route_upper_bounds, caps);
    assert!(node.state.universe.is_none());
    assert_eq!(
        configured.warm_start.as_ref().unwrap().columns.len(),
        columns.len()
    );
    // Advance the handoff's clock only after one actual private insertion.
    // This exercises a mid-transfer timeout without sleeps or load-sensitive
    // assumptions about how long the tiny graph takes to enumerate.
    let inserted = std::cell::Cell::new(0);
    let started = Instant::now();
    transfer_with(
        &mut node.master,
        &mut node.state,
        &mut configured,
        &EnumerationOptions {
            time_limit: std::time::Duration::from_secs(30),
            ..Default::default()
        },
        |destination, column| {
            destination.add_column(column.cost, &column.entries)?;
            inserted.set(inserted.get() + 1);
            Ok(())
        },
        || {
            if inserted.get() == 0 {
                started
            } else {
                started + std::time::Duration::from_secs(60)
            }
        },
    )
    .unwrap();
    assert_eq!(inserted.get(), 1);
    assert_eq!(
        node.state.enumeration_report.failure.as_deref(),
        Some("TimeLimit")
    );
    assert_eq!(node.master.costs(), columns);
    assert_eq!(node.state.routes.len(), route_count);
    assert_eq!(node.state.tags, tags);
    assert_eq!(node.state.route_upper_bounds, caps);
    assert!(node.state.universe.is_none());
    assert_eq!(
        configured.warm_start.as_ref().unwrap().columns.len(),
        columns.len()
    );
    let resumed = node.solve(Options::default()).unwrap();
    assert_eq!(resumed.master.status, cg::Status::Optimal);
    assert!((resumed.master.lp - 1.5).abs() < 1e-8);
    let retried = node.enumerate(&EnumerationOptions::default()).unwrap();
    assert!(retried.enumeration_report().complete);
    assert_eq!(retried.enumeration_report().attempts, 3);
    assert!(!root.snapshot.enumeration_report().complete);
}
#[test]
fn transfer_preserves_loaded_column_upper_bounds_and_artificials_stay_zero() {
    let problem = super::super::format::Model::from_json(include_str!(
        "../../../../../examples/networks/triangle.json"
    ))
    .unwrap()
    .problem::<2>()
    .unwrap();
    let root = solve(problem, Options::default()).unwrap();
    let mut node = root
        .snapshot
        .enumerate(&EnumerationOptions::default())
        .unwrap();
    for (column, metadata) in node.warm.columns.iter().enumerate() {
        if metadata.is_some() {
            node.master.set_upper(column, 0.).unwrap();
        }
    }
    // A duplicate transfer cannot re-add a bounded route as a fresh column.
    let repeated = node.enumerate(&EnumerationOptions::default()).unwrap();
    assert_eq!(repeated.master.costs.len(), node.master.costs.len());
    let mip = repeated
        .solve_enumerated_mip(&MipOptions::default())
        .unwrap()
        .unwrap();
    assert_eq!(mip.mip.status, MipStatus::Infeasible);
    assert!(mip.mip.solution.is_none());
    assert!(root.snapshot.solve(Options::default()).unwrap().master.lp < 2.);
}

#[cfg(feature = "highs")]
#[test]
fn end_of_node_mip_route_limit_is_raised_to_the_normal_limit() {
    let problem = || {
        super::super::format::Model::from_json(include_str!(
            "../../../../../examples/networks/triangle.json"
        ))
        .unwrap()
        .problem::<2>()
        .unwrap()
    };
    let limit = |pairs: &[(&str, f64)]| {
        let mut options = Options::default();
        options.pricing_options = pairs.iter().map(|&(k, v)| (k.into(), v)).collect();
        solve(problem(), options)
            .unwrap()
            .snapshot
            .end_of_node_mip_route_limit()
            .unwrap()
    };
    // Native postTreatment: max(normal, end-of-node).
    assert_eq!(limit(&[]), 10_000);
    assert_eq!(limit(&[("BALDESmaxNumOfEnumSolutionsForMIP", 5.)]), 5);
    assert_eq!(
        limit(&[
            ("BALDESmaxNumOfEnumSolutionsForMIP", 5.),
            ("BALDESmaxNumOfEnumSolsForEndOfNodeMIP", 20.),
        ]),
        20
    );
}

#[cfg(feature = "highs")]
#[test]
fn enumerated_mip_returns_repaired_theta_in_both_witness_encodings() {
    struct ConstantRecourse;
    impl super::super::separation::RouteRecourse for ConstantRecourse {
        fn excess(&self, _customers: &[usize]) -> crate::Result<f64> {
            Ok(0.3)
        }
    }
    let mut graph = crate::Graph::<2>::new(3, 0).unwrap();
    graph.add_arc(0, 1, 1.).unwrap();
    graph.add_arc(1, 2, 1.).unwrap();
    let problem = Problem::set_partitioning(graph, 1.).unwrap();
    let mut options = Options::default();
    options.cg.time_limit = std::time::Duration::ZERO;
    options.separation.route_recourse = Some(super::super::separation::Recourse(
        std::sync::Arc::new(ConstantRecourse),
    ));
    // No LP/cut work: the finite MIP initially pays zero Theta. Original-model
    // replay must charge recourse and publish that repaired point everywhere.
    let root = solve(problem, options).unwrap();
    let complete = root
        .snapshot
        .enumerate(&EnumerationOptions::default())
        .unwrap();
    let output = complete
        .solve_enumerated_mip(&MipOptions::default())
        .unwrap()
        .unwrap();
    let solution = output.mip.solution.as_ref().unwrap();
    assert_eq!(output.mip.status, MipStatus::Incomplete);
    assert_eq!(output.mip.bound, f64::NEG_INFINITY);
    assert!(output.master_values[0].1 >= 0.3);
    let mut reconstructed = vec![0.; output.master_values.len()];
    for (column, metadata) in complete.warm.columns.iter().enumerate() {
        if let Some((index, sign)) = metadata
            .as_ref()
            .and_then(|metadata| super::super::master_variables::decode(metadata.tag))
        {
            reconstructed[index] += sign * solution.values[column];
        }
    }
    assert_eq!(
        reconstructed,
        output
            .master_values
            .iter()
            .map(|&(_, value)| value)
            .collect::<Vec<_>>()
    );
    assert!((solution.objective - (2. + reconstructed[0])).abs() < 1e-12);
}
