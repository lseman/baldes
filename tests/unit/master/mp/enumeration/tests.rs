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
