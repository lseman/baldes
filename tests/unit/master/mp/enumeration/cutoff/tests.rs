use super::*;
use crate::master::solvers::mip::MipStatus;
fn root() -> Output<7> {
    let problem = super::super::super::format::Model::from_json(include_str!(concat!(
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
#[test]
fn all_network_cutoff_transfer_and_mip_keep_original_pricing_available() {
    let root = root();
    assert_eq!(root.master.lp, 1.5);
    let source_routes = root.snapshot.state.routes.len();
    let limits = EnumerationOptions {
        max_routes: 3,
        ..Default::default()
    };
    // The full nine-route domain exceeds this cap; only its three cheap
    // pair routes are needed below 1.75 at the common root dual point.
    let full = root.snapshot.enumerate(&limits).unwrap();
    assert!(!full.enumeration_report().complete);
    let cut = root.snapshot.enumerate_below(1.75, &limits).unwrap();
    assert_eq!(cut.coverage_cutoff(), 1.75);
    assert_eq!(cut.report().routes, 3);
    assert_eq!(cut.report().networks, [(7, 2), (42, 1)]);
    assert_eq!(cut.routes().count(), 3);
    let result = cut.solve_mip(&MipOptions::default()).unwrap();
    assert_eq!(result.coverage_cutoff, 1.75);
    assert_eq!(result.mip.status, MipStatus::ObjectiveCutoff);
    assert_eq!(result.mip.bound, 1.75);
    assert!(result.mip.solution.is_none());
    assert!(root.snapshot.state.universe.is_none());
    assert_eq!(root.snapshot.state.routes.len(), source_routes);
    let mut options = Options::default();
    options.cg.stabilize = false;
    let resumed = root.snapshot.fork().unwrap().solve(options).unwrap();
    assert_eq!(resumed.master.status, cg::Status::Optimal);
    assert_eq!(resumed.master.lp, 1.5);
    // A larger cutoff retains the global integer optimum and its witness.
    let cut = root
        .snapshot
        .enumerate_below(3., &EnumerationOptions::default())
        .unwrap();
    let result = cut.solve_mip(&MipOptions::default()).unwrap();
    assert_eq!(result.mip.status, MipStatus::Optimal);
    assert_eq!(result.mip.solution.as_ref().unwrap().objective, 2.);
    assert_eq!(
        result.route_values.iter().map(|r| r.quantity).sum::<f64>(),
        2.
    );
    assert!(result.mip.bound <= 2.);
    let stricter = cut
        .solve_mip(&MipOptions {
            objective_cutoff: 1.75,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(stricter.mip.status, MipStatus::ObjectiveCutoff);
    assert_eq!(stricter.mip.bound, 1.75);
}
#[test]
fn failed_cutoff_collection_fixed_copies_and_retry_preserve_source() {
    let root = root();
    let tag = root
        .snapshot
        .state
        .routes
        .iter()
        .position(|r| r.network == 7 && r.route.arcs == [0, 1])
        .unwrap();
    let child = root.snapshot.fix_route(tag, 1.).unwrap();
    for options in [
        EnumerationOptions {
            max_work: 0,
            ..Default::default()
        },
        EnumerationOptions {
            max_routes: 0,
            ..Default::default()
        },
        EnumerationOptions {
            time_limit: Duration::ZERO,
            ..Default::default()
        },
    ] {
        assert!(child.enumerate_below(3., &options).is_err());
        assert!(child.state.universe.is_none());
    }
    let cut = child
        .enumerate_below(3., &EnumerationOptions::default())
        .unwrap();
    let mip = cut.solve_mip(&MipOptions::default()).unwrap();
    assert_eq!(mip.mip.status, MipStatus::Optimal);
    assert_eq!(mip.mip.solution.unwrap().objective, 2.);
    assert!(mip
        .route_values
        .iter()
        .any(|r| r.route.network == 7 && r.route.route.arcs == [0, 1] && r.quantity >= 1.));
    assert!(child.state.universe.is_none());
    assert!(root.snapshot.state.fixed_routes.is_empty());
    assert!(root
        .snapshot
        .enumerate_below(f64::NAN, &EnumerationOptions::default())
        .is_err());
}
