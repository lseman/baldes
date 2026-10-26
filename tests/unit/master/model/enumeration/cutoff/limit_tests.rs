use super::*;
use crate::master::solvers::mip::MipStatus;
#[test]
fn cutoff_mip_limits_and_invalid_requests_keep_the_view_retryable() {
    let problem = super::super::super::format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/networks/triangle.json"
    )))
    .unwrap()
    .problem::<7>()
    .unwrap();
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.cg.artificial_cost = 100.;
    let root = solve(problem, options).unwrap();
    let view = root
        .snapshot
        .enumerate_below(3., &EnumerationOptions::default())
        .unwrap();
    for (limits, status) in [
        (
            MipOptions {
                node_limit: 0,
                ..Default::default()
            },
            MipStatus::NodeLimit,
        ),
        (
            MipOptions {
                time_limit: Duration::ZERO,
                ..Default::default()
            },
            MipStatus::TimeLimit,
        ),
    ] {
        let result = view.solve_mip(&limits).unwrap();
        assert_eq!(result.mip.status, status);
        assert!(result.mip.solution.is_none());
        assert!(result.route_values.is_empty());
        assert!(result.mip.bound <= view.coverage_cutoff());
        assert_eq!(view.report().coverage_cutoff, Some(3.));
    }
    assert!(view
        .solve_mip(&MipOptions {
            objective_cutoff: f64::NAN,
            ..Default::default()
        })
        .is_err());
    let retry = view.solve_mip(&MipOptions::default()).unwrap();
    assert_eq!(retry.mip.status, MipStatus::Optimal);
    assert_eq!(retry.mip.solution.unwrap().objective, 2.);
    assert!(root.snapshot.state.universe.is_none());
}
