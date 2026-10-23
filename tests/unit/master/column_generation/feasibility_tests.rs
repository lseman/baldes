use super::*;
use crate::master::clp::ClpMaster;

fn solved() -> (ClpMaster, Output) {
    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::GreaterEqual, 1., &[]).unwrap();
    master.add_column(2., &[(0, 1.)]).unwrap();
    let output = optimize(
        &mut master,
        &[],
        |_, _| {
            Ok(Answer {
                columns: vec![],
                minimum: 0.,
                exact: true,
                bound_minimum: None,
            })
        },
        &mut NoController,
        Options {
            stabilize: false,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(output.status, Status::Optimal);
    (master, output)
}

fn recovery(status: FeasibilityStatus) -> FeasibilityRecovery {
    FeasibilityRecovery {
        columns: vec![Column {
            cost: 7.,
            reduced_cost: -1.,
            entries: vec![(0, 1.)],
            acceptable: true,
            tag: 42,
            identity: "recovered".into(),
        }],
        report: FeasibilityReport {
            status,
            iterations: 1,
            pricing_calls: 1,
            simplex_iterations: 0,
            ipm_iterations: 0,
            seconds: 0.,
            artificial_activity: Some(0.),
            lower_bound: None,
            certified_infeasible: false,
        },
    }
}

#[test]
fn failed_cap_installation_leaves_the_live_master_and_warm_metadata_unchanged() {
    struct InvalidCap;
    impl Controller<ClpMaster> for InvalidCap {
        fn column_upper_bound(&self, _: i32) -> Option<f64> {
            Some(-1.)
        }
    }
    let (mut master, mut out) = solved();
    let mut loaded = out.warm_start.columns.clone();
    let before = format!("{out:?}");
    let before_loaded = format!("{loaded:?}");
    let mut seen = HashSet::new();
    let mut pool = vec![];
    let mut pool_seen = HashSet::new();
    let result = install(
        &mut master,
        &InvalidCap,
        recovery(FeasibilityStatus::Feasible),
        &mut out,
        Columns {
            loaded: &mut loaded,
            seen: &mut seen,
            pool: &mut pool,
            pool_seen: &mut pool_seen,
        },
    );
    assert!(result.is_err());
    assert_eq!(master.costs, vec![2.]);
    assert!(master.solved);
    assert_eq!(master.values, vec![1.]);
    assert_eq!(format!("{out:?}"), before);
    assert_eq!(format!("{loaded:?}"), before_loaded);
    assert!(seen.is_empty() && pool.is_empty() && pool_seen.is_empty());
}

#[test]
fn numerical_feasibility_claim_cannot_close_a_certified_master() {
    let (mut master, mut out) = solved();
    out.bound_mode = BoundMode::Certified;
    let mut loaded = out.warm_start.columns.clone();
    let mut seen = HashSet::new();
    let mut pool = vec![];
    let mut pool_seen = HashSet::new();
    let before = format!("{out:?}");
    assert!(install(
        &mut master,
        &NoController,
        recovery(FeasibilityStatus::Infeasible),
        &mut out,
        Columns {
            loaded: &mut loaded,
            seen: &mut seen,
            pool: &mut pool,
            pool_seen: &mut pool_seen
        }
    )
    .is_err());
    assert_eq!(format!("{out:?}"), before);
    assert_eq!(master.costs, vec![2.]);
}
