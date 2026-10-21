//! Global numerical oracle feedback versus independent pattern-domain optima.
use super::imported_multiple_tests::{bounds, column, fixture};
use super::*;
use multi_pricing::{
    imported::{
        self, Column, ColumnBatch, ImportOptions, ImportOracleReply, ImportPricing,
        ImportPricingKind, ImportPricingMinimum, ImportTreeStatus,
    },
    AssignmentTreeOptions, MasterVariable,
};
use std::time::Duration;
fn configured(mode: BoundMode) -> AssignmentTreeOptions {
    let mut result = AssignmentTreeOptions {
        node: options(),
        ..Default::default()
    };
    result.node.cg.bound_mode = mode;
    result
}
fn empty(point: &ImportPricing<2>, minimum: f64) -> ImportOracleReply {
    ImportOracleReply {
        batch: ColumnBatch {
            columns: vec![],
            complete: false,
        },
        minimum: Some(point.numerical_minimum(minimum).unwrap()),
    }
}
#[test]
fn numerical_minima_do_not_become_certificates_or_enumeration_completion() {
    let root = fixture(false);
    let session = root.snapshot().unwrap().begin_imports(&[7, 42]).unwrap();
    let stopped = session
        .begin_tree()
        .run_with_oracles(
            &configured(BoundMode::Certified),
            &ImportOptions::default(),
            |point, _| Ok(empty(point, f64::INFINITY)),
        )
        .unwrap();
    assert_eq!(stopped.status, ImportTreeStatus::Incomplete);
    assert_eq!(stopped.bound_mode, BoundMode::Certified);
    assert_eq!(stopped.bound, f64::NEG_INFINITY);
    assert_eq!(stopped.incumbent, 101.);
    assert!(!session.is_complete());
    let solved = session
        .begin_tree()
        .run_with_oracles(
            &configured(BoundMode::Numerical),
            &ImportOptions::default(),
            |point, _| Ok(empty(point, f64::INFINITY)),
        )
        .unwrap();
    assert_eq!(solved.status, ImportTreeStatus::Optimal);
    assert_eq!(solved.incumbent, 101.);
    assert!(session.solve(options()).is_err());
}
#[test]
fn global_coverage_handles_negative_minima_missing_groups_and_unknown_values() {
    let root = fixture(false);
    let session = root.snapshot().unwrap().begin_imports(&[7, 42]).unwrap();
    let mixed = session
        .begin_tree()
        .run_with_oracles(
            &configured(BoundMode::Numerical),
            &ImportOptions::default(),
            |point, _| {
                Ok(empty(
                    point,
                    if point.subproblem().id == 7 {
                        f64::INFINITY
                    } else {
                        -100.
                    },
                ))
            },
        )
        .unwrap();
    // A future group-42 pattern at cost -99 can replace known cost one,
    // giving full optimum 100-99=1. Group 7's oracle alone cannot prune it.
    assert_eq!(mixed.status, ImportTreeStatus::Incomplete);
    assert_eq!(mixed.bound, 1.);
    assert_eq!(mixed.incumbent, 101.);
    for unknown in [false, true] {
        let result = session
            .begin_tree()
            .run_with_oracles(
                &configured(BoundMode::Numerical),
                &ImportOptions::default(),
                |point, _| {
                    if point.subproblem().id == 7 {
                        Ok(empty(point, f64::INFINITY))
                    } else if unknown {
                        Ok(empty(point, f64::NEG_INFINITY))
                    } else {
                        Ok(ImportOracleReply::columns(ColumnBatch {
                            columns: vec![],
                            complete: false,
                        }))
                    }
                },
            )
            .unwrap();
        assert_eq!(result.status, ImportTreeStatus::Incomplete);
        assert_eq!(result.bound, f64::NEG_INFINITY);
    }
}
#[test]
fn numerical_phase_one_requires_all_groups_and_recovers_into_objective_requests() {
    let root = fixture(true);
    let session = root.snapshot().unwrap().begin_imports(&[7, 42]).unwrap();
    let impossible = session
        .begin_tree()
        .run_with_oracles(
            &configured(BoundMode::Numerical),
            &ImportOptions::default(),
            |point, _| {
                assert_eq!(point.kind(), ImportPricingKind::Feasibility);
                Ok(empty(point, f64::INFINITY))
            },
        )
        .unwrap();
    assert_eq!(impossible.status, ImportTreeStatus::Infeasible);
    let incomplete = session
        .begin_tree()
        .run_with_oracles(
            &configured(BoundMode::Numerical),
            &ImportOptions::default(),
            |point, _| {
                if point.subproblem().id == 7 {
                    Ok(empty(point, f64::INFINITY))
                } else {
                    Ok(ImportOracleReply::columns(ColumnBatch {
                        columns: vec![],
                        complete: false,
                    }))
                }
            },
        )
        .unwrap();
    assert_eq!(incomplete.status, ImportTreeStatus::Incomplete);
    let mut kinds = vec![];
    let recovered = session
        .begin_tree()
        .run_with_oracles(
            &configured(BoundMode::Numerical),
            &ImportOptions::default(),
            |point, _| {
                kinds.push(point.kind());
                let c = if point.subproblem().id == 7 {
                    column(0, 2., 0)
                } else {
                    column(0, 1., 1)
                };
                let mut query = c.clone();
                query.id = 999;
                let minimum = point.numerical_minimum(point.reduced_cost(query)?).unwrap();
                Ok(ImportOracleReply {
                    batch: ColumnBatch {
                        columns: if point.subproblem().columns.is_empty() {
                            vec![c]
                        } else {
                            vec![]
                        },
                        complete: false,
                    },
                    minimum: Some(minimum),
                })
            },
        )
        .unwrap();
    assert_eq!(recovered.status, ImportTreeStatus::Optimal);
    assert_eq!(recovered.incumbent, 3.);
    assert_eq!(
        kinds,
        vec![
            ImportPricingKind::Feasibility,
            ImportPricingKind::Feasibility,
            ImportPricingKind::Objective,
            ImportPricingKind::Objective
        ]
    );
}
#[test]
fn minima_are_scoped_to_point_group_and_phase_and_failed_rounds_roll_back() {
    let root = fixture(false);
    let session = root.snapshot().unwrap().begin_imports(&[7, 42]).unwrap();
    let start = session.begin_tree();
    let mut previous: Option<ImportPricingMinimum> = None;
    let failed = start.run_with_oracles(
        &configured(BoundMode::Numerical),
        &ImportOptions::default(),
        |point, _| {
            if let Some(minimum) = previous.take() {
                return Ok(ImportOracleReply {
                    batch: ColumnBatch {
                        columns: vec![],
                        complete: false,
                    },
                    minimum: Some(minimum),
                });
            }
            previous = Some(point.numerical_minimum(-98.)?);
            Ok(ImportOracleReply {
                batch: ColumnBatch {
                    columns: vec![column(1, 2., 0)],
                    complete: false,
                },
                minimum: None,
            })
        },
    );
    assert!(failed.is_err());
    assert!(start.solution().is_none());
    assert_eq!(session.subproblem(7).unwrap().columns.len(), 1);
    let single = root.snapshot().unwrap().begin_imports(&[7]).unwrap();
    let mut old = None;
    let stale = single.begin_tree().run_with_oracles(
        &configured(BoundMode::Numerical),
        &ImportOptions::default(),
        |point, _| {
            if let Some(minimum) = old.take() {
                return Ok(ImportOracleReply {
                    batch: ColumnBatch {
                        columns: vec![],
                        complete: false,
                    },
                    minimum: Some(minimum),
                });
            }
            let c = column(1, 2., 0);
            let minimum = point.numerical_minimum(point.reduced_cost(c.clone())? - 2.)?;
            old = Some(minimum.clone());
            Ok(ImportOracleReply {
                batch: ColumnBatch {
                    columns: vec![c],
                    complete: false,
                },
                minimum: Some(minimum),
            })
        },
    );
    assert!(stale.is_err());
    let phase = single.feasibility_pricing(7, options()).unwrap();
    let wrong_phase = phase.numerical_minimum(0.).unwrap();
    assert!(single
        .begin_tree()
        .run_with_oracles(
            &configured(BoundMode::Numerical),
            &ImportOptions::default(),
            |_, _| {
                Ok(ImportOracleReply {
                    batch: ColumnBatch {
                        columns: vec![],
                        complete: false,
                    },
                    minimum: Some(wrong_phase.clone()),
                })
            }
        )
        .is_err());
    let restored = start
        .run_with_oracles(
            &configured(BoundMode::Numerical),
            &ImportOptions::default(),
            |point, _| {
                assert_eq!(point.subproblem().columns.len(), 1);
                Ok(empty(point, f64::INFINITY))
            },
        )
        .unwrap();
    assert_eq!(restored.incumbent, 101.);
}
#[test]
fn partial_rounds_resume_with_fresh_coverage_and_do_not_upgrade_bound_modes() {
    let root = fixture(false);
    let session = root.snapshot().unwrap().begin_imports(&[7, 42]).unwrap();
    let limits = ImportOptions {
        max_batches: 1,
        ..Default::default()
    };
    let mut seen = vec![];
    let first = session
        .begin_tree()
        .run_with_oracles(&configured(BoundMode::Numerical), &limits, |point, _| {
            seen.push(point.subproblem().id);
            Ok(empty(point, f64::INFINITY))
        })
        .unwrap();
    assert_eq!(seen, vec![7]);
    assert_eq!(first.status, ImportTreeStatus::BatchLimit);
    assert_eq!(first.bound, f64::NEG_INFINITY);
    let mut larger = limits.clone();
    larger.max_batches = 2;
    let done = first
        .tree
        .run_with_oracles(&configured(BoundMode::Numerical), &larger, |point, _| {
            seen.push(point.subproblem().id);
            Ok(empty(point, f64::INFINITY))
        })
        .unwrap();
    assert_eq!(seen, vec![7, 42, 7]);
    assert_eq!(done.status, ImportTreeStatus::Optimal);
    assert!(first
        .tree
        .run_with_oracles(
            &configured(BoundMode::Certified),
            &larger,
            |_, _| unreachable!()
        )
        .is_err());
    let mut no_time = configured(BoundMode::Numerical);
    no_time.time_limit = Duration::ZERO;
    assert_eq!(
        session
            .begin_tree()
            .run_with_oracles(&no_time, &larger, |_, _| unreachable!())
            .unwrap()
            .status,
        ImportTreeStatus::TimeLimit
    );
    let contradiction = session.begin_tree().run_with_oracles(
        &configured(BoundMode::Numerical),
        &larger,
        |point, _| {
            assert!(point.numerical_minimum(f64::NAN).is_err());
            Ok(ImportOracleReply {
                batch: ColumnBatch {
                    columns: vec![column(1, 2., 0)],
                    complete: false,
                },
                minimum: Some(point.numerical_minimum(0.)?),
            })
        },
    );
    assert!(contradiction.is_err());
}
#[test]
fn adverse_unlimited_quantities_keep_numerical_oracle_bounds_unknown() {
    let root = imported::solve(
        Problem::<7> {
            rows: vec![],
            networks: vec![],
        },
        vec![],
        vec![imported::Subproblem {
            id: 7,
            variables: vec![],
            multiplicity: bounds(0., f64::INFINITY),
            columns: vec![Column {
                id: 0,
                cost: -1.,
                bounds: bounds(0., 1.),
                integer: true,
                entries: vec![],
            }],
        }],
        options(),
    )
    .unwrap();
    let session = root.snapshot().unwrap().begin_imports(&[7]).unwrap();
    let result = session
        .begin_tree()
        .run_with_oracles(
            &configured(BoundMode::Numerical),
            &ImportOptions::default(),
            |point, _| {
                Ok(ImportOracleReply {
                    batch: ColumnBatch {
                        columns: vec![],
                        complete: false,
                    },
                    minimum: Some(point.numerical_minimum(-1.)?),
                })
            },
        )
        .unwrap();
    assert_eq!(result.status, ImportTreeStatus::Incomplete);
    assert_eq!(result.bound, f64::NEG_INFINITY);
    assert_eq!(result.incumbent, -1.);
}
