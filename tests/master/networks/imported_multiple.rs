use super::*;
use baldes::master::preprocessing::Bounds;
use multi_pricing::{
    imported::{
        self, Column, ColumnBatch, ColumnEnvelope, ImportOptions, ImportPricingKind, ImportStatus,
    },
    MasterVariable,
};
pub(super) fn bounds(lower: f64, upper: f64) -> Bounds {
    Bounds { lower, upper }
}
pub(super) fn column(id: i32, cost: f64, coordinate: i32) -> Column {
    Column {
        id,
        cost,
        bounds: bounds(0., f64::INFINITY),
        integer: true,
        entries: vec![(coordinate, 1.)],
    }
}
pub(super) fn envelope(cost: f64, coordinate: i32, value: f64) -> ColumnEnvelope {
    ColumnEnvelope {
        cost_lower: cost,
        coordinates: vec![(coordinate, bounds(value, value))],
    }
}
pub(super) fn fixture(empty: bool) -> imported::ImportedOutput<2> {
    imported::solve(
        Problem::<2> {
            rows: vec![
                Row {
                    sense: RowSense::GreaterEqual,
                    rhs: 1.,
                },
                Row {
                    sense: RowSense::GreaterEqual,
                    rhs: 1.,
                },
                Row {
                    sense: RowSense::Equal,
                    rhs: 0.,
                },
            ],
            networks: vec![],
        },
        vec![
            MasterVariable {
                id: 0,
                cost: 0.,
                bounds: bounds(0., f64::INFINITY),
                integer: false,
                entries: vec![(0, 1.), (2, 1.)],
            },
            MasterVariable {
                id: 1,
                cost: 0.,
                bounds: bounds(0., f64::INFINITY),
                integer: false,
                entries: vec![(1, 1.), (2, -1.)],
            },
        ],
        vec![
            imported::Subproblem {
                id: 7,
                variables: vec![0],
                multiplicity: bounds(0., 1.),
                columns: if empty {
                    vec![]
                } else {
                    vec![column(0, 100., 0)]
                },
            },
            imported::Subproblem {
                id: 42,
                variables: vec![1],
                multiplicity: bounds(0., 1.),
                columns: if empty {
                    vec![]
                } else {
                    vec![column(0, 1., 1)]
                },
            },
        ],
        options(),
    )
    .unwrap()
}
#[test]
fn simultaneous_import_certificates_cover_every_unfinished_group() {
    let root = fixture(false);
    optimal(&root.output, 101.);
    let parent = root.snapshot().unwrap();
    let single = parent
        .begin_import(7)
        .unwrap()
        .with_envelope(envelope(2., 0, 1.))
        .unwrap();
    assert_eq!(
        single
            .objective_certificate(options())
            .unwrap()
            .unwrap()
            .lower_bound(),
        3.
    );
    let both = parent
        .begin_imports(&[42, 7])
        .unwrap()
        .with_envelope(7, envelope(2., 0, 1.))
        .unwrap();
    assert!(both.objective_certificate(options()).unwrap().is_none());
    let both = both.with_envelope(42, envelope(-100., 1, 1.)).unwrap();
    let certificate = both.objective_certificate(options()).unwrap().unwrap();
    assert_eq!(certificate.open_groups(), &[7, 42]);
    assert_eq!(certificate.lower_bound(), -98.);
    assert!(both.solve(options()).is_err());
    let step = both
        .collect(
            &ImportOptions {
                max_batches: 1,
                ..Default::default()
            },
            |group, _| {
                assert_eq!(group.id, 42);
                Ok(ColumnBatch {
                    columns: vec![],
                    complete: true,
                })
            },
        )
        .unwrap();
    assert_eq!(step.status, ImportStatus::BatchLimit);
    assert_eq!(step.session.pending_groups(), vec![7]);
    // Group 42's coarse envelope is no longer a future-column obligation.
    let certificate = step
        .session
        .objective_certificate(options())
        .unwrap()
        .unwrap();
    assert_eq!(certificate.open_groups(), &[7]);
    assert_eq!(certificate.lower_bound(), 3.);
    let improved = step
        .session
        .collect(
            &ImportOptions {
                max_batches: 1,
                ..Default::default()
            },
            |group, _| {
                assert_eq!(group.id, 7);
                Ok(ColumnBatch {
                    columns: vec![column(1, 2., 0)],
                    complete: false,
                })
            },
        )
        .unwrap();
    assert_eq!(improved.session.subproblem(7).unwrap().columns.len(), 2);
    assert_eq!(
        improved
            .session
            .objective_certificate(options())
            .unwrap()
            .unwrap()
            .lower_bound(),
        3.
    );
    let done = improved
        .session
        .collect(&ImportOptions::default(), |group, _| {
            assert_eq!(group.id, 7);
            Ok(ColumnBatch {
                columns: vec![],
                complete: true,
            })
        })
        .unwrap();
    assert_eq!(done.status, ImportStatus::Complete);
    assert!(done.session.is_complete());
    let certificate = done
        .session
        .objective_certificate(options())
        .unwrap()
        .unwrap();
    assert!(certificate.open_groups().is_empty());
    assert_eq!(certificate.lower_bound(), 3.);
    optimal(&done.session.solve(options()).unwrap().output, 3.);
    assert_eq!(both.pending_groups(), vec![7, 42]);
    assert_eq!(both.subproblem(7).unwrap().columns.len(), 1);
    optimal(&parent.solve(options()).unwrap().output, 101.);
}
#[test]
fn simultaneous_priced_collection_switches_shared_phases_and_resumes_round_robin() {
    let root = fixture(true);
    let session = root.snapshot().unwrap().begin_imports(&[7, 42]).unwrap();
    let mut seen = vec![];
    let mut source = |point: &imported::ImportPricing<2>, remaining: std::time::Duration| {
        assert!(!remaining.is_zero());
        seen.push((point.subproblem().id, point.kind()));
        let group = point.subproblem();
        if group.id == 7 && group.columns.is_empty() {
            assert_eq!(point.kind(), ImportPricingKind::Feasibility);
            Ok(ColumnBatch {
                columns: vec![column(0, 100., 0)],
                complete: false,
            })
        } else if group.id == 42 {
            assert_eq!(point.kind(), ImportPricingKind::Feasibility);
            Ok(ColumnBatch {
                columns: vec![column(0, 1., 1)],
                complete: true,
            })
        } else {
            assert_eq!(group.columns.len(), 1);
            assert_eq!(point.kind(), ImportPricingKind::Objective);
            assert!(point.reduced_cost(column(1, 2., 0)).unwrap() < 0.);
            Ok(ColumnBatch {
                columns: vec![column(1, 2., 0)],
                complete: true,
            })
        }
    };
    let limits = ImportOptions {
        max_batches: 1,
        ..Default::default()
    };
    let first = session
        .collect_priced(&limits, options(), &mut source)
        .unwrap();
    assert_eq!(first.session.pending_groups(), vec![7, 42]);
    let second = first
        .session
        .collect_priced(&limits, options(), &mut source)
        .unwrap();
    assert_eq!(second.session.pending_groups(), vec![7]);
    let done = second
        .session
        .collect_priced(&ImportOptions::default(), options(), &mut source)
        .unwrap();
    drop(source);
    assert_eq!(
        seen,
        vec![
            (7, ImportPricingKind::Feasibility),
            (42, ImportPricingKind::Feasibility),
            (7, ImportPricingKind::Objective)
        ]
    );
    assert_eq!(done.status, ImportStatus::Complete);
    optimal(&done.session.solve(options()).unwrap().output, 3.);
    assert!(session.subproblem(7).unwrap().columns.is_empty());
    assert!(session.subproblem(42).unwrap().columns.is_empty());
}
#[test]
fn simultaneous_import_limits_validation_and_rollback_preserve_parent() {
    let root = fixture(false);
    let parent = root.snapshot().unwrap();
    assert!(parent.begin_imports(&[7, 7]).is_err());
    assert!(parent.begin_imports(&[999]).is_err());
    let empty = parent.begin_imports(&[]).unwrap();
    assert!(empty.is_complete());
    let no_op = empty
        .collect_priced(
            &ImportOptions {
                max_batches: 0,
                time_limit: std::time::Duration::ZERO,
            },
            options(),
            |_, _| panic!("completed callback"),
        )
        .unwrap();
    assert_eq!(no_op.status, ImportStatus::Complete);
    optimal(&no_op.session.solve(options()).unwrap().output, 101.);
    let session = parent.begin_imports(&[7, 42]).unwrap();
    assert!(session.with_envelope(999, envelope(0., 0, 1.)).is_err());
    let zero = session
        .collect(
            &ImportOptions {
                max_batches: 2,
                time_limit: std::time::Duration::ZERO,
            },
            |_, _| panic!("zero-time callback"),
        )
        .unwrap();
    assert_eq!(zero.status, ImportStatus::TimeLimit);
    assert_eq!(zero.batches, 0);
    let zero = session
        .collect_priced(
            &ImportOptions {
                max_batches: 0,
                ..Default::default()
            },
            options(),
            |_, _| panic!("zero-batch callback"),
        )
        .unwrap();
    assert_eq!(zero.status, ImportStatus::BatchLimit);
    let mut stopped = options();
    stopped.cg.time_limit = std::time::Duration::ZERO;
    assert!(session
        .collect_priced(&ImportOptions::default(), stopped, |_, _| panic!(
            "stopped pricing callback"
        ))
        .is_err());
    let mut calls = 0;
    assert!(session
        .collect(&ImportOptions::default(), |group, _| {
            calls += 1;
            Ok(ColumnBatch {
                columns: vec![column(
                    if group.id == 7 { 1 } else { 0 },
                    2.,
                    if group.id == 7 { 0 } else { 1 },
                )],
                complete: false,
            })
        })
        .is_err());
    assert_eq!(calls, 2);
    assert_eq!(session.subproblem(7).unwrap().columns.len(), 1);
    let mut order = vec![];
    let fair = session
        .collect(
            &ImportOptions {
                max_batches: 3,
                ..Default::default()
            },
            |group, _| {
                order.push(group.id);
                Ok(ColumnBatch {
                    columns: vec![],
                    complete: group.id == 42,
                })
            },
        )
        .unwrap();
    assert_eq!(order, vec![7, 42, 7]);
    assert_eq!(fair.status, ImportStatus::BatchLimit);
    assert_eq!(fair.session.pending_groups(), vec![7]);
    assert!(fair.session.solve(options()).is_err());
    optimal(&parent.solve(options()).unwrap().output, 101.);
}
#[test]
fn simultaneous_phase_one_certificates_do_not_omit_an_open_domain() {
    let root = fixture(true);
    let base = root.snapshot().unwrap().begin_imports(&[7, 42]).unwrap();
    let partial = base.with_envelope(7, envelope(0., 0, 1.)).unwrap();
    assert!(partial
        .feasibility_certificate(options())
        .unwrap()
        .is_none());
    assert!(partial
        .infeasibility_certificate(options())
        .unwrap()
        .is_none());
    let possible = partial.with_envelope(42, envelope(0., 1, 1.)).unwrap();
    assert!(!possible
        .feasibility_certificate(options())
        .unwrap()
        .unwrap()
        .proves_infeasible());
    assert!(possible
        .infeasibility_certificate(options())
        .unwrap()
        .is_none());
    let impossible = base
        .with_envelope(7, envelope(0., 0, 0.))
        .unwrap()
        .with_envelope(42, envelope(0., 1, 0.))
        .unwrap();
    let certificate = impossible
        .infeasibility_certificate(options())
        .unwrap()
        .unwrap();
    assert_eq!(certificate.open_groups(), &[7, 42]);
    assert!(certificate.proves_infeasible());
    assert!(!impossible.is_complete());
}
