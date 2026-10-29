use super::*;
use crate::master::clp::ClpMaster;

fn column(cost: f64, tag: i32, identity: &str) -> Column {
    Column {
        cost,
        reduced_cost: 0.,
        entries: vec![(0, 1.)],
        acceptable: true,
        tag,
        identity: identity.into(),
    }
}

fn exact() -> [Phase; 1] {
    [Phase {
        check_ng: true,
        jump_arcs: true,
        labels_per_bucket: i64::MAX,
    }]
}

#[test]
fn distinct_costs_with_identical_decimal_signatures_keep_the_improving_column() {
    let cost = 1_234_567_890_123.;
    let cheaper = cost - 1.;
    assert_eq!(format!("{cost:.11e}"), format!("{cheaper:.11e}"));
    for explicit in [false, true] {
        let mut master = ClpMaster::new().unwrap();
        master.objective_offset = -cost;
        master.add_row(RowSense::GreaterEqual, 1., &[]).unwrap();
        master.add_column(cost, &[(0, 1.)]).unwrap();
        let mut options = Options {
            stabilize: false,
            gap_absolute: 0.,
            gap_relative: 0.,
            insert_non_negative: false,
            ..Default::default()
        };
        options
            .known
            .push(column(cost, 0, if explicit { "initial" } else { "" }));
        let out = optimize(
            &mut master,
            &exact(),
            |_, duals| {
                let mut candidate = column(cheaper, 0, if explicit { "cheaper" } else { "" });
                candidate.reduced_cost = cheaper - duals[0];
                Ok(Answer {
                    minimum: candidate.reduced_cost,
                    columns: vec![candidate],
                    exact: true,
                    bound_minimum: None,
                })
            },
            &mut NoController,
            options,
        )
        .unwrap();
        assert_eq!(out.status, Status::Optimal);
        assert_eq!(out.lp, -1.);
        assert_eq!(out.columns, 2);
    }
}

#[test]
fn fallback_membership_preserves_nearby_coefficients_and_normalizes_only_zero() {
    let mut first = column(1., 0, "");
    first.entries = vec![(0, 1_234_567_890_123.)];
    let mut second = first.clone();
    second.entries[0].1 += 1.;
    assert_eq!(
        format!("{:.11e}", first.entries[0].1),
        format!("{:.11e}", second.entries[0].1)
    );
    let mut membership = HashSet::new();
    assert!(membership.insert(first.signature()));
    assert!(membership.insert(second.signature()));
    first.cost = -0.;
    second = first.clone();
    second.cost = 0.;
    second.entries.push((9, -0.));
    assert_eq!(first.signature(), second.signature());
}

#[test]
fn algebraically_equal_columns_with_distinct_quantity_caps_remain_available() {
    struct Caps;
    impl Controller<ClpMaster> for Caps {
        fn column_upper_bound(&self, tag: i32) -> Option<f64> {
            (tag >= 0).then_some(1.)
        }
    }
    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::GreaterEqual, 2., &[]).unwrap();
    master.add_column(1., &[(0, 1.)]).unwrap();
    let options = Options {
        stabilize: false,
        artificial_cost: 100.,
        known: vec![column(1., 0, "")],
        ..Default::default()
    };
    let out = optimize(
        &mut master,
        &exact(),
        |_, duals| {
            let mut next = column(1., 1, "");
            next.reduced_cost = 1. - duals[0];
            Ok(Answer {
                minimum: next.reduced_cost,
                columns: vec![next],
                exact: true,
                bound_minimum: None,
            })
        },
        &mut Caps,
        options,
    )
    .unwrap();
    assert_eq!(out.status, Status::Optimal);
    assert_eq!(out.lp, 2.);
    assert!(out.inserted_tags.contains(&1));
}

struct Reuse {
    allow: bool,
}
impl Controller<ClpMaster> for Reuse {
    fn uses_inactive_column_pool(&self) -> bool {
        true
    }
    fn reprice_inactive_column(&mut self, column: &Column, _: &[f64]) -> Result<Option<Column>> {
        Ok(self.allow.then(|| column.clone()))
    }
}

fn pooled_master() -> (ClpMaster, Options) {
    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::GreaterEqual, 1., &[]).unwrap();
    master.add_column(10., &[(0, 1.)]).unwrap();
    let mut warm = WarmStart::empty(BoundMode::Numerical);
    warm.initial_rows = 1;
    warm.columns.push(Some(column(10., 0, "initial")));
    warm.generated_pool.push(column(1., 1, "cached"));
    (
        master,
        Options {
            stabilize: false,
            warm_start: Some(warm),
            ..Default::default()
        },
    )
}

#[test]
fn inactive_pool_improves_the_master_before_the_next_oracle_call() {
    let (mut master, options) = pooled_master();
    let mut calls = 0;
    let out = optimize(
        &mut master,
        &exact(),
        |_, duals| {
            calls += 1;
            assert!(duals[0] <= 1. + 1e-7);
            Ok(Answer {
                minimum: 0.,
                columns: vec![],
                exact: true,
                bound_minimum: None,
            })
        },
        &mut Reuse { allow: true },
        options,
    )
    .unwrap();
    assert_eq!(out.status, Status::Optimal);
    assert_eq!(out.lp, 1.);
    assert_eq!(calls, 1);
    assert!(!out.history[0].exact);
    assert_eq!(out.history[0].pricing_calls, 0);
    let (mut master, mut options) = pooled_master();
    options.max_iterations = 1;
    let out = optimize(
        &mut master,
        &exact(),
        |_, _| panic!("pool pass must not invoke the oracle"),
        &mut Reuse { allow: true },
        options,
    )
    .unwrap();
    assert_eq!(out.status, Status::IterationLimit);
    assert_eq!(out.numerical_bound, f64::NEG_INFINITY);
    assert_eq!(out.certified_bound, f64::NEG_INFINITY);
}

#[test]
fn a_declined_cached_path_falls_back_to_the_oracle() {
    struct Decline {
        attempts: usize,
    }
    impl Controller<ClpMaster> for Decline {
        fn uses_inactive_column_pool(&self) -> bool {
            true
        }
        fn reprice_inactive_column(&mut self, _: &Column, _: &[f64]) -> Result<Option<Column>> {
            self.attempts += 1;
            Ok(None)
        }
    }
    let (mut master, options) = pooled_master();
    let mut calls = 0;
    let mut controller = Decline { attempts: 0 };
    let out = optimize(
        &mut master,
        &exact(),
        |_, duals| {
            calls += 1;
            if calls == 1 {
                assert_eq!(duals[0], 10.);
            }
            let mut candidate = column(1., 1, "cached");
            candidate.reduced_cost = 1. - duals[0];
            Ok(Answer {
                minimum: candidate.reduced_cost,
                columns: vec![candidate],
                exact: true,
                bound_minimum: None,
            })
        },
        &mut controller,
        options,
    )
    .unwrap();
    assert_eq!(out.status, Status::Optimal);
    assert_eq!(out.lp, 1.);
    assert!(controller.attempts > 0);
    assert!(calls > 0);
    // The first exact oracle bound can close the improved LP on its next
    // solve; that valid shortcut does not require a second oracle call.
    assert_eq!(out.pricing_calls, calls);
}

#[test]
fn disabled_stabilization_does_not_request_or_allocate_an_unused_subgradient() {
    struct NoGradient;
    impl Controller<ClpMaster> for NoGradient {
        fn pricing_subgradient(&self, _: &[f64]) -> Option<Vec<f64>> {
            panic!("stabilization is disabled")
        }
    }
    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::GreaterEqual, 1., &[]).unwrap();
    master.add_column(2., &[(0, 1.)]).unwrap();
    let out = optimize(
        &mut master,
        &exact(),
        |_, _| {
            Ok(Answer {
                minimum: 0.,
                columns: vec![],
                exact: true,
                bound_minimum: None,
            })
        },
        &mut NoGradient,
        Options {
            stabilize: false,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(out.status, Status::Optimal);
}

#[test]
fn columns_only_batch_without_improvement_retries_exact_before_closure() {
    use std::cell::Cell;
    use std::rc::Rc;

    struct Retry(Rc<Cell<bool>>);
    impl Controller<ClpMaster> for Retry {
        fn pricing_columns_only(&self) -> bool {
            !self.0.get()
        }
        fn retry_exact_pricing(&mut self) -> bool {
            self.0.set(true);
            true
        }
    }
    let retried = Rc::new(Cell::new(false));
    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::GreaterEqual, 1., &[]).unwrap();
    master.add_column(1., &[(0, 1.)]).unwrap();
    let options = Options {
        stabilize: false,
        known: vec![column(1., 0, "initial")],
        ..Default::default()
    };
    let flag = retried.clone();
    let out = optimize(
        &mut master,
        &exact(),
        move |_, _| {
            Ok(Answer {
                columns: vec![],
                minimum: 0.,
                exact: flag.get(),
                bound_minimum: None,
            })
        },
        &mut Retry(retried.clone()),
        options,
    )
    .unwrap();
    assert!(retried.get());
    assert_eq!(out.status, Status::Optimal);
    assert_eq!(out.lp, 1.);
}

#[test]
fn columns_only_improvement_cannot_publish_approximate_bound() {
    use std::cell::Cell;
    use std::rc::Rc;

    struct Discovery(Rc<Cell<usize>>);
    impl Controller<ClpMaster> for Discovery {
        fn pricing_columns_only(&self) -> bool {
            self.0.get() == 1
        }
        fn pricing_bound(&self, _: &ClpMaster, _: &[f64]) -> Result<Option<f64>> {
            Ok((self.0.get() == 1).then_some(1_000.))
        }
    }
    let calls = Rc::new(Cell::new(0));
    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::GreaterEqual, 1., &[]).unwrap();
    master.add_column(1., &[(0, 1.)]).unwrap();
    let options = Options {
        stabilize: false,
        known: vec![column(1., 0, "initial")],
        ..Default::default()
    };
    let counter = calls.clone();
    let out = optimize(
        &mut master,
        &exact(),
        move |_, duals| {
            counter.set(counter.get() + 1);
            if counter.get() == 1 {
                let mut improving = column(0., 1, "improving");
                improving.reduced_cost = -duals[0];
                Ok(Answer {
                    columns: vec![improving],
                    minimum: 1_000.,
                    exact: true,
                    bound_minimum: Some(1_000.),
                })
            } else {
                Ok(Answer {
                    columns: vec![],
                    minimum: 0.,
                    exact: true,
                    bound_minimum: None,
                })
            }
        },
        &mut Discovery(calls.clone()),
        options,
    )
    .unwrap();
    assert!(calls.get() >= 2);
    assert_eq!(out.status, Status::Optimal);
    assert_eq!(out.lp, 0.);
    assert!(out.numerical_bound <= 0.);
}
