#![cfg(feature = "clp")]
use baldes::inputs::{Instance, Rounding};
use baldes::master::{cg::Status, model::Options as RootOptions};

fn r101_25() -> Instance {
    let mut instance = Instance::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/instances/Solomon/R101.txt"
    ))
    .unwrap();
    instance.truncate(25).unwrap();
    instance
}

#[test]
fn column_cleanup_shrinks_the_master_and_keeps_the_root_bound() {
    let plain = RootOptions::default();
    let reference = r101_25().solve(8, Rounding::None, plain.clone()).unwrap();
    assert_eq!(reference.master.status, Status::Optimal);

    let mut options = plain;
    options.cg.column_cleanup_threshold = 60.;
    let result = r101_25().solve(8, Rounding::None, options).unwrap();
    assert_eq!(result.master.status, Status::Optimal);
    assert!((result.master.lp - reference.master.lp).abs() < 1e-6);
    assert!(result.master.columns < reference.master.columns);
}

#[test]
fn bound_closed_pass_skips_cleanup_and_preserves_the_solution() {
    use baldes::master::{
        cg::{self, Answer, Column, Controller, LpMaster, Options},
        clp::ClpMaster,
        RowSense,
    };
    use baldes::pricing::phases::Phase;
    struct RedundantCuts {
        observed_rows: Vec<usize>,
        add_second_round: bool,
    }
    impl Controller<ClpMaster> for RedundantCuts {
        fn separates(&self) -> bool {
            true
        }
        fn separate(&mut self, master: &mut ClpMaster, _: &[i32]) -> baldes::Result<usize> {
            self.observed_rows.push(master.rhs().len());
            if self.observed_rows.len() == 1 {
                master.add_row(RowSense::LessEqual, 2., &[(0, 1.)])?;
                master.add_row(RowSense::LessEqual, 3., &[(0, 1.)])?;
                Ok(2)
            } else if self.observed_rows.len() == 2 && self.add_second_round {
                master.add_row(RowSense::LessEqual, 4., &[(0, 1.)])?;
                Ok(1)
            } else {
                Ok(0)
            }
        }
    }
    for add_second_round in [false, true] {
        let mut master = ClpMaster::new().unwrap();
        master.add_row(RowSense::Equal, 1., &[]).unwrap();
        master.add_column(1., &[(0, 1.)]).unwrap();
        let mut controller = RedundantCuts {
            observed_rows: vec![],
            add_second_round,
        };
        let result = cg::optimize(
            &mut master,
            &[Phase {
                check_ng: true,
                jump_arcs: true,
                labels_per_bucket: i64::MAX,
            }],
            |_, _| {
                Ok(Answer {
                    columns: vec![],
                    minimum: 0.,
                    exact: true,
                    bound_minimum: None,
                })
            },
            &mut controller,
            Options {
                stabilize: false,
                remove_basic_cuts: true,
                known: vec![Column {
                    cost: 1.,
                    reduced_cost: 0.,
                    entries: vec![(0, 1.)],
                    acceptable: true,
                    tag: 0,
                    identity: "unit-column".into(),
                }],
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(result.status, Status::Optimal);
        assert_eq!(result.pricing_calls, 1);
        assert_eq!(controller.observed_rows[0..2], [1, 3]);
        if add_second_round {
            assert!(controller.observed_rows.len() >= 3);
        }
        assert_eq!(master.rhs().len(), if add_second_round { 4 } else { 3 });
        assert_eq!(master.status(), 0);
        assert_eq!(master.objective(), 1.);
        assert_eq!(master.values()[0], 1.);
        assert_eq!(result.numerical_bound, 1.);
        assert_eq!(result.bound, 1.);
    }
}

#[test]
fn controller_caps_cover_initial_and_newly_priced_columns() {
    use baldes::master::{
        cg::{self, Answer, Column, Controller, LpMaster, Options},
        clp::ClpMaster,
        RowSense,
    };
    use baldes::pricing::phases::Phase;
    struct Bounded;
    impl Controller<ClpMaster> for Bounded {
        fn column_upper_bound(&self, tag: i32) -> Option<f64> {
            (tag == 0).then_some(0.25)
        }
        fn pricing_bound(&self, _: &ClpMaster, duals: &[f64]) -> baldes::Result<Option<f64>> {
            // Exact box dual for 0<=cheap<=1/4 and 0<=expensive<=1.
            Ok(Some(
                duals[0] + (-duals[0]).min(0.) * 0.25 + (10. - duals[0]).min(0.),
            ))
        }
    }
    for initially_loaded in [false, true] {
        let mut master = ClpMaster::new().unwrap();
        master.add_row(RowSense::Equal, 1., &[]).unwrap();
        let cheap = Column {
            cost: 0.,
            reduced_cost: 0.,
            entries: vec![(0, 1.)],
            acceptable: true,
            tag: 0,
            identity: "cheap".into(),
        };
        let expensive = Column {
            cost: 10.,
            tag: 1,
            identity: "expensive".into(),
            ..cheap.clone()
        };
        let known = if initially_loaded {
            vec![cheap.clone(), expensive]
        } else {
            vec![expensive]
        };
        for c in &known {
            master.add_column(c.cost, &c.entries).unwrap();
        }
        let result = cg::optimize(
            &mut master,
            &[Phase {
                check_ng: true,
                jump_arcs: true,
                labels_per_bucket: i64::MAX,
            }],
            |_, duals| {
                let mut c = cheap.clone();
                c.reduced_cost = -duals[0];
                Ok(Answer {
                    columns: vec![c],
                    minimum: -duals[0],
                    exact: true,
                    bound_minimum: None,
                })
            },
            &mut Bounded,
            Options {
                known,
                stabilize: false,
                artificial_cost: 100.,
                insert_non_negative: false,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(result.status, Status::Optimal);
        assert_eq!(result.lp, 7.5);
        assert_eq!(result.bound, 7.5);
        let column = result.column_tags.iter().position(|&tag| tag == 0).unwrap();
        assert_eq!(master.upper()[column], 0.25);
        assert_eq!(master.values()[column], 0.25);
    }
}

#[test]
fn enumeration_checkpoint_is_resumable_and_never_requests_with_positive_artificials() {
    use baldes::master::{
        cg::{self, Answer, Column, Controller, LpMaster, Options},
        clp::ClpMaster,
        RowSense,
    };
    use baldes::pricing::phases::Phase;
    struct Checkpoint(usize);
    impl Controller<ClpMaster> for Checkpoint {
        fn request_enumeration_checkpoint(&mut self, _: &ClpMaster) -> baldes::Result<bool> {
            self.0 += 1;
            Ok(true)
        }
    }
    let phases = [Phase {
        check_ng: true,
        jump_arcs: true,
        labels_per_bucket: i64::MAX,
    }];
    let price = |_, _: &[f64]| {
        Ok(Answer {
            columns: vec![],
            minimum: 0.,
            exact: true,
            bound_minimum: None,
        })
    };
    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::Equal, 1., &[]).unwrap();
    master.add_column(3., &[(0, 1.)]).unwrap();
    let mut controller = Checkpoint(0);
    let checkpoint = cg::optimize(
        &mut master,
        &phases,
        price,
        &mut controller,
        Options {
            stabilize: false,
            known: vec![Column {
                cost: 3.,
                reduced_cost: 0.,
                entries: vec![(0, 1.)],
                acceptable: true,
                tag: 42,
                identity: "checkpoint-route".into(),
            }],
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(controller.0, 1);
    assert!(checkpoint.enumeration_requested);
    assert_eq!(checkpoint.status, Status::IterationLimit);
    assert_eq!(checkpoint.warm_start.columns.len(), master.costs().len());
    assert_eq!(checkpoint.warm_start.columns[0].as_ref().unwrap().tag, 42);
    assert!(checkpoint
        .warm_start
        .generated_pool
        .iter()
        .any(|c| c.tag == 42));
    let resumed = cg::optimize(
        &mut master,
        &phases,
        price,
        &mut cg::NoController,
        Options {
            stabilize: false,
            warm_start: Some(checkpoint.warm_start),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!resumed.enumeration_requested);
    assert_eq!(resumed.status, Status::Optimal);
    assert_eq!(resumed.lp, 3.);
    assert_eq!(master.values()[0], 1.);

    let mut infeasible = ClpMaster::new().unwrap();
    infeasible.add_row(RowSense::Equal, 1., &[]).unwrap();
    let mut controller = Checkpoint(0);
    let output = cg::optimize(
        &mut infeasible,
        &phases,
        price,
        &mut controller,
        Options {
            stabilize: false,
            artificial_cost: 100.,
            max_penalty_updates: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(output.status, Status::Stalled);
    assert!(!output.enumeration_requested);
    assert_eq!(controller.0, 0);
}
