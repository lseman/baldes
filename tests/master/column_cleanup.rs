#![cfg(feature = "clp")]
use baldes::master::{column_generation::Status, root::RootOptions};
use baldes::vrptw::{Instance, Rounding};

fn r101_25() -> Instance {
    let mut instance =
        Instance::read(concat!(env!("CARGO_MANIFEST_DIR"), "/instances/R101.txt")).unwrap();
    instance.truncate(25).unwrap();
    instance
}

#[test]
fn column_cleanup_shrinks_the_master_and_keeps_the_root_bound() {
    let mut plain = RootOptions::default();
    plain.enumerate = false;
    let reference = r101_25()
        .solve_root(8, Rounding::None, plain.clone())
        .unwrap();
    assert_eq!(reference.master.status, Status::Optimal);

    let mut options = plain;
    options.cg.column_cleanup_threshold = 60.;
    let result = r101_25().solve_root(8, Rounding::None, options).unwrap();
    assert_eq!(result.master.status, Status::Optimal);
    assert!((result.master.lp - reference.master.lp).abs() < 1e-6);
    assert!(result.master.columns < reference.master.columns);
}

#[test]
fn convergence_before_pricing_keeps_the_solved_cut_model() {
    use baldes::master::{
        clp::ClpMaster,
        column_generation::{self as cg, Answer, Column, Controller, LpMaster, Options},
        RowSense,
    };
    use baldes::pricing::phases::Phase;
    struct RedundantCuts {
        observed_rows: Vec<usize>,
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
            } else {
                Ok(0)
            }
        }
    }
    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::Equal, 1., &[]).unwrap();
    master.add_column(1., &[(0, 1.)]).unwrap();
    let mut controller = RedundantCuts {
        observed_rows: vec![],
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
    // x=1 still solves the model after adding the redundant inequalities;
    // its certified bound closes the next pass without another pricing call.
    assert_eq!(result.status, Status::Optimal);
    assert_eq!(result.pricing_calls, 1);
    assert_eq!(controller.observed_rows, vec![1, 3]);
    assert_eq!(master.rhs().len(), 3);
    assert_eq!(master.status(), 0);
    assert_eq!(master.objective(), 1.);
    assert_eq!(master.values()[0], 1.);
}
