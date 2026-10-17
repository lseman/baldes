#![cfg(feature = "clp")]
use baldes::master::{column_generation::Status, root::RootOptions};
use baldes::vrptw::{Instance, Rounding};

fn r101_25() -> Instance {
    let mut instance = Instance::read(concat!(env!("CARGO_MANIFEST_DIR"), "/instances/R101.txt")).unwrap();
    instance.truncate(25).unwrap();
    instance
}

#[test]
fn column_cleanup_shrinks_the_master_and_keeps_the_root_bound() {
    let mut plain = RootOptions::default();
    plain.enumerate = false;
    let reference = r101_25().solve_root(8, Rounding::None, plain.clone()).unwrap();
    assert_eq!(reference.master.status, Status::Optimal);

    let mut options = plain;
    options.cg.column_cleanup_threshold = 60.;
    let result = r101_25().solve_root(8, Rounding::None, options).unwrap();
    assert_eq!(result.master.status, Status::Optimal);
    assert!((result.master.lp - reference.master.lp).abs() < 1e-6);
    assert!(result.master.columns < reference.master.columns);
}
