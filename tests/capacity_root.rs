#![cfg(feature = "clp")]
use baldes::master::{
    column_generation::Status,
    root::{CapacityCutConfig, RootOptions},
};
use baldes::vrptw::{Instance, Rounding};

#[test]
fn cleanup_retains_basic_capacity_rows_with_negative_normalized_slack() {
    use baldes::master::{clp::ClpMaster, RowSense};
    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::Equal, 3., &[]).unwrap();
    master.add_row(RowSense::LessEqual, -2., &[]).unwrap();
    master.add_row(RowSense::LessEqual, 4., &[]).unwrap();
    master.add_row(RowSense::LessEqual, 0., &[]).unwrap();
    master.add_row(RowSense::LessEqual, -1., &[]).unwrap();
    master.add_column(1., &[(0,1.),(1,-1.),(2,1.),(3,-1.),(4,-1.)]).unwrap();
    assert!(master.solve());
    for row in 1..5 { assert!(master.row_is_basic(row).unwrap()); }
    // -2 - (-3) = 1, divided by signed rhs -2 -> -0.5, retained.
    // rhs == -1 is not normalized and is eligible for deletion.
    assert_eq!(master.cleanup_cut_rows(1).unwrap(), vec![2,3,4]);
}

#[test]
fn capacity_root_prices_complement_rows_with_and_without_rank_one_and_fixing() {
    let instance = Instance::parse("capacity-root\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n3 1 1 1 0 100 0\n").unwrap();
    let mut plain_options = RootOptions::default();
    plain_options.enumerate = false;
    let plain = instance
        .solve_root(8, Rounding::None, plain_options)
        .unwrap();
    assert!((plain.master.lp - 1.5 * (2. + 2f64.sqrt())).abs() < 1e-8);
    for (rank1, fixing) in [(false, false), (true, false), (false, true)] {
        let mut options = RootOptions::default();
        options.enumerate = fixing;
        if rank1 {
            options = options.with_rank_one_cuts();
        }
        options = options.with_capacity_cuts(CapacityCutConfig::new(vec![0., 1., 1., 1.], 2.));
        let result = instance.solve_root(8, Rounding::None, options).unwrap();
        assert_eq!(result.master.status, Status::Optimal);
        assert!((result.master.lp - (4. + 2f64.sqrt())).abs() < 1e-8);
        assert!((result.lp_before_cuts - plain.master.lp).abs() < 1e-8);
        assert!(result.cuts > 0);
        assert!(result.master.cut_rounds > 0);
    }
}

#[test]
fn fractional_capacity_network_retains_zero_terminal_edges() {
    use baldes::cuts::mincut::MinCutDigraph;
    let mut graph = MinCutDigraph::new_with_terminals(vec![vec![0.; 3]; 3], 1, 2);
    graph.find_min_cut(1, 2);
    assert_eq!(graph.sink_side, vec![2]);
    graph.set_arc_value(0, 2, 99.);
    graph.find_min_cut(1, 2);
    assert_eq!(graph.sink_side, vec![0, 2]);
}

#[test]
fn routing_fractional_capacity_snapshot_has_twenty_cuts() {
    use baldes::cuts::capacity::CapacitySeparator;
    let mut sep = CapacitySeparator::new(
        vec![
            0., 19., 30., 16., 23., 11., 31., 15., 28., 8., 8., 7., 14., 6., 19., 11.,
        ],
        35.,
    )
    .unwrap();
    sep.parameters.complement_form = true;
    sep.parameters.use_two_path = false;
    let cuts = sep
        .separate(&[
            (vec![6], 1.),
            (vec![2], 1.),
            (vec![4, 11], 1.),
            (vec![5, 14], 1. / 3.),
            (vec![8], 1. / 3.),
            (vec![7, 14], 1. / 3.),
            (vec![13, 8], 2. / 3.),
            (vec![15, 12, 10], 1.),
            (vec![13, 9, 14], 1. / 3.),
            (vec![1, 3], 1.),
            (vec![7, 9, 5], 2. / 3.),
        ])
        .unwrap();
    assert_eq!(cuts.len(), 20);
    assert_eq!(cuts[0].customers, vec![7, 9, 14]);
    assert_eq!(cuts[0].rhs, 1);
    assert!(!cuts[0].greater_equal());
}
