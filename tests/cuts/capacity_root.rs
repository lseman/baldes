#![cfg(feature = "clp")]
use baldes::inputs::{Instance, Rounding};
use baldes::master::{
    cg::Status,
    model::{separation::CapacityCutConfig, EnumerationOptions, Options},
};

#[test]
fn cleanup_keeps_lowest_slack_basic_cut_when_none_are_protected() {
    use baldes::master::{clp::ClpMaster, RowSense};
    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::Equal, 1., &[]).unwrap();
    master.add_row(RowSense::LessEqual, 1.5, &[]).unwrap();
    master.add_row(RowSense::LessEqual, 1.25, &[]).unwrap();
    master.add_row(RowSense::LessEqual, 1., &[]).unwrap();
    master.add_row(RowSense::LessEqual, 1., &[]).unwrap();
    master
        .add_column(1., &[(0, 1.), (1, 1.), (2, 1.), (3, 1.), (4, 1.)])
        .unwrap();
    assert!(master.solve());
    for row in 1..5 {
        assert!(master.row_is_basic(row).unwrap());
    }
    assert_eq!(master.cleanup_cut_rows(1).unwrap(), vec![1, 2, 4]);
}

#[test]
fn cleanup_considers_nonbasic_lower_bound_cut_rows() {
    use baldes::master::{clp::ClpMaster, LpMaster, RowSense};
    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::GreaterEqual, 0., &[]).unwrap();
    master.add_row(RowSense::GreaterEqual, 1., &[]).unwrap();
    master.add_row(RowSense::GreaterEqual, 1., &[]).unwrap();
    master.add_column(1., &[(0, 1.), (1, 1.), (2, 1.)]).unwrap();
    assert!(master.solve());
    let nonbasic = (1..3)
        .find(|&row| !master.row_is_basic(row).unwrap())
        .expect("one duplicate binding cut must be nonbasic");
    assert!(master.row_cleanup_candidate(nonbasic).unwrap());
    assert_eq!(master.cleanup_cut_rows(1).unwrap(), vec![2]);
}

#[test]
fn cleanup_protects_nonbasic_upper_bound_cut_rows() {
    use baldes::master::{clp::ClpMaster, LpMaster, RowSense};
    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::GreaterEqual, 0., &[]).unwrap();
    master.add_row(RowSense::LessEqual, 1., &[]).unwrap();
    master.add_row(RowSense::LessEqual, 1., &[]).unwrap();
    master
        .add_column(-1., &[(0, 1.), (1, 1.), (2, 1.)])
        .unwrap();
    assert!(master.solve());
    let nonbasic = (1..3)
        .find(|&row| !master.row_is_basic(row).unwrap())
        .expect("one duplicate binding cut must be nonbasic");
    assert!(!master.row_cleanup_candidate(nonbasic).unwrap());
    assert!(!master.cleanup_cut_rows(1).unwrap().contains(&nonbasic));
}

#[test]
fn cleanup_retains_basic_capacity_rows_with_negative_normalized_slack() {
    use baldes::master::{clp::ClpMaster, RowSense};
    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::Equal, 3., &[]).unwrap();
    master.add_row(RowSense::LessEqual, -2., &[]).unwrap();
    master.add_row(RowSense::LessEqual, 4., &[]).unwrap();
    master.add_row(RowSense::LessEqual, 0., &[]).unwrap();
    master.add_row(RowSense::LessEqual, -1., &[]).unwrap();
    master
        .add_column(1., &[(0, 1.), (1, -1.), (2, 1.), (3, -1.), (4, -1.)])
        .unwrap();
    assert!(master.solve());
    for row in 1..5 {
        assert!(master.row_is_basic(row).unwrap());
    }
    // -2 - (-3) = 1, divided by signed rhs -2 -> -0.5, retained.
    // rhs == -1 is not normalized and is eligible for deletion.
    assert_eq!(master.cleanup_cut_rows(1).unwrap(), vec![2, 3, 4]);
}

#[test]
fn capacity_root_prices_complement_rows_with_rank_one_and_full_enumeration() {
    let instance = Instance::parse("capacity-root\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n3 1 1 1 0 100 0\n").unwrap();
    let plain_options = Options::default();
    let plain = instance.solve(8, Rounding::None, plain_options).unwrap();
    assert!((plain.master.lp - 1.5 * (2. + 2f64.sqrt())).abs() < 1e-8);
    for (rank1, enumerate) in [(false, false), (true, false), (false, true)] {
        let mut options = Options::default();
        options.enumeration = enumerate.then(EnumerationOptions::default);
        if rank1 {
            options.separation.rank_one.enabled = true;
        }
        options.separation.capacity = Some(CapacityCutConfig::new(vec![0., 1., 1., 1.], 2.));
        let result = instance.solve(8, Rounding::None, options).unwrap();
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

#[test]
fn kohl_time_resource_closes_r101_ten_customer_root_gap() {
    let mut instance = Instance::read("instances/Solomon/R101.txt").unwrap();
    instance.truncate(10).unwrap();
    let mut options = Options::default();
    options.cg.stabilize = false;
    let plain = instance.solve(10, Rounding::None, options.clone()).unwrap();
    assert!((plain.master.lp - 261.30231910442023).abs() < 1e-5);
    let mut capacity_options = options.clone();
    capacity_options.separation.capacity =
        Some(instance.capacity_cut_config(Rounding::None, false));
    let capacity_only = instance
        .solve(10, Rounding::None, capacity_options)
        .unwrap();
    options.separation.capacity = Some(instance.capacity_cut_config(Rounding::None, true));
    let kohl = instance.solve(10, Rounding::None, options).unwrap();
    assert_eq!(kohl.master.status, Status::Optimal);
    assert!(kohl.cuts > 0);
    assert!(kohl.master.lp > capacity_only.master.lp + 1.);
    assert!((kohl.master.lp - 269.5330).abs() < 1e-3);
}

#[test]
fn inherited_capacity_cuts_preprocess_contradictory_children_without_mutating_siblings() {
    use baldes::master::{
        model::{self, Problem, RowCoefficients},
        preprocessing::Bounds,
    };
    let instance = Instance::parse("triangle\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n3 1 1 1 0 100 0\n").unwrap();
    for complement in [false, true] {
        let graph = instance.graph(3, Rounding::None).unwrap();
        let start = graph
            .arcs
            .iter()
            .position(|a| a.tail == graph.source && a.head == 3)
            .unwrap();
        let end = graph
            .arcs
            .iter()
            .position(|a| a.tail == 3 && a.head == graph.sink)
            .unwrap();
        let arc_count = graph.arcs.len();
        let problem = Problem::set_partitioning(graph, 3.).unwrap();
        let mut config = CapacityCutConfig::new(vec![0., 1., 1., 1.], 2.);
        config.parameters.complement_form = complement;
        let mut options = Options::default();
        options.separation.capacity = Some(config);
        options.cg.stabilize = false;
        let parent = model::solve(problem, options.clone()).unwrap();
        assert!(parent.cuts > 0, "complement={complement}");
        let mut child = parent
            .snapshot
            .with_multiplicity(
                0,
                Bounds {
                    lower: 0.,
                    upper: 1.,
                },
            )
            .unwrap();
        for index in [start, end] {
            let mut arcs = vec![0.; arc_count];
            arcs[index] = 1.;
            child = child
                .with_activity_bounds(
                    Bounds {
                        lower: 1.,
                        upper: f64::INFINITY,
                    },
                    &[RowCoefficients {
                        network: 0,
                        arcs,
                        per_route: 0.,
                    }],
                )
                .unwrap();
        }
        let result = child.solve(options.clone()).unwrap();
        assert!(
            result
                .snapshot
                .preprocessing_report()
                .propagation
                .infeasible,
            "complement={complement}: {:?}",
            result.master
        );
        assert_eq!(result.master.status, Status::Infeasible);
        assert_eq!(result.master.pricing_calls, 0);
        assert!(result.values.iter().all(|value| *value == 0.));
        let resumed = result.snapshot.solve(options.clone()).unwrap();
        assert_eq!(resumed.master.status, Status::Infeasible);
        assert_eq!(resumed.master.pricing_calls, 0);
        let sibling = parent
            .snapshot
            .with_multiplicity(
                0,
                Bounds {
                    lower: 2.,
                    upper: f64::INFINITY,
                },
            )
            .unwrap()
            .solve(options.clone())
            .unwrap();
        assert_eq!(sibling.master.status, Status::Optimal);
        let replay = parent.snapshot.solve(options).unwrap();
        assert!((replay.master.lp - parent.master.lp).abs() < 1e-7);
        assert!(replay.cuts > 0);
    }
}

#[test]
fn integer_only_preprocessing_infeasibility_is_not_an_lp_proof() {
    use baldes::graph::Window;
    use baldes::master::{
        formulation::Row,
        model::{self, Network, Problem},
        RowSense,
    };
    let mut graph = baldes::Graph::<2>::new(3, 0).unwrap();
    graph.add_arc(0, 1, 1.).unwrap();
    graph.add_arc(1, 2, 1.).unwrap();
    let problem = Problem {
        rows: vec![Row {
            sense: RowSense::Equal,
            rhs: 0.5,
        }],
        networks: vec![Network {
            id: 0,
            graph,
            multiplicity: Window::new(0., 1.).unwrap(),
            arc_entries: vec![vec![(0, 1.)], vec![]],
            route_entries: vec![],
            activity_entries: vec![],
        }],
    };
    let mut options = Options::default();
    options.preprocessing.integer_domains = true;
    options.cg.stabilize = false;
    let output = model::solve(problem, options).unwrap();
    assert!(
        output
            .snapshot
            .preprocessing_report()
            .propagation
            .infeasible
    );
    assert_eq!(output.master.status, Status::Optimal, "{:?}", output.master);
    assert!((output.master.lp - 1.).abs() < 1e-8);
    assert!(output.master.pricing_calls > 0);
}
