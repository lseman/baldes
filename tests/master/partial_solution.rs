#![cfg(feature = "clp")]
use baldes::master::{
    branching::BranchConstraint,
    column_generation::Status,
    root::{
        self, CapacityCutConfig, FixedRoute, KPathCutConfig, KnapsackCutConfig, RootOptions,
        RootProblem,
    },
    RowSense,
};
use baldes::inputs::{Instance, Rounding};

fn fixture(family: usize) -> (RootProblem, RootOptions, Vec<i32>, Vec<i32>) {
    let instance = Instance::parse("triangle\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n3 1 1 1 0 100 0\n").unwrap();
    let graph = instance.graph(3, Rounding::None).unwrap();
    let arcs = |nodes: &[usize]| {
        nodes
            .windows(2)
            .map(|v| {
                graph
                    .arcs
                    .iter()
                    .find(|a| a.tail == v[0] && a.head == v[1])
                    .unwrap()
                    .id
            })
            .collect()
    };
    let pair = arcs(&[0, 1, 2, graph.sink]);
    let single = arcs(&[0, 3, graph.sink]);
    let problem = RootProblem::set_partitioning(graph, 3.).unwrap();
    let mut options = RootOptions::default();
    options.capture_state = true;
    options.cg.stabilize = false;
    options = match family {
        1 => options.with_rank_one_cuts(),
        2 => options.with_capacity_cuts(CapacityCutConfig::new(vec![0., 1., 1., 1.], 2.)),
        3 => options.with_kpath_cuts(KPathCutConfig::new(vec![0., 1., 1., 1.], 2.)),
        4 => options.with_knapsack_cuts(KnapsackCutConfig::new(vec![
            baldes::cuts::knapsack::KnapsackRow::new(0, 1, 1., 3),
        ])),
        _ => options,
    };
    (problem, options, pair, single)
}

#[test]
fn fixed_routes_keep_all_cut_families_costs_and_parent_isolation() {
    for family in 0..5 {
        let (problem, options, pair, single) = fixture(family);
        let parent = root::solve(problem, options.clone()).unwrap();
        let snapshot = parent.snapshot.as_ref().unwrap();
        if family > 0 {
            assert!(snapshot.active_cuts() > 0, "family {family}");
        }
        let fixed = root::inherit_fixed_routes(
            snapshot,
            &[FixedRoute {
                arcs: pair.clone(),
                quantity: 1,
            }],
            options.clone(),
        )
        .unwrap();
        let expected = 2. + 3. * 2f64.sqrt();
        assert_eq!(
            fixed.master.status,
            Status::Optimal,
            "family {family}: {fixed:?}"
        );
        assert!(
            (fixed.master.lp - expected).abs() < 1e-7,
            "family {family}: {fixed:?}"
        );
        assert!(
            (fixed.master.bound - expected).abs() < 1e-7,
            "family {family}: {fixed:?}"
        );
        assert!((fixed.fixed_cost - (2. + 2f64.sqrt())).abs() < 1e-8);
        assert_eq!(fixed.route_values().len(), 2);
        let cost: f64 = fixed
            .route_values()
            .iter()
            .map(|(r, q)| r.original_cost * q)
            .sum();
        assert!((cost - expected).abs() < 1e-8);
        let child = fixed.snapshot.as_ref().unwrap();
        assert_eq!(child.fixed_routes().len(), 1);
        let complete = root::inherit_fixed_routes(
            child,
            &[FixedRoute {
                arcs: single.clone(),
                quantity: 1,
            }],
            options.clone(),
        )
        .unwrap();
        assert_eq!(complete.master.status, Status::Optimal, "{complete:?}");
        assert!((complete.master.lp - expected).abs() < 1e-7);
        assert_eq!(complete.route_values().len(), 2);
        let replay = root::inherit_branches(child, &[], options.clone()).unwrap();
        assert!((replay.master.lp - fixed.master.lp).abs() < 1e-7);
        assert_eq!(replay.fixed_routes.len(), 1);
        let original = root::inherit_branches(snapshot, &[], options.clone()).unwrap();
        assert!((original.master.lp - parent.master.lp).abs() < 1e-7);
        assert!(snapshot.fixed_routes().is_empty());
        assert_eq!(snapshot.fixed_cost(), 0.);
    }
}

#[test]
fn residual_branches_and_invalid_fixings_do_not_corrupt_snapshots() {
    use baldes::pricing::branching::RyanFosterConstraint;
    let (problem, options, pair, _) = fixture(2);
    let arc = problem.arc_position(pair[0]).unwrap();
    let parent = root::solve(problem, options.clone()).unwrap();
    let snapshot = parent.snapshot.as_ref().unwrap();
    for entry in [
        FixedRoute {
            arcs: vec![],
            quantity: 1,
        },
        FixedRoute {
            arcs: pair.clone(),
            quantity: 0,
        },
        FixedRoute {
            arcs: vec![999999],
            quantity: 1,
        },
    ] {
        assert!(root::inherit_fixed_routes(snapshot, &[entry], options.clone()).is_err());
    }
    let fixed = root::inherit_fixed_routes(
        snapshot,
        &[FixedRoute {
            arcs: pair.clone(),
            quantity: 1,
        }],
        options.clone(),
    )
    .unwrap();
    let child = fixed.snapshot.as_ref().unwrap();
    for branch in [
        BranchConstraint::ArcFlow {
            arc,
            sense: RowSense::LessEqual,
            rhs: 0.,
        },
        BranchConstraint::RyanFoster(RyanFosterConstraint {
            first: 1,
            second: 2,
            together: false,
        }),
        BranchConstraint::RyanFoster(RyanFosterConstraint {
            first: 1,
            second: 3,
            together: true,
        }),
        BranchConstraint::NumberOfVehicles {
            graph_id: 0,
            sense: RowSense::LessEqual,
            rhs: 0.,
        },
    ] {
        let result = root::inherit_branches(child, &[branch], options.clone()).unwrap();
        assert!(result.preprocessing.infeasible, "{result:?}");
        assert_eq!(result.master.pricing_calls, 0);
    }
    let repeated = root::inherit_fixed_routes(
        snapshot,
        &[FixedRoute {
            arcs: pair,
            quantity: 2,
        }],
        options.clone(),
    )
    .unwrap();
    assert!(repeated.preprocessing.infeasible);
    assert_eq!(repeated.master.pricing_calls, 0);
    assert!(
        (root::inherit_branches(snapshot, &[], options)
            .unwrap()
            .master
            .lp
            - parent.master.lp)
            .abs()
            < 1e-7
    );
}

#[cfg(feature = "highs")]
#[test]
fn objective_offsets_survive_rhs_changes_clones_and_integer_transfer() {
    use baldes::master::{
        clp::ClpMaster,
        solvers::{
            highs::solve_integer,
            mip::{MipOptions, MipStatus},
        },
    };
    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::Equal, 2., &[]).unwrap();
    master.add_column(3., &[(0, 1.)]).unwrap();
    master.objective_offset = 7.;
    assert!(master.solve());
    assert_eq!(master.objective, 13.);
    let mut clone = master.try_clone().unwrap();
    clone.set_rhs(&[1.]).unwrap();
    assert!(clone.solve());
    assert_eq!(clone.objective, 10.);
    let mip = solve_integer(&clone, &[0], &MipOptions::default()).unwrap();
    assert_eq!(mip.status, MipStatus::Optimal);
    assert_eq!(mip.solution.unwrap().objective, 10.);
    assert_eq!(master.rhs, [2.]);
    assert_eq!(master.objective, 13.);
    let mut empty = ClpMaster::new().unwrap();
    empty.objective_offset = 11.;
    let mip = solve_integer(&empty, &[], &MipOptions::default()).unwrap();
    assert_eq!(mip.bound, 11.);
    assert_eq!(mip.solution.unwrap().objective, 11.);
}
