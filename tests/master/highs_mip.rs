#![cfg(feature = "highs")]
use baldes::master::{
    solvers::{
        highs::{solve_integer, HighsMaster},
        mip::{MipOptions, MipStatus},
    },
    LpMaster, RowSense,
};
use std::time::Duration;

fn triangle<M: LpMaster>(mut model: M) -> M {
    for _ in 0..3 {
        model.add_row(RowSense::Equal, 1., &[]).unwrap();
    }
    for (a, b) in [(0, 1), (0, 2), (1, 2)] {
        model.add_column(1., &[(a, 1.), (b, 1.)]).unwrap();
    }
    for v in 0..3 {
        model.add_column(1.1, &[(v, 1.)]).unwrap();
    }
    model
}
#[test]
fn highs_mip_preserves_rows_bounds_and_source_lp() {
    let mut model = triangle(HighsMaster::new().unwrap());
    assert!(model.solve());
    assert!((model.objective() - 1.5).abs() < 1e-8);
    let values = model.values.clone();
    let mip = solve_integer(&model, &[0, 1, 2, 3, 4, 5], &MipOptions::default()).unwrap();
    assert_eq!(mip.status, MipStatus::Optimal, "{mip:?}");
    assert!((mip.solution.unwrap().objective - 2.1).abs() < 1e-8);
    assert_eq!(model.values, values);
    assert!(model.solved());
    // A retained cut changes the integer solution. A retained branch and an
    // upper bound then conflict; dropping either would make the MIP feasible.
    model
        .add_row(
            RowSense::GreaterEqual,
            2.5,
            &[(0, 1.), (1, 1.), (2, 1.), (3, 1.), (4, 1.), (5, 1.)],
        )
        .unwrap();
    let mip = solve_integer(&model, &[0, 1, 2, 3, 4, 5], &MipOptions::default()).unwrap();
    assert_eq!(mip.status, MipStatus::Optimal);
    assert!((mip.solution.unwrap().objective - 3.3).abs() < 1e-8);
    model
        .add_row(RowSense::GreaterEqual, 0.2, &[(3, 1.)])
        .unwrap();
    model.set_upper(3, 0.8).unwrap();
    let mip = solve_integer(&model, &[0, 1, 2, 3, 4, 5], &MipOptions::default()).unwrap();
    assert_eq!(mip.status, MipStatus::Infeasible);
    assert!(mip.solution.is_none());
}
#[cfg(feature = "clp")]
#[test]
fn current_clp_lp_can_be_solved_as_highs_mip_without_mutation() {
    let mut model = triangle(baldes::master::clp::ClpMaster::new().unwrap());
    assert!(model.solve());
    let before = (
        model.values.clone(),
        model.duals.clone(),
        model.upper.clone(),
        model.simplex_iterations,
    );
    let mip = solve_integer(&model, &[0, 1, 2, 3, 4, 5], &MipOptions::default()).unwrap();
    assert_eq!(mip.status, MipStatus::Optimal);
    assert!((mip.solution.unwrap().objective - 2.1).abs() < 1e-8);
    assert_eq!(
        before,
        (
            model.values.clone(),
            model.duals.clone(),
            model.upper.clone(),
            model.simplex_iterations
        )
    );
    assert!(model.solved);
}
#[test]
fn mixed_integrality_and_explicit_budget_outcomes() {
    let mut model = HighsMaster::new().unwrap();
    model.add_row(RowSense::GreaterEqual, 1.5, &[]).unwrap();
    model.add_column(1., &[(0, 1.)]).unwrap();
    model.add_column(0.5, &[(0, 1.)]).unwrap();
    model.set_upper(1, 1.).unwrap();
    let mip = solve_integer(&model, &[0], &MipOptions::default()).unwrap();
    let s = mip.solution.unwrap();
    assert_eq!(mip.status, MipStatus::Optimal);
    assert!((s.objective - 1.25).abs() < 1e-8);
    assert_eq!(s.values, [1., 0.5]);
    for (options, status) in [
        (
            MipOptions {
                time_limit: Duration::ZERO,
                ..Default::default()
            },
            MipStatus::TimeLimit,
        ),
        (
            MipOptions {
                node_limit: 0,
                ..Default::default()
            },
            MipStatus::NodeLimit,
        ),
        (
            MipOptions {
                objective_cutoff: 1.,
                ..Default::default()
            },
            MipStatus::ObjectiveCutoff,
        ),
    ] {
        let mip = solve_integer(&model, &[0], &options).unwrap();
        assert_eq!(mip.status, status, "{mip:?}");
        assert!(mip.solution.is_none());
    }
    assert!(solve_integer(&model, &[0, 0], &MipOptions::default()).is_err());
    assert!(solve_integer(&model, &[2], &MipOptions::default()).is_err());
}
#[test]
fn integer_infeasibility_is_not_a_fractional_solution_or_a_cutoff() {
    let mut model = HighsMaster::new().unwrap();
    model.add_row(RowSense::Equal, 0.5, &[]).unwrap();
    model.add_column(1., &[(0, 1.)]).unwrap();
    let mip = solve_integer(&model, &[0], &MipOptions::default()).unwrap();
    assert_eq!(mip.status, MipStatus::Infeasible);
    assert!(mip.solution.is_none());
    let empty = solve_integer(&HighsMaster::new().unwrap(), &[], &MipOptions::default()).unwrap();
    assert_eq!(empty.status, MipStatus::Optimal);
    assert_eq!(empty.solution.unwrap().objective, 0.);
}
#[test]
fn small_integer_models_match_independent_enumeration() {
    let mut rng = baldes::rng::Xoroshiro128Plus::from_seed(137u64);
    for _ in 0..50 {
        let mut next = || rng.next_u32() as usize;
        let n = 3 + next() % 6;
        let costs: Vec<_> = (0..n).map(|_| next() % 17 + 1).collect();
        let weights: Vec<_> = (0..n).map(|_| next() % 5 + 1).collect();
        let target = next() % 12 + 1;
        let mut model = HighsMaster::new().unwrap();
        model
            .add_row(RowSense::GreaterEqual, target as f64, &[])
            .unwrap();
        for i in 0..n {
            model
                .add_column(costs[i] as f64, &[(0, weights[i] as f64)])
                .unwrap();
            model.set_upper(i, 1.).unwrap();
        }
        let expected = (0usize..1 << n)
            .filter(|mask| {
                (0..n)
                    .filter(|i| mask & (1 << i) != 0)
                    .map(|i| weights[i])
                    .sum::<usize>()
                    >= target
            })
            .map(|mask| {
                (0..n)
                    .filter(|i| mask & (1 << i) != 0)
                    .map(|i| costs[i])
                    .sum::<usize>()
            })
            .min();
        let mip =
            solve_integer(&model, &(0..n).collect::<Vec<_>>(), &MipOptions::default()).unwrap();
        match expected {
            Some(x) => {
                assert_eq!(mip.status, MipStatus::Optimal);
                assert!((mip.solution.unwrap().objective - x as f64).abs() < 1e-7);
            }
            None => assert_eq!(mip.status, MipStatus::Infeasible),
        }
    }
}

#[cfg(feature = "clp")]
mod routing {
    use super::*;
    use baldes::inputs::{Instance, Rounding};
    use baldes::master::{
        formulation::Row,
        model::{
            self,
            separation::{CapacityCutConfig, KPathCutConfig, KnapsackCutConfig},
            EnumerationOptions, NodeSnapshot, Options, Problem, RowCoefficients, RyanFosterMember,
        },
        preprocessing::Bounds,
    };
    fn instance() -> Instance {
        Instance::parse("triangle\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n3 1 1 1 0 100 0\n").unwrap()
    }
    fn problem() -> Problem<2> {
        Problem::set_partitioning(instance().graph(3, Rounding::None).unwrap(), 3.).unwrap()
    }
    #[derive(Clone, Copy, Debug)]
    enum Branch {
        Vehicles,
        Together,
        Apart,
        NoFirstArc,
        UnassignedCustomer,
    }
    fn restrict(node: &NodeSnapshot<2>, branch: Branch, arcs: usize) -> NodeSnapshot<2> {
        let pair = |together| {
            node.with_ryan_foster(
                &[RyanFosterMember {
                    network: 0,
                    first: 1,
                    second: 2,
                }],
                together,
            )
        };
        match branch {
            Branch::Vehicles => node.with_multiplicity(
                0,
                Bounds {
                    lower: 3.,
                    upper: 3.,
                },
            ),
            Branch::Together => pair(true),
            Branch::Apart => pair(false),
            Branch::NoFirstArc => {
                let mut coefficients = vec![0.; arcs];
                coefficients[0] = 1.;
                node.with_row(
                    Row {
                        sense: RowSense::LessEqual,
                        rhs: 0.,
                    },
                    &[RowCoefficients {
                        network: 0,
                        arcs: coefficients,
                        per_route: 0.,
                    }],
                )
            }
            // Customer 1 covers shared row 0.
            Branch::UnassignedCustomer => node.branch_assignment(0, 0, false),
        }
        .unwrap()
    }
    #[test]
    fn enumerated_mip_keeps_active_cut_families_and_typed_child_rows() {
        let instance = instance();
        let problem = problem();
        let arcs = problem.networks[0].graph.arcs.len();
        let first_arc = problem.networks[0].graph.arcs[0].id;
        for family in 0..6 {
            let mut options = Options::default();
            options.cg.stabilize = false;
            match family {
                1 => options.separation.rank_one.enabled = true,
                2 => {
                    options.separation.capacity =
                        Some(CapacityCutConfig::new(vec![0., 1., 1., 1.], 2.))
                }
                3 => options.separation.kpath = Some(KPathCutConfig::new(vec![0., 1., 1., 1.], 2.)),
                4 => {
                    options.separation.knapsack = Some(KnapsackCutConfig::new(vec![
                        baldes::cuts::knapsack::KnapsackRow::new(0, 1, 1., 3),
                    ]))
                }
                5 => options.separation.knapsack = Some(instance.fenchel_cut_config().unwrap()),
                _ => {}
            }
            let parent = model::solve(problem.clone(), options.clone()).unwrap();
            if family > 0 {
                assert!(parent.cuts > 0, "family {family}");
            }
            let snapshot = parent
                .snapshot
                .enumerate(&EnumerationOptions::default())
                .unwrap();
            assert!(
                snapshot.enumeration_report().complete,
                "family {family}: {:?}",
                snapshot.enumeration_report()
            );
            let mip = snapshot
                .solve_enumerated_mip(&MipOptions::default())
                .unwrap()
                .unwrap();
            assert_eq!(
                mip.mip.status,
                MipStatus::Optimal,
                "family {family}: {mip:?}"
            );
            assert!((mip.mip.solution.unwrap().objective - (4. + 2f64.sqrt())).abs() < 1e-6);
            for branch in [
                Branch::Vehicles,
                Branch::Together,
                Branch::Apart,
                Branch::NoFirstArc,
                Branch::UnassignedCustomer,
            ] {
                let child = restrict(&snapshot, branch, arcs)
                    .solve(options.clone())
                    .unwrap();
                if matches!(branch, Branch::UnassignedCustomer) {
                    assert_eq!(
                        child.master.status,
                        baldes::master::cg::Status::Infeasible,
                        "{child:?}"
                    );
                    continue;
                }
                let mip = child
                    .snapshot
                    .solve_enumerated_mip(&MipOptions::default())
                    .unwrap()
                    .unwrap_or_else(|| {
                        panic!(
                            "missing child universe: family={family} branch={branch:?} {:?}",
                            child.snapshot.enumeration_report()
                        )
                    });
                assert_eq!(mip.mip.status, MipStatus::Optimal, "{mip:?}");
                if matches!(branch, Branch::Vehicles) {
                    let total: f64 = mip.route_values.iter().map(|r| r.quantity).sum();
                    assert_eq!(total, 3.);
                    continue;
                }
                for fixed in &mip.route_values {
                    let r = &fixed.route.route;
                    match branch {
                        Branch::Together => {
                            assert_eq!(r.nodes.contains(&1), r.nodes.contains(&2))
                        }
                        Branch::Apart => assert!(!(r.nodes.contains(&1) && r.nodes.contains(&2))),
                        Branch::NoFirstArc => assert!(!r.arcs.contains(&first_arc)),
                        _ => unreachable!(),
                    }
                }
            }
            let again = snapshot
                .solve_enumerated_mip(&MipOptions::default())
                .unwrap()
                .unwrap();
            assert!((again.mip.solution.unwrap().objective - (4. + 2f64.sqrt())).abs() < 1e-6);
        }
    }
    #[test]
    fn optional_tree_mip_closes_only_complete_enumeration() {
        use baldes::master::tree::{self, ModelTreeOptions, TreeStatus};
        let mut options = ModelTreeOptions::default();
        options.node.cg.stabilize = false;
        options.enumerated_mip = Some(MipOptions::default());
        // Automatic enumeration needs a finite incumbent gap, supplied here
        // by the validated pricing witness.
        let result = tree::solve(problem(), options.clone()).unwrap();
        assert_eq!(result.status, TreeStatus::Optimal, "{result:?}");
        assert!(result.enumerated_mip_calls > 0, "{result:?}");
        // Without gap enumeration the incumbent-cutoff collection still feeds
        // the end-of-node MIP; the optimum must not change.
        options.node.gap_enumeration = false;
        let result = tree::solve(problem(), options.clone()).unwrap();
        assert_eq!(result.status, TreeStatus::Optimal, "{result:?}");
        assert!((result.incumbent - (4. + 2f64.sqrt())).abs() < 1e-6);
        options.node.gap_enumeration = true;
        options.enumerated_mip = Some(MipOptions {
            node_limit: 0,
            ..Default::default()
        });
        options.node_limit = 1;
        let result = tree::solve(problem(), options).unwrap();
        assert_ne!(result.status, TreeStatus::Optimal, "{result:?}");
        assert!(result.open_nodes > 0);
        assert!(result.bound < result.incumbent);
    }
}

#[cfg(feature = "clp")]
#[test]
fn transfer_refuses_coefficients_that_highs_would_drop() {
    let mut model = baldes::master::clp::ClpMaster::new().unwrap();
    model.add_row(RowSense::Equal, 1., &[]).unwrap();
    model.add_column(0., &[(0, 1e-13)]).unwrap();
    assert!(solve_integer(&model, &[0], &MipOptions::default()).is_err());
    let mut constant = HighsMaster::new().unwrap();
    constant.add_row(RowSense::Equal, 0., &[]).unwrap();
    assert_eq!(
        solve_integer(&constant, &[], &MipOptions::default())
            .unwrap()
            .status,
        MipStatus::Optimal
    );
    constant.add_row(RowSense::GreaterEqual, 1., &[]).unwrap();
    assert_eq!(
        solve_integer(&constant, &[], &MipOptions::default())
            .unwrap()
            .status,
        MipStatus::Infeasible
    );
}

#[test]
fn fractional_integer_upper_bounds_are_normalized_without_clipping_continuous_bounds() {
    // x=a+4b, x=1, a+b<=1, min 10a-b. Integer b<=1/4 means b=0,
    // so a=1 is feasible. Native presolve previously rejected this model.
    let mut model = HighsMaster::new().unwrap();
    model.add_row(RowSense::Equal, 0., &[]).unwrap();
    model.add_row(RowSense::Equal, 1., &[]).unwrap();
    model.add_row(RowSense::LessEqual, 1., &[]).unwrap();
    model.add_column(0., &[(0, 1.), (1, 1.)]).unwrap();
    model.set_upper(0, 1.).unwrap();
    model.add_column(10., &[(0, -1.), (2, 1.)]).unwrap();
    model.set_upper(1, 1.).unwrap();
    model.add_column(-1., &[(0, -4.), (2, 1.)]).unwrap();
    model.set_upper(2, 0.25).unwrap();
    let integer = solve_integer(&model, &[1, 2], &MipOptions::default()).unwrap();
    assert_eq!(integer.status, MipStatus::Optimal);
    assert_eq!(integer.solution.unwrap().values, vec![1., 1., 0.]);
    let continuous = solve_integer(&model, &[1], &MipOptions::default()).unwrap();
    assert_eq!(continuous.status, MipStatus::Optimal);
    let continuous = continuous.solution.unwrap();
    assert_eq!(continuous.objective, -0.25);
    assert_eq!(continuous.values, vec![1., 0., 0.25]);
    assert_eq!(model.upper(), &[1., 1., 0.25]);
}
