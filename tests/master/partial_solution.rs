#![cfg(feature = "clp")]
use baldes::inputs::{Instance, Rounding};
use baldes::master::{
    cg::Status,
    formulation::Row,
    model::{
        self,
        separation::{CapacityCutConfig, KPathCutConfig, KnapsackCutConfig},
        Options, Output, Problem, RowCoefficients, RyanFosterMember,
    },
    preprocessing::Bounds,
    RowSense,
};

fn fixture(family: usize) -> (Problem<2>, Options, Vec<i32>, Vec<i32>) {
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
    let problem = Problem::set_partitioning(graph, 3.).unwrap();
    let mut options = Options::default();
    options.cg.stabilize = false;
    match family {
        1 => options.separation.rank_one.enabled = true,
        2 => options.separation.capacity = Some(CapacityCutConfig::new(vec![0., 1., 1., 1.], 2.)),
        3 => options.separation.kpath = Some(KPathCutConfig::new(vec![0., 1., 1., 1.], 2.)),
        4 => {
            options.separation.knapsack = Some(KnapsackCutConfig::new(vec![
                baldes::cuts::knapsack::KnapsackRow::new(0, 1, 1., 3),
            ]))
        }
        _ => {}
    }
    (problem, options, pair, single)
}

/// Registry index of the priced route with exactly these input arc IDs.
fn route_index(output: &Output<2>, arcs: &[i32]) -> usize {
    output
        .routes
        .iter()
        .position(|r| r.route.arcs == arcs)
        .unwrap_or_else(|| panic!("route {arcs:?} was not priced: {:?}", output.routes))
}

fn positive_routes(output: &Output<2>) -> usize {
    output.values.iter().filter(|&&x| x > 1e-9).count()
}

fn fixed_cost(output: &Output<2>) -> f64 {
    output
        .fixed_routes
        .iter()
        .map(|f| f.route.route.original_cost * f.quantity)
        .sum()
}

#[test]
fn fixed_routes_keep_all_cut_families_costs_and_parent_isolation() {
    for family in 0..5 {
        let (problem, options, pair, single) = fixture(family);
        let parent = model::solve(problem, options.clone()).unwrap();
        if family > 0 {
            assert!(parent.cuts > 0, "family {family}");
        }
        let fixed = parent
            .snapshot
            .fix_route(route_index(&parent, &pair), 1.)
            .unwrap()
            .solve(options.clone())
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
        assert_eq!(fixed.fixed_routes.len(), 1);
        assert!((fixed_cost(&fixed) - (2. + 2f64.sqrt())).abs() < 1e-8);
        assert_eq!(positive_routes(&fixed), 1);
        let free: f64 = fixed
            .routes
            .iter()
            .zip(&fixed.values)
            .map(|(r, q)| r.route.original_cost * q)
            .sum();
        assert!((free + fixed_cost(&fixed) - expected).abs() < 1e-8);
        let complete = fixed
            .snapshot
            .fix_route(route_index(&fixed, &single), 1.)
            .unwrap()
            .solve(options.clone())
            .unwrap();
        assert_eq!(complete.master.status, Status::Optimal, "{complete:?}");
        assert!((complete.master.lp - expected).abs() < 1e-7);
        assert_eq!(complete.fixed_routes.len(), 2);
        assert_eq!(positive_routes(&complete), 0);
        let replay = fixed.snapshot.solve(options.clone()).unwrap();
        assert!((replay.master.lp - fixed.master.lp).abs() < 1e-7);
        assert_eq!(replay.fixed_routes.len(), 1);
        let original = parent.snapshot.solve(options.clone()).unwrap();
        assert!((original.master.lp - parent.master.lp).abs() < 1e-7);
        assert!(original.fixed_routes.is_empty());
        assert!(parent.fixed_routes.is_empty());
    }
}

fn assert_proven_infeasible(output: &Output<2>) {
    assert_eq!(output.master.status, Status::Infeasible, "{output:?}");
    assert_eq!(output.master.bound, f64::INFINITY, "{output:?}");
}

#[test]
fn residual_branches_and_invalid_fixings_do_not_corrupt_snapshots() {
    let (problem, options, pair, _) = fixture(2);
    let arc = problem.networks[0]
        .graph
        .arcs
        .iter()
        .position(|a| a.id == pair[0])
        .unwrap();
    let arcs = problem.networks[0].graph.arcs.len();
    let parent = model::solve(problem, options.clone()).unwrap();
    let index = route_index(&parent, &pair);
    for (route, quantity) in [(index, 0.), (index, 0.5), (index, -1.), (999_999, 1.)] {
        assert!(parent.snapshot.fix_route(route, quantity).is_err());
    }
    let child = parent.snapshot.fix_route(index, 1.).unwrap();
    let mut coefficients = vec![0.; arcs];
    coefficients[arc] = 1.;
    let branches = [
        child
            .with_row(
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
            .unwrap(),
        child
            .with_ryan_foster(
                &[RyanFosterMember {
                    network: 0,
                    first: 1,
                    second: 2,
                }],
                false,
            )
            .unwrap(),
        child
            .with_ryan_foster(
                &[RyanFosterMember {
                    network: 0,
                    first: 1,
                    second: 3,
                }],
                true,
            )
            .unwrap(),
        child
            .with_multiplicity(
                0,
                Bounds {
                    lower: 0.,
                    upper: 0.,
                },
            )
            .unwrap(),
    ];
    for branch in branches {
        assert_proven_infeasible(&branch.solve(options.clone()).unwrap());
    }
    let repeated = parent
        .snapshot
        .fix_route(index, 2.)
        .unwrap()
        .solve(options.clone())
        .unwrap();
    assert_proven_infeasible(&repeated);
    assert!((parent.snapshot.solve(options).unwrap().master.lp - parent.master.lp).abs() < 1e-7);
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
