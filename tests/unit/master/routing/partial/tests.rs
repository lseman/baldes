use super::*;
use crate::inputs::{Instance, Rounding};
fn fixture(family: usize) -> (RoutingContext, RootOptions, Vec<i32>, Vec<i32>) {
    let instance=Instance::parse("triangle\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n3 1 1 1 0 100 0\n").unwrap();
    let graph = instance.graph(3, Rounding::None).unwrap();
    let path = |nodes: &[usize]| {
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
    let pair = path(&[0, 1, 2, graph.sink]);
    let single = path(&[0, 3, graph.sink]);
    let problem = RoutingContext::set_partitioning(graph, 3.).unwrap();
    let mut options = RootOptions::default();
    options.capture_state = true;
    options.enumerate = false;
    options.cg.stabilize = false;
    options = match family {
        1 => options.with_rank_one_cuts(),
        2 => options.with_capacity_cuts(CapacityCutConfig::new(vec![0., 1., 1., 1.], 2.)),
        3 => options.with_kpath_cuts(KPathCutConfig::new(vec![0., 1., 1., 1.], 2.)),
        4 => options.with_knapsack_cuts(KnapsackCutConfig::new(vec![
            crate::cuts::knapsack::KnapsackRow::new(0, 1, 1., 3),
        ])),
        _ => options,
    };
    (problem, options, pair, single)
}
#[test]
fn vehicle_upper_bound_and_fixed_route_subtraction_commute() {
    let (problem, options, pair, _) = fixture(0);
    let graph_id = problem.model.networks[0].graph.graph_id.max(0);
    let parent = solve(problem, options.clone()).unwrap();
    let parent = parent.snapshot.as_ref().unwrap();
    let branch = BranchConstraint::NumberOfVehicles {
        graph_id,
        sense: RowSense::LessEqual,
        rhs: 2.,
    };
    let fixed = [FixedRoute {
        arcs: pair,
        quantity: 1,
    }];
    let branched = inherit_branches(parent, &[branch.clone()], options.clone()).unwrap();
    let (after_branch, master, _) = residual_fork(
        branched.snapshot.as_ref().unwrap(),
        &fixed,
        options.cg.artificial_cost,
    )
    .unwrap();
    assert_eq!(after_branch.model.rows[0].rhs, 1.);
    assert_eq!(master.rhs[0], 1.);
    assert_eq!(master.rhs.len(), 4);
    let partial = inherit_fixed_routes(parent, &fixed, options.clone()).unwrap();
    let (after_fixing, master, _) = partial
        .snapshot
        .as_ref()
        .unwrap()
        .fork(&[branch], options.cg.artificial_cost)
        .unwrap();
    assert_eq!(after_fixing.model.rows[0].rhs, 1.);
    assert_eq!(master.rhs[0], 1.);
    assert_eq!(master.rhs.len(), 4);
    assert_eq!(parent.master.rhs[0], 3.);
    assert_eq!(parent.branch_depth(), 0);
}
#[test]
fn fixed_cut_contributions_shift_physical_rows_once_and_leave_the_parent_intact() {
    for family in 1..5 {
        let (problem, options, pair, _) = fixture(family);
        let parent = solve(problem, options.clone()).unwrap();
        let parent = parent.snapshot.unwrap();
        let route = parent.problem.elementary_route(&pair).unwrap();
        let expected = parent.cut.entries(&parent.pricing, &route, true).unwrap();
        assert!(!expected.is_empty(), "family {family}");
        let (problem, master, _) = residual_fork(
            &parent,
            &[FixedRoute {
                arcs: pair,
                quantity: 1,
            }],
            options.cg.artificial_cost,
        )
        .unwrap();
        for (row, coefficient) in expected {
            assert_eq!(
                master.rhs[row],
                parent.master.rhs[row] - coefficient,
                "family {family}"
            );
        }
        assert_eq!(problem.model.rows[0].rhs, 2.);
        assert_eq!(parent.problem.model.rows[0].rhs, 3.);
        assert!(parent.problem.fixed_routes.is_empty());
    }
}
#[cfg(feature = "highs")]
#[test]
fn complete_residual_mip_reconstructs_fixed_routes_and_an_empty_remainder() {
    use crate::master::solvers::mip::{MipOptions, MipStatus};
    for family in 0..5 {
        let (problem, options, pair, single) = fixture(family);
        let parent = solve(problem, options.clone()).unwrap();
        let parent = parent.snapshot.unwrap();
        let partial = inherit_fixed_routes(
            &parent,
            &[FixedRoute {
                arcs: pair,
                quantity: 1,
            }],
            options.clone(),
        )
        .unwrap();
        let mut node = Rc::try_unwrap(partial.snapshot.unwrap()).unwrap();
        node.pricing.enumerate(1e12).unwrap();
        let result = node
            .solve_enumerated_mip(&MipOptions::default())
            .unwrap()
            .unwrap();
        assert_eq!(
            result.mip.status,
            MipStatus::Optimal,
            "family {family}: {result:?}"
        );
        let cost = 2. + 3. * 2f64.sqrt();
        assert!((result.mip.solution.unwrap().objective - cost).abs() < 1e-7);
        assert_eq!(result.route_values.len(), 2);
        let all = inherit_fixed_routes(
            &node,
            &[FixedRoute {
                arcs: single,
                quantity: 1,
            }],
            options,
        )
        .unwrap();
        let mut node = Rc::try_unwrap(all.snapshot.unwrap()).unwrap();
        node.pricing.enumerate(1e12).unwrap();
        let result = node
            .solve_enumerated_mip(&MipOptions::default())
            .unwrap()
            .unwrap();
        assert_eq!(result.mip.status, MipStatus::Optimal);
        assert!((result.mip.bound - cost).abs() < 1e-7);
        assert_eq!(result.route_values.len(), 2);
    }
}

#[test]
fn capacity_cuts_separated_after_fixing_use_the_complete_solution() {
    for complement in [false, true] {
        let instance=Instance::parse("four\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n3 1 1 1 0 100 0\n4 10 0 1 0 100 0\n").unwrap();
        let graph = instance.graph(3, Rounding::None).unwrap();
        let arcs = vec![
            graph
                .arcs
                .iter()
                .find(|a| a.tail == 0 && a.head == 4)
                .unwrap()
                .id,
            graph
                .arcs
                .iter()
                .find(|a| a.tail == 4 && a.head == graph.sink)
                .unwrap()
                .id,
        ];
        let problem = RoutingContext::set_partitioning(graph, 3.).unwrap();
        let mut config = CapacityCutConfig::new(vec![0., 1., 1., 1., 1.], 2.);
        config.parameters.complement_form = complement;
        let mut options = RootOptions::default().with_capacity_cuts(config);
        options.capture_state = true;
        options.enumerate = false;
        options.cg.stabilize = false;
        let parent = solve(problem, options.clone()).unwrap();
        let parent = parent.snapshot.unwrap();
        assert_eq!(parent.active_cuts(), 0);
        let output =
            inherit_fixed_routes(&parent, &[FixedRoute { arcs, quantity: 1 }], options).unwrap();
        assert!(output.master.cut_rounds > 0, "{output:?}");
        assert!(
            (output.master.lp - (24. + 2f64.sqrt())).abs() < 1e-7,
            "{output:?}"
        );
        let child = output.snapshot.unwrap();
        assert!(child.active_cuts() > 0);
        for (cut, row) in child.cut.capacity_cuts.iter().zip(&child.cut.capacity_rows) {
            if let Some(row) = row {
                let contribution =
                    cut.coefficient(&State::<2>::customers(&child.problem.fixed_routes[0].0));
                assert_eq!(child.master.rhs[*row], cut.rhs as f64 - contribution as f64);
            }
        }
    }
}

#[cfg(feature = "highs")]
#[test]
fn residual_integer_solutions_match_independent_partition_enumeration() {
    use crate::master::solvers::mip::{MipOptions, MipStatus};
    // Enumerate every capacity-two route independently and solve the
    // remaining set partition by DP, without any cut or pricing routines.
    for seed in 0..8 {
        let mut text=String::from("small\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n");
        for v in 1..=4 {
            text.push_str(&format!(
                "{v} {} {} 1 0 100 0\n",
                1 + (v * 7 + seed * 3) % 9,
                1 + (v * 3 + seed * 5) % 7
            ));
        }
        let instance = Instance::parse(&text).unwrap();
        let graph = instance.graph(3, Rounding::None).unwrap();
        let edge = |u, v| {
            graph
                .arcs
                .iter()
                .find(|a| a.tail == u && a.head == v)
                .unwrap()
        };
        let mut routes = vec![];
        for i in 1..=4 {
            routes.push((
                1usize << (i - 1),
                vec![edge(0, i).id, edge(i, graph.sink).id],
                edge(0, i).cost + edge(i, graph.sink).cost,
            ));
            for j in 1..=4 {
                if i != j {
                    routes.push((
                        (1usize << (i - 1)) | (1usize << (j - 1)),
                        vec![edge(0, i).id, edge(i, j).id, edge(j, graph.sink).id],
                        edge(0, i).cost + edge(i, j).cost + edge(j, graph.sink).cost,
                    ));
                }
            }
        }
        let mut dp = [f64::INFINITY; 16];
        dp[0] = 0.;
        for mask in 1..16 {
            for (support, _, cost) in &routes {
                if mask & support == *support {
                    dp[mask] = dp[mask].min(dp[mask ^ support] + cost);
                }
            }
        }
        let problem = RoutingContext::set_partitioning(graph, 3.).unwrap();
        let mut options = RootOptions::default();
        options.capture_state = true;
        options.enumerate = false;
        options.cg.stabilize = false;
        if seed % 2 == 1 {
            options =
                options.with_capacity_cuts(CapacityCutConfig::new(vec![0., 1., 1., 1., 1.], 2.));
        }
        let parent = solve(problem, options.clone()).unwrap();
        let parent = parent.snapshot.unwrap();
        for (support, arcs, cost) in &routes {
            let expected = cost + dp[15 ^ support];
            let output = inherit_fixed_routes(
                &parent,
                &[FixedRoute {
                    arcs: arcs.clone(),
                    quantity: 1,
                }],
                options.clone(),
            )
            .unwrap();
            let mut node = Rc::try_unwrap(output.snapshot.unwrap()).unwrap();
            node.pricing.enumerate(1e12).unwrap();
            let result = node
                .solve_enumerated_mip(&MipOptions::default())
                .unwrap()
                .unwrap();
            assert_eq!(
                result.mip.status,
                MipStatus::Optimal,
                "seed {seed}, path {arcs:?}: {result:?}"
            );
            assert!(
                (result.mip.solution.as_ref().unwrap().objective - expected).abs() < 1e-7,
                "seed {seed}, path {arcs:?}: {result:?}, expected {expected}"
            );
            let mut covered = 0usize;
            for (route, quantity) in result.route_values {
                assert_eq!(quantity, 1.);
                for v in &route.nodes[1..route.nodes.len() - 1] {
                    let bit = 1usize << (*v as usize - 1);
                    assert_eq!(covered & bit, 0);
                    covered |= bit;
                }
            }
            assert_eq!(covered, 15);
        }
    }
}
