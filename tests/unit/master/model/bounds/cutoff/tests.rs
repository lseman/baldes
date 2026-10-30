use super::*;
fn fixture(costs: [[f64; 2]; 2], lower: f64, upper: f64) -> super::super::super::NodeSnapshot<7> {
    let networks = costs
        .into_iter()
        .enumerate()
        .map(|(i, costs)| {
            let mut graph = Graph::<7>::new(4, 0).unwrap();
            for (tail, head, cost) in [(0, 1, costs[0]), (1, 3, 0.), (0, 2, costs[1]), (2, 3, 0.)] {
                graph.add_arc(tail, head, cost).unwrap();
            }
            Network {
                id: i as i32 + 1,
                graph,
                multiplicity: Window { lower, upper },
                arc_entries: vec![vec![(0, 1.)], vec![], vec![(0, 1.)], vec![]],
                route_entries: vec![],
                activity_entries: vec![],
            }
        })
        .collect();
    let problem = Problem {
        rows: vec![Row {
            sense: RowSense::Equal,
            rhs: 4.,
        }],
        networks,
    };
    let mut options = Options::default();
    options.preprocessing.enabled = false;
    options.cg.artificial_cost = 100.;
    options.cg.stabilize = false;
    solve(problem, options).unwrap().snapshot
}
#[test]
fn forced_copy_ceilings_do_not_double_count_mandatory_routes() {
    let node = fixture([[2., 5.], [3., 6.]], 2., 3.);
    let duals = vec![0.; node.state.rhs.len()];
    let mut work = 1_000_000;
    let plan = cutoff_plan(
        &node.state,
        &node.master,
        &duals,
        14.,
        &mut work,
        Instant::now() + std::time::Duration::from_secs(1),
    )
    .unwrap();
    match &plan[0] {
        NetworkCutoff::Cost {
            arc_cost_lower_bounds,
            arc_ceiling,
            other_contributions,
            route_constant_lower_bound,
            ..
        } => {
            // Feasible selection: cheap+expensive in network one, two cheap
            // routes in network two. Objective 13 improves cutoff 14.
            assert_eq!(*other_contributions, 8.);
            assert_eq!(*arc_ceiling, 6.);
            assert_eq!(*route_constant_lower_bound, 0.);
            assert_eq!(arc_cost_lower_bounds[2], 5.);
            assert!(arc_cost_lower_bounds[2] <= *arc_ceiling);
            // Subtracting the common LB 10 would incorrectly exclude it.
            assert!(arc_cost_lower_bounds[2] > 14. - 10.);
        }
        _ => panic!("missing route cutoff"),
    }
}
#[test]
fn common_point_ceilings_retain_every_exhaustive_improving_integer_selection() {
    let mut checked = 0;
    for costs in [
        [[2., 5.], [3., 6.]],
        [[-2., 5.], [1., 4.]],
        [[-2., -1.], [-3., 4.]],
    ] {
        let node = fixture(costs, 1.5, 3.);
        for dual in [-3., -1., 0., 1., 3.] {
            let mut duals = vec![0.; node.state.rhs.len()];
            duals[node.state.shared_offset] = dual;
            for incumbent in [-1., 2., 7., 14.] {
                let mut work = 1_000_000;
                let plan = cutoff_plan(
                    &node.state,
                    &node.master,
                    &duals,
                    incumbent,
                    &mut work,
                    Instant::now() + std::time::Duration::from_secs(1),
                )
                .unwrap();
                for a in 0..=3 {
                    for b in 0..=3 {
                        for c in 0..=3 {
                            for d in 0..=3 {
                                let counts = [a, b, c, d];
                                if a + b < 2
                                    || a + b > 3
                                    || c + d < 2
                                    || c + d > 3
                                    || a + b + c + d != 4
                                {
                                    continue;
                                }
                                let objective = a as f64 * costs[0][0]
                                    + b as f64 * costs[0][1]
                                    + c as f64 * costs[1][0]
                                    + d as f64 * costs[1][1];
                                for (network, p) in plan.iter().enumerate() {
                                    let NetworkCutoff::Cost {
                                        network: id,
                                        arc_cost_lower_bounds,
                                        arc_ceiling,
                                        other_contributions,
                                        route_constant_lower_bound,
                                    } = p
                                    else {
                                        panic!("unexpected empty network")
                                    };
                                    assert_eq!(*id, network as i32 + 1);
                                    for path in 0..2 {
                                        if counts[network * 2 + path] == 0 {
                                            continue;
                                        }
                                        let rc = costs[network][path] - dual;
                                        assert!(*other_contributions+rc <= objective, "invalid forced bound: {p:?}, counts={counts:?}, costs={costs:?}, dual={dual}");
                                        assert!(
                                            arc_cost_lower_bounds[2 * path]
                                                + route_constant_lower_bound
                                                <= rc
                                        );
                                        if objective < incumbent {
                                            assert!(
                                                arc_cost_lower_bounds[2 * path] <= *arc_ceiling
                                            );
                                        }
                                        checked += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(checked, 1440);
}
#[test]
fn missing_proofs_work_and_deadlines_never_return_partial_network_plans() {
    let mut node = fixture([[2., 5.], [3., 6.]], 0., 3.);
    let duals = vec![0.; node.state.rhs.len()];
    let deadline = Instant::now() + std::time::Duration::from_secs(1);
    assert!(cutoff_plan(&node.state, &node.master, &duals, 14., &mut 0, deadline).is_none());
    assert!(cutoff_plan(
        &node.state,
        &node.master,
        &duals,
        14.,
        &mut 100000,
        Instant::now()
    )
    .is_none());
    assert!(cutoff_plan(
        &node.state,
        &node.master,
        &duals,
        f64::INFINITY,
        &mut 100000,
        deadline
    )
    .is_none());
    assert!(cutoff_plan(
        &node.state,
        &node.master,
        &[f64::NAN],
        14.,
        &mut 100000,
        deadline
    )
    .is_none());
    node.state.oracles[1].network.multiplicity.upper = f64::INFINITY;
    Arc::make_mut(&mut node.state.oracles[1].network.graph).arcs[0].cost = -1.;
    assert!(cutoff_plan(
        &node.state,
        &node.master,
        &duals,
        14.,
        &mut 100000,
        deadline
    )
    .is_none());
}

#[test]
fn paired_front_activity_row_compilation_preserves_the_selected_value_binding() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    let mut source = corpus["cases"][0]["model"].clone();
    source["rows"] = serde_json::json!([{"sense":"ge","rhs":4.}]);
    source["pricing_variables"] = serde_json::json!([{"network":42,"id":901,"cost":1.,"accumulated_resource":0,"entries":[[0,1.]]}]);
    let model = crate::master::model::format::Model::from_json(&source.to_string()).unwrap();
    let (problem, _) = model.compile::<2>().unwrap();
    assert_eq!(problem.networks[0].activity_entries.len(), 1);
    assert_eq!(problem.networks[0].activity_entries[0].resource, 0);
}

#[test]
fn integer_zero_activity_networks_need_no_source_walk_proof() {
    let mut node = fixture([[2., 5.], [3., 6.]], 0., 0.5);
    for oracle in &mut node.state.oracles {
        oracle.network.multiplicity.upper = 0.5;
    }
    let duals = vec![0.; node.state.rhs.len()];
    let plan = cutoff_plan(
        &node.state,
        &node.master,
        &duals,
        1.,
        &mut 0,
        Instant::now() + std::time::Duration::from_secs(1),
    )
    .unwrap();
    assert!(plan
        .iter()
        .all(|p| matches!(p, NetworkCutoff::Empty { .. })));
}

#[test]
fn source_path_preparation_uses_the_callers_budget_beyond_fixed_path_and_scan_caps() {
    let count = 5000;
    let mut graph = Graph::<2>::new(4, 0).unwrap();
    for _ in 0..count {
        graph.add_arc(0, 1, 0.).unwrap();
    }
    // Each of the5000 prefixes explores100 dead-end parallel arcs before
    // its sink arc. Complete coverage needs more than250000 scan units.
    for _ in 0..100 {
        graph.add_arc(1, 2, 0.).unwrap();
    }
    graph.add_arc(1, 3, 0.).unwrap();
    let problem = Problem {
        rows: vec![],
        networks: vec![Network {
            id: 42,
            graph,
            multiplicity: Window::new(0., 1.).unwrap(),
            arc_entries: vec![vec![]; count + 101],
            route_entries: vec![],
            activity_entries: vec![],
        }],
    };
    let mut options = Options::default();
    options.cg.time_limit = std::time::Duration::ZERO;
    options.preprocessing.enabled = false;
    let node = solve(problem, options).unwrap().snapshot;
    let oracle = &node.state.oracles[0];
    let mut work = 4_000_000;
    let paths = super::columns::physical_paths(
        oracle,
        &mut work,
        Instant::now() + std::time::Duration::from_secs(2),
    )
    .unwrap();
    assert_eq!(paths.len(), count);
    assert_eq!(paths.first().unwrap(), &[0, count + 100]);
    assert_eq!(paths.last().unwrap(), &[count - 1, count + 100]);
    assert!(work < 4_000_000 - 250_000 && work > 2_000_000);

    let mut streamed_work = 4_000_000;
    let mut streamed = Vec::new();
    let streamed_count = super::columns::visit_physical_paths(
        oracle,
        &mut streamed_work,
        Instant::now() + std::time::Duration::from_secs(2),
        |path, _| {
            streamed.push(path.to_vec());
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(streamed_count, count);
    assert_eq!(streamed, paths);
    assert!(streamed_work < 4_000_000 - 250_000 && streamed_work > 2_000_000);

    assert!(super::columns::physical_paths(
        oracle,
        &mut 8,
        Instant::now() + std::time::Duration::from_secs(2)
    )
    .is_none());
}
