use super::*;
use crate::inputs::{Instance, Rounding};

fn rc101(customers: usize) -> Problem {
    let root = env!("CARGO_MANIFEST_DIR");
    let mut instance = Instance::read(format!("{root}/instances/Solomon/RC101.txt")).unwrap();
    instance.truncate(customers).unwrap();
    let graph = instance.graph(8, Rounding::None).unwrap();
    Problem::set_partitioning(graph, customers as f64).unwrap()
}

fn options() -> Options {
    let mut options = Options::default();
    options.cg.stabilize = false;
    options
}

#[test]
fn cut_row_remapping_preserves_both_ryan_foster_branch_rows() {
    let problem = rc101(25);
    let mut configured = options();
    configured.separation.rank_one.enabled = true;
    let root = solve_with_master_variables(problem, vec![], configured.clone()).unwrap();
    let root_state = &root.snapshot.state;
    let protected = super::super::separation::protected_rows(root_state);
    let cut_row = (0..root_state.shared_rows.len())
        .find(|row| !protected.contains(&(root_state.shared_offset + row)))
        .expect("expected a removable cut row");
    let graph = &root.snapshot.state.oracles[0].network.graph;
    let witness = &root.snapshot.state.routes[0].route;
    let arc_member = ArcSetMember {
        network: 0,
        first: vec![witness.arcs[0]],
        second: vec![
            graph
                .arcs
                .iter()
                .find(|arc| !witness.arcs.contains(&arc.id))
                .unwrap()
                .id,
        ],
    };
    let vertex_member = RyanFosterMember {
        network: 0,
        first: 1,
        second: 2,
    };
    let mut child = root
        .snapshot
        .with_ryan_foster(&[vertex_member], true)
        .unwrap()
        .with_arc_set_branch(&[arc_member], true)
        .unwrap();
    let state = &mut child.state;
    let offset = state.shared_offset;
    let vertex_row = state.oracles[0].ryan_foster_rows[0].0;
    let arc_row = state.oracles[0].arc_set_branch_rows[0].0;
    assert!(cut_row < vertex_row && vertex_row < arc_row);
    assert!(super::super::separation::supports_row_cleanup(state));
    let protected = super::super::separation::protected_rows(state);
    assert!(!protected.contains(&(offset + cut_row)));
    assert!(protected.contains(&(offset + vertex_row)));
    assert!(protected.contains(&(offset + arc_row)));
    let original_entries: Vec<_> = state
        .routes
        .iter()
        .map(|route| super::entries(&state.oracles[0], &route.route).unwrap())
        .collect();
    assert!(original_entries
        .iter()
        .any(|entries| entries.iter().any(|&(row, _)| row == arc_row)));
    let mut warm = child.warm.clone().into_owned();
    let columns: Vec<_> = warm
        .cut_artificials
        .iter()
        .filter_map(|&(row, column)| (row == offset + cut_row).then_some(column))
        .collect();
    let (map, column_map) = child.master.remove(&[offset + cut_row], &columns).unwrap();
    super::super::separation::remap_rows(state, &map).unwrap();
    assert_eq!(state.oracles[0].ryan_foster_rows[0].0, vertex_row - 1);
    assert_eq!(state.oracles[0].arc_set_branch_rows[0].0, arc_row - 1);
    for (route, original) in state.routes.iter().zip(original_entries) {
        let expected: Vec<_> = original
            .into_iter()
            .map(|(row, value)| (map[offset + row].unwrap() - offset, value))
            .collect();
        assert_eq!(
            super::entries(&state.oracles[0], &route.route).unwrap(),
            expected
        );
    }
    state.cut_artificial_bases = state
        .cut_artificial_bases
        .iter()
        .filter_map(|(&column, &cost)| Some((column_map[column]?, cost)))
        .collect();
    warm.columns = warm
        .columns
        .into_iter()
        .zip(&column_map)
        .filter_map(|(column, mapped)| mapped.map(|_| column))
        .collect();
    for column in warm
        .columns
        .iter_mut()
        .flatten()
        .chain(&mut warm.generated_pool)
    {
        column.entries = column
            .entries
            .iter()
            .filter_map(|&(row, value)| map[row].map(|row| (row, value)))
            .collect();
    }
    warm.artificials = warm
        .artificials
        .iter()
        .filter_map(|&column| column_map[column])
        .collect();
    warm.cut_artificials = warm
        .cut_artificials
        .iter()
        .filter_map(|&(row, column)| Some((map[row]?, column_map[column]?)))
        .collect();
    child.warm = warm.into();
    let solved = child.solve(configured).unwrap();
    for (route, &value) in solved.routes.iter().zip(&solved.values) {
        if value > 1e-8 {
            let oracle = &solved.snapshot.state.oracles[0];
            assert!(super::entries(oracle, &route.route).unwrap().is_empty());
        }
    }
}

#[test]
fn warm_branched_cleanup_retires_generated_cuts_and_preserves_branches() {
    let (problem, config) = super::super::separation::mixed_arc_problem_and_config();
    let mut configured = options();
    configured.preprocessing.enabled = false;
    configured.separation.rank_one_incidence = Some(config);
    let mut root_options = configured.clone();
    root_options.cg.max_cut_rounds = 0;
    let root = solve_with_master_variables(problem, vec![], root_options).unwrap();
    let oracle = &root.snapshot.state.oracles[0];
    let original_cut_count = oracle.solver.cuts.cuts.len();
    let vertex_member = RyanFosterMember {
        network: oracle.network.id,
        first: 1,
        second: 3,
    };
    let arc_member = ArcSetMember {
        network: oracle.network.id,
        first: vec![0],
        second: vec![7],
    };
    let child = root
        .snapshot
        .with_ryan_foster(&[vertex_member], false)
        .unwrap()
        .with_arc_set_branch(&[arc_member], false)
        .unwrap();
    let mut uncut_options = configured.clone();
    uncut_options.cg.max_cut_rounds = 0;
    let uncut_child = child.fork().unwrap();
    let solved = child.solve(configured).unwrap();
    let uncut = uncut_child.solve(uncut_options).unwrap();
    let oracle = &solved.snapshot.state.oracles[0];
    assert_eq!(uncut.master.status, crate::master::cg::Status::Optimal);
    assert_eq!(solved.master.status, crate::master::cg::Status::Optimal);
    assert!(solved.master.lp >= uncut.master.lp - 1e-6);
    assert!(solved.master.cut_rounds >= 1);
    assert!(oracle
        .solver
        .cuts
        .retired
        .iter()
        .any(|&cut| cut >= original_cut_count));
    assert_eq!(oracle.ryan_foster_rows.len(), 1);
    assert_eq!(oracle.arc_set_branch_rows.len(), 1);
    let protected = super::super::separation::protected_rows(&solved.snapshot.state);
    let offset = solved.snapshot.state.shared_offset;
    assert!(protected.contains(&(offset + oracle.ryan_foster_rows[0].0)));
    assert!(protected.contains(&(offset + oracle.arc_set_branch_rows[0].0)));
    for (route, &value) in solved.routes.iter().zip(&solved.values) {
        if value > 1e-8 && route.network == oracle.network.id {
            assert!(super::entries(oracle, &route.route).unwrap().is_empty());
        }
    }
}

fn prepared(problem: &Problem) -> RyanFosterBranching {
    let mut branching = RyanFosterBranching::new(2.);
    branching
        .prepare(
            problem.networks.iter().map(|n| (n.id, &n.graph)),
            None,
            None,
        )
        .unwrap();
    branching
}

#[test]
fn candidates_are_fractional_cooccurrences_and_children_obey_the_pair() {
    let problem = rc101(25);
    let branching = prepared(&problem);
    let root = solve_with_master_variables(problem, vec![], options()).unwrap();
    let proposals = branching.proposals(&root).unwrap();
    assert!(!proposals.is_empty(), "the RC101 root is fractional");
    for p in &proposals {
        assert!(p.value > 1e-6 && p.value < 0.999999, "{}", p.value);
        assert!(p.name.starts_with("PS_"));
        assert_eq!(
            p.origin,
            Some(super::model_strong::ProducerFamily::PackingRyanFoster)
        );
    }
    let best = proposals
        .iter()
        .min_by(|a, b| (a.value - 0.5).abs().total_cmp(&(b.value - 0.5).abs()))
        .unwrap();
    let Restriction::RyanFoster { members } = &best.restriction else {
        unreachable!()
    };
    let member = members[0];
    for together in [false, true] {
        let child = root.snapshot.with_ryan_foster(members, together).unwrap();
        let solved = child.solve(options()).unwrap();
        assert!(solved.master.bound >= root.master.bound - 1e-6);
        let constraint = RyanFosterConstraint {
            first: member.first,
            second: member.second,
            together,
        };
        let oracle = &solved.snapshot.state.oracles[0];
        for (route, &value) in solved.routes.iter().zip(&solved.values) {
            if value > 1e-9 {
                let vertices: Vec<_> = path_vertices(oracle, &route.route).unwrap().collect();
                assert!(
                    constraint.satisfied_by(vertices.iter().copied()),
                    "{together} {vertices:?}"
                );
            }
        }
    }
    // The parent snapshot is unchanged by forming children.
    assert!(root.snapshot.state.oracles[0].ryan_foster_rows.is_empty());
}

#[test]
fn several_networks_require_packing_sets_and_customer_vertices() {
    let mut problem = rc101(5);
    let mut second = problem.networks[0].clone();
    second.id = 1;
    problem.networks.push(second);
    let graphs = || problem.networks.iter().map(|n| (n.id, &n.graph));
    assert!(RyanFosterBranching::new(2.)
        .prepare(graphs(), None, None)
        .is_err());
    let mut labelled = RyanFosterBranching::new(2.);
    labelled.sets = vec![
        NetworkPackingAssignment::new(0, vec![(1, 0), (2, 1)]),
        NetworkPackingAssignment::new(1, vec![(1, 0)]),
    ];
    labelled.prepare(graphs(), None, None).unwrap();
    // This layout needs set-valued membership rather than a direct vertex pair.
    assert!(labelled.members(0, 1).is_none());
    labelled.sets = vec![NetworkPackingAssignment::new(0, vec![(0, 0)])];
    assert!(labelled.prepare(graphs(), None, None).is_err());
    assert!(RyanFosterBranching::new(f64::NAN)
        .prepare(graphs().take(1), None, None)
        .is_err());
}

fn vertex_set_fixture(
    multiple_vertices: bool,
) -> (Problem<2>, RyanFosterBranching, Vec<NetworkRoute>) {
    use crate::graph::Window;
    let mut networks = vec![];
    let mut registrations = vec![];
    let mut routes = vec![];
    for id in [7, 42, 99, 101] {
        let mut labels = match id {
            7 => vec![(1, 0), (3, 1)],
            42 => vec![(1, 0)],
            99 => vec![(3, 1)],
            _ => vec![(2, 2)],
        };
        if multiple_vertices && matches!(id, 7 | 42) {
            labels.push((2, 0));
        }
        let paths = match id {
            7 => {
                let mut paths = vec![vec![], vec![1], vec![3], vec![1, 3]];
                if multiple_vertices {
                    paths.extend([vec![2], vec![2, 3]]);
                }
                paths
            }
            42 if multiple_vertices => vec![vec![], vec![1], vec![2]],
            42 => vec![vec![], vec![1]],
            99 => vec![vec![], vec![3]],
            _ => vec![vec![2]],
        };
        let mut edges = BTreeSet::new();
        for path in paths {
            let vertices: Vec<_> = std::iter::once(0).chain(path).chain([4]).collect();
            edges.extend(vertices.windows(2).map(|edge| (edge[0], edge[1])));
        }
        let mut graph = Graph::<2>::new(5, 0).unwrap();
        for (index, (tail, head)) in edges.into_iter().enumerate() {
            graph
                .add_arc_with_id(
                    10 + 7 * index as i32,
                    tail,
                    head,
                    if tail == 0 && id != 101 { 1. } else { 0. },
                )
                .unwrap();
        }
        let arc_entries = graph
            .arcs
            .iter()
            .map(|arc| {
                labels
                    .iter()
                    .find(|&&(vertex, set)| vertex == arc.head && set < 2)
                    .map_or_else(Vec::new, |&(_, set)| vec![(set, 1.)])
            })
            .collect();
        // Enumerate the tiny DAG independently of the pricing engine.
        fn visit(graph: &Graph<2>, vertex: usize, path: &mut Vec<i32>, paths: &mut Vec<Vec<i32>>) {
            if vertex == graph.sink {
                paths.push(path.clone());
                return;
            }
            for arc in graph.arcs.iter().filter(|arc| arc.tail == vertex) {
                path.push(arc.id);
                visit(graph, arc.head, path, paths);
                path.pop();
            }
        }
        let mut paths = vec![];
        visit(&graph, graph.source, &mut vec![], &mut paths);
        routes.extend(paths.into_iter().map(|arcs| NetworkRoute {
            network: id,
            route: graph.elementary_route(&arcs).unwrap().unwrap(),
        }));
        networks.push(Network {
            id,
            graph,
            multiplicity: Window::new(
                if id == 101 { 1. } else { 0. },
                if id == 7 { 0.5 } else { 1. },
            )
            .unwrap(),
            arc_entries,
            route_entries: vec![],
            activity_entries: vec![],
        });
        registrations.push(NetworkPackingAssignment::new(id, labels));
    }
    let problem = Problem {
        rows: vec![
            Row {
                sense: RowSense::Equal,
                rhs: 1.
            };
            2
        ],
        networks,
    };
    let mut branching = RyanFosterBranching::new(100.);
    branching.sets = registrations;
    branching
        .prepare(
            problem.networks.iter().map(|n| (n.id, &n.graph)),
            None,
            None,
        )
        .unwrap();
    (problem, branching, routes)
}

fn visits_vertex_pair(branching: &RyanFosterBranching, route: &NetworkRoute) -> (bool, bool) {
    let labels = &branching.prepared.labels[&route.network];
    let contains = |set| {
        route
            .route
            .nodes
            .iter()
            .any(|&v| labels[v as usize] == Some(set))
    };
    (contains(0), contains(1))
}

#[test]
fn vertex_set_children_preserve_missing_and_multiple_members_through_pricing_and_enumeration() {
    for multiple_vertices in [false, true] {
        for mode in [
            crate::master::cg::BoundMode::Numerical,
            crate::master::cg::BoundMode::Certified,
        ] {
            let (problem, branching, routes) = vertex_set_fixture(multiple_vertices);
            let mut configured = options();
            configured.cg.bound_mode = mode;
            configured.preprocessing.enabled = false;
            configured.reduced_cost_fixing = false;
            configured.gap_enumeration = false;
            let root = solve_with_master_variables(problem, vec![], configured.clone()).unwrap();
            assert!((root.master.lp - 1.5).abs() < 1e-6);
            let proposals = branching.proposals(&root).unwrap();
            let proposal = proposals
                .iter()
                .find(|p| p.name == "PS_0_1")
                .expect("fractional pair must not be discarded");
            assert!((proposal.value - 0.5).abs() < 1e-6);
            let Restriction::ArcSetRyanFoster { members } = &proposal.restriction else {
                panic!("expected set-valued restrictions")
            };
            assert_eq!(members.len(), 3);
            assert!(members
                .iter()
                .any(|m| m.network == 42 && m.second.is_empty()));
            assert!(members
                .iter()
                .any(|m| m.network == 99 && m.first.is_empty()));
            for together in [false, true] {
                let accepts = |route: &NetworkRoute| {
                    let (a, b) = visits_vertex_pair(&branching, route);
                    if together {
                        a == b
                    } else {
                        !(a && b)
                    }
                };
                let child = root
                    .snapshot
                    .with_arc_set_branch(members, together)
                    .unwrap();
                let row = child.state.shared_rows.len() - 1;
                for route in &routes {
                    let oracle = child
                        .state
                        .oracles
                        .iter()
                        .find(|o| o.network.id == route.network)
                        .unwrap();
                    assert_eq!(
                        oracle
                            .solver
                            .route_satisfies_relaxation(&route.route.nodes, &route.route.arcs)
                            .unwrap(),
                        accepts(route) || !crate::master::branching::RYAN_FOSTER_IN_PRICING,
                        "route validation: {together} {} {:?}",
                        route.network,
                        route.route.nodes,
                    );
                    let coefficient = super::entries(oracle, &route.route)
                        .unwrap()
                        .into_iter()
                        .find(|&(r, _)| r == row)
                        .map_or(0., |(_, value)| value);
                    assert_eq!(
                        coefficient,
                        if accepts(route) { 0. } else { 1. },
                        "{together} {} {:?}",
                        route.network,
                        route.route.nodes
                    );
                }
                let complete = child.enumerate(&EnumerationOptions::default()).unwrap();
                assert!(
                    complete.enumeration_report().complete,
                    "{:?}",
                    complete.enumeration_report()
                );
                let actual: BTreeSet<_> = complete
                    .state
                    .universe
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|&tag| {
                        let route = &complete.state.routes[tag];
                        (route.network, route.route.arcs.clone())
                    })
                    .collect();
                let expected: BTreeSet<_> = routes
                    .iter()
                    .filter(|route| accepts(route))
                    .map(|route| (route.network, route.route.arcs.clone()))
                    .collect();
                assert_eq!(
                    actual, expected,
                    "together={together}, multi={multiple_vertices}"
                );
                let solved = child.fork().unwrap().solve(configured.clone()).unwrap();
                if together {
                    // Only the half-capacity network can visit both sets.
                    assert!(
                        super::super::model_strong::proven_infeasible(&solved),
                        "{:?}",
                        solved.master
                    );
                } else {
                    if crate::master::branching::RYAN_FOSTER_IN_PRICING {
                        assert_eq!(solved.master.status, crate::master::cg::Status::Optimal);
                    } else {
                        assert!(matches!(
                            solved.master.status,
                            crate::master::cg::Status::Optimal | crate::master::cg::Status::Stalled
                        ));
                        assert!(solved.master.bound <= 2. + 1e-6);
                    }
                    assert!((solved.master.lp - 2.).abs() < 1e-6);
                    for (route, &value) in solved.routes.iter().zip(&solved.values) {
                        if value > 1e-8 {
                            assert!(accepts(route));
                        }
                    }
                }
                assert!(child
                    .state
                    .oracles
                    .iter()
                    .find(|o| o.network.id == 101)
                    .unwrap()
                    .arc_set_branch_rows
                    .is_empty());
            }
            assert!(root
                .snapshot
                .state
                .oracles
                .iter()
                .all(|o| o.arc_set_branch_rows.is_empty() && o.ryan_foster_rows.is_empty()));
            assert!((root.snapshot.solve(configured).unwrap().master.lp - 1.5).abs() < 1e-6);
        }
    }
}

#[test]
fn vertex_set_tree_bounds_respect_independent_integer_route_covers() {
    for multiple_vertices in [false, true] {
        let (problem, branching, routes) = vertex_set_fixture(multiple_vertices);
        let mut optimum = f64::INFINITY;
        for mask in 0..1_usize << routes.len() {
            let mut cover = [0; 2];
            let mut counts = BTreeMap::<i32, usize>::new();
            let mut cost = 0.;
            for (index, route) in routes.iter().enumerate() {
                if mask & (1 << index) == 0 {
                    continue;
                }
                let (a, b) = visits_vertex_pair(&branching, route);
                cover[0] += usize::from(a);
                cover[1] += usize::from(b);
                *counts.entry(route.network).or_default() += 1;
                cost += route.route.original_cost;
            }
            if cover == [1, 1]
                && problem.networks.iter().all(|n| {
                    let quantity = counts.get(&n.id).copied().unwrap_or(0) as f64;
                    quantity >= n.multiplicity.lower && quantity <= n.multiplicity.upper
                })
            {
                optimum = optimum.min(cost);
            }
        }
        assert_eq!(optimum, 2.);
        for mode in [
            crate::master::cg::BoundMode::Numerical,
            crate::master::cg::BoundMode::Certified,
        ] {
            for strong in [false, true] {
                let mut configured = ModelTreeOptions::default();
                configured.node.cg.stabilize = false;
                configured.node.cg.bound_mode = mode;
                configured.node.preprocessing.enabled = false;
                configured.node.reduced_cost_fixing = false;
                configured.node.gap_enumeration = false;
                configured.branching_registry.branching =
                    crate::master::tree::NetworkBranchingPolicy::Phased;
                configured.branching_registry.ryan_foster = Some(branching.clone());
                configured.branching_registry.strong_branching = strong.then(Default::default);
                let output = crate::master::tree::solve(problem.clone(), configured).unwrap();
                assert!(output.bound <= optimum + 1e-6, "{output:?}");
                if crate::master::branching::RYAN_FOSTER_IN_PRICING {
                    assert_eq!(
                        output.status,
                        crate::master::tree::TreeStatus::Optimal,
                        "{output:?}"
                    );
                }
                if output.status == crate::master::tree::TreeStatus::Optimal {
                    assert!((output.incumbent - optimum).abs() < 1e-6, "{output:?}");
                    assert!((output.bound - optimum).abs() < 1e-6, "{output:?}");
                } else {
                    assert_eq!(
                        output.status,
                        crate::master::tree::TreeStatus::Incomplete,
                        "{output:?}"
                    );
                    assert!(output.open_nodes > 0, "{output:?}");
                }
                if strong {
                    assert!(output
                        .mapped_branch_evaluations
                        .iter()
                        .any(|e| e.name == "PS_0_1"));
                }
            }
        }
    }
}

#[test]
fn vertex_set_branches_keep_fixed_reductions_exact_when_the_child_is_infeasible() {
    let (problem, branching, _) = vertex_set_fixture(true);
    for mode in [
        crate::master::cg::BoundMode::Numerical,
        crate::master::cg::BoundMode::Certified,
    ] {
        let mut configured = options();
        configured.cg.bound_mode = mode;
        configured.preprocessing.enabled = false;
        configured.reduced_cost_fixing = false;
        configured.gap_enumeration = false;
        let root =
            solve_with_master_variables(problem.clone(), vec![], configured.clone()).unwrap();
        let proposals = branching.proposals(&root).unwrap();
        let Restriction::ArcSetRyanFoster { members } = &proposals
            .iter()
            .find(|p| p.name == "PS_0_1")
            .unwrap()
            .restriction
        else {
            panic!("expected set-valued restrictions")
        };
        let complete = root
            .snapshot
            .enumerate(&EnumerationOptions::default())
            .unwrap();
        let tag = complete
            .state
            .universe
            .as_ref()
            .unwrap()
            .iter()
            .copied()
            .find(|&tag| {
                let route = &complete.state.routes[tag];
                route.network == 42 && route.route.nodes == [0, 1, 4]
            })
            .unwrap();
        let fixed = complete.fix_route(tag, 1.).unwrap();
        let child = fixed.with_arc_set_branch(members, true).unwrap();
        let route = &child.state.fixed_routes[0].route;
        let oracle = child
            .state
            .oracles
            .iter()
            .find(|o| o.network.id == route.network)
            .unwrap();
        assert!(!super::super::lifecycle::integer_route_feasible(oracle, &route.route).unwrap());
        assert!(super::super::bounds::exact_fixed_reduction(&child.state, &child.master).unwrap());
        let solved = child.solve(configured.clone()).unwrap();
        assert!(
            super::super::model_strong::proven_infeasible(&solved),
            "{:?}",
            solved.master
        );
        let restored = fixed.solve(configured).unwrap();
        assert_eq!(restored.master.status, crate::master::cg::Status::Optimal);
        assert!((restored.master.lp - 2.).abs() < 1e-6);
    }
}

#[test]
fn ryan_foster_tree_preserves_default_optimum_bounds_and_open_nodes() {
    let problem = rc101(25);
    let solve = |ryan_foster: Option<RyanFosterBranching>| {
        let mut configured = ModelTreeOptions::default();
        configured.branching_registry.mapped_variables = problem.arc_flow_variables(1.);
        configured.branching_registry.ryan_foster = ryan_foster;
        configured.node_limit = 2000;
        crate::master::tree::solve(problem.clone(), configured).unwrap()
    };
    let default = solve(None);
    let paired = solve(Some(RyanFosterBranching::new(2.)));
    assert_eq!(default.status, crate::master::tree::TreeStatus::Optimal);
    assert!(paired.processed_nodes > 1);
    assert!(paired.bound <= default.incumbent + 1e-6, "{paired:?}");
    if crate::master::branching::RYAN_FOSTER_IN_PRICING {
        assert_eq!(paired.status, crate::master::tree::TreeStatus::Optimal);
    }
    if paired.status == crate::master::tree::TreeStatus::Optimal {
        assert!((default.incumbent - paired.incumbent).abs() < 1e-6);
    } else {
        assert_eq!(paired.status, crate::master::tree::TreeStatus::Incomplete);
        assert!(paired.open_nodes > 0);
    }
}

#[test]
fn explicit_incidence_candidates_install_arc_sensitive_children() {
    let (problem, config) = super::super::separation::mixed_arc_problem_and_config();
    let mut branching = RyanFosterBranching::new(2.);
    branching
        .prepare(
            problem.networks.iter().map(|n| (n.id, &n.graph)),
            None,
            Some(&config),
        )
        .unwrap();
    let mut root = solve_with_master_variables(problem, vec![], options()).unwrap();
    let mut route = root.routes[0].route.clone();
    route.arcs = vec![0, 3, 15];
    root.routes = vec![NetworkRoute { network: 7, route }];
    root.values = vec![0.5];
    let proposals = branching.proposals(&root).unwrap();
    let proposal = proposals.iter().find(|p| p.name == "PS_0_2").unwrap();
    let Restriction::ArcSetRyanFoster { members } = &proposal.restriction else {
        panic!("expected arc-set branch")
    };
    assert_eq!(members.len(), 2);
    for together in [false, true] {
        let child = root
            .snapshot
            .with_arc_set_branch(members, together)
            .unwrap();
        let row = child.state.shared_rows.len() - 1;
        for oracle in &child.state.oracles {
            let constraint = &oracle.arc_set_branch_rows.last().unwrap().1;
            for original in root
                .snapshot
                .state
                .routes
                .iter()
                .filter(|r| r.network == oracle.network.id)
            {
                let entries = super::entries(oracle, &original.route).unwrap();
                let coefficient = entries.iter().any(|&(r, _)| r == row);
                assert_eq!(
                    coefficient,
                    !constraint.satisfied_by(original.route.arcs.iter().copied())
                );
            }
        }
        let solved = child.solve(options()).unwrap();
        for (route, &value) in solved.routes.iter().zip(&solved.values) {
            if value > 1e-8 {
                let oracle = solved
                    .snapshot
                    .state
                    .oracles
                    .iter()
                    .find(|o| o.network.id == route.network)
                    .unwrap();
                let constraint = &oracle.arc_set_branch_rows.last().unwrap().1;
                assert!(constraint.satisfied_by(route.route.arcs.iter().copied()));
            }
        }
        assert!(root
            .snapshot
            .state
            .oracles
            .iter()
            .all(|o| o.arc_set_branch_rows.is_empty()));
    }
}

#[test]
fn explicit_incidence_candidate_values_match_native_capture() {
    use crate::cuts::separation::{ArcData, GraphData, Parameters, VertexData};
    let capture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/cuts/ryan_foster_candidate_values_native.json"
    ))
    .unwrap();
    let case: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/cuts/ryan_foster_candidate_model_native.json"
    ))
    .unwrap();
    let problem = crate::master::model::format::Model::from_json(&case["model"].to_string())
        .unwrap()
        .problem::<2>()
        .unwrap();
    let graphs = problem
        .networks
        .iter()
        .map(|network| {
            let graph = &network.graph;
            let vertices = (0..10)
                .map(|id| {
                    let sets = if id == 1 {
                        vec![1]
                    } else if (3..=8).contains(&id) {
                        vec![id - 1]
                    } else {
                        vec![]
                    };
                    VertexData {
                        id,
                        elem_sets: sets.clone(),
                        packing_sets: sets.clone(),
                        covering_sets: sets,
                    }
                })
                .collect();
            let arcs = graph
                .arcs
                .iter()
                .map(|arc| {
                    let sets = if arc.head == 1 && arc.tail != 4 && arc.tail != 5 {
                        vec![0]
                    } else {
                        vec![]
                    };
                    ArcData {
                        id: arc.id,
                        tail: arc.tail as i32,
                        head: arc.head as i32,
                        elem_sets: sets.clone(),
                        packing_sets: sets.clone(),
                        covering_sets: sets,
                    }
                })
                .collect();
            GraphData {
                id: network.id,
                source: 0,
                sink: 9,
                elem_set_count: 8,
                packing_set_count: 8,
                covering_set_count: 8,
                vertices,
                arcs,
                ..Default::default()
            }
        })
        .collect();
    let config = RankOneIncidenceConfig {
        rows: (0..8).collect(),
        graphs,
        parameters: Parameters::default(),
    };
    let mut branching = RyanFosterBranching::new(2.);
    branching
        .prepare(
            problem.networks.iter().map(|n| (n.id, &n.graph)),
            None,
            Some(&config),
        )
        .unwrap();
    let mut output = solve_with_master_variables(problem, vec![], options()).unwrap();
    let template = output.routes[0].route.clone();
    assert!(output.snapshot.state.fixed_routes.is_empty());
    assert_eq!(capture["exit_code"], 0);
    for hit in capture["hits"].as_array().unwrap() {
        output.routes.clear();
        output.values.clear();
        for captured in hit["routes"].as_array().unwrap() {
            let mut route = template.clone();
            route.arcs = captured["arcs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap() as i32)
                .collect();
            output.routes.push(NetworkRoute {
                network: captured["graph"].as_i64().unwrap() as i32,
                route,
            });
            output.values.push(captured["quantity"].as_f64().unwrap());
        }
        let name = format!("PS_{}_{}", hit["pair"][0], hit["pair"][1]);
        let expected = hit["value"].as_f64().unwrap();
        let proposals = branching.proposals(&output).unwrap();
        let actual = proposals
            .iter()
            .find(|p| p.name == name)
            .map_or(0., |p| p.value);
        assert!(
            (actual - expected).abs() < 1e-9,
            "{name}: {actual} != {expected}"
        );
    }
}

#[test]
fn enforcement_switch_changes_pricing_but_preserves_master_branch_and_bounds() {
    let (problem, _, routes) = vertex_set_fixture(false);
    let mut configured = options();
    configured.preprocessing.enabled = false;
    configured.reduced_cost_fixing = false;
    configured.gap_enumeration = false;
    let root = solve_with_master_variables(problem, vec![], configured.clone()).unwrap();
    let member = RyanFosterMember {
        network: 7,
        first: 1,
        second: 3,
    };
    let arc_member = ArcSetMember {
        network: 7,
        first: root.snapshot.state.oracles[0]
            .network
            .graph
            .arcs
            .iter()
            .filter(|a| a.head == 1)
            .map(|a| a.id)
            .collect(),
        second: root.snapshot.state.oracles[0]
            .network
            .graph
            .arcs
            .iter()
            .filter(|a| a.head == 3)
            .map(|a| a.id)
            .collect(),
    };
    for arc_sets in [false, true] {
        let mut child = if arc_sets {
            root.snapshot
                .with_arc_set_branch(std::slice::from_ref(&arc_member), false)
                .unwrap()
        } else {
            root.snapshot.with_ryan_foster(&[member], false).unwrap()
        };
        let oracle = &child.state.oracles[0];
        let violating = routes
            .iter()
            .find(|r| r.network == 7 && r.route.nodes.contains(&1) && r.route.nodes.contains(&3))
            .unwrap();
        assert_eq!(
            super::entries(oracle, &violating.route)
                .unwrap()
                .last()
                .unwrap()
                .1,
            1.
        );
        assert!(
            !super::super::lifecycle::integer_route_feasible(oracle, &violating.route).unwrap()
        );
        assert_eq!(
            oracle
                .solver
                .route_satisfies_relaxation(&violating.route.nodes, &violating.route.arcs)
                .unwrap(),
            !crate::master::branching::RYAN_FOSTER_IN_PRICING
        );
        let mut duals = vec![0.; child.master.rhs.len()];
        duals[child.state.shared_offset] = 10.;
        duals[child.state.shared_offset + 1] = 10.;
        *duals.last_mut().unwrap() = -100.;
        let priced = child.state.price(0, &duals).unwrap();
        let expected_minimum = if crate::master::branching::RYAN_FOSTER_IN_PRICING {
            -9.
        } else {
            -19.
        };
        assert!(
            (priced.minimum - expected_minimum).abs() < 1e-6,
            "{priced:?}"
        );
        let solved = child.solve(configured.clone()).unwrap();
        // Independent covers: apart routes cost at least two; singleton routes attain two.
        assert!(
            solved.master.numerical_bound <= 2. + 1e-6,
            "{:?}",
            solved.master
        );
        assert!(solved.master.lp >= 2. - 1e-6, "{:?}", solved.master);
        if solved.master.status == cg::Status::Optimal {
            assert!(
                (solved.master.bound - 2.).abs() < 1e-6,
                "{:?}",
                solved.master
            );
        }
        assert_ne!(solved.master.status, cg::Status::Infeasible);
    }
    assert!(root.snapshot.state.oracles[0]
        .solver
        .ryan_foster_constraints()
        .is_empty());
}
